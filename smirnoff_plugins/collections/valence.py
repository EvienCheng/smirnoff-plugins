from functools import lru_cache
from typing import Dict, Iterable, Literal, Set, Tuple, Type, Union

from openff.interchange import Interchange
from openff.interchange.components.potentials import Potential
from openff.interchange.interop.openmm._valence import _is_constrained
from openff.interchange.models import PotentialKey, VirtualSiteKey
from openff.interchange.smirnoff._base import SMIRNOFFCollection
from openff.toolkit import Quantity
from openff.toolkit import unit as off_unit
from openff.toolkit.typing.engines.smirnoff.parameters import ParameterHandler
from openmm import openmm

from smirnoff_plugins.handlers.valence import UreyBradleyHandler


@lru_cache
def _cache_urey_bradley_parameter_lookup(
    potential_key: PotentialKey,
    parameter_handler: ParameterHandler,
) -> dict[str, Quantity]:
    parameter = parameter_handler.parameters[potential_key.id]

    return {
        parameter_name: getattr(parameter, parameter_name)
        for parameter_name in ["k", "length"]
    }


class SMIRNOFFUreyBradleyCollection(SMIRNOFFCollection):
    is_plugin: bool = True

    type: Literal["UreyBradleys"] = "UreyBradleys"

    expression: Literal["k/2*(r-length)**2"] = "k/2*(r-length)**2"

    @classmethod
    def allowed_parameter_handlers(cls) -> Iterable[Type[ParameterHandler]]:
        """Return an iterable of allowed types of ParameterHandler classes."""
        return (UreyBradleyHandler,)

    @classmethod
    def supported_parameters(cls) -> Iterable[str]:
        """Return an iterable of supported parameter attributes."""
        return "smirks", "id", "k", "length"

    @classmethod
    def potential_parameters(cls) -> Iterable[str]:
        """Return a subset of `supported_parameters` that are meant to be included in potentials."""
        return "k", "length"

    @classmethod
    def valence_terms(cls, topology):
        """Return all angles in this topology."""
        return [(angle[0], angle[2]) for angle in topology.angles]

    def store_potentials(self, parameter_handler: UreyBradleyHandler) -> None:
        """Store the potentials from the parameter handler."""
        for potential_key in self.key_map.values():
            self.potentials.update(
                {
                    potential_key: Potential(
                        parameters=_cache_urey_bradley_parameter_lookup(
                            potential_key,
                            parameter_handler,
                        ),
                    ),
                },
            )

    def modify_openmm_forces(
        self,
        interchange: Interchange,
        system: openmm.System,
        add_constrained_forces: bool,
        constrained_pairs: Set[Tuple[int, ...]],
        particle_map: Dict[Union[int, "VirtualSiteKey"], int],
    ) -> None:
        # Mainly taken from
        # https://github.com/openforcefield/openff-interchange/blob/83383b8b3af557c167e4a3003495e0e5ffbeff73/openff/interchange/interop/openmm/_valence.py#L50

        harmonic_bond_force = openmm.HarmonicBondForce()
        harmonic_bond_force.setName("UreyBradleyForce")
        system.addForce(harmonic_bond_force)

        has_constraint_handler = "Constraints" in interchange.collections

        for top_key, pot_key in self.key_map.items():
            openff_indices = top_key.atom_indices
            openmm_indices = tuple(particle_map[index] for index in openff_indices)

            if len(openmm_indices) != 2:
                raise ValueError(
                    f"Expected 2 indices for Urey-Bradley potential, got {len(openmm_indices)}: {openmm_indices}",
                )

            if has_constraint_handler and not add_constrained_forces:
                if _is_constrained(
                    constrained_pairs,
                    (openmm_indices[0], openmm_indices[1]),
                ):
                    # This 1-3 length is constrained, so not add a bond force
                    continue

            params = self.potentials[pot_key].parameters
            k = params["k"].m_as(
                off_unit.kilojoule / off_unit.nanometer**2 / off_unit.mol,
            )
            length = params["length"].m_as(off_unit.nanometer)

            harmonic_bond_force.addBond(
                particle1=openmm_indices[0],
                particle2=openmm_indices[1],
                length=length,
                k=k,
            )

from smirnoff_plugins.handlers.valence import HarmonicHeightHandler
from typing import ClassVar

class SMIRNOFFHarmonicHeightCollection(SMIRNOFFCollection):
    is_plugin: ClassVar[bool] = True

    type: ClassVar[Literal["HarmonicHeight"]] = "HarmonicHeight"

    expression: ClassVar[str] = (
        "0.5 * k * (h - h0)^2;"
        "h = abs((Nx*(x2-x1) + Ny*(y2-y1) + Nz*(z2-z1)) / sqrt(Nx^2 + Ny^2 + Nz^2));"
        "Nx = (y3 - y1)*(z4 - z1) - (z3 - z1)*(y4 - y1);"
        "Ny = (z3 - z1)*(x4 - x1) - (x3 - x1)*(z4 - z1);"
        "Nz = (x3 - x1)*(y4 - y1) - (y3 - y1)*(x4 - x1);"
    )

    @classmethod
    def allowed_parameter_handlers(cls) -> Iterable[Type[ParameterHandler]]:
        return (HarmonicHeightHandler,)

    @classmethod
    def supported_parameters(cls) -> Iterable[str]:
        return "smirks", "id", "k", "h0"

    @classmethod
    def potential_parameters(cls) -> Iterable[str]:
        return "k", "h0"

    @classmethod
    def valence_terms(cls, topology):
        unique_terms = []
        seen = set()
        for improper in topology.impropers:
            atoms = tuple(improper)
            central = atoms[1]
            others = sorted([atoms[0], atoms[2], atoms[3]])
            canonical = (others[0], central, others[1], others[2])
            if canonical not in seen:
                seen.add(canonical)
                unique_terms.append(canonical)
        return unique_terms
    
    # def store_potentials(self, parameter_handler: HarmonicHeightHandler) -> None:
    #     # for potential_key in self.key_map.values():
    #     #     param = parameter_handler.parameters[potential_key.id]
    #     #     self.potentials[potential_key] = Potential(
    #     #         parameters={"k": param.k, "h0": param.h0}
    #     #     )

    def store_potentials(self, parameter_handler: HarmonicHeightHandler) -> None:
        seen_params = {}

        for potential_key in self.key_map.values():
            param = parameter_handler.parameters[potential_key.id]

            key_tuple = tuple(
                getattr(param, pname).m_as(unit) if hasattr(getattr(param, pname), "m_as") else getattr(param, pname)
                for pname, unit in [("k", "kilojoule / mole / nanometer**2"), ("h0", "nanometer")]
            )

            if key_tuple not in seen_params:
                seen_params[key_tuple] = Potential(
                    parameters={pname: getattr(param, pname) for pname in self.potential_parameters()}
                )

            self.potentials[potential_key] = seen_params[key_tuple]



    def modify_openmm_forces(
        self,
        interchange: Interchange,
        system: openmm.System,
        add_constrained_forces: bool,
        constrained_pairs: Set[Tuple[int, ...]],
        particle_map: Dict[Union[int, VirtualSiteKey], int],
    ) -> None:
        force = openmm.CustomCompoundBondForce(
            4,
            self.expression
        )
        force.addPerBondParameter("k")
        force.addPerBondParameter("h0")
        force.setName("HarmonicHeight")
        system.addForce(force)

        for top_key, pot_key in self.key_map.items():
            indices = [particle_map[i] for i in top_key.atom_indices]
            params = self.potentials[pot_key].parameters
            k = params["k"].m_as("kilojoule / mole / nanometer**2")
            h0 = params["h0"].m_as("nanometer")

            force.addBond(indices, [k, h0])

from smirnoff_plugins.handlers.valence import LeeKrimmHandler

class SMIRNOFFLeeKrimmCollection(SMIRNOFFCollection):
    type: ClassVar[Literal["LeeKrimm"]] = "LeeKrimm"
    is_plugin: ClassVar[bool] = True
    acts_as: str = "LeeKrimm"

    expression: ClassVar[str] = (
        "V2 * ((abs(h)^t) / (1 - abs(h)^s))^2 + "
        "V4 * ((abs(h)^t) / (1 - abs(h)^s))^4;"
        "h = abs((Nx*(x2-x1) + Ny*(y2-y1) + Nz*(z2-z1)) / sqrt(Nx^2 + Ny^2 + Nz^2));"
        "Nx = (y3 - y1)*(z4 - z1) - (z3 - z1)*(y4 - y1);"
        "Ny = (z3 - z1)*(x4 - x1) - (x3 - x1)*(z4 - z1);"
        "Nz = (x3 - x1)*(y4 - y1) - (y3 - y1)*(x4 - x1);"
    )

    @classmethod
    def allowed_parameter_handlers(cls):
        from smirnoff_plugins.handlers.valence import LeeKrimmHandler
        return (LeeKrimmHandler,)

    @classmethod
    def supported_parameters(cls) -> Iterable[str]:
        return "smirks", "id", "V2", "V4", "t", "s"

    @classmethod
    def potential_parameters(cls) -> Iterable[str]:
        return "V2", "V4", "t", "s"
    
    @classmethod
    def valence_terms(cls, topology):
        return topology.impropers

    def store_potentials(self, parameter_handler):
        for potential_key in self.key_map.values():
            param = parameter_handler.parameters[potential_key.id]
            self.potentials[potential_key] = Potential(
                parameters={
                    "V2": param.V2,
                    "V4": param.V4,
                    "t": param.t * off_unit.dimensionless,
                    "s": param.s * off_unit.dimensionless,
                }
            )

    def modify_openmm_forces(
        self,
        interchange: Interchange,
        system: openmm.System,
        add_constrained_forces: bool,
        constrained_pairs: Set[tuple[int, ...]],
        particle_map: Dict[Union[int, VirtualSiteKey], int],
    ) -> None:
        force = openmm.CustomCompoundBondForce(4, self.expression)
        force.addPerBondParameter("V2")
        force.addPerBondParameter("V4")
        force.addPerBondParameter("t")
        force.addPerBondParameter("s")
        force.setName("LeeKrimm")

        for key, val in self.key_map.items():
            atom_indices = [particle_map[i] for i in key.atom_indices]
            params = self.potentials[val].parameters
            force.addBond(
                atom_indices,
                [
                    params["V2"].m_as("kilojoule / mole"),
                    params["V4"].m_as("kilojoule / mole"),
                    params["t"],
                    params["s"],
                ],
            )

        system.addForce(force)


from smirnoff_plugins.handlers.valence import TwoMinimaHandler

class SMIRNOFFTwoMinimaCollection(SMIRNOFFCollection):
    type: ClassVar[Literal["TwoMinima"]] = "TwoMinima"
    is_plugin: ClassVar[bool] = True
    acts_as: str = "TwoMinima"


    expression: ClassVar[str] = (
        "k1 * (1 + cos(periodicity * theta - phase)) - "
        "k2 * (1 + cos(2 * periodicity * theta + phase));"
        "theta = acos(h_clamped);"
        "h_clamped = min(1, max(-1, h));"
        "h = (Nx*(x2-x1) + Ny*(y2-y1) + Nz*(z2-z1)) / sqrt(Nx^2 + Ny^2 + Nz^2);"
        "Nx = (y3-y1)*(z4-z1) - (z3-z1)*(y4-y1);"
        "Ny = (z3-z1)*(x4-x1) - (x3-x1)*(z4-z1);"
        "Nz = (x3-x1)*(y4-y1) - (y3-y1)*(x4-x1);"
    )

    @classmethod
    def allowed_parameter_handlers(cls):
        from smirnoff_plugins.handlers.valence import TwoMinimaHandler
        return (TwoMinimaHandler,)

    @classmethod
    def supported_parameters(cls):
        return "smirks", "id", "k1", "k2", "periodicity", "phase"

    @classmethod
    def potential_parameters(cls):
        return "k1", "k2", "periodicity", "phase"
    
    @classmethod
    def valence_terms(cls, topology):
        return topology.impropers
    
    
    def store_potentials(self, parameter_handler):
        for potential_key in self.key_map.values():
            param = parameter_handler.parameters[potential_key.id]
            self.potentials[potential_key] = Potential(
                parameters={
                    "k1": param.k1,
                    "k2": param.k2,
                    "periodicity": param.periodicity * off_unit.dimensionless,
                    "phase": param.phase,
                }
            )


    def modify_openmm_forces(
        self,
        interchange,
        system,
        add_constrained_forces: bool,
        constrained_pairs: Set[Tuple[int, ...]],
        particle_map: Dict[Union[int, "VirtualSiteKey"], int],
    ):
        force = openmm.CustomCompoundBondForce(4, self.expression)
        force.addPerBondParameter("k1")
        force.addPerBondParameter("k2")
        force.addPerBondParameter("periodicity")
        force.addPerBondParameter("phase")
        force.setName("TwoMinima")

        for key, val in self.key_map.items():
            atom_indices = [particle_map[i] for i in key.atom_indices]
            params = self.potentials[val].parameters
            force.addBond(
                atom_indices,
                [
                    params["k1"].m_as("kilojoule / mole"),
                    params["k2"].m_as("kilojoule / mole"),
                    params["periodicity"],
                    params["phase"].m_as("radian"),
                ],
            )

        system.addForce(force)
    
from smirnoff_plugins.handlers.valence import HarmonicAngleHandler

class SMIRNOFFHarmonicAngleCollection(SMIRNOFFCollection):
    is_plugin: ClassVar[bool] = True
    type: ClassVar[str] = "HarmonicAngle"

    expression: ClassVar[str] = "0.5 * k * (theta - theta0)^2;"

    @classmethod
    def allowed_parameter_handlers(cls) -> Iterable[Type[ParameterHandler]]:
        return (HarmonicAngleHandler,)

    @classmethod
    def supported_parameters(cls) -> Iterable[str]:
        return "smirks", "id", "k", "theta0"

    @classmethod
    def potential_parameters(cls) -> Iterable[str]:
        return "k", "theta0"

    @classmethod
    def valence_terms(cls, topology):
        """Return all unique angle tuples in canonical order: (atom1, central, atom2)."""
        unique_terms = []
        seen = set()
        for angle in topology.angles:
            central = angle[1]
            others = tuple(sorted([angle[0], angle[2]]))
            canonical = (others[0], central, others[1])
            if canonical not in seen:
                seen.add(canonical)
                unique_terms.append(canonical)
        return unique_terms

    def store_potentials(self, parameter_handler: HarmonicAngleHandler) -> None:
        seen_params = {}

        for potential_key in self.key_map.values():
            param = parameter_handler.parameters[potential_key.id]

            key_tuple = (
                param.k.m_as("kilojoule / mole / radian**2") 
                if hasattr(param.k, "m_as") else param.k,
                param.theta0.m_as("radian") 
                if hasattr(param.theta0, "m_as") else param.theta0,
            )

            if key_tuple not in seen_params:
                seen_params[key_tuple] = Potential(
                    parameters={pname: getattr(param, pname) for pname in self.potential_parameters()}
                )

            self.potentials[potential_key] = seen_params[key_tuple]

    def modify_openmm_forces(
        self,
        interchange: Interchange,
        system: openmm.System,
        add_constrained_forces: bool,
        constrained_pairs: Set[Tuple[int, ...]],
        particle_map: Dict[Union[int, VirtualSiteKey], int],
    ) -> None:
        """Add a harmonic angle force to OpenMM."""
        force = openmm.CustomAngleForce(self.expression)
        force.addPerAngleParameter("k")
        force.addPerAngleParameter("theta0")
        force.setName("HarmonicAngle")
        system.addForce(force)

        for top_key, pot_key in self.key_map.items():
            indices = [particle_map[i] for i in top_key.atom_indices]
            params = self.potentials[pot_key].parameters
            k = params["k"].m_as("kilojoule / mole / radian**2")
            theta0 = params["theta0"].m_as("radian")
            force.addAngle(indices[0], indices[1], indices[2], [k, theta0])