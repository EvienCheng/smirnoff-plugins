import abc

from openff.toolkit import unit
from openmm import CustomCompoundBondForce
from openff.toolkit.typing.engines.smirnoff.parameters import (
    ElectrostaticsHandler,
    IncompatibleParameterError,
    LibraryChargeHandler,
    ParameterAttribute,
    ParameterHandler,
    ParameterType,
    ToolkitAM1BCCHandler,
    VirtualSiteHandler,
    _allow_only,
    vdWHandler,
    ImproperTorsionHandler,
    BondHandler
)


class _CustomBondedHandler(ParameterHandler, abc.ABC):
    """Base class for custom bonded parameter handlers (bonds, angles, torsions)."""

    # cutoff = ParameterAttribute(default=9.0 * unit.angstroms, unit=unit.angstrom)
    
    # periodicity = ParameterAttribute(
    #     default=1, converter=int
    # )

    # periodic_method = ParameterAttribute(
    #     default="cutoff", converter=_allow_only(["cutoff"])
    #     )
    
    # nonperiodic_method = ParameterAttribute(
    #     default="no-cutoff", converter=_allow_only(["no-cutoff"])
    #     )

    # switch_width = ParameterAttribute(default=1.0 * unit.angstroms, unit=unit.angstrom)

    class BondType(ParameterType):
        """A SMIRNOFF bond type

        .. warning :: This API is experimental and subject to change.
        """

        _ELEMENT_NAME = "Bond"

        length = ParameterAttribute(default=None, unit=unit.angstrom)
        k = ParameterAttribute(
            default=None, unit=unit.kilocalorie / unit.mole / unit.angstrom**2
        )

        # # fractional bond order params
        # length_bondorder = MappedParameterAttribute(default=None, unit=unit.angstrom)
        # k_bondorder = MappedParameterAttribute(
        #     default=None, unit=unit.kilocalorie / unit.mole / unit.angstrom**2
        # )

        def __init__(self, **kwargs):
            # these checks enforce mutually-exclusive parameterattribute specifications
            has_k = "k" in kwargs.keys()
            has_k_bondorder = any(["k_bondorder" in key for key in kwargs.keys()])
            has_length = "length" in kwargs.keys()
            has_length_bondorder = any(
                ["length_bondorder" in key for key in kwargs.keys()]
            )

            # Are these errors too general? What about ParametersMissingError/ParametersOverspecifiedError?
            if has_k:
                if has_k_bondorder:
                    raise SMIRNOFFSpecError(
                        "BOTH k and k_bondorder* cannot be specified simultaneously."
                    )
            else:
                if not has_k_bondorder:
                    raise SMIRNOFFSpecError(
                        "Either k or k_bondorder* must be specified."
                    )
            if has_length:
                if has_length_bondorder:
                    raise SMIRNOFFSpecError(
                        "BOTH length and length_bondorder* cannot be specified simultaneously."
                    )
            else:
                if not has_length_bondorder:
                    raise SMIRNOFFSpecError(
                        "Either length or length_bondorder* must be specified."
                    )

            super().__init__(**kwargs)
            
    def check_handler_compatibility(self, other_handler: ParameterHandler):
        """Checks whether this ParameterHandler encodes compatible physics as another
        ParameterHandler. This is called if a second handler is attempted to be
        initialized for the same tag.
        
        Parameters
        ----------
        other_handler : a ParameterHandler object
            The handler to compare to.
        
        Raises
        ------
        IncompatibleParameterError if handler_kwargs are incompatible with existing
        parameters.
        """
    
        if self.__class__ != other_handler.__class__:
            raise IncompatibleParameterError(
                f"{self.__class__.__name__} and {other_handler.__class__.__name__} are not compatible."
            )

        int_attrs_to_compare = ["periodicity"]

        self._check_attributes_are_equal(
            other_handler,
            identical_attrs=int_attrs_to_compare,
            tolerance_attrs=[],
            tolerance=self._SCALETOL,
        )



class HarmonicHeightHandler(_CustomBondedHandler):

    class HarmonicHeightType(ParameterType):
        """Defines parameters for harmonic height restraint."""

        _VALENCE_TYPE = "ImproperTorsion"
        _ELEMENT_NAME = "ImproperTorsion"

        k = ParameterAttribute(
            default=None, unit=unit.kilojoule_per_mole / unit.nanometer**2
        )
        h0 = ParameterAttribute(
            default=None, unit=unit.nanometer
        )

    _TAGNAME = "HarmonicHeight"
    _INFOTYPE = HarmonicHeightType

    def create_force(self, system, topology):
        """Applies the harmonic height potential to the OpenMM system."""
        
        force = CustomCompoundBondForce(4, 
            "0.5 * k * (h - h0)^2;"
            "h = abs((Nx*(x2-x1) + Ny*(y2-y1) + Nz*(z2-z1)) / sqrt(Nx^2 + Ny^2 + Nz^2));"
            "Nx = (y3-y1)*(z4-z1) - (z3-z1)*(y4-y1);"
            "Ny = (z3-z1)*(x4-x1) - (x3-x1)*(z4-z1);"
            "Nz = (x3-x1)*(y4-y1) - (y3-y1)*(x4-x1);"
        )
        force.addPerBondParameter("k")
        force.addPerBondParameter("h0")

        for parameter in self.parameters:
            smirks = parameter.smirks
            k = parameter.k
            h0 = parameter.h0

            for match in topology.chemical_environment_matches(smirks):
                atom_indices = tuple(match) 

                force.addBond(
                    atom_indices,
                    [k.m_as(unit.kilojoule_per_mole / unit.nanometer**2),
                     h0.m_as(unit.nanometer)]
                )

        system.addForce(force)

class LeeKrimmHandler(_CustomBondedHandler):

    class LeeKrimmType(ParameterType):
        """Defines parameters for Lee Krimm restraint."""

        _VALENCE_TYPE = "ImproperTorsion"
        _ELEMENT_NAME = "ImproperTorsion"

        V2 = ParameterAttribute(
            default=None, unit=unit.kilojoule_per_mole
        )
        V4 = ParameterAttribute(
            default=None, unit=unit.kilojoule_per_mole
        )
        t = ParameterAttribute(default=2.0, unit = unit.dimensionless)
        s = ParameterAttribute(default=1.0, unit = unit.dimensionless)

    _TAGNAME = "LeeKrimm"
    _INFOTYPE = LeeKrimmType

    def create_force(self, system, topology):
        """Applies the OOP potential to the OpenMM system."""
        
        force = CustomCompoundBondForce(4, 
            "V2 * ((abs(h)^t) / (1 - abs(h)^s))^2 + "
            "V4 * ((abs(h)^t) / (1 - abs(h)^s))^4;"
            "h = abs((Nx*(x2-x1) + Ny*(y2-y1) + Nz*(z2-z1)) / sqrt(Nx^2 + Ny^2 + Nz^2));"
            "Nx = (y3-y1)*(z4-z1) - (z3-z1)*(y4-y1);"
            "Ny = (z3-z1)*(x4-x1) - (x3-x1)*(z4-z1);"
            "Nz = (x3-x1)*(y4-y1) - (y3-y1)*(x4-x1);"
        )
        
        force.addPerBondParameter("V2")
        force.addPerBondParameter("V4")
        force.addPerBondParameter("t")
        force.addPerBondParameter("s")

        for parameter in self.parameters:
            smirks = parameter.smirks
            V2 = parameter.V2
            V4 = parameter.V4
            t = parameter.t
            s = parameter.s

            for match in topology.chemical_environment_matches(smirks):
                atom_indices = tuple(match) 

                force.addBond(
                    atom_indices,
                    [V2.m_as(unit.kilojoule_per_mole),
                     V4.m_as(unit.kilojoule_per_mole),
                     t.m_as(unit.dimensionless),
                     s.m_as(unit.dimensionless)]
                )

        system.addForce(force)

class TwoMinimaHandler(_CustomBondedHandler):

    class TwoMinimaType(ParameterType):
        """Defines parameters for TwoMinima restraint."""

        _VALENCE_TYPE = "ImproperTorsion"
        _ELEMENT_NAME = "ImproperTorsion"

        k1 = ParameterAttribute(default=None, unit=unit.kilojoule_per_mole)
        k2 = ParameterAttribute(default=None, unit=unit.kilojoule_per_mole)
        periodicity = ParameterAttribute(default=1.0)
        phase = ParameterAttribute(default=0.0, unit=unit.radian)

    _TAGNAME = "TwoMinima"
    _INFOTYPE = TwoMinimaType

    def create_force(self, system, topology):
        """Applies the two minima OOP potential to the OpenMM system."""

        force = CustomCompoundBondForce(4, 
            "k1 * (1 + cos(periodicity * theta - phase)) - "
            "k2 * (1 + cos(2 * periodicity * theta + phase));"
            "theta = acos(h);"
            "h = abs((Nx*(x2-x1) + Ny*(y2-y1) + Nz*(z2-z1)) / sqrt(Nx^2 + Ny^2 + Nz^2));"
            "Nx = (y3-y1)*(z4-z1) - (z3-z1)*(y4-y1);"
            "Ny = (z3-z1)*(x4-x1) - (x3-x1)*(z4-z1);"
            "Nz = (x3-x1)*(y4-y1) - (y3-y1)*(x4-x1);"
        )

        force.addPerBondParameter("k1")
        force.addPerBondParameter("k2")
        force.addPerBondParameter("periodicity")
        force.addPerBondParameter("phase")

        for parameter in self.parameters:
            smirks = parameter.smirks
            k1 = parameter.k1
            k2 = parameter.k2
            periodicity = parameter.periodicity
            phase = parameter.phase

            for match in topology.chemical_environment_matches(smirks):
                atom_indices = tuple(match)

                force.addBond(
                    atom_indices,
                    [
                        k1.m_as(unit.kilojoule_per_mole),
                        k2.m_as(unit.kilojoule_per_mole),
                        periodicity,
                        phase.m_as(unit.radian)
                    ]
                )

        system.addForce(force)
