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

    cutoff = ParameterAttribute(default=9.0 * unit.angstroms, unit=unit.angstrom)
    
    periodicity = ParameterAttribute(
        default=1, converter=int
    )

    periodic_method = ParameterAttribute(
        default="cutoff", converter=_allow_only(["cutoff"])
        )
    
    nonperiodic_method = ParameterAttribute(
        default="no-cutoff", converter=_allow_only(["no-cutoff"])
        )

    switch_width = ParameterAttribute(default=1.0 * unit.angstroms, unit=unit.angstrom)


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


