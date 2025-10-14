from openff.toolkit import unit
from openff.toolkit.typing.engines.smirnoff.parameters import (
    ConstraintHandler,
    ParameterAttribute,
    ParameterHandler,
    ParameterType,
)


class UreyBradleyHandler(ParameterHandler):
    """A custom SMIRNOFF handler for Urey-Bradley interactions."""

    class UreyBradleyType(ParameterType):
        """A custom SMIRNOFF type for Urey-Bradley interactions."""

        _ELEMENT_NAME = "UreyBradley"

        k = ParameterAttribute(
            default=None, unit=unit.kilojoule_per_mole / unit.nanometer**2
        )
        length = ParameterAttribute(default=None, unit=unit.nanometers)

    _TAGNAME = "UreyBradleys"
    _INFOTYPE = UreyBradleyType
    _DEPENDENCIES = [ConstraintHandler]


class HarmonicHeightHandler(ParameterHandler):
    """Handler for HarmonicHeight improper-like interactions."""

    class HarmonicHeightType(ParameterType):
        _ELEMENT_NAME = "HarmonicHeight"

        k = ParameterAttribute(
            default=None, unit=unit.kilojoule_per_mole / unit.nanometer**2
        )
        h0 = ParameterAttribute(default=None, unit=unit.nanometer)

    _TAGNAME = "HarmonicHeights"
    _INFOTYPE = HarmonicHeightType


class HarmonicAngleHandler(ParameterHandler):
    """Handler for HarmonicAngle improper-like interactions."""

    class HarmonicAngleType(ParameterType):
        _ELEMENT_NAME = "HarmonicAngle"

        k = ParameterAttribute(
            default=None, unit=unit.kilocalorie_per_mole / unit.radians**2
        )
        theta0 = ParameterAttribute(default=None, unit=unit.radians)

    _TAGNAME = "HarmonicAngle"
    _INFOTYPE = HarmonicAngleType


class LeeKrimmHandler(ParameterHandler):
    """Handler for Lee-Krimm improper-like interactions."""

    class LeeKrimmType(ParameterType):
        _ELEMENT_NAME = "LeeKrimm"

        V2 = ParameterAttribute(unit=unit.kilojoule_per_mole)
        V4 = ParameterAttribute(unit=unit.kilojoule_per_mole)
        t = ParameterAttribute(default=2.0)
        s = ParameterAttribute(default=1.0)

    _TAGNAME = "LeeKrimm"
    _INFOTYPE = LeeKrimmType
