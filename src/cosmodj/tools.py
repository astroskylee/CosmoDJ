# Authors: Tian Li, Coleman Krawczyk, Wolfgang Enzi, Andy Lundgren

"""SLCOSMO-style compatibility helpers.

This module provides lightweight wrappers around the main CosmoDJ distance
functions for older code that expects a ``tools`` namespace.
"""

from __future__ import annotations

from typing import Mapping

from .distances import (
    Cosmology,
    Planck18Cosmology,
    angular_diameter_distance as _angular_diameter_distance,
    angular_diameter_distances as _angular_diameter_distances,
    time_delay_distance as _time_delay_distance,
)


C_KM_S = 299792.458


def _h0(cosmology: Cosmology | Mapping[str, float] | None = None):
    if cosmology is None:
        return Planck18Cosmology.H0
    if isinstance(cosmology, Cosmology):
        return cosmology.H0
    if "H0" in cosmology:
        return cosmology["H0"]
    return cosmology["h0"]


def angular_diameter_distance(
    z,
    cosmology: Cosmology | Mapping[str, float] | None = None,
    n: int = 20,
):
    """Return angular-diameter distance in Mpc using the tools-style default."""

    return _angular_diameter_distance(z, cosmology, n=n)


def dldsdls(
    zl,
    zs,
    cosmology: Cosmology | Mapping[str, float] | None = None,
    n: int = 20,
):
    """Return ``(D_l, D_s, D_ls)`` angular-diameter distances in Mpc."""

    return _angular_diameter_distances(zl, zs, cosmology, n=n)


def dldsdlsdldsdls(
    zl,
    zs,
    cosmology: Cosmology | Mapping[str, float] | None = None,
    n: int = 20,
):
    """Alias for :func:`dldsdls` matching the requested alternative name."""

    return dldsdls(zl, zs, cosmology, n=n)


def time_delay_distance(
    zl,
    zs,
    cosmology: Cosmology | Mapping[str, float] | None = None,
    n: int = 20,
):
    """Return lensing time-delay distance in Mpc."""

    return _time_delay_distance(zl, zs, cosmology, n=n)


def compute_distances(
    zl,
    zs,
    cosmology: Cosmology | Mapping[str, float] | None = None,
    n: int = 20,
):
    """Return historical dimensionless ``(D_l, D_s, D_ls)`` distances.

    This mirrors the old SLCOSMO ``tool.compute_distances`` convention by
    multiplying physical Mpc distances by ``H0 / c``.
    """

    dl, ds, dls = dldsdls(zl, zs, cosmology, n=n)
    scale = _h0(cosmology) / C_KM_S
    return dl * scale, ds * scale, dls * scale


class tool:
    """Class-style compatibility wrapper for older ``tool.dldsdls`` code."""

    c_km_s = C_KM_S

    angular_diameter_distance = staticmethod(angular_diameter_distance)
    dldsdls = staticmethod(dldsdls)
    dldsdlsdldsdls = staticmethod(dldsdlsdldsdls)
    time_delay_distance = staticmethod(time_delay_distance)
    compute_distances = staticmethod(compute_distances)


__all__ = [
    "C_KM_S",
    "angular_diameter_distance",
    "compute_distances",
    "dldsdls",
    "dldsdlsdldsdls",
    "time_delay_distance",
    "tool",
]
