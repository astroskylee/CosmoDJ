"""JAX cosmological distance calculations.

Distances use the CPL dark-energy parameterization,
``w(a) = w0 + wa * (1 - a)``. Planck18 is represented only as a
parameter container; runtime distance calculations are performed here with JAX.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from jax import config

config.update("jax_enable_x64", True)

from astropy.constants import c as speed_of_light
import jax.numpy as jnp

from .quadrature import gauss_legendre_integrate


C_KM_S = float(speed_of_light.to_value("km/s"))


@dataclass(frozen=True)
class Cosmology:
    """CPL cosmology container.

    Parameters are dimensionless except ``H0`` in km/s/Mpc, ``m_nu_eV`` in eV,
    and ``nu_y`` which stores ``m_nu / (k_B T_nu0)`` for massive species.
    """

    Omegam: float = 0.32
    Omegak: float = 0.0
    w0: float = -1.0
    wa: float = 0.0
    H0: float = 70.0
    Omegade: float | None = None
    Ogamma0: float = 0.0
    Neff: float = 0.0
    neff_per_nu: float | None = None
    nmasslessnu: int = 0
    nu_y: tuple[float, ...] = ()
    m_nu_eV: tuple[float, ...] = ()
    Tcmb0: float = 0.0


Planck18Cosmology = Cosmology(
    Omegam=0.30966,
    Omegak=0.0,
    w0=-1.0,
    wa=0.0,
    H0=67.66,
    Omegade=0.6888463055445441,
    Ogamma0=5.402015137139352e-05,
    Neff=3.046,
    neff_per_nu=1.0153333333333332,
    nmasslessnu=2,
    nu_y=(357.9121209673803,),
    m_nu_eV=(0.0, 0.0, 0.06),
    Tcmb0=2.7255,
)
"""Planck 2018 parameters; calculations are performed by CosmoDJ/JAX."""

def _resolve_cosmology(cosmology=None):
    if cosmology is None:
        return Planck18Cosmology
    return cosmology


def _as_cosmology(cosmology: Cosmology | Mapping[str, float] | None = None) -> Cosmology:
    cosmology = _resolve_cosmology(cosmology)
    if isinstance(cosmology, Cosmology):
        return cosmology

    h0 = cosmology.get("H0", cosmology.get("h0"))
    if h0 is None:
        raise KeyError("Cosmology mapping must include 'H0' or 'h0'.")

    return Cosmology(
        Omegam=cosmology["Omegam"],
        Omegak=cosmology.get("Omegak", 0.0),
        w0=cosmology.get("w0", -1.0),
        wa=cosmology.get("wa", 0.0),
        H0=h0,
        Omegade=cosmology.get("Omegade", cosmology.get("Ode0", None)),
        Ogamma0=cosmology.get("Ogamma0", 0.0),
        Neff=cosmology.get("Neff", 0.0),
        neff_per_nu=cosmology.get("neff_per_nu", None),
        nmasslessnu=cosmology.get("nmasslessnu", 0),
        nu_y=tuple(cosmology.get("nu_y", ())),
        m_nu_eV=tuple(cosmology.get("m_nu_eV", cosmology.get("m_nu", ()))),
        Tcmb0=cosmology.get("Tcmb0", 0.0),
    )


def _neff_per_nu(cosmology: Cosmology):
    if cosmology.neff_per_nu is not None:
        return cosmology.neff_per_nu
    n_nu = cosmology.nmasslessnu + len(cosmology.nu_y)
    if n_nu == 0:
        return 0.0
    return cosmology.Neff / n_nu


def nu_relative_density(z, cosmology: Cosmology | Mapping[str, float] | None = None):
    """Return neutrino energy density relative to photon energy density.

    This follows the Komatsu et al. 2011 fitting formula used by Astropy for
    massive neutrinos.
    """

    cosmo = _as_cosmology(cosmology)
    z_arr = jnp.asarray(z, dtype=jnp.float64)
    prefac = 0.22710731766

    if cosmo.Neff == 0:
        return jnp.zeros_like(z_arr)

    if len(cosmo.nu_y) == 0:
        return prefac * cosmo.Neff * jnp.ones_like(z_arr)

    p = 1.83
    invp = 0.54644808743
    k = 0.3173
    nu_y = jnp.asarray(cosmo.nu_y, dtype=jnp.float64)
    curr_nu_y = nu_y / (1.0 + jnp.expand_dims(z_arr, axis=-1))
    rel_mass_per = (1.0 + (k * curr_nu_y) ** p) ** invp
    rel_mass = jnp.sum(rel_mass_per, axis=-1) + cosmo.nmasslessnu
    return prefac * _neff_per_nu(cosmo) * rel_mass


def _omega_nu0(cosmology: Cosmology):
    return cosmology.Ogamma0 * nu_relative_density(0.0, cosmology)


def _omega_de0(cosmology: Cosmology):
    if cosmology.Omegade is not None:
        return cosmology.Omegade
    return (
        1.0
        - cosmology.Omegam
        - cosmology.Omegak
        - cosmology.Ogamma0
        - _omega_nu0(cosmology)
    )


def dark_energy_scale(z, cosmology: Cosmology | Mapping[str, float] | None = None):
    """Return CPL dark-energy density scaling relative to z=0."""

    cosmo = _as_cosmology(cosmology)
    z_arr = jnp.asarray(z, dtype=jnp.float64)
    zp1 = 1.0 + z_arr
    return zp1 ** (3.0 * (1.0 + cosmo.w0 + cosmo.wa)) * jnp.exp(
        -3.0 * cosmo.wa * z_arr / zp1
    )


def e_z(z, cosmology: Cosmology | Mapping[str, float] | None = None):
    """Dimensionless Hubble parameter ``E(z) = H(z) / H0``."""

    cosmo = _as_cosmology(cosmology)
    z_arr = jnp.asarray(z, dtype=jnp.float64)
    zp1 = 1.0 + z_arr
    omega_nu_z = cosmo.Ogamma0 * nu_relative_density(z_arr, cosmo)
    ez2 = (
        cosmo.Omegam * zp1**3
        + cosmo.Omegak * zp1**2
        + cosmo.Ogamma0 * zp1**4
        + omega_nu_z * zp1**4
        + _omega_de0(cosmo) * dark_energy_scale(z_arr, cosmo)
    )
    return jnp.sqrt(ez2)


def _transverse_from_radial(chi, Omegak):
    chi_arr = jnp.asarray(chi, dtype=jnp.float64)
    ok = jnp.asarray(Omegak, dtype=jnp.float64)
    sqrt_abs_ok = jnp.sqrt(jnp.maximum(jnp.abs(ok), 1.0e-300))
    d_pos = jnp.sinh(sqrt_abs_ok * chi_arr) / sqrt_abs_ok
    d_neg = jnp.sin(sqrt_abs_ok * chi_arr) / sqrt_abs_ok
    d_curved = jnp.where(ok > 0.0, d_pos, d_neg)
    return jnp.where(jnp.abs(ok) < 1.0e-14, chi_arr, d_curved)


def comoving_radial_distance(
    z,
    cosmology: Cosmology | Mapping[str, float] | None = None,
    n: int = 256,
):
    """Line-of-sight comoving distance from observer to redshift ``z`` in Mpc."""

    cosmo = _as_cosmology(cosmology)
    chi = gauss_legendre_integrate(lambda z_eval: 1.0 / e_z(z_eval, cosmo), 0.0, z, n=n)
    return (C_KM_S / cosmo.H0) * chi


def transverse_comoving_distance(
    z,
    cosmology: Cosmology | Mapping[str, float] | None = None,
    n: int = 256,
):
    """Transverse comoving distance from observer to redshift ``z`` in Mpc."""

    cosmo = _as_cosmology(cosmology)
    chi = gauss_legendre_integrate(lambda z_eval: 1.0 / e_z(z_eval, cosmo), 0.0, z, n=n)
    dm_dimensionless = _transverse_from_radial(chi, cosmo.Omegak)
    return (C_KM_S / cosmo.H0) * dm_dimensionless


def angular_diameter_distance(
    z,
    cosmology: Cosmology | Mapping[str, float] | None = None,
    n: int = 256,
):
    """Angular-diameter distance from observer to redshift ``z`` in Mpc."""

    z_arr = jnp.asarray(z, dtype=jnp.float64)
    return transverse_comoving_distance(z_arr, cosmology, n=n) / (1.0 + z_arr)


def angular_diameter_distance_z1z2(
    z1,
    z2,
    cosmology: Cosmology | Mapping[str, float] | None = None,
    n: int = 256,
):
    """Angular-diameter distance between two redshifts in Mpc."""

    z1_arr = jnp.asarray(z1, dtype=jnp.float64)
    z2_arr = jnp.asarray(z2, dtype=jnp.float64)

    cosmo = _as_cosmology(cosmology)
    chi1 = gauss_legendre_integrate(lambda z_eval: 1.0 / e_z(z_eval, cosmo), 0.0, z1_arr, n=n)
    chi2 = gauss_legendre_integrate(lambda z_eval: 1.0 / e_z(z_eval, cosmo), 0.0, z2_arr, n=n)
    dm12_dimensionless = _transverse_from_radial(chi2 - chi1, cosmo.Omegak)
    return (C_KM_S / cosmo.H0) * dm12_dimensionless / (1.0 + z2_arr)


def angular_diameter_distances(
    zl,
    zs,
    cosmology: Cosmology | Mapping[str, float] | None = None,
    n: int = 256,
):
    """Return ``(D_l, D_s, D_ls)`` angular-diameter distances in Mpc."""

    return (
        angular_diameter_distance(zl, cosmology, n=n),
        angular_diameter_distance(zs, cosmology, n=n),
        angular_diameter_distance_z1z2(zl, zs, cosmology, n=n),
    )


def luminosity_distance(
    z,
    cosmology: Cosmology | Mapping[str, float] | None = None,
    n: int = 256,
):
    """Luminosity distance from observer to redshift ``z`` in Mpc."""

    z_arr = jnp.asarray(z, dtype=jnp.float64)
    return (1.0 + z_arr) ** 2 * angular_diameter_distance(z_arr, cosmology, n=n)


def time_delay_distance(
    zl,
    zs,
    cosmology: Cosmology | Mapping[str, float] | None = None,
    n: int = 256,
):
    """Lensing time-delay distance ``(1 + zl) D_l D_s / D_ls`` in Mpc."""

    dl, ds, dls = angular_diameter_distances(zl, zs, cosmology, n=n)
    return (1.0 + jnp.asarray(zl, dtype=jnp.float64)) * dl * ds / dls
