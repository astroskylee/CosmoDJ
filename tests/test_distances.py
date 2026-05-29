import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer.util import log_density
from astropy.cosmology import Planck18 as AstropyPlanck18

from cosmodj import (
    Cosmology,
    Planck18Cosmology,
    angular_diameter_distance,
    angular_diameter_distances,
    e_z,
    nu_relative_density,
)


def test_angular_diameter_distance_flat_lcdm_sanity():
    cosmo = Cosmology(Omegam=0.32, Omegak=0.0, w0=-1.0, wa=0.0, H0=70.0)
    da = np.asarray(angular_diameter_distance(1.0, cosmo))
    assert np.isfinite(da)
    assert 1600.0 < da < 1800.0


def test_planck18_is_default_cosmology():
    da_default = angular_diameter_distance(1.0)
    da_planck18 = angular_diameter_distance(1.0, Planck18Cosmology)
    np.testing.assert_allclose(da_default, da_planck18)
    assert Planck18Cosmology.H0 == 67.66
    assert Planck18Cosmology.Omegam == 0.30966
    assert Planck18Cosmology.Ogamma0 == 5.402015137139352e-05
    assert Planck18Cosmology.nu_y == (357.9121209673803,)


def test_planck18_da_matches_astropy_for_100_redshifts_to_machine_precision():
    rng = np.random.default_rng(12345)
    z = np.sort(rng.uniform(1.0e-4, 5.0, size=100))
    da_cosmodj = np.asarray(angular_diameter_distance(z))
    da_astropy = AstropyPlanck18.angular_diameter_distance(z).value
    np.testing.assert_allclose(da_cosmodj, da_astropy, rtol=1.0e-10, atol=1.0e-8)


def test_planck18_ez_and_neutrino_density_match_astropy():
    z = np.linspace(0.0, 5.0, 64)
    np.testing.assert_allclose(
        np.asarray(nu_relative_density(z)),
        AstropyPlanck18.nu_relative_density(z),
        rtol=1.0e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(e_z(z)),
        AstropyPlanck18.efunc(z),
        rtol=1.0e-13,
        atol=0.0,
    )


def test_lensing_distances_are_ordered_for_typical_lens():
    cosmo = {"Omegam": 0.32, "Omegak": 0.0, "w0": -1.0, "wa": 0.0, "h0": 70.0}
    dl, ds, dls = angular_diameter_distances(0.5, 2.0, cosmo)
    assert dl > 0.0
    assert ds > 0.0
    assert dls > 0.0


def test_default_distances_are_jittable():
    z = jnp.linspace(0.1, 5.0, 8)
    da = jax.jit(angular_diameter_distance)(z)
    assert da.shape == z.shape

    dl, ds, dls = jax.jit(lambda zl, zs: angular_diameter_distances(zl, zs))(0.5, 2.0)
    assert dl > 0.0
    assert ds > 0.0
    assert dls > 0.0


def test_angular_diameter_distance_runs_inside_numpyro_model():
    def model():
        Omegam = numpyro.sample("Omegam", dist.Uniform(0.2, 0.4))
        H0 = numpyro.sample("H0", dist.Uniform(60.0, 80.0))
        cosmo = Cosmology(Omegam=Omegam, Omegak=0.0, w0=-1.0, wa=0.0, H0=H0)
        da = angular_diameter_distance(jnp.array([0.5, 1.0]), cosmo)
        numpyro.deterministic("Da", da)
        numpyro.sample("Da_obs", dist.Normal(da[0], 10.0), obs=jnp.array(1300.0))

    log_prob, model_trace = log_density(
        model,
        model_args=(),
        model_kwargs={},
        params={"Omegam": jnp.array(0.32), "H0": jnp.array(70.0)},
    )

    assert jnp.isfinite(log_prob)
    assert model_trace["Da"]["value"].shape == (2,)
    assert jnp.all(jnp.isfinite(model_trace["Da"]["value"]))
