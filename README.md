# CosmoDJ

CosmoDJ is a lightweight JAX package for cosmological distance calculations, written for strong-lensing cosmology forecasts and NumPyro workflows. The package currently provides angular-diameter distances, transverse and radial comoving distances, luminosity distances, and time-delay distances for CPL dark-energy cosmologies.

The default cosmology is `Planck18Cosmology`, implemented as a CosmoDJ parameter object. Runtime distance calculations are performed with JAX rather than Astropy. Astropy is used for physical constants and in tests as a reference implementation.

## Installation

For local development:

```bash
cd /mnt/d/lensing/Package_Tian/CosmoDJ
pip install -e .
```

After PyPI release:

```bash
pip install CosmoDJ
```

## Basic Usage

```python
from cosmodj import angular_diameter_distance, angular_diameter_distances

Da = angular_diameter_distance(1.0)          # Mpc, default Planck18Cosmology
Dl, Ds, Dls = angular_diameter_distances(0.5, 2.0)  # Mpc
```

`angular_diameter_distance` accepts scalar or array-like redshifts:

```python
import jax.numpy as jnp
from cosmodj import angular_diameter_distance

z = jnp.array([0.5, 1.0, 2.0])
Da = angular_diameter_distance(z)
```

## Custom Cosmology

```python
from cosmodj import Cosmology, angular_diameter_distance

cosmo = Cosmology(
    Omegam=0.32,
    Omegak=0.0,
    w0=-1.0,
    wa=0.0,
    H0=70.0,
)

Da = angular_diameter_distance(1.0, cosmo)
```

Dictionary inputs are also supported:

```python
from cosmodj import angular_diameter_distances

cosmo = {"Omegam": 0.32, "Omegak": 0.0, "w0": -1.0, "wa": 0.0, "h0": 70.0}
Dl, Ds, Dls = angular_diameter_distances(0.5, 2.0, cosmo)
```

## NumPyro Example

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from cosmodj import Cosmology, angular_diameter_distance


def model():
    Omegam = numpyro.sample("Omegam", dist.Uniform(0.2, 0.4))
    H0 = numpyro.sample("H0", dist.Uniform(60.0, 80.0))

    cosmo = Cosmology(Omegam=Omegam, Omegak=0.0, w0=-1.0, wa=0.0, H0=H0)
    z = jnp.array([0.5, 1.0])
    Da = angular_diameter_distance(z, cosmo)

    numpyro.sample("Da_obs", dist.Normal(Da, 20.0), obs=jnp.array([1250.0, 1650.0]))
```

## JIT Example

```python
import jax
import jax.numpy as jnp

from cosmodj import angular_diameter_distance

z = jnp.linspace(0.1, 5.0, 100)
Da = jax.jit(angular_diameter_distance)(z)
```

## Validation

The Planck18 implementation includes photon and neutrino density terms, including the massive-neutrino fitting formula used by Astropy. In the current test suite, CosmoDJ agrees with Astropy Planck18 at approximately machine precision for `E(z)`, `nu_relative_density(z)`, and angular-diameter distances over random redshifts in `0 < z < 5`.

Example validation result:

```text
Da max_rel ~ 4e-15
E(z) max_rel ~ 2e-16
nu_relative_density max_rel = 0
```

## Citation

If you use this package in a publication, please cite:

```bibtex
@ARTICLE{2024MNRAS.527.5311L,
       author = {{Li}, Tian and {Collett}, Thomas E. and {Krawczyk}, Coleman M. and {Enzi}, Wolfgang},
        title = "{Cosmology from large populations of galaxy-galaxy strong gravitational lenses}",
      journal = {\mnras},
     keywords = {gravitational lensing: strong, galaxies: structure, cosmological parameters, dark energy, cosmology: observations, Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2024,
        month = jan,
       volume = {527},
       number = {3},
        pages = {5311-5323},
          doi = {10.1093/mnras/stad3514},
archivePrefix = {arXiv},
       eprint = {2307.09271},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.5311L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

