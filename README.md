# CosmoDJ

CosmoDJ is a lightweight JAX package for cosmological distance calculations, written for strong-lensing cosmology forecasts and NumPyro workflows. The package currently provides angular-diameter distances, transverse and radial comoving distances, luminosity distances, and time-delay distances for CPL dark-energy cosmologies.

The default cosmology is `Planck18Cosmology`, implemented as a CosmoDJ parameter object. Runtime distance calculations are performed with JAX rather than Astropy. Astropy is used for physical constants and in tests as a reference implementation.

## Installation

For local development:

```bash
cd /yourpath
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
