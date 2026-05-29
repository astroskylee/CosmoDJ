"""Numerical integration helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


@lru_cache(maxsize=None)
def _legendre_nodes_weights(n: int):
    return np.polynomial.legendre.leggauss(n)


def gauss_legendre_integrate(
    func: Callable,
    a,
    b,
    *args,
    n: int = 64,
    **kwargs,
):
    """Integrate ``func(x, *args, **kwargs)`` from ``a`` to ``b``.

    The implementation follows the same fixed-order Gauss-Legendre rule used
    in the current lensing scripts, but uses JAX arrays and supports scalar or
    broadcast-compatible array limits.
    """

    a_arr = jnp.asarray(a, dtype=jnp.float64)
    b_arr = jnp.asarray(b, dtype=jnp.float64)
    x_np, w_np = _legendre_nodes_weights(n)
    x = jnp.asarray(x_np, dtype=jnp.float64)
    w = jnp.asarray(w_np, dtype=jnp.float64)

    shape = jnp.broadcast_shapes(a_arr.shape, b_arr.shape)
    a_b = jnp.broadcast_to(a_arr, shape)
    b_b = jnp.broadcast_to(b_arr, shape)

    node_shape = (1,) * len(shape) + (x.shape[0],)
    x_eval = 0.5 * (
        (b_b - a_b)[..., None] * x.reshape(node_shape)
        + (b_b + a_b)[..., None]
    )
    values = func(x_eval, *args, **kwargs)
    return 0.5 * (b_b - a_b) * jnp.sum(w.reshape(node_shape) * values, axis=-1)
