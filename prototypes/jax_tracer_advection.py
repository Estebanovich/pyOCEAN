"""
Prototype: the flux-form WENO5 tracer advection RHS of SigmaBaroclinic in JAX
(jax.numpy + jit; XLA fuses the array expressions), to compare with the fused
Numba kernel ``core.dynamics.sigma_baroclinic._advect_div`` on the same data.

Same rules as the reference ``SigmaBaroclinic._rhs_numpy``:
  * WENO5-Z (Borges et al. 2008) where the 5-cell upwind stencil is valid, WENO3
    where the 3 central cells are, first-order upwind otherwise; cells outside the
    domain are invalid with value 0;
  * interior faces only; open edges take the boundary cell on outflow and the
    external value on inflow; closed edges carry no flux.
Both upwind reconstructions are evaluated and selected with ``where`` (branch-free,
as XLA and GPUs prefer).

Decision context: docs/design/performance.md (JAX backend = roadmap stage D).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _wval(v1, v2, v3, v4, v5, k1, k2, k3, k4, k5, z_variant=True):
    q0 = (2 * v1 - 7 * v2 + 11 * v3) / 6
    q1 = (-v2 + 5 * v3 + 2 * v4) / 6
    q2 = (2 * v3 + 5 * v4 - v5) / 6
    b0 = 13 / 12 * (v1 - 2 * v2 + v3) ** 2 + 0.25 * (v1 - 4 * v2 + 3 * v3) ** 2
    b1 = 13 / 12 * (v2 - 2 * v3 + v4) ** 2 + 0.25 * (v2 - v4) ** 2
    b2 = 13 / 12 * (v3 - 2 * v4 + v5) ** 2 + 0.25 * (3 * v3 - 4 * v4 + v5) ** 2
    if z_variant:
        tau, ez = jnp.abs(b2 - b0), 1e-40
        a0, a1, a2 = 0.1 * (1 + tau / (ez + b0)), 0.6 * (1 + tau / (ez + b1)), 0.3 * (1 + tau / (ez + b2))
    else:
        a0, a1, a2 = 0.1 / (1e-6 + b0) ** 2, 0.6 / (1e-6 + b1) ** 2, 0.3 / (1e-6 + b2) ** 2
    w5 = (a0 * q0 + a1 * q1 + a2 * q2) / (a0 + a1 + a2)
    c0, c1 = (-v2 + 3 * v3) / 2, (v3 + v4) / 2
    e0, e1 = (1 / 3) / (1e-6 + (v3 - v2) ** 2) ** 2, (2 / 3) / (1e-6 + (v4 - v3) ** 2) ** 2
    w3 = (e0 * c0 + e1 * c1) / (e0 + e1)
    return jnp.where(k1 & k2 & k3 & k4 & k5, w5, jnp.where(k2 & k3 & k4, w3, v3))


def _faces(c, ok, q, axis):
    """
    WENO values on the faces 1 … n−1 along ``axis`` (between cells j−1 and j), upwind w.r.t.
    q (same shape as the returned faces, n−1 along axis).  c, ok broadcast over the tracer axis.
    """
    n = c.shape[axis]
    pad = [(0, 0)] * c.ndim
    pad[axis] = (3, 3)
    cp = jnp.pad(c, pad)
    op = jnp.pad(ok, pad[-ok.ndim:] if ok.ndim < c.ndim else pad)

    def sl(a, s):                                       # cells j + s for faces j = 1 … n−1 (padded index j + 3 + s)
        return jax.lax.slice_in_dim(a, 4 + s, 4 + s + n - 1, axis=axis - (c.ndim - a.ndim))
    v = {s: sl(cp, s) for s in range(-3, 3)}
    k = {s: sl(op, s) for s in range(-3, 3)}
    pos = _wval(v[-3], v[-2], v[-1], v[0], v[1], k[-3], k[-2], k[-1], k[0], k[1])
    neg = _wval(v[2], v[1], v[0], v[-1], v[-2], k[2], k[1], k[0], k[-1], k[-2])
    return jnp.where(q >= 0, pos, neg)


@partial(jax.jit, static_argnames=("open_w", "open_e", "open_s", "open_n"))
def advect_rhs(c, Qx, Qy, Fz, ok, ext, open_w=False, open_e=False, open_s=False, open_n=False):
    """−∇·(Q c) for the stack c (ntr, N, ny, nx); Qx (N, ny, nx+1), Qy (N, ny+1, nx), Fz (N+1, ny, nx)."""
    qx_in = Qx[None, :, :, 1:-1]
    fx_in = qx_in * _faces(c, ok, qx_in, axis=3)
    qy_in = Qy[None, :, 1:-1, :]
    fy_in = qy_in * _faces(c, ok, qy_in, axis=2)
    qz_in = Fz[None, 1:-1]
    fz_in = qz_in * _faces(c, ok, qz_in, axis=1)

    def edge(q, inside, outside, inflow_positive, is_open):
        if not is_open:
            return jnp.zeros_like(inside)
        inflow = q > 0 if inflow_positive else q < 0
        return q * jnp.where(inflow, outside, inside)
    fw = edge(Qx[None, :, :, :1], c[..., :1], ext[..., :1], True, open_w)
    fe = edge(Qx[None, :, :, -1:], c[..., -1:], ext[..., -1:], False, open_e)
    fs = edge(Qy[None, :, :1, :], c[:, :, :1, :], ext[:, :, :1, :], True, open_s)
    fn = edge(Qy[None, :, -1:, :], c[:, :, -1:, :], ext[:, :, -1:, :], False, open_n)
    fx = jnp.concatenate([fw, fx_in, fe], axis=3)
    fy = jnp.concatenate([fs, fy_in, fn], axis=2)
    zero = jnp.zeros_like(c[:, :1])
    fz = jnp.concatenate([zero, fz_in, zero], axis=1)
    return -(jnp.diff(fx, axis=3) + jnp.diff(fy, axis=2) + jnp.diff(fz, axis=1))
