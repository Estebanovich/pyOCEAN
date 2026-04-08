"""
Equation of state (EOS) for pyOCEAN — Phase 1.

Two formulations are provided:

  1. **TEOS-10** (default)
     Full nonlinear seawater EOS via the ``gsw`` library.
     Accurate to < 0.1 kg/m³ across the full T–S–p range.
     Reference: TEOS-10 (2010), gsw v3 documentation.

  2. **Linear EOS**
     First-order Taylor expansion around a reference state (T₀, S₀, ρ₀):

         ρ = ρ₀ [1 − α(T − T₀) + β(S − S₀)]          (Vallis 2006, eq. 1.156)

     Useful for idealised experiments where density stratification is weak
     and a simple, transparent formulation is preferred over accuracy.

Pressure approximation
----------------------
The hydrostatic pressure at layer centre k is approximated from the
z-coordinate using a constant reference density ρ₀:

    p(k) ≈ ρ₀ · g · |z_t[k]|           [Pa]
    p_dbar(k) = p(k) / 1e4              [dbar]

This is equivalent to the MITgcm "pressure = ρ₀ g z" approximation used in
a Boussinesq model.  The full dynamical pressure is solved separately in
``core/dynamics/pressure.py``.

Public API
----------
    density_teos10(T, S, p_dbar)            → ρ  [kg/m³]  NumPy arrays
    density_linear(T, S, *, rho0, α, β, …) → ρ  [kg/m³]  NumPy arrays
    compute_density(state, z_grid, eos, …) → xr.DataArray ρ(k, j_t, i_t)

References
----------
TEOS-10 (2010) — The International Thermodynamic Equation of Seawater.
    IOC, SCOR and IAPSO, UNESCO.
McDougall & Barker (2011) — gsw oceanographic toolbox of TEOS-10. SCOR/IAPSO.
Vallis (2006) — Atmospheric and Oceanic Fluid Dynamics, §1.6 & §2.4.
"""

from __future__ import annotations

from typing import Union, Callable
import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

G: float = 9.81      # m/s²  gravitational acceleration (Vallis 2006, §1.2)
RHO_0: float = 1025.0  # kg/m³ Boussinesq reference density


# ---------------------------------------------------------------------------
# 1. TEOS-10
# ---------------------------------------------------------------------------

def density_teos10(
    T: np.ndarray,
    S: np.ndarray,
    p_dbar: np.ndarray,
) -> np.ndarray:
    """
    Compute in-situ density using the full TEOS-10 equation of state.

    For idealised Cartesian domains the following approximations are applied
    (consistent with MITgcm's nonlinear EOS in Boussinesq mode):

    * Practical Salinity SP ≈ Absolute Salinity SA  (error < 1% in open ocean)
    * Potential temperature θ ≈ Conservative Temperature CT  (error ≪ 0.01°C
      for θ < 40°C and moderate pressures)

    For realistic regional runs these approximations should be replaced by the
    full gsw conversion chain (SA_from_SP → CT_from_pt → rho).

    The equation of state is:

        ρ = gsw.rho(SA, CT, p)                    (TEOS-10, McDougall 2011)

    Parameters
    ----------
    T : np.ndarray
        Potential temperature [°C].  Any shape.
    S : np.ndarray
        Practical salinity [PSU].  Same shape as T.
    p_dbar : np.ndarray
        Sea pressure [dbar].  Broadcastable with T and S.

    Returns
    -------
    np.ndarray
        In-situ density ρ [kg/m³].  Same shape as T (after broadcasting).

    Examples
    --------
    >>> rho = density_teos10(np.array([10.0]), np.array([35.0]), np.array([0.0]))
    >>> 1025.0 < float(rho) < 1030.0
    True
    """
    import gsw  # optional import: not needed if eos="linear"

    T = np.asarray(T, dtype=float)
    S = np.asarray(S, dtype=float)
    p_dbar = np.broadcast_to(np.asarray(p_dbar, dtype=float), T.shape).copy()

    # Idealised approximations: SA ≈ SP, CT ≈ θ
    SA = S         # g/kg ≈ PSU
    CT = T         # °C   ≈ θ

    rho = gsw.rho(SA, CT, p_dbar)
    return rho


# ---------------------------------------------------------------------------
# 2. Linear EOS
# ---------------------------------------------------------------------------

def density_linear(
    T: np.ndarray,
    S: np.ndarray,
    *,
    rho0: float = RHO_0,
    alpha: float = 2.0e-4,
    beta: float = 7.4e-4,
    T0: float = 10.0,
    S0: float = 35.0,
) -> np.ndarray:
    """
    Compute density using the linear (first-order Taylor) equation of state.

    The linear EOS is:

        ρ = ρ₀ [1 − α(T − T₀) + β(S − S₀)]        (Vallis 2006, eq. 1.156)

    where:
        α = −(1/ρ₀)(∂ρ/∂T)|_{S₀,p₀}   thermal expansion  [1/°C]
        β =  (1/ρ₀)(∂ρ/∂S)|_{T₀,p₀}   haline contraction [1/PSU]

    Default values correspond to typical mid-ocean conditions near 10°C and
    35 PSU at the surface (Vallis 2006, Table 1.1).

    Note: the linear EOS ignores pressure dependence of density (no
    compressibility), which is appropriate for Boussinesq models where the
    buoyancy is the only dynamically relevant density variable.

    Parameters
    ----------
    T : np.ndarray
        Potential temperature [°C].  Any shape.
    S : np.ndarray
        Salinity [PSU].  Same shape as T.
    rho0 : float, optional
        Reference density [kg/m³]. Default 1025.0.
    alpha : float, optional
        Thermal expansion coefficient [1/°C]. Default 2.0e-4.
    beta : float, optional
        Haline contraction coefficient [1/PSU]. Default 7.4e-4.
    T0 : float, optional
        Reference potential temperature [°C]. Default 10.0.
    S0 : float, optional
        Reference salinity [PSU]. Default 35.0.

    Returns
    -------
    np.ndarray
        Density ρ [kg/m³].  Same shape as T.

    Examples
    --------
    >>> rho = density_linear(np.array([10.0]), np.array([35.0]))
    >>> float(rho)
    1025.0
    >>> # Warmer water is less dense
    >>> rho_warm = density_linear(np.array([15.0]), np.array([35.0]))
    >>> float(rho_warm) < 1025.0
    True
    >>> # Saltier water is denser
    >>> rho_salty = density_linear(np.array([10.0]), np.array([36.0]))
    >>> float(rho_salty) > 1025.0
    True
    """
    T = np.asarray(T, dtype=float)
    S = np.asarray(S, dtype=float)

    # ρ = ρ₀ [1 − α(T − T₀) + β(S − S₀)]    (Vallis 2006, eq. 1.156)
    rho = rho0 * (1.0 - alpha * (T - T0) + beta * (S - S0))
    return rho


# ---------------------------------------------------------------------------
# 3. High-level dispatcher
# ---------------------------------------------------------------------------

# Type alias for a custom EOS callable
EOSCallable = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def compute_density(
    state,
    z_grid: xr.Dataset,
    eos: Union[str, EOSCallable] = "teos10",
    **kwargs,
) -> xr.DataArray:
    """
    Compute the 3-D in-situ density field from a ModelState.

    The hydrostatic sea pressure at each layer centre is approximated by:

        p(k) = ρ₀ · g · |z_t[k]|     [Pa]
        p_dbar(k) = p(k) / 1e4        [dbar]

    This is column-uniform (no horizontal pressure variation), consistent
    with a Boussinesq model where only the buoyancy perturbation matters.

    Parameters
    ----------
    state : ModelState
        Current model state. Uses ``state.T`` and ``state.S``.
    z_grid : xr.Dataset
        1-D vertical grid from ``create_z_grid``; provides ``z_t`` [m].
    eos : str or callable, optional
        EOS selection:

        * ``"teos10"`` (default) — full nonlinear TEOS-10 via gsw.
        * ``"linear"``           — linear EOS with default coefficients.
        * A callable ``f(T, S, p_dbar) → rho`` for a custom EOS.

    **kwargs
        Additional keyword arguments forwarded to the EOS function.

        For ``eos="linear"``:
            rho0, alpha, beta, T0, S0  (see ``density_linear`` docstring).

        For ``eos="teos10"`` or a custom callable: kwargs are ignored.

    Returns
    -------
    xr.DataArray, dims (k, j_t, i_t)
        In-situ density ρ [kg/m³] on the T-cell grid.

    Raises
    ------
    ValueError
        If ``eos`` is not a recognised string or callable.

    Examples
    --------
    >>> from core.grid import create_cartesian_grid, create_z_grid
    >>> from core.state import ModelState
    >>> hg = create_cartesian_grid(nx=10, ny=8, dx=1e3, dy=1e3)
    >>> vg = create_z_grid(nz=5, dz=[10.0]*5)
    >>> st = ModelState.from_grids(hg, vg)
    >>> rho = compute_density(st, vg, eos="linear")
    >>> rho.dims
    ('k', 'j_t', 'i_t')
    >>> float(rho.mean())
    1025.0
    """
    # ---- Resolve EOS function -------------------------------------------
    if callable(eos):
        eos_fn: EOSCallable = eos
    elif eos == "teos10":
        eos_fn = density_teos10
    elif eos == "linear":
        eos_fn = lambda T, S, p: density_linear(T, S, **kwargs)
    else:
        raise ValueError(
            f"Unknown eos='{eos}'. Use 'teos10', 'linear', or pass a callable."
        )

    # ---- Extract T and S as NumPy arrays --------------------------------
    T_np = state.T.values    # (nz, ny, nx)
    S_np = state.S.values    # (nz, ny, nx)

    # ---- Build 3-D pressure array from z-level depths -------------------
    # p(k) ≈ ρ₀ · g · |z_t[k]|  [Pa]  → convert to dbar (1 dbar = 1e4 Pa)
    # Broadcast: z_t has shape (nz,), expand to (nz, 1, 1) for broadcasting
    z_t = z_grid["z_t"].values                          # (nz,) in metres, ≤ 0
    p_dbar = RHO_0 * G * np.abs(z_t) / 1e4              # (nz,)  [dbar]
    p_dbar_3d = p_dbar[:, np.newaxis, np.newaxis]        # (nz, 1, 1)
    p_dbar_3d = np.broadcast_to(p_dbar_3d, T_np.shape).copy()

    # ---- Compute density -------------------------------------------------
    rho_np = eos_fn(T_np, S_np, p_dbar_3d)              # (nz, ny, nx)

    # ---- Wrap as xr.DataArray with same coords as T ----------------------
    rho = xr.DataArray(
        rho_np,
        dims=state.T.dims,
        coords=state.T.coords,
        attrs={
            "units": "kg/m3",
            "long_name": "In-situ density",
            "standard_name": "sea_water_density",
            "eos": str(eos) if isinstance(eos, str) else "custom",
        },
    )
    return rho
