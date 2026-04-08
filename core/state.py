"""
Model state container for pyOCEAN — Phase 1.

``ModelState`` holds all prognostic variables on the Arakawa C-grid as
xarray DataArrays with named dimensions, units, and physical descriptions.
It is the single object passed between dynamics, mixing, and I/O modules.

Prognostic variables and their C-grid locations
------------------------------------------------

    u(k, j_t, i_u)      Zonal velocity       [m/s]  — U-face (western cell face)
    v(k, j_v, i_t)      Meridional velocity  [m/s]  — V-face (southern cell face)
    w(k_f, j_t, i_t)    Vertical velocity    [m/s]  — T-column interfaces (k_f = 0 is surface)
    T(k, j_t, i_t)      Potential temperature [°C]  — T-cell centre
    S(k, j_t, i_t)      Salinity             [PSU]  — T-cell centre
    eta(j_t, i_t)        Sea surface height    [m]  — T-column (no vertical dim)

Kinematic boundary conditions enforced here:
    w[k_f=0]    = 0  (rigid-lid approximation for Phase 1; replaced by dη/dt in Phase 2)
    w[k_f=nz]   = 0  (no normal flow at solid bottom)

The ``time`` attribute records the current simulation time [s].

Usage
-----
    from core.grid import create_cartesian_grid, create_z_grid
    from core.state import ModelState

    hgrid = create_cartesian_grid(nx=100, ny=50, dx=2e3, dy=2e3)
    vgrid = create_z_grid(nz=20, dz=[25.0]*20)
    state = ModelState.from_grids(hgrid, vgrid)

    # Access variables
    u_np = state.u.values          # NumPy array for numerics
    state.T.isel(k=0).plot()       # xarray/Matplotlib

    # Deep copy (needed by Adams–Bashforth 2 for previous time-step)
    state_prev = state.copy()

    # Export to NetCDF
    state.to_dataset().to_netcdf("output.nc")

References
----------
Durran (2010) — Numerical Methods for Fluid Dynamics, §2 (variable staggering).
Marshall et al. (1997) — MITgcm finite-volume formulation.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# ModelState
# ---------------------------------------------------------------------------

class ModelState:
    """
    Container for the prognostic variables of pyOCEAN.

    All fields are xarray DataArrays sharing the coordinate system derived
    from the horizontal and vertical grids.  The class intentionally stores
    no reference to the grid datasets themselves — the grids are only needed
    at construction time to build the coordinate arrays.

    Parameters
    ----------
    u : xr.DataArray, dims (k, j_t, i_u)
        Zonal velocity [m/s].
    v : xr.DataArray, dims (k, j_v, i_t)
        Meridional velocity [m/s].
    w : xr.DataArray, dims (k_f, j_t, i_t)
        Vertical velocity [m/s] at layer interfaces.
    T : xr.DataArray, dims (k, j_t, i_t)
        Potential temperature [°C].
    S : xr.DataArray, dims (k, j_t, i_t)
        Salinity [PSU].
    eta : xr.DataArray, dims (j_t, i_t)
        Sea surface height [m].
    time : float, optional
        Current simulation time [s]. Default 0.0.
    """

    def __init__(
        self,
        u: xr.DataArray,
        v: xr.DataArray,
        w: xr.DataArray,
        T: xr.DataArray,
        S: xr.DataArray,
        eta: xr.DataArray,
        time: float = 0.0,
    ) -> None:
        self.u = u
        self.v = v
        self.w = w
        self.T = T
        self.S = S
        self.eta = eta
        self.time = float(time)

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_grids(
        cls,
        h_grid: xr.Dataset,
        v_grid: xr.Dataset,
        T_init: float = 10.0,
        S_init: float = 35.0,
        time: float = 0.0,
    ) -> "ModelState":
        """
        Create a zero-velocity ModelState from horizontal and vertical grids.

        Velocity components (u, v, w) are initialised to zero.
        Temperature T and salinity S are initialised to uniform values.
        Sea surface height eta is initialised to zero.

        Kinematic BCs are enforced:
            w[:, :, :] = 0  (vertical velocity is zero everywhere initially,
            and w at surface/bottom interfaces is always zero)

        Parameters
        ----------
        h_grid : xr.Dataset
            Horizontal grid from ``create_cartesian_grid`` or
            ``create_spherical_grid``.
        v_grid : xr.Dataset
            Vertical grid from ``create_z_grid`` or ``uniform_z_grid``.
        T_init : float, optional
            Uniform initial potential temperature [°C]. Default 10.0.
        S_init : float, optional
            Uniform initial salinity [PSU]. Default 35.0.
        time : float, optional
            Initial simulation time [s]. Default 0.0.

        Returns
        -------
        ModelState
            Zero-velocity state with uniform T and S.

        Examples
        --------
        >>> from core.grid import create_cartesian_grid, create_z_grid
        >>> hg = create_cartesian_grid(nx=10, ny=8, dx=1e3, dy=1e3)
        >>> vg = create_z_grid(nz=5, dz=[10.0]*5)
        >>> st = ModelState.from_grids(hg, vg)
        >>> st.u.dims
        ('k', 'j_t', 'i_u')
        """
        nx = int(h_grid.attrs["nx"])
        ny = int(h_grid.attrs["ny"])
        nz = int(v_grid.attrs["nz"])

        # Coordinate arrays from the grids
        i_t = h_grid.coords["i_t"].values   # (nx,)
        j_t = h_grid.coords["j_t"].values   # (ny,)
        i_u = h_grid.coords["i_u"].values   # (nx+1,)
        j_v = h_grid.coords["j_v"].values   # (ny+1,)
        k   = v_grid.coords["k"].values     # (nz,)
        k_f = v_grid.coords["k_f"].values   # (nz+1,)

        # Common coord dicts reused across variables
        coords_3d_t = {"k": k, "j_t": j_t, "i_t": i_t}
        coords_3d_u = {"k": k, "j_t": j_t, "i_u": i_u}
        coords_3d_v = {"k": k, "j_v": j_v, "i_t": i_t}
        coords_3d_w = {"k_f": k_f, "j_t": j_t, "i_t": i_t}
        coords_2d   = {"j_t": j_t, "i_t": i_t}

        # ---- Zonal velocity u : (k, j_t, i_u) ----------------------------
        u = xr.DataArray(
            np.zeros((nz, ny, nx + 1), dtype=float),
            dims=["k", "j_t", "i_u"],
            coords=coords_3d_u,
            attrs={"units": "m/s", "long_name": "Zonal velocity",
                   "grid_location": "U-face (western cell face)"},
        )

        # ---- Meridional velocity v : (k, j_v, i_t) -----------------------
        v = xr.DataArray(
            np.zeros((nz, ny + 1, nx), dtype=float),
            dims=["k", "j_v", "i_t"],
            coords=coords_3d_v,
            attrs={"units": "m/s", "long_name": "Meridional velocity",
                   "grid_location": "V-face (southern cell face)"},
        )

        # ---- Vertical velocity w : (k_f, j_t, i_t) -----------------------
        # w[k_f=0]  = 0 (surface: dη/dt replaces w in Phase 2)
        # w[k_f=nz] = 0 (bottom: no normal flow through solid bottom)
        w = xr.DataArray(
            np.zeros((nz + 1, ny, nx), dtype=float),
            dims=["k_f", "j_t", "i_t"],
            coords=coords_3d_w,
            attrs={"units": "m/s", "long_name": "Vertical velocity",
                   "grid_location": "T-column interface (k_f=0 is surface)"},
        )

        # ---- Potential temperature T : (k, j_t, i_t) ---------------------
        T = xr.DataArray(
            np.full((nz, ny, nx), T_init, dtype=float),
            dims=["k", "j_t", "i_t"],
            coords=coords_3d_t,
            attrs={"units": "degC", "long_name": "Potential temperature",
                   "standard_name": "sea_water_potential_temperature",
                   "grid_location": "T-cell centre"},
        )

        # ---- Salinity S : (k, j_t, i_t) ----------------------------------
        S = xr.DataArray(
            np.full((nz, ny, nx), S_init, dtype=float),
            dims=["k", "j_t", "i_t"],
            coords=coords_3d_t,
            attrs={"units": "PSU", "long_name": "Practical salinity",
                   "standard_name": "sea_water_salinity",
                   "grid_location": "T-cell centre"},
        )

        # ---- Sea surface height eta : (j_t, i_t) -------------------------
        eta = xr.DataArray(
            np.zeros((ny, nx), dtype=float),
            dims=["j_t", "i_t"],
            coords=coords_2d,
            attrs={"units": "m", "long_name": "Sea surface height",
                   "standard_name": "sea_surface_height_above_geoid",
                   "grid_location": "T-cell centre"},
        )

        return cls(u=u, v=v, w=w, T=T, S=S, eta=eta, time=time)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def zeros_like(cls, other: "ModelState") -> "ModelState":
        """
        Return a new ModelState with the same grid as *other* but all fields
        set to zero.

        Useful for accumulating tendency arrays (∂u/∂t, ∂T/∂t, …) in the
        timestepper without allocating grids again.

        Parameters
        ----------
        other : ModelState
            Reference state whose coordinate system is copied.

        Returns
        -------
        ModelState
            Zero-filled state with identical coordinates.
        """
        return cls(
            u=xr.zeros_like(other.u),
            v=xr.zeros_like(other.v),
            w=xr.zeros_like(other.w),
            T=xr.zeros_like(other.T),
            S=xr.zeros_like(other.S),
            eta=xr.zeros_like(other.eta),
            time=0.0,
        )

    # ------------------------------------------------------------------
    # Copy / arithmetic helpers
    # ------------------------------------------------------------------

    def copy(self) -> "ModelState":
        """
        Return a deep copy of this state.

        Used by the Adams–Bashforth 2 timestepper to preserve the previous
        time-step values without aliasing.

        Returns
        -------
        ModelState
            Independent copy with all arrays duplicated.
        """
        return ModelState(
            u=self.u.copy(deep=True),
            v=self.v.copy(deep=True),
            w=self.w.copy(deep=True),
            T=self.T.copy(deep=True),
            S=self.S.copy(deep=True),
            eta=self.eta.copy(deep=True),
            time=self.time,
        )

    def __add__(self, other: "ModelState") -> "ModelState":
        """
        Element-wise addition of two states.

        Used in AB2: ``state_new = state_old + dt * tendency``.
        ``time`` is taken from the left-hand operand.
        """
        return ModelState(
            u=self.u + other.u,
            v=self.v + other.v,
            w=self.w + other.w,
            T=self.T + other.T,
            S=self.S + other.S,
            eta=self.eta + other.eta,
            time=self.time,
        )

    def __mul__(self, scalar: float) -> "ModelState":
        """
        Scalar multiplication — used to scale tendency arrays by dt.

        ``dt * tendency`` requires ``tendency.__rmul__`` which delegates here.
        """
        return ModelState(
            u=self.u * scalar,
            v=self.v * scalar,
            w=self.w * scalar,
            T=self.T * scalar,
            S=self.S * scalar,
            eta=self.eta * scalar,
            time=self.time,
        )

    def __rmul__(self, scalar: float) -> "ModelState":
        """Right-multiply: scalar * state."""
        return self.__mul__(scalar)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def to_dataset(self) -> xr.Dataset:
        """
        Combine all prognostic variables into a single xarray Dataset.

        The ``time`` attribute is stored as a dataset-level attribute so that
        ``state.to_dataset().to_netcdf("output.nc")`` produces a self-contained
        snapshot file.

        Returns
        -------
        xr.Dataset
            Dataset with variables ``u``, ``v``, ``w``, ``T``, ``S``, ``eta``
            and a global attribute ``time_s`` (simulation time in seconds).

        Examples
        --------
        >>> ds = state.to_dataset()
        >>> ds.to_netcdf("snapshot_t0.nc")
        """
        ds = xr.Dataset(
            {"u": self.u, "v": self.v, "w": self.w,
             "T": self.T, "S": self.S, "eta": self.eta},
            attrs={
                "description": "pyOCEAN model state snapshot",
                "time_s": self.time,
            },
        )
        return ds

    # ------------------------------------------------------------------
    # Diagnostics / repr
    # ------------------------------------------------------------------

    @property
    def shape(self) -> dict:
        """
        Dictionary of variable names → array shapes.

        Useful for quick sanity checks.

        Returns
        -------
        dict
            E.g. ``{"u": (nz, ny, nx+1), "v": (nz, ny+1, nx), ...}``.
        """
        return {
            "u":   self.u.shape,
            "v":   self.v.shape,
            "w":   self.w.shape,
            "T":   self.T.shape,
            "S":   self.S.shape,
            "eta": self.eta.shape,
        }

    def __repr__(self) -> str:
        nz, ny_t, nx_u = self.u.shape
        nx = nx_u - 1
        ny = ny_t
        return (
            f"ModelState(nx={nx}, ny={ny}, nz={nz}, time={self.time:.1f} s)\n"
            f"  u   {self.u.dims}  {self.u.shape}\n"
            f"  v   {self.v.dims}  {self.v.shape}\n"
            f"  w   {self.w.dims}  {self.w.shape}\n"
            f"  T   {self.T.dims}  {self.T.shape}\n"
            f"  S   {self.S.dims}  {self.S.shape}\n"
            f"  eta {self.eta.dims}  {self.eta.shape}\n"
        )
