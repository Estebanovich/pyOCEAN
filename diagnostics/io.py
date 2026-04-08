"""
NetCDF4 I/O for pyOCEAN — Phase 1.

Provides two complementary interfaces for writing and reading model output:

1. **Snapshot I/O** — read/write a single ModelState to a self-contained
   NetCDF4 file (e.g. restart files, one-shot validation output):

       write_snapshot(state, "restart_t3600.nc")
       state = read_snapshot("restart_t3600.nc")

2. **DiagnosticsWriter** — accumulate multiple ModelState snapshots during a
   run and flush them to a single NetCDF4 file with an unlimited ``time``
   dimension (e.g. hourly model history output):

       with DiagnosticsWriter("history.nc", variables=["T", "S", "eta"]) as dw:
           for step in range(n_steps):
               state = ab2_step(...)
               if step % output_interval == 0:
                   dw.write(state)

File format
-----------
All files are standard NetCDF4 (HDF5 backend via the netCDF4 library, called
through xarray).  Variables follow CF Conventions 1.8 where possible:

    time          — unlimited dimension, seconds since model start [s]
    k             — layer index (0 = surface) [–]
    k_f           — interface index (0 = surface interface) [–]
    j_t, i_t      — T-cell row/column indices [m] (or degrees for spherical)
    i_u           — U-face column index [m]
    j_v           — V-face row index [m]

    u(time, k, j_t, i_u)   Zonal velocity         [m/s]
    v(time, k, j_v, i_t)   Meridional velocity    [m/s]
    w(time, k_f, j_t, i_t) Vertical velocity      [m/s]
    T(time, k, j_t, i_t)   Potential temperature  [°C]
    S(time, k, j_t, i_t)   Salinity               [PSU]
    eta(time, j_t, i_t)     Sea surface height     [m]

Snapshot files (no ``time`` dimension) use the same variable/dimension names
with the ``time_s`` global attribute recording the simulation time.

Design notes
------------
* ``DiagnosticsWriter`` accumulates datasets in memory and writes once on
  ``close()``.  This is simple and reliable for Phase 1 domains (small grids,
  short runs).  For Phase 2+ operational runs, consider streaming writes
  using ``netCDF4`` directly with ``unlimited_dims``.
* ``read_snapshot`` always returns a fresh ``ModelState``; it does not cache
  open file handles.
* A subset of variables can be requested in ``DiagnosticsWriter`` to reduce
  output size (e.g. T and S only for tracer diagnostics).

References
----------
Unidata (2023). NetCDF User's Guide.
CF Conventions Working Group (2022). CF Conventions Version 1.10.
xarray Developers (2023). xarray documentation: to_netcdf.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import xarray as xr

from core.state import ModelState

# All prognostic variables in canonical output order
_ALL_VARS: tuple[str, ...] = ("u", "v", "w", "T", "S", "eta")


# ---------------------------------------------------------------------------
# Snapshot I/O
# ---------------------------------------------------------------------------

def write_snapshot(
    state: ModelState,
    path: str | Path,
    variables: Iterable[str] | None = None,
) -> None:
    """
    Write a single ModelState snapshot to a NetCDF4 file.

    The file can be used as a restart file or for single-time validation
    output.  The simulation time is stored in the global attribute ``time_s``
    [s] and in a scalar coordinate variable ``time`` [s].

    Parameters
    ----------
    state : ModelState
        State to write.
    path : str or Path
        Output file path.  An existing file is overwritten.
    variables : iterable of str, optional
        Subset of variable names to write.  Default: all six prognostic
        variables (``"u"``, ``"v"``, ``"w"``, ``"T"``, ``"S"``, ``"eta"``).

    Examples
    --------
    >>> write_snapshot(state, "output/restart_t3600.nc")
    >>> write_snapshot(state, "tracer.nc", variables=["T", "S"])
    """
    vars_to_write = list(variables) if variables is not None else list(_ALL_VARS)
    full_ds = state.to_dataset()
    ds = xr.Dataset(
        {v: full_ds[v] for v in vars_to_write},
        attrs={
            "description": "pyOCEAN model state snapshot",
            "time_s": state.time,
            "Conventions": "CF-1.8",
        },
    )
    ds.to_netcdf(Path(path), mode="w")


def read_snapshot(path: str | Path) -> ModelState:
    """
    Read a ModelState snapshot from a NetCDF4 file written by
    :func:`write_snapshot`.

    Parameters
    ----------
    path : str or Path
        Input file path.

    Returns
    -------
    ModelState
        Reconstructed model state.  The simulation time is read from the
        ``time_s`` global attribute (falls back to 0.0 if absent).

    Raises
    ------
    KeyError
        If any of the six prognostic variables is missing from the file.

    Examples
    --------
    >>> state = read_snapshot("output/restart_t3600.nc")
    >>> state.time   # seconds
    3600.0
    """
    ds = xr.open_dataset(Path(path))
    time = float(ds.attrs.get("time_s", 0.0))
    state = ModelState(
        u=ds["u"],
        v=ds["v"],
        w=ds["w"],
        T=ds["T"],
        S=ds["S"],
        eta=ds["eta"],
        time=time,
    )
    ds.close()
    return state


# ---------------------------------------------------------------------------
# Time-series diagnostics writer
# ---------------------------------------------------------------------------

class DiagnosticsWriter:
    """
    Accumulate ModelState snapshots and flush them to a NetCDF4 file with
    an unlimited ``time`` dimension.

    Usage as a context manager (recommended)::

        with DiagnosticsWriter("history.nc", variables=["T", "eta"]) as dw:
            for step in range(n_steps):
                state = timestepper(...)
                if step % save_interval == 0:
                    dw.write(state)
        # file is flushed automatically on exit

    Or explicitly::

        dw = DiagnosticsWriter("history.nc")
        dw.write(state_t0)
        dw.write(state_t1)
        dw.close()   # MUST call close() to flush to disk

    Parameters
    ----------
    path : str or Path
        Output file path.  An existing file is overwritten on ``close()``.
    variables : iterable of str, optional
        Subset of prognostic variables to save.  Default: all six.

    Notes
    -----
    Snapshots are held in memory until ``close()`` is called.  For Phase 1
    domains (small grids, short runs) this is negligible.  For Phase 2+
    operational runs, the list is still flushed at once, which limits memory
    usage only for the diagnostics subset — not the full model state.
    """

    def __init__(
        self,
        path: str | Path,
        variables: Iterable[str] | None = None,
    ) -> None:
        self._path = Path(path)
        self._vars: list[str] = list(variables) if variables is not None else list(_ALL_VARS)
        self._datasets: list[xr.Dataset] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def write(self, state: ModelState) -> None:
        """
        Append a ModelState snapshot to the in-memory buffer.

        Each snapshot is tagged with ``state.time`` [s] along the ``time``
        dimension.  Duplicate time values are allowed (e.g. for sub-stepping
        output); the order of insertion is preserved in the final file.

        Parameters
        ----------
        state : ModelState
            Snapshot to record.
        """
        full_ds = state.to_dataset()
        ds = xr.Dataset(
            {v: full_ds[v] for v in self._vars if v in full_ds},
            attrs=full_ds.attrs,
        )
        # Prepend a scalar 'time' dimension so xr.concat stacks along it
        ds = ds.expand_dims({"time": [state.time]})
        self._datasets.append(ds)

    def close(self) -> None:
        """
        Flush all buffered snapshots to disk and release memory.

        If no snapshots have been written, this is a no-op.
        ``close()`` is idempotent — calling it a second time after an empty
        buffer is safe.

        The output file contains a ``time`` unlimited dimension and standard
        CF global attributes.
        """
        if not self._datasets:
            return
        combined = xr.concat(self._datasets, dim="time")
        combined.attrs.update({
            "description": "pyOCEAN model diagnostics time series",
            "Conventions": "CF-1.8",
        })
        combined.to_netcdf(self._path, mode="w", unlimited_dims=["time"])
        self._datasets.clear()

    @property
    def n_snapshots(self) -> int:
        """Number of snapshots buffered so far (not yet written to disk)."""
        return len(self._datasets)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "DiagnosticsWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
