"""
Tests for diagnostics/io.py.

Validation strategy:
  1. Round-trip accuracy — values, shapes, dims, coordinates unchanged.
  2. Attribute preservation — units, long_name, time_s.
  3. DiagnosticsWriter — file creation, time dimension, variable selection.
  4. Context-manager protocol.
  5. Edge cases — empty writer, single-variable output, zero time.

Covers
------
  write_snapshot      : creates file, values match, time_s stored
  read_snapshot       : values match after roundtrip, time preserved
  DiagnosticsWriter   : write/close, time dim unlimited, subset vars,
                        n_snapshots counter, context manager, empty close
"""

import numpy as np
import pytest
import xarray as xr

from core.grid.horizontal import create_cartesian_grid
from core.grid.vertical import uniform_z_grid
from core.state import ModelState
from diagnostics.io import write_snapshot, read_snapshot, DiagnosticsWriter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NX, NY, NZ = 6, 4, 3
DX = DY = 1000.0
DZ = 20.0
F0 = 1e-4


@pytest.fixture
def hgrid():
    return create_cartesian_grid(nx=NX, ny=NY, dx=DX, dy=DY, f0=F0)


@pytest.fixture
def vgrid():
    return uniform_z_grid(nz=NZ, dz_uniform=DZ)


@pytest.fixture
def state(hgrid, vgrid):
    """State with non-trivial values for roundtrip checks."""
    st = ModelState.from_grids(hgrid, vgrid, T_init=15.0, S_init=35.0, time=3600.0)
    # Assign non-trivial values so we can tell if values are preserved
    T_vals = st.T.values.copy()
    T_vals[0, :, :] = 18.0   # warm surface
    T_vals[-1, :, :] = 12.0  # cold bottom
    st.T = xr.DataArray(T_vals, dims=st.T.dims, coords=st.T.coords, attrs=st.T.attrs)

    u_vals = st.u.values.copy()
    u_vals[:] = 0.05
    st.u = xr.DataArray(u_vals, dims=st.u.dims, coords=st.u.coords, attrs=st.u.attrs)

    eta_vals = np.random.default_rng(42).uniform(-0.01, 0.01, (NY, NX))
    st.eta = xr.DataArray(eta_vals, dims=st.eta.dims, coords=st.eta.coords,
                           attrs=st.eta.attrs)
    return st


# ===========================================================================
# write_snapshot / read_snapshot
# ===========================================================================

class TestSnapshotRoundtrip:

    def test_file_is_created(self, state, tmp_path):
        p = tmp_path / "snap.nc"
        write_snapshot(state, p)
        assert p.exists(), "write_snapshot should create the file"

    def test_T_values_preserved(self, state, tmp_path):
        p = tmp_path / "snap.nc"
        write_snapshot(state, p)
        st2 = read_snapshot(p)
        assert np.allclose(st2.T.values, state.T.values)

    def test_S_values_preserved(self, state, tmp_path):
        p = tmp_path / "snap.nc"
        write_snapshot(state, p)
        st2 = read_snapshot(p)
        assert np.allclose(st2.S.values, state.S.values)

    def test_u_values_preserved(self, state, tmp_path):
        p = tmp_path / "snap.nc"
        write_snapshot(state, p)
        st2 = read_snapshot(p)
        assert np.allclose(st2.u.values, state.u.values)

    def test_eta_values_preserved(self, state, tmp_path):
        p = tmp_path / "snap.nc"
        write_snapshot(state, p)
        st2 = read_snapshot(p)
        assert np.allclose(st2.eta.values, state.eta.values)

    def test_time_preserved(self, state, tmp_path):
        p = tmp_path / "snap.nc"
        write_snapshot(state, p)
        st2 = read_snapshot(p)
        assert st2.time == pytest.approx(state.time)

    def test_zero_time_preserved(self, hgrid, vgrid, tmp_path):
        st = ModelState.from_grids(hgrid, vgrid, time=0.0)
        p = tmp_path / "t0.nc"
        write_snapshot(st, p)
        st2 = read_snapshot(p)
        assert st2.time == 0.0

    def test_shapes_preserved(self, state, tmp_path):
        p = tmp_path / "snap.nc"
        write_snapshot(state, p)
        st2 = read_snapshot(p)
        assert st2.u.shape == state.u.shape
        assert st2.v.shape == state.v.shape
        assert st2.w.shape == state.w.shape
        assert st2.T.shape == state.T.shape
        assert st2.S.shape == state.S.shape
        assert st2.eta.shape == state.eta.shape

    def test_dims_preserved(self, state, tmp_path):
        p = tmp_path / "snap.nc"
        write_snapshot(state, p)
        st2 = read_snapshot(p)
        assert st2.T.dims == state.T.dims
        assert st2.u.dims == state.u.dims

    def test_units_attr_preserved(self, state, tmp_path):
        p = tmp_path / "snap.nc"
        write_snapshot(state, p)
        st2 = read_snapshot(p)
        assert "units" in st2.T.attrs
        assert st2.T.attrs["units"] == state.T.attrs["units"]

    def test_variable_subset_written(self, state, tmp_path):
        """write_snapshot with variables=["T","S"] should omit u, v, w, eta."""
        p = tmp_path / "tracer.nc"
        write_snapshot(state, p, variables=["T", "S"])
        ds = xr.open_dataset(p)
        assert "T" in ds
        assert "S" in ds
        assert "u" not in ds
        assert "eta" not in ds
        ds.close()

    def test_conventions_global_attr(self, state, tmp_path):
        p = tmp_path / "snap.nc"
        write_snapshot(state, p)
        ds = xr.open_dataset(p)
        assert "Conventions" in ds.attrs
        ds.close()


# ===========================================================================
# DiagnosticsWriter
# ===========================================================================

class TestDiagnosticsWriter:

    def test_file_created_on_close(self, state, tmp_path):
        p = tmp_path / "hist.nc"
        dw = DiagnosticsWriter(p)
        dw.write(state)
        dw.close()
        assert p.exists()

    def test_context_manager_creates_file(self, state, tmp_path):
        p = tmp_path / "hist.nc"
        with DiagnosticsWriter(p) as dw:
            dw.write(state)
        assert p.exists()

    def test_empty_close_is_noop(self, tmp_path):
        p = tmp_path / "empty.nc"
        dw = DiagnosticsWriter(p)
        dw.close()
        assert not p.exists(), "Empty writer should not create a file"

    def test_n_snapshots_counter(self, state, tmp_path):
        p = tmp_path / "hist.nc"
        with DiagnosticsWriter(p) as dw:
            assert dw.n_snapshots == 0
            dw.write(state)
            assert dw.n_snapshots == 1
            dw.write(state)
            assert dw.n_snapshots == 2

    def test_time_dimension_present(self, state, tmp_path):
        p = tmp_path / "hist.nc"
        with DiagnosticsWriter(p) as dw:
            dw.write(state)
        ds = xr.open_dataset(p)
        assert "time" in ds.dims
        ds.close()

    def test_multiple_snapshots_time_dim(self, hgrid, vgrid, tmp_path):
        """Three writes at different times → time dim length == 3."""
        p = tmp_path / "hist3.nc"
        times = [0.0, 3600.0, 7200.0]
        with DiagnosticsWriter(p) as dw:
            for t in times:
                st = ModelState.from_grids(hgrid, vgrid, T_init=15.0 + t / 3600.0,
                                           time=t)
                dw.write(st)
        ds = xr.open_dataset(p)
        assert ds.sizes["time"] == 3
        assert np.allclose(ds["time"].values, times)
        ds.close()

    def test_T_values_in_time_series(self, hgrid, vgrid, tmp_path):
        """T values at each time step are correctly stored."""
        p = tmp_path / "T_series.nc"
        T_vals = [10.0, 12.0, 14.0]
        with DiagnosticsWriter(p, variables=["T"]) as dw:
            for T_init in T_vals:
                st = ModelState.from_grids(hgrid, vgrid, T_init=T_init,
                                           time=float(T_init))
                dw.write(st)
        ds = xr.open_dataset(p)
        for i, T_expected in enumerate(T_vals):
            assert np.allclose(ds["T"].isel(time=i).values, T_expected)
        ds.close()

    def test_variable_subset_in_output(self, state, tmp_path):
        """Variables not requested should be absent from the file."""
        p = tmp_path / "subset.nc"
        with DiagnosticsWriter(p, variables=["T", "S"]) as dw:
            dw.write(state)
        ds = xr.open_dataset(p)
        assert "T" in ds
        assert "S" in ds
        assert "u" not in ds
        assert "eta" not in ds
        ds.close()

    def test_shapes_in_time_series(self, state, tmp_path):
        """Variable shapes in the file match state shapes."""
        p = tmp_path / "shapes.nc"
        with DiagnosticsWriter(p) as dw:
            dw.write(state)
        ds = xr.open_dataset(p)
        assert ds["T"].shape == (1, NZ, NY, NX)       # (time, k, j_t, i_t)
        assert ds["u"].shape == (1, NZ, NY, NX + 1)   # (time, k, j_t, i_u)
        assert ds["eta"].shape == (1, NY, NX)          # (time, j_t, i_t)
        ds.close()

    def test_context_manager_flushes_on_exception(self, state, tmp_path):
        """DiagnosticsWriter.close() is called even when an exception occurs."""
        p = tmp_path / "except.nc"
        try:
            with DiagnosticsWriter(p) as dw:
                dw.write(state)
                raise RuntimeError("simulated error")
        except RuntimeError:
            pass
        # File should still be flushed despite the exception
        assert p.exists()
