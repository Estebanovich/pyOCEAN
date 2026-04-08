"""core.grid — Grid generation and coordinate utilities."""

from .horizontal import create_cartesian_grid, create_spherical_grid, set_land_mask
from .vertical import (
    compute_hfac,
    create_z_grid,
    stretched_z_grid,
    uniform_z_grid,
)
from .bathymetry import (
    apply_min_depth,
    build_bathymetry,
    fill_isolated_cells,
    load_gebco,
    regrid_to_model,
    smooth_bathymetry,
)

__all__ = [
    "create_cartesian_grid",
    "create_spherical_grid",
    "set_land_mask",
    "create_z_grid",
    "uniform_z_grid",
    "stretched_z_grid",
    "compute_hfac",
    "load_gebco",
    "regrid_to_model",
    "smooth_bathymetry",
    "fill_isolated_cells",
    "apply_min_depth",
    "build_bathymetry",
]
