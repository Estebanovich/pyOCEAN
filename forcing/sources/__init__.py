"""Forcing adapters: one module per external product (ERA5, WRF, ESPC-HYCOM, TPXO, …)."""

from forcing.sources.base import (
    BBox,
    ForcingSource,
    available_sources,
    get_source,
    register_source,
)

# Importing an adapter module registers it under its short name.
from forcing.sources import era5, hycom_z, wrf  # noqa: E402,F401

__all__ = ["BBox", "ForcingSource", "available_sources", "get_source", "register_source"]
