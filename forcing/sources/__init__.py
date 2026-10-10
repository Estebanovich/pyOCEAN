"""Forcing adapters: one module per external product (ERA5, WRF, ESPC-HYCOM, TPXO, …)."""

from forcing.sources.base import (
    BBox,
    ForcingSource,
    available_sources,
    get_source,
    register_source,
)

__all__ = ["BBox", "ForcingSource", "available_sources", "get_source", "register_source"]
