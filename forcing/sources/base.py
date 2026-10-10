"""
Forcing adapter interface and source registry.

An adapter translates one external product into the canonical form defined
in ``forcing/canonical.py``.  It is responsible only for:

  * opening the product for a time window and bounding box (``open``), and
  * renaming variables and converting units, signs and time stamps
    (``to_canonical``).

Grid and time interpolation are shared (``forcing/interp.py``) and are
never done inside an adapter.  ``ForcingSource.load`` chains both steps and
validates the result, so a unit-conversion bug in a new adapter fails
immediately instead of silently corrupting a run.

Adapters register themselves under a short name so that a configuration
file can mix products per forcing category, e.g. atmosphere from WRF,
lateral boundaries from ESPC-HYCOM and tides from TPXO, which is the
operational setup of HYCOM-IOA.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, ClassVar

import xarray as xr

from forcing.canonical import validate_canonical


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BBox:
    """
    Geographic bounding box used to subset a forcing product.

    Attributes
    ----------
    lon_min, lon_max : float
        Longitude limits [°E], −180–180 convention.
    lat_min, lat_max : float
        Latitude limits [°N].
    """

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    def __post_init__(self) -> None:
        if self.lat_min >= self.lat_max:
            raise ValueError(f"lat_min ({self.lat_min}) must be < lat_max ({self.lat_max})")
        if self.lon_min >= self.lon_max:
            raise ValueError(f"lon_min ({self.lon_min}) must be < lon_max ({self.lon_max})")

    @classmethod
    def from_grid(cls, grid: xr.Dataset, pad: float = 1.0) -> "BBox":
        """
        Bounding box enclosing every T, U and V point of a model grid.

        Parameters
        ----------
        grid : xr.Dataset
            pyOCEAN spherical horizontal grid.
        pad : float
            Margin added on every side [degrees], so that bilinear
            interpolation at the domain edge has source points on both sides.
        """
        lons = [grid[f"lon_{p}"] for p in ("t", "u", "v")]
        lats = [grid[f"lat_{p}"] for p in ("t", "u", "v")]
        return cls(
            lon_min=min(float(x.min()) for x in lons) - pad,
            lon_max=max(float(x.max()) for x in lons) + pad,
            lat_min=min(float(y.min()) for y in lats) - pad,
            lat_max=max(float(y.max()) for y in lats) + pad,
        )


# ---------------------------------------------------------------------------
# Adapter base class
# ---------------------------------------------------------------------------

class ForcingSource(ABC):
    """
    Abstract forcing adapter.

    Subclasses set the class attribute ``kind`` ('atmosphere' or 'ocean')
    and implement ``open`` and ``to_canonical``.
    """

    kind: ClassVar[str]

    @abstractmethod
    def open(self, t0: datetime, t1: datetime, bbox: BBox) -> xr.Dataset:
        """Return the raw product subset covering [t0, t1] and ``bbox`` (lazy if possible)."""

    @abstractmethod
    def to_canonical(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename variables and convert units, signs and time stamps to canonical form."""

    def load(self, t0: datetime, t1: datetime, bbox: BBox) -> xr.Dataset:
        """
        Open, translate and validate the product for a time window.

        Returns
        -------
        xr.Dataset
            Canonical dataset, with provenance attributes ``pyocean_source``
            and ``pyocean_kind``.

        Raises
        ------
        ValueError
            If ``t0 >= t1`` or the adapter output violates the canonical
            contract.
        """
        if t0 >= t1:
            raise ValueError(f"t0 ({t0}) must be earlier than t1 ({t1})")
        ds = self.to_canonical(self.open(t0, t1, bbox))
        ds.attrs["pyocean_source"] = type(self).__name__
        ds.attrs["pyocean_kind"] = self.kind
        return validate_canonical(ds, self.kind)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[ForcingSource]] = {}


def register_source(name: str) -> Callable[[type[ForcingSource]], type[ForcingSource]]:
    """
    Class decorator registering an adapter under ``name``.

    Example
    -------
    >>> @register_source("era5")
    ... class ERA5Source(ForcingSource): ...
    """
    def decorator(cls: type[ForcingSource]) -> type[ForcingSource]:
        if not (isinstance(cls, type) and issubclass(cls, ForcingSource)):
            raise TypeError(f"{cls!r} is not a ForcingSource subclass")
        if name in _REGISTRY:
            raise ValueError(f"forcing source {name!r} is already registered")
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_source(name: str, **kwargs) -> ForcingSource:
    """Instantiate the adapter registered as ``name`` with ``kwargs``."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown forcing source {name!r}; available: {available_sources()}"
        ) from None
    return cls(**kwargs)


def available_sources() -> list[str]:
    """Sorted names of all registered adapters."""
    return sorted(_REGISTRY)
