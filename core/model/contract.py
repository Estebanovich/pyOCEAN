"""
Component contract of the 3-D core (N1, spec 2026-10-28-n1-component-core-design.md §2).

A component is a small object with one role in the time step (``Role``).  It declares the variables it reads and
writes (names of ``core.model.registry.REGISTRY``), the state it carries from one step to the next
(``state_fields``) and the configuration problems it can detect before the first step (``check`` → ``Issue``).
The design follows the four reference models read for N1 (docs/design/n1_refs/):

* parameters documented where they are declared, with units and citation — MOM6 ``get_param``
  (MOM_file_parser.F90; mom6.md §3);
* all configuration errors collected and reported once — MITgcm ``CONFIG_CHECK`` (model/src/config_check.F,
  errCount, mitgcm.md §2) and CROCO ``check_kwds`` (read_inp.F:3111-3134, croco.md §2);
* messages that name the value, the valid options and the fix — Oceananigans (hydrostatic_free_surface_model.jl
  argument checks, oceananigans.md §5).
"""
from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np


class Role(str, Enum):
    """Roles of the components, one per place in the fixed step order (spec §4)."""
    EOS = "eos"
    VERTICAL_COORDINATE = "vertical_coordinate"
    PGF = "pgf"
    MOMENTUM = "momentum"
    TIME_STEPPER = "time_stepper"
    FREE_SURFACE = "free_surface"
    CLOSURE = "closure"
    BOTTOM_DRAG = "bottom_drag"
    LAYER_TRANSPORT = "layer_transport"
    TRACER_ADVECTION = "tracer_advection"
    TRACER_DIFFUSION = "tracer_diffusion"
    OPEN_BOUNDARY = "open_boundary"
    INTERIOR_SPONGE = "interior_sponge"
    SURFACE_FLUXES = "surface_fluxes"
    DIAGNOSTICS = "diagnostics"


@dataclass(frozen=True)
class Issue:
    """One configuration problem found before the first step."""
    level: Literal["error", "warning"]
    component: str
    message: str


class PyOceanConfigWarning(UserWarning):
    """A configuration that runs but probably does not do what was meant (e.g. an argument ignored in σ)."""


class ConfigurationError(ValueError):
    """Every configuration error found by the pre-flight, reported together."""


def param(default: Any, *, units: str, doc: str, cite: str = "", benchmark: str = "") -> Any:
    """
    Dataclass field with its documentation: units, meaning, the paper (or model source) it comes from and the
    benchmark that validated the default.  The model documentation is generated from these metadata.
    """
    return dataclasses.field(default=default, metadata={"units": units, "doc": doc, "cite": cite,
                                                        "benchmark": benchmark})


@dataclass
class BuildContext:
    """What a component needs to know about the rest of the model to check its configuration."""
    vertical: Literal["sigma_eulerian", "ale"]
    regrid: str | None = None
    open_edges: tuple[str, ...] = ()
    has_state_provider: bool = False
    dt: float = 0.0
    n_tracers: int = 0
    given: frozenset[str] = frozenset()


@runtime_checkable
class Component(Protocol):
    """The contract every component of the 3-D core satisfies (spec §2)."""
    name: str
    role: Role
    reads: tuple[str, ...]
    writes: tuple[str, ...]

    def check(self, ctx: BuildContext) -> list[Issue]:
        ...

    def state_fields(self) -> dict[str, np.ndarray]:
        ...


def raise_issues(issues: list[Issue]) -> None:
    """Emit every warning, then raise one ``ConfigurationError`` listing every error (if any)."""
    for i in issues:
        if i.level == "warning":
            warnings.warn(f"[{i.component}] {i.message}", PyOceanConfigWarning, stacklevel=3)
    errors = [i for i in issues if i.level == "error"]
    if len(errors) == 1:
        raise ConfigurationError(errors[0].message)
    if errors:
        lines = "\n".join(f"  - [{i.component}] {i.message}" for i in errors)
        raise ConfigurationError(f"{len(errors)} configuration errors:\n{lines}")
