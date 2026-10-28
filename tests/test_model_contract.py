"""N1 task 1: component contract (core/model/contract.py) and variable registry (core/model/registry.py)."""
from __future__ import annotations

import dataclasses
import warnings

import pytest

from core.dynamics.sigma_baroclinic import SigmaState
from core.model.contract import ConfigurationError, Issue, PyOceanConfigWarning, Role, param, raise_issues
from core.model.registry import REGISTRY


def test_raise_issues_reports_every_error_once_and_warns():
    issues = [Issue("error", "pgf", "pgf='bad' is not valid"), Issue("warning", "ale", "regrid ignored in σ"),
              Issue("error", "momentum", "adv='bad' is not valid")]
    with pytest.warns(PyOceanConfigWarning, match="regrid ignored"):
        with pytest.raises(ConfigurationError) as e:
            raise_issues(issues)
    assert "pgf='bad'" in str(e.value) and "adv='bad'" in str(e.value)
    assert isinstance(e.value, ValueError)


def test_raise_issues_warnings_only_does_not_raise():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        raise_issues([Issue("warning", "x", "just a warning")])
    assert len(w) == 1 and issubclass(w[0].category, PyOceanConfigWarning)


def test_param_carries_units_doc_and_citation():
    @dataclasses.dataclass(frozen=True)
    class Spec:
        kv: float = param(1e-5, units="m2 s-1", doc="tracer vertical diffusivity", cite="Vallis 2006")
    f = dataclasses.fields(Spec)[0]
    assert Spec().kv == 1e-5
    assert f.metadata["units"] == "m2 s-1" and f.metadata["cite"] == "Vallis 2006"
    assert f.metadata["doc"] == "tracer vertical diffusivity"


def test_roles_cover_the_spec():
    assert {r.value for r in Role} >= {"eos", "vertical_coordinate", "pgf", "momentum", "time_stepper",
                                       "free_surface", "closure", "bottom_drag", "layer_transport",
                                       "tracer_advection", "tracer_diffusion", "open_boundary",
                                       "interior_sponge", "surface_fluxes", "diagnostics"}


def test_registry_covers_every_state_field():
    names = {f.name for f in dataclasses.fields(SigmaState)} - {"time"}
    assert names <= set(REGISTRY)
    for spec in REGISTRY.values():
        assert spec.kind in ("prognostic", "history", "component_state", "diagnostic")
        assert spec.location in ("t", "u", "v", "w", "q", "scalar")
    assert REGISTRY["u"].location == "u" and REGISTRY["u"].kind == "prognostic"
    assert REGISTRY["g_hist"].kind == "history"
    assert {"closure.tke", "closure.eps", "closure.akv"} <= set(REGISTRY)
