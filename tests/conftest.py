"""
pytest configuration: cap Numba at the 5 performance cores before any test module imports Numba (Esteban's
rule: never the efficiency cores).  Kernels called directly by tests otherwise use all 11 cores.
"""
import os

os.environ["NUMBA_NUM_THREADS"] = str(min(int(os.environ.get("NUMBA_NUM_THREADS", "5")), 5))

# ---------------------------------------------------------------------------------------------------------------
# Test categories that depend on the machine (used by CI, .github/workflows/tests.yml)
#
# * golden: compares a run with state hashes or arrays stored on the development machine (Apple M3 Pro, base conda
#   env, docs/manual/getting-started/installation.md).  The free-surface solve and BLAS round-off differ between
#   builds of SciPy/OpenBLAS, so these hashes hold only on that machine.  They guard "no physics change" there.
#   Skipped when PYOCEAN_SKIP_GOLDEN=1.
# * localdata: needs files outside the repository (MITgcm thesis runs on /Volumes, ~/pyOCEAN_refs model sources).
#   Skipped when PYOCEAN_SKIP_LOCALDATA=1.
# ---------------------------------------------------------------------------------------------------------------
import pytest  # noqa: E402

_GOLDEN = (
    "tests/test_n1_golden.py::",
    "tests/test_closure_component.py::test_closure_extraction_is_bitwise",
    "tests/test_closure_component.py::test_gls_stratified_bitwise_after_extraction",
    "tests/test_preflight.py::test_ignored_args_warn_and_keep_bits",
    "tests/test_periodic.py::test_no_periodic_is_bitwise",
    "tests/test_smagorinsky_model.py::test_zero_smagorinsky_is_bitwise",
    "tests/test_biharmonic_model.py::test_zero_biharmonic_is_bitwise",
    "tests/test_surface_heat.py::test_no_heat_is_bitwise",
    "tests/test_model_driver.py::test_keep_terms_does_not_change_bits",
    "tests/test_model_driver.py::test_driver_reuse_model_two_runs",
    "tests/test_n2a_budget.py::test_budget_off_is_bitwise",
    "tests/test_n2a_salinity.py::test_uniform_S_at_s_ref_leaves_dynamics_bitwise",
    "tests/test_ale_loglaw.py::test_quadratic_drag_unchanged",
    "tests/test_sigma_baroclinic.py::test_sigma_resting_exterior_bit_for_bit",
    "tests/test_boundary_tides_3d.py::test_tides_off_bit_for_bit",
)
_LOCALDATA = (
    "tests/test_mom6_bayiw.py::",
    "tests/test_bayiw_baroclinic.py::test_realistic_case_uses_the_density_equivalent_temperature",
    "tests/test_dome_benchmark.py::test_mom6_input_has_the_common_physics_and_the_dome_configs",
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "golden: compares with state hashes stored on the development machine")
    config.addinivalue_line("markers", "localdata: needs data files outside the repository")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_golden = os.environ.get("PYOCEAN_SKIP_GOLDEN") == "1"
    skip_local = os.environ.get("PYOCEAN_SKIP_LOCALDATA") == "1"
    for item in items:
        nodeid = item.nodeid if item.nodeid.startswith("tests/") else "tests/" + item.nodeid
        if nodeid.startswith(_GOLDEN):
            item.add_marker(pytest.mark.golden)
            if skip_golden:
                item.add_marker(pytest.mark.skip(reason="golden hashes of the development machine"))
        if nodeid.startswith(_LOCALDATA):
            item.add_marker(pytest.mark.localdata)
            if skip_local:
                item.add_marker(pytest.mark.skip(reason="needs data outside the repository"))
