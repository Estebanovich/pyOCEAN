"""
The test session must never use the efficiency cores (Esteban's rule: ≤ 5 threads, the 5 performance cores of
the M3 Pro).  Kernels called directly from tests take Numba's default thread count, so tests/conftest.py caps it.
"""
import numba


def test_numba_threads_are_capped_at_the_performance_cores():
    assert numba.config.NUMBA_NUM_THREADS <= 5
    assert numba.get_num_threads() <= 5
