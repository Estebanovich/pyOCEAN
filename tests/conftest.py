"""
pytest configuration: cap Numba at the 5 performance cores before any test module imports Numba (Esteban's
rule: never the efficiency cores).  Kernels called directly by tests otherwise use all 11 cores.
"""
import os

os.environ["NUMBA_NUM_THREADS"] = str(min(int(os.environ.get("NUMBA_NUM_THREADS", "5")), 5))
