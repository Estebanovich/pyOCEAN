"""
Thread count of the compiled (Numba) kernels of the 3-D core.  One list object shared by every module
(``core.dynamics.sigma_baroclinic`` re-exports it), so ``set_threads`` reaches every component.
"""
from __future__ import annotations

import numba

_THREADS = [1]


def set_threads(n: int) -> None:
    """
    Threads for the compiled kernels (default 1).  Rows are independent, so results are
    bit-identical for any thread count.  On a laptop keep the total over concurrent runs at
    or below the number of performance cores.
    """
    _THREADS[0] = max(1, min(int(n), numba.config.NUMBA_NUM_THREADS))
