"""
HYCOM timer statistics (xctmrp, mod_xc_mp.h:5436-5447): ``(3x,a6,'   calls =',i9,'   time =',f11.5,...)`` lines
after each ``timer statistics`` header.  HYCOM prints one block after the start-up (mod_hycom.F90:2169, then resets
the timers) and one at the end, so the last block is the time stepping alone.

Usage: python validation/ilicak_benchmark/hycom/timers.py <stdout> [simulated hours]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_LINE = re.compile(r"^\s*(\w+)\s+calls =\s*(\d+)\s+time =\s*([\d.]+)", re.M)


def parse(stdout: str) -> dict[str, float]:
    """name → time [s] (and name_calls → calls) of the last ``timer statistics`` block."""
    last = stdout.rsplit("timer statistics", 1)[-1]
    out: dict[str, float] = {}
    for name, calls, t in _LINE.findall(last):
        out[name] = float(t)
        out[f"{name}_calls"] = int(calls)
    return out


if __name__ == "__main__":
    t = parse(Path(sys.argv[1]).read_text(errors="replace"))
    hours = float(sys.argv[2]) if len(sys.argv) > 2 else None
    split = "  ".join(f"{k} {t[k]:.3f}" for k in ("cnuity", "tsadvc", "momtum", "barotp", "hybgen") if k in t)
    rate = f"  s/h {t['total'] / hours:.4f}" if hours and "total" in t else ""
    print(f"total {t.get('total', float('nan')):.3f}  steps {t.get('cnuity_calls', 0)}{rate}  [{split}]")
