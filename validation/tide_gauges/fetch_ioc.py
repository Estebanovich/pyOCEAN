"""
Download raw sea-level data of the Mexican stations from the IOC Sea Level
Station Monitoring Facility (https://www.ioc-sealevelmonitoring.org), which
relays the SMN-UNAM network in near real time.  No QC is applied by IOC.

The service returns at most ~31 days per request, so the period is fetched
in 30-day chunks and cached as JSON under ~/pyOCEAN_refs/sealevel/ioc/.

Usage:  python validation/tide_gauges/fetch_ioc.py [END_DATE] [DAYS]
        (defaults: today, 90 days)
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

API = "https://www.ioc-sealevelmonitoring.org/service.php"
CACHE = Path.home() / "pyOCEAN_refs" / "sealevel" / "ioc"
CHUNK_DAYS = 30


def _get(url: str, path: Path, retries: int = 3) -> None:
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            json.loads(data)                                    # must be valid JSON
            path.write_bytes(data)
            return
        except Exception as err:                                # network or server hiccup
            if k == retries - 1:
                print(f"  FAILED {url}: {err}")
            time.sleep(5 * (k + 1))


def station_list() -> list[dict]:
    """Mexican stations (country MEX), one entry per (code, sensor)."""
    path = CACHE / "stationlist.json"
    if not path.exists():
        _get(f"{API}?query=stationlist&showall=all", path)
    return [s for s in json.loads(path.read_text()) if s.get("country") == "MEX"]


def fetch(end: date, days: int) -> None:
    raw = CACHE / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    codes = sorted({s["Code"] for s in station_list()})
    start = end - timedelta(days=days)
    for code in codes:
        a = start
        while a < end:
            b = min(a + timedelta(days=CHUNK_DAYS), end)
            path = raw / f"{code}_{a:%Y%m%d}_{b:%Y%m%d}.json"
            if not path.exists():
                _get(f"{API}?query=data&code={code}&timestart={a:%Y-%m-%d}&timestop={b:%Y-%m-%d}&format=json",
                     path)
                time.sleep(1.0)                                 # be polite to the service
            a = b
        print(code, flush=True)


if __name__ == "__main__":
    end = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 90
    fetch(end, days)
