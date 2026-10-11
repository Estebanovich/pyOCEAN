"""
Reader for native HYCOM ``.a/.b`` files.

HYCOM's binary format (Wallcraft et al. 2009, HYCOM User's Guide §4) is
used for grids (regional.grid), bathymetry (regional.depth), archives
(archv.*), restart, relaxation and forcing files — including the nests and
WRF-derived forcing of HYCOM-IOA.

``.a`` file
    Sequence of 2-D records.  Each record is idm·jdm big-endian IEEE
    float32 values (i varying fastest), padded with zeros to a multiple of
    4096 words.  Land / void points hold the HYCOM special value 2¹⁰⁰,
    returned here as NaN.

``.b`` file
    Text.  Header lines ``<value> '<name>' = <comment>`` (e.g. idm, jdm,
    iversn) and one line per ``.a`` record, in order, giving its name and
    min/max.  Three dialects are recognised:

    * archive:   ``temp     =  52272  45006.000  2 34.000  -2.1E+00  2.9E+01``
                 (name, time step, model day, layer k, target density, min, max)
    * grid/depth: ``plon:  min,max =  -140.00000  -76.04000``
    * forcing:   ``airtmp: day,span,range = 46263.0000 0.041667  2.4E+01  2.6E+01``
    * other labelled lines, min/max last: ``rmu: range = …``, ``chl: month,range = 01 …``,
      ``river precip: month,range = 1 …``, relaxation
      ``tem: month,layer,dens,range = 01 03 19.000 …`` (layer → k)
    * depth:     ``min,max depth =   5.0000000  6106.40186``

The min/max on each ``.b`` line is checked against the data read from the
``.a`` file, which catches wrong dimensions, offsets or truncated files.

References
----------
Wallcraft, A.J., Metzger, E.J. & Carroll, S.N. (2009). Software Design
    Description for the HYbrid Coordinate Ocean Model (HYCOM) Version 2.2.
    NRL/MR/7320--09-9166.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_WORDS_PER_BLOCK = 4096
_SPVAL_THRESHOLD = 2.0 ** 99      # HYCOM data void is 2^100

_HEADER = re.compile(r"^\s*([-+0-9.eE]+)\s+'(\w+)\s*'\s*=")
_ARCHIVE = re.compile(
    r"^\s*(\S+?)\s*=\s*(\d+)\s+([-+0-9.eE]+)\s+(\d+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$")
# "name: labels = v1 … min max"  (labels e.g. "min,max", "range", "day,span,range", "month,range")
_LABELLED = re.compile(r"^\s*([A-Za-z][\w.\- ]*?)\s*:\s*([a-z, ]+?)\s*=\s*((?:[-+0-9.eE]+\s*)+)$")
# "min,max name = min max"  (regional.depth)
_MINMAX_ANON = re.compile(r"^\s*(?:(\w[\w.]*)\s+-\s+)?min\s*,\s*max\s*=\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$")
_MINMAX_NAMED = re.compile(r"^\s*min,max\s+(\w+)\s*=\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)")
_IJDM = re.compile(r"i/jdm\s*=\s*(\d+)\s+(\d+)")


@dataclass
class BRecord:
    """One ``.b`` record line: field name, layer and the min/max of the ``.a`` record."""

    name: str
    min: float
    max: float
    k: int | None = None
    day: float | None = None
    density: float | None = None


@dataclass
class BFile:
    """Parsed ``.b`` file: header values and the ordered list of records."""

    header: dict[str, float] = field(default_factory=dict)
    records: list[BRecord] = field(default_factory=list)


def hycom_npad(idm: int, jdm: int) -> int:
    """
    Number of padding words after each ``.a`` record.

        npad = 4096·⌈idm·jdm / 4096⌉ − idm·jdm
    """
    return (-idm * jdm) % _WORDS_PER_BLOCK


def read_b(path: str | Path) -> BFile:
    """
    Parse a HYCOM ``.b`` file (archive, grid/depth or forcing dialect).

    Header values are stored as numbers keyed by their name (``idm``,
    ``jdm``, ``iversn``, …; integers where exact).  Lines that are neither a
    header value nor a record line (titles, column captions) are ignored.
    """
    out = BFile()
    previous = ""
    for line in Path(path).read_text(errors="replace").splitlines():
        n_before = len(out.records)
        _parse_b_line(line, out, Path(path).name[:-2] if str(path).endswith(".b") else Path(path).stem)
        # "i/jdm = …" counts only on the line right before the first record
        # (HYCOM tool convention); title lines may quote other grids.
        if n_before == 0 and out.records and (m := _IJDM.search(previous)):
            out.header.setdefault("idm", int(m[1]))
            out.header.setdefault("jdm", int(m[2]))
        previous = line
    return out


def _parse_b_line(line: str, out: BFile, default_name: str = "") -> None:
    """Append a record or header value parsed from one ``.b`` line, if any."""
    if m := _ARCHIVE.match(line):
        out.records.append(BRecord(name=m[1], k=int(m[4]), day=float(m[3]),
                                   density=float(m[5]), min=float(m[6]), max=float(m[7])))
    elif m := _LABELLED.match(line):
        values = [float(v) for v in m[3].split()]
        labels = [x.strip() for x in m[2].split(",")]
        day = values[labels.index("day")] if "day" in labels else None
        k = int(values[labels.index("layer")]) if "layer" in labels else None
        dens = values[labels.index("dens")] if "dens" in labels else None
        out.records.append(BRecord(name=m[1], k=k, day=day, density=dens,
                                   min=values[-2], max=values[-1]))
    elif m := _MINMAX_ANON.match(line):              # "[NAME - ]min, max = a b" (else named after the file)
        out.records.append(BRecord(name=m[1] or default_name, min=float(m[2]), max=float(m[3])))
    elif m := _MINMAX_NAMED.match(line):
        out.records.append(BRecord(name=m[1], min=float(m[2]), max=float(m[3])))
    elif m := _HEADER.match(line):
        value = float(m[1])
        out.header[m[2]] = int(value) if value.is_integer() else value


def read_field(
    path_b: str | Path,
    name: str,
    k: int | None = None,
    index: int = 0,
    idm: int | None = None,
    jdm: int | None = None,
    check: bool = True,
) -> np.ndarray:
    """
    Read one 2-D field from a ``.a`` file, located through its ``.b`` file.

    Parameters
    ----------
    path_b : str or Path
        The ``.b`` file; the ``.a`` file is the same path with suffix ``.a``.
    name : str
        Field name as written in the ``.b`` file (e.g. 'temp', 'thknss', 'plon').
    k : int, optional
        Layer number (archive files; 0 for 2-D archive fields).
    index : int
        Which occurrence of (name, k) to read, e.g. the time record of a
        forcing file.  Default 0.
    idm, jdm : int, optional
        Array dimensions; taken from the ``.b`` header when present
        (archives, regional.grid, forcing ``i/jdm`` line).
    check : bool
        Compare the data min/max against the ``.b`` line (default True).

    Returns
    -------
    np.ndarray, shape (jdm, idm)
        Field in float64, NaN at land/void points.

    Raises
    ------
    KeyError
        If no record matches.
    ValueError
        If dimensions are unknown, the ``.a`` file is too short, or the
        min/max check fails.
    """
    path_b = Path(path_b)
    b = read_b(path_b)
    idm = idm if idm is not None else b.header.get("idm")
    jdm = jdm if jdm is not None else b.header.get("jdm")
    if idm is None or jdm is None:
        raise ValueError(f"{path_b.name}: idm/jdm not in the .b header; pass idm= and jdm=")

    matches = [i for i, r in enumerate(b.records) if r.name == name and (k is None or r.k == k)]
    if len(matches) <= index:
        available = sorted({(r.name, r.k) for r in b.records}, key=str)
        raise KeyError(f"{path_b.name}: no record {index} for field {name!r} k={k}; available: {available}")
    return _read_record(path_b, b, matches[index], idm, jdm, check)


def read_record(
    path_b: str | Path,
    irec: int,
    idm: int | None = None,
    jdm: int | None = None,
    check: bool = True,
) -> np.ndarray:
    """
    Read the ``irec``-th record (0-based, in file order) of a ``.a/.b`` pair.

    For files whose records are identified by position rather than name,
    e.g. HYCOM ``tidal.salReIm`` (Re, Im of each constituent in turn).
    Same checks and return value as ``read_field``.
    """
    path_b = Path(path_b)
    b = read_b(path_b)
    idm = idm if idm is not None else b.header.get("idm")
    jdm = jdm if jdm is not None else b.header.get("jdm")
    if idm is None or jdm is None:
        raise ValueError(f"{path_b.name}: idm/jdm not in the .b header; pass idm= and jdm=")
    if not 0 <= irec < len(b.records):
        raise KeyError(f"{path_b.name}: record {irec} out of range (0–{len(b.records) - 1})")
    return _read_record(path_b, b, irec, idm, jdm, check)


def _read_record(path_b: Path, b: BFile, irec: int, idm: int, jdm: int, check: bool) -> np.ndarray:
    rec = b.records[irec]
    name, k = rec.name, rec.k

    n = idm * jdm
    stride = n + hycom_npad(idm, jdm)
    path_a = path_b.with_suffix(".a")
    expected = 4 * stride * len(b.records)
    actual = path_a.stat().st_size
    if actual > expected or actual % (4 * stride):
        raise ValueError(
            f"{path_a.name}: size {actual} B is not {len(b.records)} records of "
            f"{idm}×{jdm} (+{hycom_npad(idm, jdm)} pad) = {expected} B — check idm/jdm"
        )
    with open(path_a, "rb") as f:
        f.seek(4 * stride * irec)
        data = np.fromfile(f, dtype=">f4", count=n)
    if data.size < n:
        raise ValueError(f"{path_a.name} is too short for record {irec} ({idm}×{jdm})")

    x = data.astype(np.float64).reshape(jdm, idm)
    x[np.abs(x) >= _SPVAL_THRESHOLD] = np.nan

    if check and np.isfinite(x).any():
        lo, hi = np.nanmin(x), np.nanmax(x)
        tol = 1e-4 * max(abs(rec.min), abs(rec.max), 1e-30)
        if abs(lo - rec.min) > tol or abs(hi - rec.max) > tol:
            raise ValueError(
                f"{path_b.name}: min/max of record {irec} ({name}, k={k}) are "
                f"[{lo:.6g}, {hi:.6g}] but the .b file says [{rec.min:.6g}, {rec.max:.6g}]"
            )
    return x
