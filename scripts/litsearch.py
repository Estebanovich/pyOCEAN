#!/usr/bin/env python
"""
Full-text search of the local literature library (first step before any
Consensus search; see CLAUDE.md "Systematic literature checks").

The library keeps every paper as ``Author_Year - Title.pdf`` plus a text copy
``Author_Year - Title.md``.  ``--convert`` creates the missing .md files with
``pdftotext`` (poppler).  The search ranks files by how many of the query terms
they contain and prints the matching lines.

Usage:
    python scripts/litsearch.py "reference potential energy" "lock exchange"
    python scripts/litsearch.py --convert            # make .md for new PDFs first
    python scripts/litsearch.py --lib /other/folder "WENO"
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_LIB = Path("/Users/estebancruz/Documents/TESIS_InternalWaves-2/Literatura")


@dataclass
class Hit:
    name: str
    path: Path
    score: int
    lines: list[str] = field(default_factory=list)


def convert_missing(lib: Path) -> list[Path]:
    """Create ``.md`` text copies for PDFs that lack one (pdftotext -layout off)."""
    if shutil.which("pdftotext") is None:
        raise RuntimeError("pdftotext not found (brew install poppler)")
    made = []
    for pdf in sorted(lib.rglob("*.pdf")):
        md = pdf.with_suffix(".md")
        if not md.exists():
            subprocess.run(["pdftotext", "-enc", "UTF-8", str(pdf), str(md)], check=True)
            made.append(md)
    return made


def search(terms: list[str], lib: Path = DEFAULT_LIB, max_lines: int = 4) -> list[Hit]:
    """Files containing any of ``terms`` (case-insensitive), most terms first."""
    pats = [re.compile(re.escape(t), re.IGNORECASE) for t in terms]
    hits = []
    for md in sorted(lib.rglob("*.md")):
        text = md.read_text(errors="ignore")
        flat = re.sub(r"\s+", " ", text)                     # phrases broken across lines still match
        found = [p for p in pats if p.search(flat)]
        if not found:
            continue
        lines = [ln.strip() for ln in text.splitlines() if any(p.search(ln) for p in found)][:max_lines]
        hits.append(Hit(md.name, md, len(found), lines))
    return sorted(hits, key=lambda h: (-h.score, h.name))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("terms", nargs="*")
    ap.add_argument("--lib", type=Path, default=DEFAULT_LIB)
    ap.add_argument("--convert", action="store_true", help="create .md for PDFs without one")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    if args.convert:
        for md in convert_missing(args.lib):
            print("converted", md.name)
    if args.terms:
        hits = search(args.terms, args.lib)
        if not hits:
            print("no matches in the local library → search Consensus and add the papers to "
                  "docs/literature/reading_list.md")
        for h in hits[:args.top]:
            print(f"[{h.score}/{len(args.terms)}] {h.name}")
            for ln in h.lines:
                print("      ", ln[:160])


if __name__ == "__main__":
    main()
