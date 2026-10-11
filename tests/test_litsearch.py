"""Tests for scripts/litsearch.py — full-text search of the local literature library."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from litsearch import search  # noqa: E402


def _lib(tmp_path: Path) -> Path:
    (tmp_path / "sub").mkdir()
    (tmp_path / "Ilicak_2012 - Spurious dianeutral mixing.md").write_text(
        "We compute the reference potential energy.\nSpurious mixing scales with the grid Reynolds number.\n")
    (tmp_path / "sub" / "Other_2000 - Seiches.md").write_text("Internal seiche in a lake.\nNo mixing here.\n")
    return tmp_path


def test_ranks_files_by_number_of_matching_terms(tmp_path):
    hits = search(["reference potential energy", "grid Reynolds"], _lib(tmp_path))
    assert hits[0].name.startswith("Ilicak_2012")
    assert hits[0].score == 2
    assert any("Reynolds" in line for line in hits[0].lines)


def test_search_is_recursive_and_case_insensitive(tmp_path):
    hits = search(["INTERNAL SEICHE"], _lib(tmp_path))
    assert [h.name for h in hits] == ["Other_2000 - Seiches.md"]


def test_no_match_returns_empty(tmp_path):
    assert search(["wetting and drying"], _lib(tmp_path)) == []


def test_scanned_pdfs_without_text_are_reported(tmp_path):
    from litsearch import scanned
    (tmp_path / "Old_1991 - Scanned.md").write_text("Downloaded from somewhere\n")
    (tmp_path / "New_2020 - Text.md").write_text("word " * 1000)
    assert [p.name for p in scanned(tmp_path)] == ["Old_1991 - Scanned.md"]
