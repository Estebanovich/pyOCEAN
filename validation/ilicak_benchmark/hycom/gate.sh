#!/usr/bin/env bash
# HYCOM 2.3 feasibility gate (spec §A3): build from a copied config without -mavx (arm64), run a resting basin for
# 10 steps on 5 MPI ranks. Time box: ~1 h of work; on failure stop and report (no further HYCOM work).
# HYCOM-src is not modified; the build copy in RUNS/hycom/build gets the config flags and the guard below.
set -euo pipefail
SRC=~/pyOCEAN/HYCOM-src
B=/Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark/hycom/build
rm -rf "$B"; mkdir -p "$B"; cp -R "$SRC"/. "$B"/
cd "$B"
CFG=$(ls config/*generic*gnu*relo* 2>/dev/null | head -1 || true)
[[ -n "$CFG" ]] || { echo "GATE FAILED: no generic-gnu-relo config in HYCOM-src/config" >&2; exit 1; }
sed -i '' -e 's/-mavx//g' -e 's/-m64//g' "$CFG"                 # x86-only flags (arm64 gfortran)
# HYCOM bug (class d, approved by Esteban 2026-10-14): mod_hycom.F90:1656 and 1674 zero hnudge and salfac when
# tidnud = 0 / tidflg <= 0, but mod_tides.F90:217, 234-239 allocate them only with tides on (tidflg > 0, and
# tidnud /= 0 for hnudge) → segfault in any run without tides (Jan-2025 move of both arrays into mod_tides; every
# other use, mod_barotp.F90:130-168, is guarded).  Guard the two lines in the build copy only (HYCOM-src stays
# untouched); no effect on physics or cost.
for v in hnudge salfac; do
  sed -i '' -e "s/^\( *\)$v(:,:)=0.0 *!not used/\1if (allocated($v)) $v(:,:)=0.0 !not used/" mod_hycom.F90
  grep -q "if (allocated($v)) $v" mod_hycom.F90 || { echo "GATE FAILED: $v patch did not apply" >&2; exit 1; }
done
( make hycom ARCH=generic-gnu-relo TYPE=mpi CPP_EXTRAS="-DEOS_SIG0 -DEOS_7T" > make.log 2>&1 ) || { tail -30 make.log; echo "GATE FAILED: build" >&2; exit 1; }
ls -la hycom* | head
echo "GATE: build OK — cost runs: run_hycom.sh"
