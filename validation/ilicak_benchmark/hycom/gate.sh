#!/usr/bin/env bash
# HYCOM 2.3 feasibility gate (spec §A3): build from a copied config without -mavx (arm64), run a resting basin for
# 10 steps on 5 MPI ranks. Time box: ~1 h of work; on failure stop and report (no further HYCOM work).
# HYCOM sources are not modified; the build lives in RUNS/hycom/build.
set -euo pipefail
SRC=~/pyOCEAN/HYCOM-src
B=/Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark/hycom/build
rm -rf "$B"; mkdir -p "$B"; cp -R "$SRC"/. "$B"/
cd "$B"
CFG=$(ls config/*generic*gnu*relo* 2>/dev/null | head -1 || true)
[[ -n "$CFG" ]] || { echo "GATE FAILED: no generic-gnu-relo config in HYCOM-src/config" >&2; exit 1; }
sed -i '' -e 's/-mavx//g' -e 's/-m64//g' "$CFG"                 # x86-only flags (arm64 gfortran)
( make hycom ARCH=generic-gnu-relo TYPE=mpi CPP_EXTRAS="-DEOS_SIG0 -DEOS_7T" > make.log 2>&1 ) || { tail -30 make.log; echo "GATE FAILED: build" >&2; exit 1; }
ls -la hycom* | head
echo "GATE: build OK — next: the resting-basin case (10 steps, mpiexec -n 5)"
