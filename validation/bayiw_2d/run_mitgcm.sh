#!/usr/bin/env bash
# BayIW 2-D section in MITgcm (spec docs/superpowers/specs/2026-10-12-bayiw-2d-section-design.md §3).
#   run_mitgcm.sh build                        : compile once (sNx = 3, sNy = 352, Nr = 50, one process, no MPI)
#   run_mitgcm.sh run <case> <bay> [periodic]  : write inputs (section.py) and run 60 h in
#                                                RUNS/mitgcm/<case>_<bay>[_periodic]/
# Nx = 3: land walls in columns 0 and 2 as pyOCEAN's closed one-column channel; 'periodic' makes the three
# columns wet (x-invariant rotating slice).  MITgcm source unchanged; optfile and code/ are Esteban's.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MIT=/Volumes/ESTEBAN_MIT/MITgcm
BASE=/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_2d/mitgcm
CODE_SRC=$MIT/verification/BayIW_Rectan_2layer/code
if [[ "$1" == "build" ]]; then
  mkdir -p "$BASE/code" "$BASE/build"
  cp "$CODE_SRC"/{packages.conf,OBCS_OPTIONS.h,DIAGNOSTICS_SIZE.h,MNC_SIZE.h} "$BASE/code/"
  sed -e 's/sNx = *[0-9]*,/sNx =   3,/' -e 's/sNy = *[0-9]*,/sNy = 352,/' \
      -e 's/nPx = *[0-9]*,/nPx =   1,/' -e 's/nPy = *[0-9]*,/nPy =   1,/' "$CODE_SRC/SIZE.h" > "$BASE/code/SIZE.h"
  grep -E "sNx =|sNy =|OLx =|nPx =|nPy =|Nr  =" "$BASE/code/SIZE.h"
  rm -rf "$BASE/build"; mkdir -p "$BASE/build"; cd "$BASE/build"
  export NETCDF_ROOT=/opt/homebrew     # netcdf-fortran and netcdf-c both linked there (Homebrew)
  "$MIT/tools/genmake2" -rootdir "$MIT" -mods ../code \
      -optfile "$MIT/tools/build_options/darwin_arm64_gfortran_esteban" > genmake.log 2>&1
  make depend > make_depend.log 2>&1
  make -j 5 > make.log 2>&1
  ls -la mitgcmuv
elif [[ "$1" == "run" ]]; then
  VAR="${4:-}"
  RUN="$BASE/$2_$3${VAR:+_$VAR}"
  PER=False; [[ "$VAR" == "periodic" ]] && PER=True
  rm -rf "$RUN"; mkdir -p "$RUN"
  cd "$ROOT" && python -c "from pathlib import Path; from validation.bayiw_2d import section; \
section.write_mitgcm_inputs('$2', '$3', Path('$RUN'), periodic=$PER)"
  cd "$RUN"; cp "$BASE/build/mitgcmuv" .
  ./mitgcmuv > output.txt 2>&1 || true
  if grep -q "PROGRAM MAIN: Execution ended Normally" output.txt; then echo "OK $RUN"; \
  else echo "FAILED $RUN"; tail -20 output.txt; exit 1; fi
fi
