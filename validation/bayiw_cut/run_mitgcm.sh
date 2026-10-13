#!/usr/bin/env bash
# BayIW cut in MITgcm (spec §1.2): build once (200 × 160, 5 MPI processes), run phase 1 (radiate) / 2 (nest).
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 build | run <case> <bay> radiate|nest" >&2; exit 2; }
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MIT=/Volumes/ESTEBAN_MIT/MITgcm
BASE=/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_cut/mitgcm
CODE_SRC=$MIT/verification/BayIW_Rectan_2layer/code
case "$1" in
  build)
    mkdir -p "$BASE/code"; rm -rf "$BASE/build"; mkdir -p "$BASE/build"
    cp "$CODE_SRC"/{packages.conf,OBCS_OPTIONS.h,DIAGNOSTICS_SIZE.h,MNC_SIZE.h} "$BASE/code/"
    sed -e 's/sNx = *[0-9]*,/sNx =  40,/' -e 's/sNy = *[0-9]*,/sNy = 160,/' \
        -e 's/nPx = *[0-9]*,/nPx =   5,/' -e 's/nPy = *[0-9]*,/nPy =   1,/' "$CODE_SRC/SIZE.h" > "$BASE/code/SIZE.h"
    grep -E "sNx =|sNy =|nPx =|nPy =|Nr  =" "$BASE/code/SIZE.h"
    cd "$BASE/build"; export NETCDF_ROOT=/opt/homebrew
    "$MIT/tools/genmake2" -rootdir "$MIT" -mods ../code -mpi \
        -optfile "$MIT/tools/build_options/darwin_arm64_gfortran_esteban" > genmake.log 2>&1
    make depend > make_depend.log 2>&1; make -j 5 > make.log 2>&1; ls -la mitgcmuv ;;
  run)
    [[ $# -eq 4 ]] || { echo "usage: $0 run <case> <bay> radiate|nest" >&2; exit 2; }
    RUN="$BASE/$2_$3_$4"; rm -rf "$RUN"; mkdir -p "$RUN"
    cd "$ROOT" && python -c "from pathlib import Path; from validation.bayiw_cut import cut; \
cut.write_mitgcm_inputs('$2', '$3', Path('$RUN'), '$4'); \
cut.write_obcs_files('$2', '$3', Path('$RUN')) if '$4' == 'nest' else None"
    cd "$RUN"; cp "$BASE/build/mitgcmuv" .
    /usr/bin/time -p mpirun -np 5 ./mitgcmuv > output.txt 2>&1 || true
    if grep -q "PROGRAM MAIN: Execution ended Normally" STDOUT.0000 2>/dev/null; then echo "OK $RUN"; \
    else echo "FAILED $RUN"; tail -20 STDOUT.0000 2>/dev/null; tail -20 output.txt; exit 1; fi ;;
  *) echo "usage: $0 build | run <case> <bay> radiate|nest" >&2; exit 2 ;;
esac
