#!/usr/bin/env bash
# MITgcm verification/internal_wave, unchanged code and inputs, run for 10 forcing periods (spec §3.1).
#   run_mitgcm.sh build | run
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 build|run" >&2; exit 2; }
MIT=/Volumes/ESTEBAN_MIT/MITgcm
SRC=$HOME/pyOCEAN_refs/MITgcm/verification/internal_wave
BASE=/Volumes/ESTEBAN_MIT/pyOCEAN_runs/internal_wave/mitgcm
case "$1" in
  build)
    rm -rf "$BASE/build"; mkdir -p "$BASE/build"; cd "$BASE/build"
    export NETCDF_ROOT=/opt/homebrew
    "$MIT/tools/genmake2" -rootdir "$MIT" -mods "$SRC/code" \
        -optfile "$MIT/tools/build_options/darwin_arm64_gfortran_esteban" > genmake.log 2>&1
    make depend > make_depend.log 2>&1
    make -j 5 > make.log 2>&1
    ls -la mitgcmuv ;;
  run)
    RUN="$BASE/run"; rm -rf "$RUN"; mkdir -p "$RUN"; cd "$RUN"
    cp "$SRC"/input/* .
    # 10 periods of 44 567 s at Δt = 500 s = 892 steps; state every 1800 s
    sed -i '' -e 's/nTimeSteps=100,/nTimeSteps=892,/' -e 's/dumpFreq=50000.,/dumpFreq=1800.,/' data
    grep -E "nTimeSteps|dumpFreq" data
    cp "$BASE/build/mitgcmuv" .
    ./mitgcmuv > output.txt 2>&1 || true
    if grep -q "PROGRAM MAIN: Execution ended Normally" output.txt; then echo "OK $RUN"; \
    else echo "FAILED $RUN"; tail -20 output.txt; exit 1; fi ;;
  *) echo "usage: $0 build|run" >&2; exit 2 ;;
esac
