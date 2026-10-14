#!/usr/bin/env bash
# MITgcm runs of the four-model benchmark part A (spec 2026-10-13-ilicak-four-model-benchmark-design §A2).
#   run_mitgcm.sh build <case>                              : genmake2 + make in RUNS/builds/mitgcm_<case> (serial)
#   run_mitgcm.sh run <case> zstar <nu> [dt] [hours] [tag]  : write, run, read → series.npz
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNS=/Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark
MIT=/Volumes/ESTEBAN_MIT/MITgcm
case "${1:-}" in
  build)
    B="$RUNS/builds/mitgcm_$2"; rm -rf "$B"; mkdir -p "$B/build"
    (cd "$ROOT" && python -c "from pathlib import Path; from validation.ilicak_benchmark import mitgcm; \
mitgcm.write_build('$2', Path('$B/code'))")
    cd "$B/build"; export NETCDF_ROOT=/opt/homebrew
    "$MIT/tools/genmake2" -rootdir "$MIT" -mods ../code -optfile "$MIT/tools/build_options/darwin_arm64_gfortran_esteban" \
        > genmake.log 2>&1
    make depend > make_depend.log 2>&1 && make -j 5 > make.log 2>&1 || true
    [[ -x mitgcmuv ]] || { tail -20 make.log; echo "FAILED build $2" >&2; exit 1; }
    echo "OK $B/build/mitgcmuv" ;;
  run)
    CASE=$2; NU=$4; DT=${5:-}; HRS=${6:-}; TAG=${7:-}
    B="$RUNS/builds/mitgcm_$CASE/build"
    [[ -x "$B/mitgcmuv" ]] || "$0" build "$CASE"
    D=$(cd "$ROOT" && python -c "from validation.ilicak_benchmark import cases; \
print(cases.run_dir('$CASE', 'mitgcm', 'zstar', float('$NU'), '$TAG'))")
    rm -rf "$D"
    (cd "$ROOT" && python -c "from pathlib import Path; from validation.ilicak_benchmark import mitgcm; \
mitgcm.write_run('$CASE', float('$NU'), Path('$D'), dt=${DT:-None}, hours=${HRS:-None})")
    cd "$D"; cp "$B/mitgcmuv" .
    ./mitgcmuv > output.txt 2>&1 || true
    grep -q "PROGRAM MAIN: Execution ended Normally" output.txt || { tail -15 output.txt; echo "FAILED $D" >&2; exit 1; }
    (cd "$ROOT" && python -c "
from pathlib import Path
from validation.ilicak_benchmark import cases, mitgcm, series
c = cases.get('$CASE'); dt = float('${DT:-0}') or c.dt; hrs = float('${HRS:-0}') or c.hours
s = mitgcm.read(Path('$D'), '$CASE')
meta = dict(model='MITgcm', coord='zstar', case='$CASE', nu=float('$NU'), dt=dt, n_steps=int(round(hrs * 3600 / dt)),
            wall_s=mitgcm.wall_s(Path('$D')), threads=1)
print(series.save(Path('$D') / 'series.npz', s, meta))" 2>&1 | grep -v Warn) ;;
  *) echo "usage: $0 build <case> | run <case> zstar <nu> [dt] [hours] [tag]" >&2; exit 2 ;;
esac
