#!/usr/bin/env bash
# CROCO runs of the four-model benchmark part A (spec 2026-10-13-ilicak-four-model-benchmark-design §A2).
#   run_croco.sh build <case>                              : jobcomp in RUNS/builds/croco_<case> (serial, -O2)
#   run_croco.sh run <case> sigma <nu> [dt] [hours] [tag]  : write, run, read → series.npz
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNS=/Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark
case "${1:-}" in
  build)
    B="$RUNS/builds/croco_$2"; rm -rf "$B"
    (cd "$ROOT" && python -c "from pathlib import Path; from validation.ilicak_benchmark import croco; \
croco.write_build('$2', Path('$B'))")
    cd "$B" && ./jobcomp > build.log 2>&1 || true
    [[ -x "$B/croco" ]] || { tail -20 "$B/build.log"; echo "FAILED build $2" >&2; exit 1; }
    echo "OK $B/croco" ;;
  run)
    CASE=$2; NU=$4; DT=${5:-}; HRS=${6:-}; TAG=${7:-}
    B="$RUNS/builds/croco_$CASE"
    [[ -x "$B/croco" ]] || "$0" build "$CASE"
    D=$(cd "$ROOT" && python -c "from validation.ilicak_benchmark import cases; \
print(cases.run_dir('$CASE', 'croco', 'sigma', float('$NU'), '$TAG'))")
    rm -rf "$D"
    (cd "$ROOT" && python -c "from pathlib import Path; from validation.ilicak_benchmark import croco; \
croco.write_run('$CASE', float('$NU'), Path('$D'), dt=${DT:-None}, hours=${HRS:-None})")
    cd "$D"
    /usr/bin/time -p "$B/croco" croco.in > croco.log 2> time.txt || true
    grep -q "MAIN: DONE" croco.log || { tail -15 croco.log; echo "FAILED $D" >&2; exit 1; }
    (cd "$ROOT" && python -c "
from pathlib import Path
from validation.ilicak_benchmark import cases, croco, series
c = cases.get('$CASE'); dt = float('${DT:-0}') or c.dt; hrs = float('${HRS:-0}') or c.hours
s = croco.read(Path('$D'), '$CASE')
meta = dict(model='CROCO', coord='sigma', case='$CASE', nu=float('$NU'), dt=dt, n_steps=int(round(hrs * 3600 / dt)),
            wall_s=croco.wall_s(Path('$D')), threads=1)
print(series.save(Path('$D') / 'series.npz', s, meta))" 2>&1 | grep -v Warn) ;;
  *) echo "usage: $0 build <case> | run <case> sigma <nu> [dt] [hours] [tag]" >&2; exit 2 ;;
esac
