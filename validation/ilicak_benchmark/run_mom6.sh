#!/usr/bin/env bash
# MOM6 runs of the four-model benchmark part A (spec 2026-10-13-ilicak-four-model-benchmark-design §A2).
#   run_mom6.sh run <case> <coord> <nu> [dt] [hours] [tag]   : write, run on NP ranks (default 1), read → series.npz
# The repro build (-O2) of validation/mom6_bayiw/run_mom6.sh build is reused; MPICH may hang in MPI_Finalize, so the
# launcher waits for "Total runtime" and stops only its own ranks.
set -euo pipefail
EXE=~/pyOCEAN_refs/MOM6/.testing/build/repro/MOM6
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[[ "${1:-}" == "run" && $# -ge 4 ]] || { echo "usage: $0 run <case> <coord> <nu> [dt] [hours] [tag]" >&2; exit 2; }
CASE=$2; COORD=$3; NU=$4; DT=${5:-}; HRS=${6:-}; TAG=${7:-}
D=$(cd "$ROOT" && python -c "from validation.ilicak_benchmark import cases; \
print(cases.run_dir('$CASE', 'mom6', '$COORD', float('$NU'), '$TAG'))")
rm -rf "$D"; mkdir -p "$D"
(cd "$ROOT" && python -c "from pathlib import Path; from validation.ilicak_benchmark import mom6; \
mom6.write_case('$CASE', '$COORD', float('$NU'), Path('$D'), dt=${DT:-None}, hours=${HRS:-None})")
cd "$D"
mpirun -n "${NP:-1}" "$EXE" > stdout.txt 2>&1 &
pid=$!
for _ in $(seq 1 7200); do
  grep -q "Total runtime" stdout.txt 2>/dev/null && break
  kill -0 $pid 2>/dev/null || break
  sleep 1
done
sleep 2; pkill -P $pid 2>/dev/null || true; kill $pid 2>/dev/null || true
grep -q "Total runtime" stdout.txt || { grep -m3 FATAL stdout.txt || tail -5 stdout.txt; echo "FAILED $D" >&2; exit 1; }
(cd "$ROOT" && python -c "
import json
from pathlib import Path
from validation.ilicak_benchmark import cases, mom6, series
c = cases.get('$CASE'); dt = float('${DT:-0}') or c.dt; hrs = float('${HRS:-0}') or c.hours
s = mom6.read(Path('$D'), '$CASE')
meta = dict(model='MOM6', coord='$COORD', case='$CASE', nu=float('$NU'), dt=dt, n_steps=int(round(hrs * 3600 / dt)),
            wall_s=mom6.wall_s(Path('$D')), threads=int('${NP:-1}'))
print(series.save(Path('$D') / 'series.npz', s, meta))" 2>&1 | grep -v Warn)
