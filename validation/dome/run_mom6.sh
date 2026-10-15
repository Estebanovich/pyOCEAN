#!/usr/bin/env bash
# MOM6 ρ runs of the DOME benchmark (spec 2026-10-14-dome-overflow §6.1).
#   run_mom6.sh <res_km> <variant: common|kv_low> [tag] [days]  : write, run on NP ranks (default 1), read → series.npz
# The repro build (-O2) of validation/mom6_bayiw/run_mom6.sh build is reused; MPICH may hang in MPI_Finalize, so the
# launcher waits for "Total runtime" and stops only its own ranks.
set -euo pipefail
EXE=~/pyOCEAN_refs/MOM6/.testing/build/repro/MOM6
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[[ $# -ge 2 ]] || { echo "usage: $0 <res_km> <common|kv_low> [tag] [days]" >&2; exit 2; }
RES=$1; VAR=$2; TAG=${3:-$2}; DAYS=${4:-}
D=$(cd "$ROOT" && python -c "from validation.dome import cases; print(cases.run_dir(float('$RES'), 'mom6', 'rho', '$TAG'))")
rm -rf "$D"; mkdir -p "$D"
(cd "$ROOT" && python -c "from pathlib import Path; from validation.dome import mom6; \
mom6.write_case(float('$RES'), Path('$D'), '$VAR', days=${DAYS:-None})")
cd "$D"
mpirun -n "${NP:-1}" "$EXE" > stdout.txt 2>&1 &
pid=$!
for _ in $(seq 1 14400); do
  grep -q "Total runtime" stdout.txt 2>/dev/null && break
  kill -0 $pid 2>/dev/null || break
  sleep 1
done
sleep 2; pkill -P $pid 2>/dev/null || true; kill $pid 2>/dev/null || true
grep -q "Total runtime" stdout.txt || { grep -m3 FATAL stdout.txt || tail -5 stdout.txt; echo "FAILED $D" >&2; exit 1; }
(cd "$ROOT" && python -c "
from pathlib import Path
from validation.dome import cases, mom6, series
c = cases.get(float('$RES'))
s = mom6.read(Path('$D'), float('$RES'))
meta = dict(model='MOM6', coord='rho', res_km=float('$RES'), variant='$VAR', dt=c.dt,
            n_steps=int(round(s['t'][-1] / c.dt)), wall_s=mom6.wall_s(Path('$D')), threads=int('${NP:-1}'))
print(series.save(Path('$D') / 'series.npz', s, meta))" 2>&1 | grep -v Warn)
