#!/usr/bin/env bash
# HYCOM 2.3 cost run of the Ilicak overflow (plan 2026-10-13-ilicak-benchmark-hycom-addendum, Task 3).
#   run_hycom.sh <npes> <baclin> <batrop> <hours> [nu_h]
# Writes the case to RUNS/hycom/run_np<npes>_dt<baclin>, runs the gate build (gate.sh) on <npes> MPI ranks and prints
# the time-stepping wall time, the steps and s per simulated hour (timers.py).  MPICH may hang in MPI_Finalize, so
# the launcher waits for 'mpi_finalize called' (every HYCOM stop, normal or not, mod_xc_mp.h xcstop) and stops only
# its own ranks (as run_mom6.sh); the run succeeded only with 'normal stop'.
set -euo pipefail
[[ $# -ge 4 ]] || { echo "usage: $0 <npes> <baclin> <batrop> <hours> [nu_h]" >&2; exit 2; }
NP=$1; BACLIN=$2; BATROP=$3; HRS=$4; NU=${5:-100}
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RUNS=/Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark/hycom
EXE=$RUNS/build/hycom
[[ -x "$EXE" ]] || { echo "no HYCOM build at $EXE: run gate.sh first" >&2; exit 1; }
D=$RUNS/run_np${NP}_dt${BACLIN}
rm -rf "$D"; mkdir -p "$D"
(cd "$ROOT" && python -c "from validation.ilicak_benchmark.hycom import blkdat; \
blkdat.write_case('$D', npes=$NP, baclin=$BACLIN, hours=$HRS, nu_h=$NU, batrop=$BATROP)")
cd "$D"
mpirun -n "$NP" "$EXE" > stdout.txt 2>&1 &
pid=$!
for _ in $(seq 1 14400); do
  if grep -qs "mpi_finalize called" stdout.txt; then break; fi
  kill -0 $pid 2>/dev/null || break
  sleep 1
done
sleep 2
for c in $(pgrep -P $pid); do pkill -P $c 2>/dev/null || true; kill $c 2>/dev/null || true; done   # ranks under hydra
kill $pid 2>/dev/null || true
grep -qs "normal stop" stdout.txt summary_out || {
  grep -m3 -i "error\|neg. dp" stdout.txt || tail -15 stdout.txt; echo "FAILED $D" >&2; exit 1; }
(cd "$ROOT" && python validation/ilicak_benchmark/hycom/timers.py "$D/stdout.txt" "$HRS")
