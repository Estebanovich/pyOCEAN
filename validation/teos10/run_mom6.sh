#!/usr/bin/env bash
# MOM6 runs of the TEOS-10 rest case (criterion 4, MOM6 mode; validation/teos10/mom6.py).
#   [DRAG=1] run_mom6.sh <eos> <coord> [n_layers]   : the one-step initial-PGF run and the 10-day run, then series.npz
#   eos: linear | wright_full | teos10;  coord: zstar | hybrid
# The repro build (-O2) of validation/mom6_bayiw/run_mom6.sh build is reused; MPICH may hang in MPI_Finalize, so the
# launcher waits for "Total runtime" and stops only its own ranks (as validation/ilicak_benchmark/run_mom6.sh).
# Outputs in /Volumes/ESTEBAN_MIT/pyOCEAN_runs/teos10/mom6_rest/<eos>_<coord>[_nN]/{pgf0,run,series.npz}.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
EXE=~/pyOCEAN_refs/MOM6/.testing/build/repro/MOM6
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[[ $# -ge 2 ]] || { echo "usage: $0 <eos> <coord> [n_layers]" >&2; exit 2; }
EOS=$1; COORD=$2; NL=${3:-None}; DR=$([[ "${DRAG:-0}" == 1 ]] && echo True || echo False)

run_one() {   # $1 = mode (pgf0 | run)
  local D
  D=$(cd "$ROOT" && python -c "from validation.teos10 import mom6; print(mom6.run_dir('$EOS', '$COORD', '$1', $NL, $DR))")
  rm -rf "$D"; mkdir -p "$D"
  (cd "$ROOT" && python -W ignore -c "from validation.teos10 import mom6; mom6.write_case('$EOS', '$COORD', '$1', $NL, drag=$DR)")
  cd "$D"
  mpirun -n 1 "$EXE" > stdout.txt 2>&1 &
  local pid=$!
  for _ in $(seq 1 7200); do
    grep -q "Total runtime" stdout.txt 2>/dev/null && break
    kill -0 $pid 2>/dev/null || break
    sleep 1
  done
  sleep 2; pkill -P $pid 2>/dev/null || true; kill $pid 2>/dev/null || true
  grep -q "Total runtime" stdout.txt || { grep -m3 FATAL stdout.txt || tail -5 stdout.txt; echo "FAILED $D" >&2; exit 1; }
  cd "$ROOT"
}

run_one pgf0
run_one run
(cd "$ROOT" && python -W ignore -c "from validation.teos10 import mom6; print(mom6.save('$EOS', '$COORD', $NL, $DR))")
