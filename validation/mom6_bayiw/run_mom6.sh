#!/usr/bin/env bash
# MOM6 for the BayIW benchmark (spec 2026-10-13-mom6-bayiw-benchmark-design §2).
#   run_mom6.sh build      : optimized build (.testing build/repro, FCFLAGS -g -O2); build/symmetric (-O0) untouched
#   run_mom6.sh run <dir>  : run the case in <dir> on 5 MPI ranks
# MPICH (Homebrew) may hang in MPI_Finalize after MOM6 has finished: wait for the timing line, then stop it.
set -euo pipefail
T=~/pyOCEAN_refs/MOM6/.testing
EXE=$T/build/repro/MOM6
RUNS=/Volumes/ESTEBAN_MIT/pyOCEAN_runs/mom6_bayiw
case "$1" in
  build)
    mkdir -p "$RUNS"
    cd "$T" && FC=mpif90 MPIFC=mpif90 make -j5 build/repro/MOM6 > "$RUNS/build_repro.log" 2>&1   # FC as build/symmetric
    ls -la "$EXE"; grep -m1 "^FCFLAGS" "$T/build/repro/Makefile" ;;
  run)
    cd "$2"
    NP=${NP:-5}
    mpirun -n "$NP" "$EXE" > stdout.txt 2>&1 &
    pid=$!
    for _ in $(seq 1 3600); do
      grep -q "Total runtime" stdout.txt 2>/dev/null && break
      kill -0 $pid 2>/dev/null || break
      sleep 1
    done
    sleep 2; pkill -f "$EXE" 2>/dev/null || true
    grep -E "Total runtime|FATAL" stdout.txt | head -2 || tail -5 stdout.txt ;;
  *) echo "usage: $0 build | run <dir>" >&2; exit 2 ;;
esac
