#!/usr/bin/env bash
# MOM6 for the BayIW benchmark (spec 2026-10-13-mom6-bayiw-benchmark-design §2).
#   run_mom6.sh build      : optimized build (.testing build/repro, FCFLAGS -g -O2); build/symmetric (-O0) untouched
#   run_mom6.sh run <dir>  : run the case in <dir> on 5 MPI ranks
#   run_mom6.sh section <case> <bay> <zstar|hycom1|hybgen> [hours] : the 2-D section (60 h), saved as section series
#   run_mom6.sh cost <zstar|hycom1> [dt] [hours]           : the cut, two layers with bay (3 h), clock table
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
  section)
    D="$RUNS/section/$2_$3_$4"; ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
    rm -rf "$D"
    (cd "$ROOT" && python -c "from pathlib import Path; from validation.mom6_bayiw import config; \
config.write_case('section', '$2', '$3', '$4', Path('$D'), hours=${5:-60.0})")
    "$0" run "$D"
    (cd "$ROOT" && python -c "from validation.mom6_bayiw import read_mom6; \
print(read_mom6.save_series(read_mom6.section_series('$D', '$2', '$3'), '$2', '$3', '$4'))" 2>&1 | grep -v Warn) ;;
  cost)
    DT=${3:-120.0}; HRS=${4:-3.0}; D="$RUNS/cost/$2_dt${DT}_h${HRS}"; ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
    rm -rf "$D"
    (cd "$ROOT" && python -c "from pathlib import Path; from validation.mom6_bayiw import config; \
config.write_case('cut', '2layer', 'bay', '$2', Path('$D'), hours=$HRS, dt=$DT)")
    "$0" run "$D"
    (cd "$ROOT" && python -c "from validation.mom6_bayiw import read_mom6; c = read_mom6.clock_table('$D/stdout.txt'); \
print(f\"$2 dt $DT: main loop {c['Main loop']:.2f} s for $HRS h = {c['Main loop'] / $HRS:.2f} s per simulated hour\")" 2>&1 | grep -v Warn) ;;
  *) echo "usage: $0 build | run <dir> | section <case> <bay> <coord> [hours] | cost <coord> [dt] [hours]" >&2; exit 2 ;;
esac
