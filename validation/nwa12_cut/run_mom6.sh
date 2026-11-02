#!/usr/bin/env bash
# MOM6 + SIS2 on the NWA12 cut (regional chain, plan Task 10).
#
#   validation/nwa12_cut/run_mom6.sh pgf0     # one coupling step of 600 s: PFu, PFv of the initial state (criterion 7)
#   validation/nwa12_cut/run_mom6.sh tides    # tides only, 30 days (criterion 4)
#   validation/nwa12_cut/run_mom6.sh full     # MOM6 reference, NWA12's MANIZZA_05 shortwave, 123 days (R23)
#   validation/nwa12_cut/run_mom6.sh full_jerlov   # parity control: the same with pyOCEAN's Jerlov-I shortwave
#
# The run directory /Volumes/ESTEBAN_MIT/pyOCEAN_runs/nwa12_cut/runs/mom6_<mode>/ is written by
# mom6_inputs.py --run-dir (INPUT/ links to inputs/mom6/, templates of this directory filled for the mode).
# At most 5 MPI ranks (P-cores only, CLAUDE.md); NP=… changes the count (MOM_layout is written for 5: LAYOUT 1,5).
# MPICH may hang in MPI_Finalize after the end of the run (validation/teos10/run_mom6.sh), so the launcher waits for
# the coupler's final clock report ("Total runtime") or for the ranks to exit, then stops only its own ranks.
# Writes timing.json (wall seconds, simulated days, ranks) next to the outputs.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
[[ $# -eq 1 ]] || { echo "usage: $0 {pgf0|tides|full|full_jerlov}" >&2; exit 2; }
MODE=$1
NP=${NP:-5}
(( NP <= 5 )) || { echo "NP ≤ 5 (P-cores only)" >&2; exit 2; }
EXE=${EXE:-$HOME/pyOCEAN_refs/build/nwa12_cut/model/coupler_main}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[[ -x "$EXE" ]] || { echo "no executable $EXE: run validation/nwa12_cut/build_mom6.sh" >&2; exit 1; }

D=$(cd "$ROOT" && python -W ignore validation/nwa12_cut/mom6_inputs.py --run-dir "$MODE")
cd "$D"
t0=$(date +%s)
mpirun -n "$NP" "$EXE" > stdout.txt 2> stderr.txt &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  grep -q "Total runtime" stdout.txt 2>/dev/null && break
  grep -qE "FATAL from PE|^ *FATAL:" stdout.txt stderr.txt 2>/dev/null && break   # mpp_error / MOM_error markers
  sleep 20
done
t1=$(date +%s)
sleep 5
pkill -P "$pid" 2>/dev/null || true
kill "$pid" 2>/dev/null || true
python - "$D" "$((t1 - t0))" "$NP" <<'EOF'
import json, sys
from pathlib import Path
d, wall, n = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
r = json.loads((d / "run.json").read_text())
ok = "Total runtime" in (d / "stdout.txt").read_text(errors="replace")
(d / "timing.json").write_text(json.dumps(dict(mode=r["mode"], days=r["days"], seconds=r["seconds"], wall_s=wall,
                                               nproc=n, completed=ok), indent=1))
days = r["days"] if r["days"] > 0 else r["seconds"] / 86400.0
print(f"{d.name}: {'completed' if ok else 'FAILED'} in {wall} s ({wall / days:.1f} s per simulated day, {n} ranks)")
EOF
grep -q "Total runtime" stdout.txt || { grep -h -m5 -A3 -E "FATAL from PE|^ *FATAL:" stdout.txt stderr.txt || tail -20 stderr.txt; exit 1; }
