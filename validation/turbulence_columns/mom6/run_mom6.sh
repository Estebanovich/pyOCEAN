#!/bin/bash
# MOM6 CVMix single-column KPP tests (MOM6-examples/ocean_only/CVmix_SCM_tests/<case>/KPP) as the reference of
# pyOCEAN's closure='kpp' columns (N3 task 9; spec criterion 2; docs/design/n2c_refs/kpp_mom6_cvmix.md §5).
#   case = wind_only | cooling_only | mech_only | skin_warming_wind
# The run uses the distributed files unchanged: common_KPP/{MOM_input,input.nml,diag_table,zgrid.nc} and the
# case's KPP/MOM_override (400 layers of 1 m, 2 × 2 reentrant, Δt 1200 s, 5 days, output every 2 h).
# Usage: validation/turbulence_columns/mom6/run_mom6.sh <case>
# Output: /Volumes/ESTEBAN_MIT/pyOCEAN_runs/turbulence_columns/mom6/<case>/{prog,prog_z,visc,surffluxes}.nc
# MPICH (Homebrew) hangs in MPI_Finalize after MOM6 has finished (as validation/mom6_lock_exchange/run_mom6.sh):
# the script waits for the final timing table in stdout and then stops the process.
set -euo pipefail
case=${1:?wind_only, cooling_only, mech_only or skin_warming_wind}
EX=~/pyOCEAN_refs/MOM6-examples/ocean_only/CVmix_SCM_tests
EXE=~/pyOCEAN_refs/MOM6/.testing/build/symmetric/MOM6
R=/Volumes/ESTEBAN_MIT/pyOCEAN_runs/turbulence_columns/mom6/$case
[ -f "$EX/$case/KPP/MOM_override" ] || { echo "unknown case $case"; exit 1; }
rm -rf "$R"
mkdir -p "$R/INPUT" "$R/RESTART"
cp "$EX"/common_KPP/{MOM_input,input.nml,diag_table,zgrid.nc} "$R/"
cp "$EX/$case/KPP/MOM_override" "$R/"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
cd "$R"
t0=$(date +%s)
mpirun -n 1 "$EXE" > stdout.txt 2>&1 &
pid=$!
for _ in $(seq 1 600); do
  grep -q "Total runtime" stdout.txt 2>/dev/null && break
  kill -0 $pid 2>/dev/null || break
  sleep 1
done
sleep 2
pkill -f "$EXE" 2>/dev/null || true
echo "wall $(( $(date +%s) - t0 )) s" >> stdout.txt
grep -E "Total runtime|FATAL" stdout.txt || tail -5 stdout.txt
