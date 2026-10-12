#!/bin/bash
# Run a MOM6-examples ocean_only case with a chosen kinetic-energy scheme, to measure what the
# KE scheme (KE_ARAKAWA vs KE_GUDONOV, MOM_CoriolisAdv.F90 gradKE) does in MOM6 itself.
# Usage: validation/mom6_ke_schemes/run_case.sh <case under ocean_only, e.g. sloshing/rho> KE_ARAKAWA|KE_GUDONOV [DAYMAX] [ENERGYSAVEDAYS]
# (DAYMAX and ENERGYSAVEDAYS in the case's TIMEUNIT)
# Output: /Volumes/ESTEBAN_MIT/pyOCEAN_runs/mom6_ke/<case with / → _>_<KE>/ (ocean.stats.nc, prog.nc, ...)
# MPICH (Homebrew) hangs in MPI_Finalize after MOM6 has finished: the script waits for the
# final timing table in stdout and then stops the process.
set -euo pipefail
case=${1:?case}; ke=${2:?KE scheme}; daymax=${3:-}; esave=${4:-}
EX=~/pyOCEAN_refs/MOM6-examples/ocean_only/$case
EXE=~/pyOCEAN_refs/MOM6/.testing/build/symmetric/MOM6
R=/Volumes/ESTEBAN_MIT/pyOCEAN_runs/mom6_ke/${case//\//_}_$ke
rm -rf "$R"; mkdir -p "$R/INPUT" "$R/RESTART"
for f in MOM_input input.nml diag_table; do cp "$EX/$f" "$R/"; done
[ -d "$EX/INPUT" ] && cp -R "$EX/INPUT/." "$R/INPUT/"
cp "$EX/MOM_override" "$R/MOM_override"                 # the case's own overrides choose its coordinate
echo "" >> "$R/MOM_override"
if grep -q "^KE_SCHEME" "$R/MOM_input" "$R/MOM_override"; then echo "#override KE_SCHEME = \"$ke\"" >> "$R/MOM_override"
else echo "KE_SCHEME = \"$ke\"" >> "$R/MOM_override"; fi
if [ -n "$daymax" ]; then echo "#override DAYMAX = $daymax" >> "$R/MOM_override"; fi       # DAYMAX is always in MOM_input
if [ -n "$esave" ]; then
  if grep -q "^ENERGYSAVEDAYS" "$R/MOM_input" "$R/MOM_override"; then echo "#override ENERGYSAVEDAYS = $esave" >> "$R/MOM_override"
  else echo "ENERGYSAVEDAYS = $esave" >> "$R/MOM_override"; fi
fi
cd "$R"
mpirun -n 1 "$EXE" > stdout.txt 2>&1 &
pid=$!
for _ in $(seq 1 1800); do
  grep -q "Total runtime" stdout.txt 2>/dev/null && break
  kill -0 $pid 2>/dev/null || break
  sleep 1
done
sleep 2
pkill -f "$EXE" 2>/dev/null || true
grep -E "Total runtime|FATAL" stdout.txt | head -2 || tail -5 stdout.txt
