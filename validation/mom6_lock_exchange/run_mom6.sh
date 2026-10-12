#!/bin/bash
# MOM6 lock exchange (Ilıcak et al. 2012, MOM6-examples/ocean_only/lock_exchange) as the reference
# for pyOCEAN's hybrid ALE lock exchange: 128 × 500 m, 20 m, Δρ = 5 kg m⁻³, 6 h, output every 1800 s.
#   mode = layer  pure isopycnal layers (the example as distributed)
#   mode = rho    continuous isopycnal ALE: regrid to the layer densities every step (P1M_H4, PPM_H4)
#   mode = budget layer mode, first 30 min, every step (250 s), with the momentum-budget terms
#                 PFu, CAu, gKEu, rvxv, diffu, du_dt_visc, u_BT_accel and the layer transport uh
# Usage: validation/mom6_lock_exchange/run_mom6.sh layer|rho|budget
# Output: /Volumes/ESTEBAN_MIT/pyOCEAN_runs/mom6_lock_exchange/<mode>/prog.nc
# MPICH (Homebrew) hangs in MPI_Finalize after MOM6 has finished: the script waits for the
# final timing table in stdout and then stops the process.
set -euo pipefail
mode=${1:?layer, rho or budget}
EX=~/pyOCEAN_refs/MOM6-examples/ocean_only/lock_exchange
EXE=~/pyOCEAN_refs/MOM6/.testing/build/symmetric/MOM6
R=/Volumes/ESTEBAN_MIT/pyOCEAN_runs/mom6_lock_exchange/$mode
mkdir -p "$R/INPUT" "$R/RESTART"
cp "$EX"/{MOM_input,input.nml,diag_table} "$R/"
if [ "$mode" = budget ]; then
  sed -i '' 's/"prog",     1000,"seconds"/"prog",     250,"seconds"/' "$R/diag_table"
  for f in PFu CAu gKEu rvxv diffu du_dt_visc u_BT_accel uh; do
    echo "\"ocean_model\",\"$f\",\"$f\",\"prog\",\"all\",.false.,\"none\",1" >> "$R/diag_table"
  done
else
  sed -i '' 's/"prog",     1000,"seconds"/"prog",     1800,"seconds"/' "$R/diag_table"
fi
{
  if [ "$mode" = budget ]; then echo '#override DAYMAX = 1800.0'; else echo '#override DAYMAX = 21600.0'; fi
  if [ "$mode" = rho ]; then
    echo 'USE_REGRIDDING = True'
    echo 'REGRIDDING_COORDINATE_MODE = "RHO"'
    echo 'INTERPOLATION_SCHEME = "P1M_H4"'
    echo 'BOUNDARY_EXTRAPOLATION = True'
    echo 'REMAPPING_SCHEME = "PPM_H4"'
  fi
} > "$R/MOM_override"
cd "$R"
mpirun -n 1 "$EXE" > stdout.txt 2>&1 &
pid=$!
for _ in $(seq 1 600); do
  grep -q "Total runtime" stdout.txt 2>/dev/null && break
  kill -0 $pid 2>/dev/null || break
  sleep 1
done
sleep 2
pkill -f "$EXE" 2>/dev/null || true
grep -E "Total runtime|FATAL" stdout.txt || tail -5 stdout.txt
