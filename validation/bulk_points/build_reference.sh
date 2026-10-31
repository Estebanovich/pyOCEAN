#!/bin/bash
# Build the FMS NCAR bulk reference (N2b criterion 1) and write tests/data/bulk_fms_ncar_reference.npz.
#
# Sources (outside the repo, public GitHub):
#   ~/pyOCEAN_refs/FMScoupler  NOAA-GFDL/FMScoupler tag 2025.02 (git 5b700a4), the release of
#                              FMS 2025.02 (FMS here: 2025.02.01); shared/surface_flux.F90 unchanged
#   ~/pyOCEAN_refs/ice_param   NOAA-GFDL/ice_param (git 1d19a57); ocean_rough.F90 only because
#                              surface_flux.F90:120 uses it (not called with do_iter_monin_obukhov = F)
# FMS objects/modules: the libFMS.a that validation/mom6_column/build_reference.sh builds
# (-fdefault-real-8 -fdefault-double-8, as GFDL builds).  -ffp-contract=off so gfortran does not
# fuse a*b+c on arm64: the reference is the algorithm, reproducible by NumPy to round-off.
set -euo pipefail
REF=$HOME/pyOCEAN_refs
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
DEPS=$REF/MOM6/.testing/build/deps
OUT=$REF/build/bulk_points
[ -f "$DEPS/lib/libFMS.a" ] || { echo "run validation/mom6_column/build_reference.sh first"; exit 1; }
git -C "$REF/FMScoupler" checkout -q 2025.02
mkdir -p "$OUT"
FFLAGS="-O2 -fdefault-real-8 -fdefault-double-8 -ffp-contract=off -I$DEPS/include -J$OUT"
mpif90 $FFLAGS -c "$REF/ice_param/ocean_rough.F90" -o "$OUT/ocean_rough.o"
mpif90 $FFLAGS -c "$REF/FMScoupler/shared/surface_flux.F90" -o "$OUT/surface_flux.o"
mpif90 $FFLAGS "$HERE/fms_ncar_driver.F90" "$OUT/surface_flux.o" "$OUT/ocean_rough.o" \
       "$DEPS/lib/libFMS.a" $(nf-config --flibs) $(nc-config --libs) -o "$OUT/fms_ncar_driver"

cd "$OUT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
(cd "$ROOT" && python -m validation.bulk_points.points "$OUT/points.txt")
# OM4_025/input.nml:108-115; the z2 table uses the multilevel routine with T, q at 2 m (ERA5)
write_nml() {
  cat > input.nml <<NML
&surface_flux_nml
    ncar_ocean_flux = .true.
    raoult_sat_vap = .true.
    $1
/
&sat_vapor_pres_nml
    construct_table_wrt_liq = .true.
    construct_table_wrt_liq_and_ice = .true.
/
&monin_obukhov_nml
    neutral = .true.
/
NML
}
write_nml ""
./fms_ncar_driver points.txt out_z10.txt 10.0
write_nml "ncar_ocean_flux_multilevel = .true., bulk_zu = 10.0, bulk_zt = 2.0, bulk_zq = 2.0"
./fms_ncar_driver points.txt out_z2.txt 10.0

cd "$ROOT"
python - "$OUT" <<'PY'
import sys
import numpy as np
from validation.bulk_points.points import INPUT_FIELDS, OUTPUT_FIELDS, POINTS
out = sys.argv[1]
data = {f"in_{k}": POINTS[k] for k in INPUT_FIELDS}
for tag in ("z10", "z2"):
    a = np.loadtxt(f"{out}/out_{tag}.txt")
    assert a.shape == (POINTS["u10"].size, len(OUTPUT_FIELDS)), a.shape
    assert np.all(np.isfinite(a))
    for i, k in enumerate(OUTPUT_FIELDS):
        data[f"{tag}_{k}"] = a[:, i]
data["fmscoupler_commit"] = np.array("5b700a4fb585b4c8f47c479d9431178a1707c33a")
np.savez_compressed("tests/data/bulk_fms_ncar_reference.npz", **data)
print("wrote tests/data/bulk_fms_ncar_reference.npz")
PY

# --- MITgcm pkg/exf (step 7, no test table): EXF_BULKFORMULAE (MITgcm git 27256b2) unchanged,
# with ALLOW_BULK_LARGEYEAGER04 and EXF_CALC_ATMRHO; one point every second row (sNy = 2 n).
# Output ~/pyOCEAN_refs/build/mitgcm_exf/out.txt, compared by compare_mitgcm.py.
M=$REF/MITgcm
MB=$REF/build/mitgcm_exf
mkdir -p "$MB/inc"
NPTS=$(python -c "import sys; sys.path.insert(0, '$ROOT'); from validation.bulk_points.points import POINTS; print(POINTS['u10'].size)")
cat > "$MB/inc/SIZE.h" <<SZ
      INTEGER sNx, sNy, OLx, OLy, nSx, nSy, nPx, nPy, Nx, Ny, Nr
      PARAMETER ( sNx = 1, sNy = $((2 * NPTS)), OLx = 2, OLy = 2, nSx = 1,
     &           nSy = 1, nPx = 1, nPy = 1, Nx = sNx*nSx*nPx,
     &           Ny = sNy*nSy*nPy, Nr = 1 )
      INTEGER MAX_OLX, MAX_OLY
      PARAMETER ( MAX_OLX = OLx, MAX_OLY = OLy )
SZ
printf '#define ALLOW_EXF\n' > "$MB/inc/PACKAGES_CONFIG.h"
sed -e 's/^# undef  ALLOW_BULK_LARGEYEAGER04/# define ALLOW_BULK_LARGEYEAGER04/' \
    -e 's/^#define EXF_INTERP_USE_DYNALLOC/#define EXF_INTERP_USE_DYNALLOC\
#define EXF_CALC_ATMRHO/' "$M/pkg/exf/EXF_OPTIONS.h" > "$MB/inc/EXF_OPTIONS.h"
# as genmake2: C preprocessing, then tools/set64bitConst.sh turns "1. _d 0" into "1.D0"
CPPF=(-E -cpp -P -DWORDLENGTH=4 -I"$MB/inc" -I"$M/pkg/exf" -I"$M/model/inc" -I"$M/eesupp/inc")
MFLAGS=(-O2 -ffp-contract=off -fdefault-real-8 -fdefault-double-8 -ffixed-line-length-132)
# copy the source next to the modified EXF_OPTIONS.h: #include "..." searches the source's own
# directory first, which would pick pkg/exf/EXF_OPTIONS.h (LY04 off); inc/ is separate
# because .F and .f are one file on the case-insensitive macOS file system
cp "$M/pkg/exf/exf_bulkformulae.F" "$MB/inc/exf_bulkformulae.F"
for src in "$MB/inc/exf_bulkformulae.F" "$HERE/mitgcm_exf_driver.F"; do
  gfortran "${CPPF[@]}" "$src" | bash "$M/tools/set64bitConst.sh" > "$MB/$(basename "$src" .F).f"
done
gfortran "${MFLAGS[@]}" -c "$MB/exf_bulkformulae.f" -o "$MB/exf_bulkformulae.o"
gfortran "${MFLAGS[@]}" "$MB/mitgcm_exf_driver.f" "$MB/exf_bulkformulae.o" -o "$MB/mitgcm_exf_driver"
"$MB/mitgcm_exf_driver" "$OUT/points.txt" "$MB/out.txt"
echo "wrote $MB/out.txt"
