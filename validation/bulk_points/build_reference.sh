#!/bin/bash
# Build the bulk reference tables of N2b criterion 1: tests/data/bulk_croco_coare_reference.npz
# (CROCO COARE, below) and tests/data/bulk_fms_ncar_reference.npz (FMS NCAR).
#
# Sources (outside the repo, public GitHub):
#   ~/pyOCEAN_refs/FMScoupler  NOAA-GFDL/FMScoupler tag 2025.02 (git 5b700a4), the release of
#                              FMS 2025.02 (FMS here: 2025.02.01); shared/surface_flux.F90 unchanged
#   ~/pyOCEAN_refs/ice_param   NOAA-GFDL/ice_param (git 1d19a57); ocean_rough.F90 only because
#                              surface_flux.F90:120 uses it (not called with do_iter_monin_obukhov = F)
# FMS objects/modules: the libFMS.a that validation/mom6_column/build_reference.sh builds
# (-fdefault-real-8 -fdefault-double-8, as GFDL builds).  -ffp-contract=off so gfortran does not
# fuse a*b+c on arm64: the reference is the algorithm, reproducible by NumPy to round-off.
#
# Usage: build_reference.sh [coare]   ("coare": only the CROCO COARE table, no FMS needed)
set -euo pipefail
REF=$HOME/pyOCEAN_refs
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1

# --- CROCO COARE 3.0 (N2b Task 2): tests/data/bulk_croco_coare_reference.npz ----------------
# ~/pyOCEAN_refs/croco git ab63b58, OCEAN/bulk_flux.F, default options (cppdefs.h:178-185):
# BULK_FLUX, BULK_GUSTINESS, BULK_LW; SALINITY for evap (:634-646); MASKING, WET_DRY, SST_SKIN,
# READ_PATM, ABL1D, CFB, WASP, ECUME undefined.  bulk_point = local declarations (:44-116) +
# the scalar block of bulk_flux_tile (:199-659, with its "# ifndef ABL1D" guard at :198 and
# the "# endif" of :669 that closes the ABL1D branch opened at :593) + the functions
# (:758-974), copied by sed; the
# only edits replace (i,j) arrays by scalars, and line 207 Q = spec_hum(RH,psurf,TairC)
# by Q = RH, with RH = q_air given (pyOCEAN takes q; spec_hum reads RH >= 2 as g/kg, so it
# cannot carry q < 2 g/kg).  Compiled as jobcomp:249 (-O2 -fdefault-real-8 -fdefault-double-8
# -std=legacy) plus -ffp-contract=off (no a*b+c fusion on arm64, as the FMS table above).
CR=$REF/croco/OCEAN
CB=$REF/build/croco_bulk
mkdir -p "$CB"
[ "$(git -C "$REF/croco" rev-parse --short=7 HEAD)" = ab63b58 ] || { echo "croco is not at ab63b58"; exit 1; }
{
  cat <<'F77'
      subroutine bulk_point(wspd_in, TairC_in, Q_in, TseaC_in, radlw_in, rho0,
     &   hfsen_o, hflat_o, hflw_o, evap_o, Cd_o, rhoAir_o, aer, cer,
     &   Wstar_o, Tstar_o, Qstar_o, delW_o, Qsea_o, charn_o, patm_o)
      implicit none
# include "params_bulk.h"
      REAL, PARAMETER :: g = 9.81, Cp = 3985.0, vonKar = 0.41
      REAL :: wspd_in, TairC_in, Q_in, TseaC_in, radlw_in, rho0
      REAL :: hfsen_o, hflat_o, hflw_o, evap_o, Cd_o, rhoAir_o, aer, cer
      REAL :: Wstar_o, Tstar_o, Qstar_o, delW_o, Qsea_o, charn_o, patm_o
      REAL :: srflx_in, prate_in, S_in, stflx_T, stflx_S
F77
  sed -n '44,116p' "$CR/bulk_flux.F"
  cat <<'F77'
      srflx_in = 0.0
      prate_in = 0.0
      S_in = 35.0
      rho0i=1.0/rho0
      cpi=1.0/cp
      iexns  = (psurf*ip00)**(-rdocpd)
F77
  sed -n '198,659p' "$CR/bulk_flux.F" | sed \
      -e 's/wspd(i,j)/wspd_in/' -e 's/tair(i,j)/TairC_in/' -e 's/rhum(i,j)/Q_in/' \
      -e 's/spec_hum (RH,psurf,TairC)/RH/' \
      -e 's/t(i,j,N,nrhs,itemp)/TseaC_in/g' -e 's/t(i,j,N,nrhs,isalt)/S_in/g' \
      -e 's/radlw(i,j)/radlw_in/' -e 's/srflx(i,j)/srflx_in/' -e 's/prate(i,j)/prate_in/g' \
      -e 's/stflx(i,j,itemp)/stflx_T/g' -e 's/stflx(i,j,isalt)/stflx_S/g' \
      -e 's/aer(i,j)/aer/' -e 's/cer(i,j)/cer/'
  echo '# endif'
  cat <<'F77'
      hfsen_o = hfsen
      hflat_o = hflat
      hflw_o = hflw
      evap_o = evap
      Cd_o = Cd
      rhoAir_o = rhoAir
      Wstar_o = Wstar(1)
      Tstar_o = Tstar
      Qstar_o = Qstar
      delW_o = delW(1)
      Qsea_o = Qsea
      charn_o = charn
      patm_o = patm
      end subroutine bulk_point
F77
  sed -n '758,974p' "$CR/bulk_flux.F"
} > "$CB/croco_bulk_point.F"
CFLAGS=(-cpp -DBULK_FLUX -DBULK_GUSTINESS -DBULK_LW -DSALINITY -I"$CR" -O2 -fdefault-real-8
        -fdefault-double-8 -std=legacy -ffp-contract=off -ffixed-line-length-none)
gfortran "${CFLAGS[@]}" -c "$CB/croco_bulk_point.F" -o "$CB/croco_bulk_point.o"
gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffp-contract=off \
         "$HERE/croco_bulk_driver.F90" "$CB/croco_bulk_point.o" -o "$CB/croco_bulk_driver"
(cd "$ROOT" && python -m validation.bulk_points.points "$CB/points.txt" coare)
"$CB/croco_bulk_driver" "$CB/points.txt" "$CB/out.txt"
(cd "$ROOT" && python - "$CB" <<'PY'
import sys
import numpy as np
from validation.bulk_points.points import COARE_INPUT_FIELDS, COARE_OUTPUT_FIELDS, COARE_POINTS
out = sys.argv[1]
a = np.loadtxt(f"{out}/out.txt")
assert a.shape == (COARE_POINTS["u10"].size, len(COARE_OUTPUT_FIELDS)), a.shape
assert np.all(np.isfinite(a))
data = {f"in_{k}": COARE_POINTS[k] for k in COARE_INPUT_FIELDS}
data.update({k: a[:, i] for i, k in enumerate(COARE_OUTPUT_FIELDS)})
data["croco_commit"] = np.array("ab63b58")
np.savez_compressed("tests/data/bulk_croco_coare_reference.npz", **data)
print("wrote tests/data/bulk_croco_coare_reference.npz")
PY
)
[ "${1:-}" = coare ] && exit 0

# --- FMS NCAR (N2b Task 1) and MITgcm exf ----------------------------------------------------
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
