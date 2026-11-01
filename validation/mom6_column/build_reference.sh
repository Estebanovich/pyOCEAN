#!/bin/bash
# Build FMS + MOM6 (objects and modules) with the MOM6 .testing machinery, then the column drivers.
# Output: ~/pyOCEAN_refs/MOM6/.testing/build/{deps,symmetric} and ~/pyOCEAN_refs/build/column/.
# Uses ≤ 5 make jobs (P-cores).  Needs autoconf/automake/libtool (Homebrew), mpif90, nf-config,
# and the MOM6 submodules pkg/GSW-Fortran and pkg/CVMix-src (git submodule update --init).
set -euo pipefail
REF=$HOME/pyOCEAN_refs
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$REF/MOM6/.testing"
make -j5 FC=mpif90 MPIFC=mpif90 MPICC=mpicc FMS_URL="file://$REF/FMS" FMS_COMMIT=2025.02.01 build/symmetric/MOM6
B=$REF/MOM6/.testing/build/symmetric
DEPS=$REF/MOM6/.testing/build/deps            # .testing Makefile: DEPS = $(BUILD)/deps
OBJS=$(ls "$B"/*.o | grep -v -E "/(MOM_driver|ocean_solo|MOM_sum_driver)\.o$")
mkdir -p "$REF/build/column"
# hybgen column routines are private in MOM_hybgen_regrid.F90: extract them verbatim into a module
# with a public mirror of the hybgen_regrid_CS fields they read (MOM_hybgen_regrid.F90:765-992).
HYB=$REF/build/column/hybgen_column_mom6.F90
{
  echo "module hybgen_column_mom6"
  echo "use MOM_error_handler, only : MOM_error, FATAL, WARNING"
  echo "implicit none ; private"
  echo "public :: hybgen_column_init, hybgen_column_regrid"
  echo "type, public :: hybgen_regrid_CS"
  echo "  real :: min_thickness, rho_eps, dp_far_from_bot, dp_far_from_sfc, h_thin, onem"
  echo "  integer :: nk"
  echo "end type hybgen_regrid_CS"
  echo "contains"
  awk '/^subroutine hybgen_column_init\(/{on=1} on{print} /^end subroutine hybgen_column_regrid/{on=0}' \
      "$REF/MOM6/src/ALE/MOM_hybgen_regrid.F90"
  echo "end module hybgen_column_mom6"
} > "$HYB"
# Jackson et al. (2008) shear mixing: kappa_shear_column, calculate_projected_state and
# find_kappa_tke are private in MOM_kappa_shear.F90 (:866-2069).  Extract them verbatim with the
# Kappa_shear_CS type (:36-128, made public) and the module's use lines (:8-21).  Four integer
# counters are inserted (they change no arithmetic): Newton iterations (after :1825), Picard
# truncation-ramp events (after :1802 and :1817), Newton aborts (after :2035) and κ–Q iterations
# (after :1701).
KS_SRC=$REF/MOM6/src/parameterizations/vertical/MOM_kappa_shear.F90
KSM=$REF/build/column/kappa_shear_column_mom6.F90
{
  echo "module kappa_shear_column_mom6"
  sed -n '8,21p' "$KS_SRC"
  echo "implicit none ; private"
  echo "#include <MOM_memory.h>"
  echo "public :: kappa_shear_column"
  echo "integer, public :: ks_n_newton = 0, ks_n_ramp = 0, ks_n_abort = 0, ks_n_iter = 0"
  awk '/^type, public :: Kappa_shear_CS ; private/{on=1} on{sub(/ ; private$/, ""); print} /^end type Kappa_shear_CS/{on=0}' "$KS_SRC"
  echo "contains"
  awk '/^subroutine kappa_shear_column\(/{on=1}
       on{print}
       on && /^ *else ! do_Newton is \.true\./{print "      ks_n_newton = ks_n_newton + 1"}
       on && /kappa\(K\) = 2\.0 \* \(kappa\(K\) - /{print "          ks_n_ramp = ks_n_ramp + 1"}
       on && /Newton_err = 0\.5\*Newton_err/{print "      ks_n_abort = ks_n_abort + 1"}
       on && /^  do itt=1,CS%max_RiNo_it/{print "    ks_n_iter = ks_n_iter + 1"}
       /^end subroutine find_kappa_tke/{on=0}' "$KS_SRC"
  echo "end module kappa_shear_column_mom6"
} > "$KSM"
# Drag-driven BBL mixing: add_LOTW_BBL_diffusivity (:1606-1789) and set_BBL_TKE (:1947-2152)
# of MOM_set_diffusivity.F90 need the private set_diffusivity_CS.  Extract both verbatim with
# the CS type (:62-203, made public) and the module's use lines (:8-45); no line is changed.
SD_SRC=$REF/MOM6/src/parameterizations/vertical/MOM_set_diffusivity.F90
BBM=$REF/build/column/bbl_column_mom6.F90
{
  echo "module bbl_column_mom6"
  sed -n '8,45p' "$SD_SRC"
  echo "implicit none ; private"
  echo "#include <MOM_memory.h>"
  echo "public :: add_LOTW_BBL_diffusivity, set_BBL_TKE"
  awk '/^type, public :: set_diffusivity_CS ; private/{on=1} on{sub(/ ; private$/, ""); print} /^end type set_diffusivity_CS/{on=0}' "$SD_SRC"
  echo "contains"
  awk '/^subroutine add_LOTW_BBL_diffusivity\(/{on=1} on{print} /^end subroutine add_LOTW_BBL_diffusivity/{on=0}' "$SD_SRC"
  awk '/^subroutine set_BBL_TKE\(/{on=1} on{print} /^end subroutine set_BBL_TKE/{on=0}' "$SD_SRC"
  echo "end module bbl_column_mom6"
} > "$BBM"
# Nonlinear EOS (TEOS-10 in the 3-D core, task 1): the module-level parameter lines of
# MOM_EOS_Wright_full.F90 (:23-37) and MOM_EOS_Roquet_rho.F90 (:14-156) are copied verbatim into
# include files, with one write statement per coefficient, so eos_driver coef prints them as the
# compiler folds the parameter expressions.
for m in Wright_full:wright_full Roquet_rho:roquet_rho; do
  SRC=$REF/MOM6/src/equation_of_state/MOM_EOS_${m%%:*}.F90 ; OUT=$REF/build/column/${m##*:}
  grep '^real, parameter ::' "$SRC" > "${OUT}_params.inc"
  sed -E 's/^real, parameter :: *([A-Za-z0-9_]+) *=.*/write(u,"(a,1x,es25.17)") "\1", \1/' \
      "${OUT}_params.inc" > "${OUT}_write.inc"
done
# Nonlinear FV pressure gradient (TEOS-10 in the 3-D core, task 3): ALE_PLM_edge_values
# (MOM_ALE.F90:1537-1596) reads the private ALE_CS only for answer_date.  Copy it verbatim into a
# module with a public stand-in type holding that one field and the module's PLM use line (:55-56).
PLM=$REF/build/column/ale_plm_column_mom6.F90
{
  echo "module ale_plm_column_mom6"
  echo "use MOM_grid, only : ocean_grid_type"
  echo "use MOM_verticalGrid, only : verticalGrid_type"
  echo "use PLM_functions, only : PLM_extrapolate_slope, PLM_monotonized_slope, PLM_slope_wa"
  echo "implicit none ; private"
  echo "#include <MOM_memory.h>"
  echo "public :: ALE_PLM_edge_values"
  echo "type, public :: ALE_CS"
  echo "  integer :: answer_date"
  echo "end type ALE_CS"
  echo "contains"
  awk '/^subroutine ALE_PLM_edge_values\(/{on=1} on{print} /^end subroutine ALE_PLM_edge_values/{on=0}' \
      "$REF/MOM6/src/ALE/MOM_ALE.F90"
  echo "end module ale_plm_column_mom6"
} > "$PLM"
for d in remap_driver regrid_driver hybgen_driver kappa_shear_driver bbl_driver eos_driver pgf_driver; do
  [ -f "$HERE/$d.F90" ] || continue
  EXTRA=""; [ "$d" = hybgen_driver ] && EXTRA="$HYB"
  # kappa_shear: preprocess MOM_memory.h; no FMA contraction, so the comparison with NumPy can
  # reach round-off (gfortran contracts a*b+c by default on arm64)
  [ "$d" = kappa_shear_driver ] && EXTRA="-ffp-contract=off -I$REF/MOM6/config_src/memory/dynamic_symmetric -I$REF/MOM6/src/framework $KSM"
  [ "$d" = bbl_driver ] && EXTRA="-ffp-contract=off -I$REF/MOM6/config_src/memory/dynamic_symmetric -I$REF/MOM6/src/framework $BBM"
  [ "$d" = eos_driver ] && EXTRA="-ffp-contract=off -I$REF/build/column"
  [ "$d" = pgf_driver ] && EXTRA="-ffp-contract=off -I$REF/MOM6/config_src/memory/dynamic_symmetric -I$REF/MOM6/src/framework $PLM"
  mpif90 -O2 -fdefault-real-8 -fdefault-double-8 -I"$B" -I"$DEPS/include" -J"$REF/build/column" \
         $EXTRA "$HERE/$d.F90" $OBJS "$DEPS/lib/libFMS.a" "$DEPS/lib/libgsw.a" "$DEPS/lib/libcvmix.a" \
         $(nf-config --flibs) $(nc-config --libs) -o "$REF/build/column/$d"
done
echo "built: $(ls "$REF/build/column" | grep -v '\.mod$' | tr '\n' ' ')"
# Reference table of the nonlinear EOS for tests/test_eos_nonlinear.py (committed, < 1 MB)
python "$HERE/eos_columns.py" "$REF/build/column/eos_driver" "$HERE/../../tests/data/mom6_eos_reference.npz"
# Reference table of the nonlinear FV pressure gradient for tests/test_pgf_nonlinear.py (< 1 MB)
python "$HERE/eos_columns.py" pgf "$REF/build/column/pgf_driver" "$HERE/../../tests/data/mom6_pgf_plm_reference.npz"
