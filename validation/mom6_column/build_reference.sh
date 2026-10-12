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
for d in remap_driver regrid_driver hybgen_driver; do
  [ -f "$HERE/$d.F90" ] || continue
  EXTRA=""; [ "$d" = hybgen_driver ] && EXTRA="$HYB"
  mpif90 -O2 -fdefault-real-8 -fdefault-double-8 -I"$B" -I"$DEPS/include" -J"$REF/build/column" \
         $EXTRA "$HERE/$d.F90" $OBJS "$DEPS/lib/libFMS.a" "$DEPS/lib/libgsw.a" "$DEPS/lib/libcvmix.a" \
         $(nf-config --flibs) $(nc-config --libs) -o "$REF/build/column/$d"
done
echo "built: $(ls "$REF/build/column" | grep -v '\.mod$' | tr '\n' ' ')"
