#!/usr/bin/env bash
# Build MOM6 + SIS2 + the FMS coupler ('full') + atmos_null + land_null, the NWA12 component list without COBALT
# (CEFI-regional-MOM6 git 36b8f49, yamls/NWA12/compile.yaml), in ~/pyOCEAN_refs/build/nwa12_cut/.
#
#   validation/nwa12_cut/build_mom6.sh            # FMS, then the model executable  …/model/coupler_main (link MOM6_SIS2)
#   FMS_SRC=… BUILD=… validation/nwa12_cut/build_mom6.sh    # another FMS source / build directory
#
# Components (local clones; CEFI submodule commits where they exist): MOM6 ~/pyOCEAN_refs/MOM6 (git 0e9a561,
# GFDL main; CEFI pins its dev/cefi fork af82618, not available locally), SIS2 c287b24, icebergs fd6478f,
# ice_param 1d19a57, atmos_null fc39f82, land_null 57e5942, FMScoupler 175a9da (~/pyOCEAN_refs/FMScoupler_cefi),
# FMS 16bb3af (~/pyOCEAN_refs/FMS_2025.03, CEFI's submodule commit; `git describe` 2026.01.01).  The local FMS
# 2025.02.01 (~/pyOCEAN_refs/FMS, untouched) cannot build this coupler: 175a9da calls fms_error_mesg and
# fms_check_nml_error (full_coupler_mod.F90:425, 443), which libFMS.F90 exports only from 2025.03 on (16bb3af
# libFMS.F90:395-396); controller ruling for Task 10.  COBALT is replaced by MOM6's GFDL_ocean_BGC
# stubs (config_src/external), Icepack by SIS2's Icepack_interfaces stubs, as in CEFI's path list.
#
# Dependencies are generated with MOM6's ac/makedep (as MOM6's .testing builds FMS and MOM6); every object goes to
# the build directory, no source tree is touched.  Memory: dynamic_symmetric (MOM6 and SIS2, as CEFI).  Flags: the
# repro set of MOM6's .testing (-O2, 8-byte reals) for GCC; CPP defines of compile.yaml (FMS: INTERNAL_FILE_NML,
# use_libMPI, use_netCDF, MAXFIELDMETHODS_=600; no use_yaml, so data_table and diag_table are the legacy text
# files; coupler: _USE_LEGACY_LAND_, use_AM3_physics).  MAKE_JOBS ≤ 5 (P-cores only, CLAUDE.md).
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
REFS=${REFS:-$HOME/pyOCEAN_refs}
B=${BUILD:-$REFS/build/nwa12_cut}
FMS_SRC=${FMS_SRC:-$REFS/FMS_2025.03}
MOM6=$REFS/MOM6
MAKEDEP=$MOM6/ac/makedep
JOBS=${MAKE_JOBS:-5}
(( JOBS <= 5 )) || { echo "MAKE_JOBS ≤ 5 (P-cores only)" >&2; exit 2; }

NC_INC=$(nc-config --includedir)
NF_FLAGS=$(nf-config --fflags)
NC_LIBS="$(nf-config --flibs) $(nc-config --libs)"
FCBASE="-g -O2 -fdefault-real-8 -fdefault-double-8 -fcray-pointer -ffree-line-length-none -fallow-argument-mismatch -fallow-invalid-boz"

# ---------------------------------------------------------------------------------------------------------- FMS
FMS_DEFS="-DHAVE_MPI=1 -Duse_libMPI=1 -Duse_netCDF=1 -DHAVE_NETCDF_H=1 -D__NR_gettid=SYS_gettid -D__APPLE__=1 \
-DSTDC_HEADERS=1 -DHAVE_STDLIB_H=1 -DHAVE_STRING_H=1 -DHAVE_UNISTD_H=1 -DINTERNAL_FILE_NML -DMAXFIELDMETHODS_=600"
mkdir -p "$B/fms"
cd "$B/fms"
cat > Makefile <<EOF
FC = mpif90
CC = mpicc
AR = ar
ARFLAGS = rv
DEFS = $FMS_DEFS
CPPFLAGS = -I$NC_INC
FCFLAGS = $FCBASE $NF_FLAGS
CFLAGS = -g -O2
-include Makefile.dep
EOF
if [[ ! -f Makefile.dep ]]; then
  python3 "$MAKEDEP" $FMS_DEFS -o Makefile.dep -e -x libFMS.a -s "$FMS_SRC/test_fms" "$FMS_SRC"
fi
echo "== FMS ($FMS_SRC, $(git -C "$FMS_SRC" describe --tags --always 2>/dev/null)) → $B/fms" >&2
make -j"$JOBS" libFMS.a > build.log 2>&1 || { tail -40 build.log; echo "FMS build FAILED ($B/fms/build.log)" >&2; exit 1; }

# -------------------------------------------------------------------------------------------------------- model
SRC=(
  "$MOM6/config_src/memory/dynamic_symmetric" "$MOM6/config_src/drivers/FMS_cap" "$MOM6/src"
  "$MOM6/config_src/external/GFDL_ocean_BGC" "$MOM6/config_src/external/ODA_hooks"
  "$MOM6/config_src/external/stochastic_physics" "$MOM6/config_src/external/drifters"
  "$MOM6/config_src/external/database_comms" "$MOM6/config_src/external/MARBL" "$MOM6/config_src/infra/FMS2"
  "$REFS/SIS2/config_src/dynamic_symmetric" "$REFS/SIS2/config_src/external/Icepack_interfaces" "$REFS/SIS2/src"
  "$REFS/icebergs/src" "$REFS/ice_param" "$REFS/atmos_null" "$REFS/land_null"
  "$REFS/FMScoupler_cefi/shared" "$REFS/FMScoupler_cefi/full"
)
MODEL_DEFS="-DINTERNAL_FILE_NML -Duse_libMPI -Duse_netCDF -DUSE_FMS2_IO -DMAX_FIELDS_=100 -DNOT_SET_AFFINITY \
-D_USE_MOM6_DIAG -D_USE_LEGACY_LAND_ -Duse_AM3_physics"
mkdir -p "$B/model"
cd "$B/model"
cat > Makefile <<EOF
FC = mpif90
CC = mpicc
LD = mpif90
DEFS = $MODEL_DEFS
CPPFLAGS = -I$NC_INC -I$B/fms -I$FMS_SRC/include -I$MOM6/config_src/memory/dynamic_symmetric \
 -I$REFS/SIS2/config_src/dynamic_symmetric -I$MOM6/src/framework
FCFLAGS = $FCBASE $NF_FLAGS -I$B/fms
CFLAGS = -g -O2
LDFLAGS =
LIBS = $B/fms/libFMS.a $NC_LIBS
-include Makefile.dep
EOF
if [[ ! -f Makefile.dep ]]; then
  # one program (FMScoupler full/coupler_main.F90) → target coupler_main (makedep's -x fails for a single program
  # under Python 3, ac/makedep:222); CVMix and GSW come through MOM6/src (src/parameterizations/CVmix,
  # src/equation_of_state/TEOS10 link pkg/), so pkg/ is not listed again
  python3 "$MAKEDEP" $MODEL_DEFS -o Makefile.dep -e "${SRC[@]}"
fi
echo "== model → $B/model/MOM6_SIS2" >&2
make -j"$JOBS" coupler_main > build.log 2>&1 || { grep -m20 -B2 -A6 "Error" build.log | tail -60; echo "model build FAILED ($B/model/build.log)" >&2; exit 1; }
{
  echo "MOM6 $(git -C "$MOM6" rev-parse --short HEAD)"; echo "FMS $(git -C "$FMS_SRC" describe --tags --always)"
  for c in SIS2 icebergs ice_param atmos_null land_null FMScoupler_cefi; do echo "$c $(git -C "$REFS/$c" rev-parse --short HEAD)"; done
  echo "FCFLAGS $FCBASE"; echo "built $(date '+%Y-%m-%d %H:%M')"
} > "$B/model/VERSIONS"
ln -sf coupler_main "$B/model/MOM6_SIS2"
ls -la "$B/model/coupler_main"
