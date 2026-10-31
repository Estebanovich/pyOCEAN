#!/usr/bin/env bash
# MITgcm column of the ERA5 bulk benchmark (pyOCEAN N2b task 7, validation/bulk_column/README.md).
#
#   validation/bulk_column/mitgcm/run_mitgcm.sh build
#   validation/bulk_column/mitgcm/run_mitgcm.sh run [days]       # default 45; inputs from forcing.py
#
# Build (serial gfortran, binary I/O, no NetCDF) in ~/pyOCEAN_refs/build/bulk_column/mitgcm from
# ~/pyOCEAN_refs/MITgcm (git 27256b2) with code/ of this directory plus three files derived from MITgcm's own at
# build time (nothing else is changed):
#   CPP_OPTIONS.h  = model/inc/CPP_OPTIONS.h with SHORTWAVE_HEATING defined (penetrating SW; ATMOSPHERIC_LOADING
#                    stays defined, as shipped: exf needs it to read apressure, and a uniform p_a has no gradient
#                    in a 1 x 1 periodic column)
#   EXF_OPTIONS.h  = pkg/exf/EXF_OPTIONS.h with ALLOW_BULK_LARGEYEAGER04 and EXF_CALC_ATMRHO defined (rho_a from
#                    apressure, atemp and aqh, exf_bulkformulae.F:287-291; the default undefs both, see
#                    validation/bulk_points/README.md); EXF_LWDOWN_WITH_EMISSIVITY is defined by default
#   swfrac.F       = model/src/swfrac.F with jwtype = 5 (Jerlov III) instead of the hard-wired 2 (IA)
#   MOM_COMMON_OPTIONS.h = pkg/mom_common's with ALLOW_BOTTOMDRAG_ROUGHNESS defined (zRoughBot: log-law C_d of the
#                    bottom cell, mom_init_fixed.F:189-215)
# Outputs to /Volumes/ESTEBAN_MIT/pyOCEAN_runs/bulk_column/mitgcm_<days>d (never into the repo).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
export MITGCM_ROOT=${MITGCM_ROOT:-$HOME/pyOCEAN_refs/MITgcm}
MIT=$MITGCM_ROOT
B=$HOME/pyOCEAN_refs/build/bulk_column/mitgcm
RUNS=/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bulk_column
export PATH="/opt/homebrew/bin:$PATH" FC=/opt/homebrew/bin/gfortran
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1

case "${1:-}" in
  build)
    rm -rf "$B"; mkdir -p "$B/code" "$B/build"
    cp "$HERE"/code/* "$B/code/"
    sed -e 's/^#undef SHORTWAVE_HEATING/#define SHORTWAVE_HEATING/' "$MIT/model/inc/CPP_OPTIONS.h" > "$B/code/CPP_OPTIONS.h"
    grep -q "^#define SHORTWAVE_HEATING" "$B/code/CPP_OPTIONS.h"
    grep -q "^#define ATMOSPHERIC_LOADING" "$B/code/CPP_OPTIONS.h"
    sed -e 's/^# undef  ALLOW_BULK_LARGEYEAGER04/# define ALLOW_BULK_LARGEYEAGER04/' \
        -e 's/^# undef EXF_CALC_ATMRHO/# define EXF_CALC_ATMRHO/' "$MIT/pkg/exf/EXF_OPTIONS.h" > "$B/code/EXF_OPTIONS.h"
    grep -q "^# define ALLOW_BULK_LARGEYEAGER04" "$B/code/EXF_OPTIONS.h"
    grep -q "^# define EXF_CALC_ATMRHO" "$B/code/EXF_OPTIONS.h"
    grep -q "^# define EXF_LWDOWN_WITH_EMISSIVITY" "$B/code/EXF_OPTIONS.h"
    sed -e 's/^      jwtype=2$/      jwtype=5/' "$MIT/model/src/swfrac.F" > "$B/code/swfrac.F"
    [[ $(grep -c "^      jwtype=5$" "$B/code/swfrac.F") == 2 ]] || { echo "swfrac.F patch failed" >&2; exit 1; }
    sed -e 's/^#undef ALLOW_BOTTOMDRAG_ROUGHNESS/#define ALLOW_BOTTOMDRAG_ROUGHNESS/' \
        "$MIT/pkg/mom_common/MOM_COMMON_OPTIONS.h" > "$B/code/MOM_COMMON_OPTIONS.h"
    grep -q "^#define ALLOW_BOTTOMDRAG_ROUGHNESS" "$B/code/MOM_COMMON_OPTIONS.h"
    cd "$B/build"
    "$MIT/tools/genmake2" -rootdir "$MIT" -mods ../code \
        -optfile "$ROOT/validation/mitgcm_exf/darwin_arm64_gfortran_nonetcdf" > genmake.log 2>&1 \
        || { tail -30 genmake.log; exit 1; }
    make depend > make_depend.log 2>&1 || { tail -30 make_depend.log; exit 1; }
    make -j 5 > make.log 2>&1 || { grep -n -i error make.log | head -30; exit 1; }
    test -x mitgcmuv && echo "OK $B/build/mitgcmuv" ;;
  run)
    DAYS=${2:-45}
    D="$RUNS/mitgcm_${DAYS}d"
    [[ -f "$D/uwind.bin" ]] || (cd "$ROOT" && python -m validation.bulk_column.forcing --days "$DAYS")
    [[ -x "$B/build/mitgcmuv" ]] || "$0" build
    cd "$D"
    rm -f ./*.data ./*.meta STDOUT.* STDERR.* output.txt data data.* eedata
    cp "$HERE"/input/* .
    NSTEPS=$(( DAYS * 86400 / 120 ))
    sed -i '' -e "s/^ nTimeSteps = [0-9]*,/ nTimeSteps = $NSTEPS,/" data
    cp "$B/build/mitgcmuv" .
    start=$(date +%s)
    ./mitgcmuv > output.txt 2>&1 || true
    grep -q "PROGRAM MAIN: Execution ended Normally" STDOUT.0000 output.txt 2>/dev/null \
        || { tail -20 output.txt; tail -20 STDOUT.0000 2>/dev/null; echo "FAILED $D" >&2; exit 1; }
    echo "MITgcm $DAYS d: $(( $(date +%s) - start )) s wall, outputs in $D" ;;
  *) echo "usage: $0 build | run [days]" >&2; exit 2 ;;
esac
