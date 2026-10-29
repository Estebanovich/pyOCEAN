#!/usr/bin/env bash
# Run one CROCO SINGLE_COLUMN case built by build_croco.sh (N3 task 8).
#
#   validation/turbulence_columns/croco/run_croco.sh CASE CLOSURE
#
# The input file is CROCO's own TEST_CASES/croco.in.<KEY> unchanged except the output paths (and NINFO = 120 to keep
# the log short); it is passed on the command line, which also avoids the KATO_PHILLIPS filename typo of
# read_inp.F:127.  Outputs (scm_his.nc, scm_rst.nc, croco.log) go to
# /Volumes/ESTEBAN_MIT/pyOCEAN_runs/turbulence_columns/croco/<case>_<closure>/ (never into the repo).
set -euo pipefail

CASE=${1:?case}
CLOSURE=${2:?closure (gls|kpp|kppsfc|kppraw)}
SRC=${CROCO_SRC:-$HOME/pyOCEAN_refs/croco/OCEAN}
RUN=$HOME/pyOCEAN_refs/croco_runs/scm/${CASE}_${CLOSURE}
OUT=${PYOCEAN_RUNS:-/Volumes/ESTEBAN_MIT/pyOCEAN_runs}/turbulence_columns/croco/${CASE}_${CLOSURE}

case $CASE in
  kato_phillips) IN=croco.in.KATO_PHILIPS ;;
  willis_deardorff) IN=croco.in.WILLIS_DEARDORFF ;;
  diurnal_cycle) IN=croco.in.DIURNALCYCLE ;;
  forced_ekbbl) IN=croco.in.FORCED_EKBBL ;;
  forced_dbleek) IN=croco.in.FORCED_DBLEEK ;;
  forced_nonrotbbl) IN=croco.in.FORCED_NONROTBBL ;;
  forced_oscnonrotbbl) IN=croco.in.FORCED_OSCNONROTBBL ;;
  *) echo "unknown case $CASE" >&2; exit 2 ;;
esac
test -x "$RUN/croco" || { echo "build first: build_croco.sh $CASE $CLOSURE" >&2; exit 1; }

mkdir -p "$OUT"
cd "$OUT"
rm -f scm_his.nc scm_avg.nc scm_rst.nc
# output paths into $OUT; NINFO (4th number of the time_stepping line) 1 -> 120; history output also of Akv, Akt,
# HBL, HBBL, the stresses and Shfl (line 33) and of the GLS TKE, GLS (= epsilon for k-epsilon) and Lscale (line 35):
# SINGLE_COLUMN does not define AVERAGES (only the REGIONAL block does, cppdefs.h:310), so scm_avg.nc is never
# written and the hourly snapshots of scm_his.nc are the reference
sed -E -e "s#^( *)scm_(his|avg|rst)\.nc#\1$OUT/scm_\2.nc#" \
    -e "4s#^( *[0-9]+ +[0-9]+ +[0-9]+ +)1 *\$#\1120#" \
    -e "33s#.*#                             F   F     F   T    T    F    F     F       T   T    T     T     T    T    T    T    T   F  10*F#" \
    -e "35s#.*#                      T     T    T#" \
    "$SRC/../TEST_CASES/$IN" > croco.in
grep -q "$OUT/scm_his.nc" croco.in || { echo "output path substitution failed" >&2; exit 1; }
start=$(date +%s)
OMP_NUM_THREADS=1 "$RUN/croco" croco.in > croco.log 2>&1 || { tail -30 croco.log; exit 1; }
echo "$CASE $CLOSURE: $(( $(date +%s) - start )) s wall, outputs in $OUT"
