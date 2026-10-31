#!/usr/bin/env bash
# Run the CROCO column built by build_croco.sh (pyOCEAN N2b task 7).
#
#   validation/bulk_column/croco/run_croco.sh [days]          # default 45
#
# croco.in of this directory with NTIMES = days x 720 (dt = 120 s) and the output/bulk paths moved to
# /Volumes/ESTEBAN_MIT/pyOCEAN_runs/bulk_column/croco_<days>d (never into the repo), where forcing.py writes
# croco_blk.nc.  Hourly snapshots (scm_his.nc) and hourly means of the fluxes (scm_avg.nc).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
RUN=$HOME/pyOCEAN_refs/croco_runs/bulk_column
DAYS=${1:-45}
OUT=/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bulk_column/croco_${DAYS}d
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1

test -x "$RUN/croco" || { echo "build first: build_croco.sh" >&2; exit 1; }
[[ -f "$OUT/croco_blk.nc" ]] || (cd "$ROOT" && python -m validation.bulk_column.forcing --days "$DAYS")
cd "$OUT"
rm -f scm_his.nc scm_avg.nc scm_rst.nc croco.log
NT=$(( DAYS * 86400 / 120 ))
sed -E -e "s#^( *)scm_(his|avg|rst)\.nc#\1$OUT/scm_\2.nc#" \
       -e "s#^( *)croco_blk\.nc#\1$OUT/croco_blk.nc#" \
       -e "s#^( *)32400( +120 +30 +720)\$#\1$NT\2#" \
       -e "s#^( *)32400( +-1)\$#\1$NT\2#" "$HERE/croco.in" > croco.in
grep -q "$OUT/croco_blk.nc" croco.in && grep -q "^ *$NT  *120" croco.in || { echo "croco.in substitution failed" >&2; exit 1; }
start=$(date +%s)
"$RUN/croco" croco.in > croco.log 2>&1 || { tail -30 croco.log; exit 1; }
grep -q -i "blowup\|MAIN: Abnormal" croco.log && { tail -30 croco.log; echo "CROCO failed" >&2; exit 1; }
echo "CROCO $DAYS d: $(( $(date +%s) - start )) s wall, outputs in $OUT"
