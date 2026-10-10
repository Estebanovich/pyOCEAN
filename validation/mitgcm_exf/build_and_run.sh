#!/usr/bin/env bash
# Build (serial, gfortran) and run the MITgcm exf bulk-formula validation
# experiment in two variants:
#   ly04 : ALLOW_BULK_LARGEYEAGER04                      (code/)
#   ly09 : ALLOW_BULK_LARGEYEAGER04 + ALLOW_DRAG_LARGEYEAGER09 (code_ly09/ over code/)
#
# Usage:  ./build_and_run.sh [ly04|ly09|all]   (default: all)
# Env:    MITGCM_ROOT (default ~/pyOCEAN_refs/MITgcm), PYTHON (default /opt/anaconda3/bin/python)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
export MITGCM_ROOT="${MITGCM_ROOT:-$HOME/pyOCEAN_refs/MITgcm}"
PYTHON="${PYTHON:-/opt/anaconda3/bin/python}"
OPTFILE="$HERE/darwin_arm64_gfortran_nonetcdf"
export FC=/opt/homebrew/bin/gfortran
export PATH="/opt/homebrew/bin:$PATH"

variants=("${1:-all}")
[[ "${variants[0]}" == all ]] && variants=(ly04 ly09)

"$PYTHON" "$HERE/make_inputs.py"

for v in "${variants[@]}"; do
  case "$v" in
    ly04) mods="$HERE/code" ;;
    ly09) mods="$HERE/code_ly09 $HERE/code" ;;
    *) echo "unknown variant $v" >&2; exit 1 ;;
  esac

  bld="$HERE/build/$v"
  run="$HERE/run/$v"
  mkdir -p "$bld" "$run"

  echo "=== building $v in $bld"
  (
    cd "$bld"
    if [[ ! -x mitgcmuv ]]; then
      "$MITGCM_ROOT/tools/genmake2" -rootdir "$MITGCM_ROOT" -mods "$mods" \
          -optfile "$OPTFILE" > genmake.log 2>&1 || { tail -40 genmake.log; exit 1; }
      make depend > make_depend.log 2>&1 || { tail -40 make_depend.log; exit 1; }
      make -j 8 > make.log 2>&1 || { grep -n -i "error" make.log | head -40; exit 1; }
    fi
  )

  echo "=== running $v in $run"
  (
    cd "$run"
    rm -f ./*.data ./*.meta STDOUT.* STDERR.* ./*.txt ./*.log
    ln -sf "$HERE"/input/* .
    "$bld/mitgcmuv" > output.txt 2>&1 || { tail -40 output.txt; exit 1; }
    grep -q "NORMAL END" STDOUT.0000 2>/dev/null || grep -q "NORMAL END" output.txt \
      || { echo "run $v did not end normally"; tail -40 output.txt; exit 1; }
    ls exfDiag.*.data
  )
done

echo "=== done; now run: $PYTHON $HERE/compare.py"
