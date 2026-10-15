#!/usr/bin/env bash
# MITgcm z runs of the DOME benchmark (spec 2026-10-14-dome-overflow §6.2).
#   run_mitgcm.sh build <res_km>                          : genmake2 + make in RUNS/builds/mitgcm_<res>km (serial)
#   run_mitgcm.sh run <res_km> <legg|common> [tag] [days]  : write, run, read → series.npz
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNS=/Volumes/ESTEBAN_MIT/pyOCEAN_runs/dome
MIT=/Volumes/ESTEBAN_MIT/MITgcm
case "${1:-}" in
  build)
    B="$RUNS/builds/mitgcm_${2}km"; rm -rf "$B"; mkdir -p "$B/build"
    (cd "$ROOT" && python -c "from pathlib import Path; from validation.dome import mitgcm; \
mitgcm.write_build(float('$2'), Path('$B/code'))")
    cd "$B/build"; export NETCDF_ROOT=/opt/homebrew
    "$MIT/tools/genmake2" -rootdir "$MIT" -mods ../code -optfile "$MIT/tools/build_options/darwin_arm64_gfortran_esteban" \
        > genmake.log 2>&1
    make depend > make_depend.log 2>&1 && make -j 5 > make.log 2>&1 || true
    [[ -x mitgcmuv ]] || { tail -20 make.log; echo "FAILED build $2" >&2; exit 1; }
    echo "OK $B/build/mitgcmuv" ;;
  run)
    RES=$2; VAR=$3; TAG=${4:-$3}; DAYS=${5:-}
    B="$RUNS/builds/mitgcm_${RES}km/build"
    [[ -x "$B/mitgcmuv" ]] || "$0" build "$RES"
    D=$(cd "$ROOT" && python -c "from validation.dome import cases; print(cases.run_dir(float('$RES'), 'mitgcm', 'z', '$TAG'))")
    rm -rf "$D"
    (cd "$ROOT" && python -c "from pathlib import Path; from validation.dome import mitgcm; \
mitgcm.write_run(float('$RES'), Path('$D'), '$VAR', days=${DAYS:-None})")
    cd "$D"; cp "$B/mitgcmuv" .
    ./mitgcmuv > output.txt 2>&1 || true
    grep -q "PROGRAM MAIN: Execution ended Normally" output.txt || { tail -15 output.txt; echo "FAILED $D" >&2; exit 1; }
    (cd "$ROOT" && python -c "
from pathlib import Path
from validation.dome import cases, mitgcm, series
s = mitgcm.read(Path('$D'), float('$RES'))
c = cases.get(float('$RES'))
meta = dict(model='MITgcm', coord='z', res_km=float('$RES'), variant='$VAR', dt=600.0 * c.dx / 10e3,
            n_steps=int(round(s['t'][-1] / (600.0 * c.dx / 10e3))), wall_s=mitgcm.wall_s(Path('$D')), threads=1)
print(series.save(Path('$D') / 'series.npz', s, meta))" 2>&1 | grep -v Warn) ;;
  *) echo "usage: $0 build <res_km> | run <res_km> <legg|common> [tag] [days]" >&2; exit 2 ;;
esac
