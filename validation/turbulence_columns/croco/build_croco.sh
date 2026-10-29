#!/usr/bin/env bash
# Build one CROCO SINGLE_COLUMN executable (N3 task 8; docs/design/n2c_refs/croco_column_cases.md).
#
#   validation/turbulence_columns/croco/build_croco.sh CASE CLOSURE
#
# CASE    : kato_phillips | willis_deardorff | diurnal_cycle | forced_ekbbl | forced_dbleek |
#           forced_nonrotbbl | forced_oscnonrotbbl
# CLOSURE : gls (the shipped SINGLE_COLUMN block: GLS k-epsilon, Canuto A, cppdefs.h:779-792)
#           kpp (#undef GLS_MIXING, #define LMD_MIXING with the regional preset of cppdefs.h:255-259:
#                LMD_SKPP, LMD_BKPP, LMD_RIMIX, LMD_CONVEC, LMD_NONLOCAL; diurnal_cycle also DIURNAL_SRFLUX,
#                which only selects Ric = 0.15 in lmd_skpp2005.F:97-103 — CROCO's own setting for a diurnal
#                shortwave cycle, McWilliams et al. 2009)
#           kppsfc (surface KPP alone, the pyOCEAN 'kpp' analogue: LMD_MIXING + LMD_SKPP + LMD_NONLOCAL, without the
#                bottom KPP, the interior Ri mixing and the convective adjustment; DIURNAL_SRFLUX as above)
#           kppraw (= kpp without the alfabeta.F fix below, to document the CROCO bug)
#
# CROCO bug fixed locally for kpp and kppsfc (class d, N3 task 8 README): with the linear EOS, alfabeta.F:133 (and
# :138) returns alpha = |Tcoef| (beta = |Scoef|) in kg m-3 K-1, while the NONLIN_EOS branch divides by rho0
# (alfabeta.F:116-118) and lmd_skpp2005.F:180-186 needs alpha in K-1 (Bo = g alpha stflx [m2 s-3]); the surface
# buoyancy flux of KPP is then rho0 = 1024 times too large and free convection mixes the whole 50 m column in 6 h
# (Willis-Deardorff).  Only KPP calls alfabeta_tile (lmd_skpp1994.F:177, lmd_skpp2005.F:167).  The run directory
# gets a copy of alfabeta.F with alpha = |Tcoef|/rho0, beta = |Scoef|/rho0 in the linear branch (jobcomp uses local
# .F files before the source tree).
#
# The run directory is ~/pyOCEAN_refs/croco_runs/scm/<case>_<closure>/ (the jobcomp of tidal_flat3d with the three
# arm64 changes of validation/thacker/README.md).  Only cppdefs.h is changed; param.h, ana_*.F, analytical.F and
# prsgrd.F are CROCO's own (the vertical resolution comes from param.h:78-93 through the case key).
set -euo pipefail

CASE=${1:?case}
CLOSURE=${2:?closure (gls|kpp|kppsfc|kppraw)}
SRC=${CROCO_SRC:-$HOME/pyOCEAN_refs/croco/OCEAN}
JOBCOMP=${CROCO_JOBCOMP:-$HOME/pyOCEAN_refs/croco_runs/tidal_flat3d/jobcomp}
RUN=$HOME/pyOCEAN_refs/croco_runs/scm/${CASE}_${CLOSURE}

case $CASE in
  kato_phillips) KEY=KATO_PHILIPS ;;
  willis_deardorff) KEY=WILLIS_DEARDORFF ;;
  diurnal_cycle) KEY=DIURNAL_CYCLE ;;
  forced_ekbbl) KEY=FORCED_EKBBL ;;
  forced_dbleek) KEY=FORCED_DBLEEK ;;
  forced_nonrotbbl) KEY=FORCED_NONROTBBL ;;
  forced_oscnonrotbbl) KEY=FORCED_OSCNONROTBBL ;;
  *) echo "unknown case $CASE" >&2; exit 2 ;;
esac
[[ $CLOSURE =~ ^(gls|kpp|kppsfc|kppraw)$ ]] || { echo "closure must be gls, kpp, kppsfc or kppraw" >&2; exit 2; }

mkdir -p "$RUN"
cd "$RUN"
cp -f "$JOBCOMP" jobcomp
cp -f "$SRC/param.h" param.h

# cppdefs.h: REGIONAL off, SINGLE_COLUMN on, exactly one case key inside the SINGLE_COLUMN block
python3 - "$SRC/cppdefs.h" cppdefs.h "$KEY" "$CLOSURE" "$CASE" <<'EOF'
import re, sys
src, dst, key, closure, case = sys.argv[1:]
txt = open(src).read()
txt = txt.replace("#define REGIONAL        /* REGIONAL Applications */",
                  "#undef  REGIONAL        /* REGIONAL Applications */", 1)
txt = txt.replace("#undef  SINGLE_COLUMN   /* 1DV vertical mixing Example */",
                  "#define SINGLE_COLUMN   /* 1DV vertical mixing Example */", 1)
i0 = txt.index("#elif defined SINGLE_COLUMN")
i1 = txt.index("#elif defined", i0 + 10)
blk = txt[i0:i1]
keys = ["KATO_PHILIPS", "WILLIS_DEARDORFF", "DIURNAL_CYCLE", "FORCED_EKBBL", "FORCED_DBLEEK",
        "FORCED_NONROTBBL", "FORCED_OSCNONROTBBL"]
for k in keys:
    blk = re.sub(r"# (define|undef) +%s\b" % k, "# %s %s" % ("define" if k == key else "undef ", k), blk)
if closure in ("kpp", "kppsfc", "kppraw"):
    extra = ["# define LMD_SKPP", "# define LMD_BKPP", "# define LMD_RIMIX", "# define LMD_CONVEC",
             "# define LMD_NONLOCAL"]
    if closure == "kppsfc":
        extra = ["# define LMD_SKPP", "# define LMD_NONLOCAL"]
    if case == "diurnal_cycle":
        extra.append("# define DIURNAL_SRFLUX")
    blk = blk.replace("# undef  LMD_MIXING\n# define GLS_MIXING",
                      "# define LMD_MIXING\n# undef  GLS_MIXING\n" + "\n".join(extra))
    assert "# define LMD_MIXING" in blk, "KPP substitution failed"
txt = txt[:i0] + blk + txt[i1:]
assert txt.count("#define SINGLE_COLUMN") == 1 and txt.count("#undef  REGIONAL") == 1
open(dst, "w").write(txt)
EOF

rm -f alfabeta.F
if [[ $CLOSURE == kpp || $CLOSURE == kppsfc ]]; then
  sed -e 's#^\( *\)alpha(i,j)=abs(Tcoef)$#\1alpha(i,j)=abs(Tcoef)/rho0#' \
      -e 's#^\( *\)beta(i,j)=abs(Scoef)$#\1beta(i,j)=abs(Scoef)/rho0#' "$SRC/alfabeta.F" > alfabeta.F
  grep -q "alpha(i,j)=abs(Tcoef)/rho0" alfabeta.F || { echo "alfabeta.F patch failed" >&2; exit 1; }
fi

echo "building $CASE ($CLOSURE) in $RUN"
NPROCS=${NPROCS:-5} ./jobcomp > build.log 2>&1 || { tail -30 build.log; exit 1; }
test -x croco && echo "OK: $RUN/croco"
