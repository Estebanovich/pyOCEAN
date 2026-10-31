#!/usr/bin/env bash
# Build the CROCO column of the ERA5 bulk benchmark (pyOCEAN N2b task 7, validation/bulk_column/README.md).
#
#   validation/bulk_column/croco/build_croco.sh
#
# CROCO ~/pyOCEAN_refs/croco (git ab63b58), SINGLE_COLUMN with a new case key BULK_COLUMN; the run directory
# ~/pyOCEAN_refs/croco_runs/bulk_column/ gets the jobcomp of tidal_flat3d (arm64 changes of
# validation/thacker/README.md) and local copies of the files below (jobcomp uses local .F/.h before the source
# tree).  Every change is a configuration of the case or a documented fix, none touches bulk_flux.F:
#
#   cppdefs.h     REGIONAL off, SINGLE_COLUMN on, case key BULK_COLUMN alone; in its block: SALINITY, KPP for the
#                 surface layer only (LMD_MIXING + LMD_SKPP + LMD_NONLOCAL, the 'kppsfc' set of
#                 validation/turbulence_columns), BULK_FLUX with CROCO's default options (BULK_GUSTINESS,
#                 BULK_LW; cppdefs.h:178-185) instead of ANA_SMFLUX/ANA_SRFLUX/ANA_STFLUX, ANA_BSFLUX, AVERAGES
#   param.h       BULK_COLUMN: LLm0 = MMm0 = 5, N = 36
#   ana_grid.F    BULK_COLUMN: depth = 36 m, f0 = 2 Omega sin(40.24 deg) = 9.421258e-5 s-1, beta = 0
#   ana_initial.F BULK_COLUMN: T, S of the WOA23 profile (bulk_column_init.h, written by
#                 validation.bulk_column.forcing.croco_init_header)
#   analytical.F  BULK_COLUMN: bottom heat flux 0 (the SINGLE_COLUMN branch sets it only for the stratified cases)
#   lmd_swfrac.F  lmd_Jwt = 5 (Jerlov III) instead of the hard-wired 1, in both routines of the file (cases.py: the
#                 same type in the three models)
#   alfabeta.F    linear-EOS alpha = |Tcoef|/rho0, beta = |Scoef|/rho0: the class-(d) CROCO error found in N3
#                 (alfabeta.F:133,138 omit 1/rho0; the KPP surface buoyancy flux is rho0 times too large),
#                 corrected as in validation/turbulence_columns/croco/build_croco.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SRC=${CROCO_SRC:-$HOME/pyOCEAN_refs/croco/OCEAN}
JOBCOMP=${CROCO_JOBCOMP:-$HOME/pyOCEAN_refs/croco_runs/tidal_flat3d/jobcomp}
RUN=$HOME/pyOCEAN_refs/croco_runs/bulk_column

mkdir -p "$RUN"
cd "$RUN"
rm -f ./*.F ./*.h
cp -f "$JOBCOMP" jobcomp
(cd "$ROOT" && python -c "from pathlib import Path; from validation.bulk_column.forcing import croco_init_header; \
croco_init_header(Path('$RUN/bulk_column_init.h'))")

python3 - "$SRC" "$RUN" <<'EOF'
import re, sys
src, run = sys.argv[1:]
def sub(txt, old, new, name):
    assert txt.count(old) == 1, f"{name}: patch anchor not found exactly once: {old[:60]!r}"
    return txt.replace(old, new)

# --- cppdefs.h
txt = open(f"{src}/cppdefs.h").read()
txt = sub(txt, "#define REGIONAL        /* REGIONAL Applications */",
          "#undef  REGIONAL        /* REGIONAL Applications */", "cppdefs")
txt = sub(txt, "#undef  SINGLE_COLUMN   /* 1DV vertical mixing Example */",
          "#define SINGLE_COLUMN   /* 1DV vertical mixing Example */", "cppdefs")
i0 = txt.index("#elif defined SINGLE_COLUMN")
i1 = txt.index("#elif defined", i0 + 10)
blk = txt[i0:i1]
blk = re.sub(r"# define +KATO_PHILIPS\b", "# undef  KATO_PHILIPS", blk)
blk = sub(blk, "# undef  OPENMP", "# define BULK_COLUMN\n# undef  OPENMP", "cppdefs block")
blk = sub(blk, "# undef  LMD_MIXING\n# define GLS_MIXING",
          "# define LMD_MIXING\n# define LMD_SKPP\n# define LMD_NONLOCAL\n# undef  GLS_MIXING", "cppdefs block")
blk = sub(blk, "# define ANA_SMFLUX\n# define ANA_SRFLUX\n# define ANA_STFLUX\n# define ANA_BTFLUX",
          "# define SALINITY\n# define BULK_FLUX\n# define BULK_GUSTINESS\n# define BULK_LW\n# undef  SST_SKIN\n"
          "# undef  READ_PATM\n# undef  ANA_SMFLUX\n# undef  ANA_SRFLUX\n# undef  ANA_STFLUX\n# define ANA_BTFLUX\n"
          "# define ANA_BSFLUX\n# define AVERAGES", "cppdefs block")
assert len(re.findall(r"# define +(KATO_PHILIPS|WILLIS_DEARDORFF|DIURNAL_CYCLE|FORCED_\w+)\b", blk)) == 0
txt = txt[:i0] + blk + txt[i1:]
open(f"{run}/cppdefs.h", "w").write(txt)

# --- param.h
txt = open(f"{src}/param.h").read()
txt = sub(txt, "#elif defined SINGLE_COLUMN\n# ifdef KATO_PHILIPS",
          "#elif defined SINGLE_COLUMN\n# if defined BULK_COLUMN\n      parameter (LLm0=5 ,   MMm0=5,    N=36)\n"
          "# elif defined KATO_PHILIPS", "param.h")
open(f"{run}/param.h", "w").write(txt)

# --- ana_grid.F
txt = open(f"{src}/ana_grid.F").read()
txt = sub(txt, "# elif defined SINGLE_COLUMN\n#  if defined KATO_PHILIPS || defined WILLIS_DEARDORFF",
          "# elif defined SINGLE_COLUMN\n#  if defined BULK_COLUMN\n                     depth=36.\n"
          "                     f0=9.421258e-5\n                     beta=0.\n"
          "#  elif defined KATO_PHILIPS || defined WILLIS_DEARDORFF", "ana_grid.F")
open(f"{run}/ana_grid.F", "w").write(txt)

# --- ana_initial.F
txt = open(f"{src}/ana_initial.F").read()
txt = sub(txt, "#   elif defined SINGLE_COLUMN && defined WILLIS_DEARDORFF",
          "#   elif defined SINGLE_COLUMN && defined BULK_COLUMN\n#    include \"bulk_column_init.h\"\n"
          "#   elif defined SINGLE_COLUMN && defined WILLIS_DEARDORFF", "ana_initial.F")
open(f"{run}/ana_initial.F", "w").write(txt)

# --- analytical.F (bottom heat flux)
txt = open(f"{src}/analytical.F").read()
txt = sub(txt, "            btflx(i,j,itemp)=-1.0e-5*0.01\n#    endif",
          "            btflx(i,j,itemp)=-1.0e-5*0.01\n#    else\n            btflx(i,j,itemp)=0.\n#    endif",
          "analytical.F")
open(f"{run}/analytical.F", "w").write(txt)

# --- lmd_swfrac.F (Jerlov III)
txt = open(f"{src}/lmd_swfrac.F").read()
assert txt.count("      lmd_Jwt=1 ") == 2, "lmd_swfrac.F: expected the two lmd_Jwt=1 (lmd_swfrac_tile, lmd_swfrac_ER_tile)"
txt = txt.replace("      lmd_Jwt=1 ", "      lmd_Jwt=5 ")
open(f"{run}/lmd_swfrac.F", "w").write(txt)

# --- alfabeta.F (linear EOS, N3 class-d fix)
txt = open(f"{src}/alfabeta.F").read()
txt = re.sub(r"(?m)^( *)alpha\(i,j\)=abs\(Tcoef\)$", r"\1alpha(i,j)=abs(Tcoef)/rho0", txt)
txt = re.sub(r"(?m)^( *)beta\(i,j\)=abs\(Scoef\)$", r"\1beta(i,j)=abs(Scoef)/rho0", txt)
assert "alpha(i,j)=abs(Tcoef)/rho0" in txt and "beta(i,j)=abs(Scoef)/rho0" in txt, "alfabeta.F patch failed"
open(f"{run}/alfabeta.F", "w").write(txt)
print("patched: cppdefs.h param.h ana_grid.F ana_initial.F analytical.F lmd_swfrac.F alfabeta.F")
EOF

echo "building CROCO bulk column in $RUN"
NPROCS=${NPROCS:-5} ./jobcomp > build.log 2>&1 || { tail -30 build.log; exit 1; }
test -x croco && echo "OK: $RUN/croco"
