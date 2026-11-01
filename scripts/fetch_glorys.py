#!/usr/bin/env python
"""
Download a GLORYS12 subset from Copernicus Marine (``copernicusmarine subset``) for ``GLORYSSource``.

Variables thetao, so, uo, vo, zos of the daily 1/12° product (default: the multi-year reanalysis
``cmems_mod_glo_phy_my_0.083deg_P1D-m``, which is what CEFI uses for NWA12).  Needs ``copernicusmarine login``
once.  Example::

    python scripts/fetch_glorys.py --start 2017-04-29 --end 2017-09-03 \
        --west -76.5 --east -67.5 --south 35.5 --north 42.75 --outdir ~/pyOCEAN_runs/nwa12/glorys

Nothing is written inside the repository: ``--outdir`` is the data directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

VARIABLES = ["thetao", "so", "uo", "vo", "zos"]
DEFAULT_DATASET = "cmems_mod_glo_phy_my_0.083deg_P1D-m"


def build_arguments(args: argparse.Namespace) -> dict:
    """Keyword arguments of ``copernicusmarine.subset`` for the parsed command line."""
    if args.west >= args.east or args.south >= args.north:
        raise ValueError("need west < east and south < north")
    out = Path(args.outdir).expanduser()
    return dict(
        dataset_id=args.dataset, variables=VARIABLES,
        minimum_longitude=args.west, maximum_longitude=args.east,
        minimum_latitude=args.south, maximum_latitude=args.north,
        start_datetime=f"{args.start}T00:00:00", end_datetime=f"{args.end}T00:00:00",
        minimum_depth=args.min_depth, maximum_depth=args.max_depth,
        output_directory=str(out),
        output_filename=f"glorys12_{args.start.replace('-', '')}_{args.end.replace('-', '')}_box.nc",
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="first day, YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="last day, YYYY-MM-DD")
    for k in ("west", "east", "south", "north"):
        ap.add_argument(f"--{k}", type=float, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--min-depth", type=float, default=0.49)
    ap.add_argument("--max-depth", type=float, default=5728.0)
    args = ap.parse_args(argv)
    kwargs = build_arguments(args)
    Path(kwargs["output_directory"]).mkdir(parents=True, exist_ok=True)
    import copernicusmarine
    copernicusmarine.subset(**kwargs)


if __name__ == "__main__":
    main()
