"""
Check SWDDIF2 (diffuse solar radiation) anomalies in WRF 3km (RDPS) output.
Scans +6h forecast files for each analysis time (00/06/12/18 UTC) in April 2026
and reports timestamps where SWDDIF2 >= threshold.

Directory structure:
  {base_dir}/DD/HH/rdps_pres_r030_h006.{YYYYMMDDHH}.nc        (raw file)
  {base_dir}/DD/HH/r030_v040_easia_prs.2byte.ft006.{YYYYMMDDHH}.nc  (light file)

Usage:
    python check_swddif2_anomaly.py /ARCV/NWP/RAWD/MODL/RDPS/NE57/202604
    python check_swddif2_anomaly.py /ARCV/NWP/RAWD/MODL/RDPS/NE57/202604 --threshold 500
    python check_swddif2_anomaly.py /ARCV/NWP/RAWD/MODL/RDPS/NE57/202604 --start 10 --end 15
    python check_swddif2_anomaly.py /ARCV/NWP/RAWD/MODL/RDPS/NE57/202604 --start 20 --end 20 --hours 0 12
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr


THRESHOLD_DEFAULT = 500.0
VARNAME_DEFAULT   = "SWDDIF2"
ANAL_HOURS        = [0, 6, 12, 18]

RAW_PATTERN   = "rdps_pres_r030_h006.{analtim}.nc"
LIGHT_PATTERN = "r030_v040_easia_prs.2byte.ft006.{analtim}.nc"


def read_swddif2(fpath, varname):
    """Read varname from NetCDF file via xarray (scale_factor/add_offset applied automatically)."""
    result = {"file": fpath.name, "exists": fpath.exists()}
    if not fpath.exists():
        return result

    try:
        with xr.open_dataset(fpath, mask_and_scale=True) as ds:
            if varname not in ds:
                candidates = [v for v in ds.data_vars if "SWDDIF" in v.upper() or "DIF" in v.upper()]
                result["error"] = f"Variable '{varname}' not found"
                if candidates:
                    result["candidates"] = candidates
                return result

            data     = ds[varname].values.astype(np.float32)
            max_val  = float(np.nanmax(data))
            mean_val = float(np.nanmean(data))
            n_above  = int(np.sum(data >= THRESHOLD_DEFAULT))

            result.update({"max": max_val, "mean": mean_val, "n_above": n_above,
                           "shape": data.shape, "units": ds[varname].attrs.get("units", "-")})
    except Exception as e:
        result["error"] = str(e)

    return result


def check_file(fpath, varname, threshold):
    """Read file and flag if any value exceeds threshold."""
    r = read_swddif2(fpath, varname)
    if "n_above" in r:
        r["anomaly"] = r["n_above"] > 0
    return r


def print_result_table(rows, threshold, label):
    """Print table of timestamps where anomalies were found."""
    anomalies = [r for r in rows if r.get("anomaly")]

    print(f"\n{'─'*70}")
    print(f"  [{label}]  threshold >= {threshold}")
    print(f"{'─'*70}")

    if not anomalies:
        print("  No anomalies found.")
        return

    print(f"  {'Analysis time(UTC)':<20} {'Max':>10} {'Mean':>10} {'>={:.0f} grids'.format(threshold):>14}  File")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*14}  {'-'*35}")

    prev_day = None
    for r in anomalies:
        cur_day = r["analtim"][:8]
        if cur_day != prev_day and prev_day is not None:
            print()
        prev_day = cur_day
        ts = f"{r['analtim'][:4]}-{r['analtim'][4:6]}-{r['analtim'][6:8]} {r['analtim'][8:10]}UTC"
        print(f"  {ts:<20} {r['max']:>10.2f} {r['mean']:>10.4f} {r['n_above']:>14}  {r['file']}")

    daily = defaultdict(lambda: {"max": 0.0, "n_above": 0, "hours": []})
    for r in anomalies:
        d = r["analtim"][:8]
        daily[d]["max"]     = max(daily[d]["max"], r["max"])
        daily[d]["n_above"] += r["n_above"]
        daily[d]["hours"].append(int(r["analtim"][8:10]))

    print(f"\n  [Daily summary]")
    print(f"  {'Date':<12} {'Daily max':>10} {'Total grids':>14}  Analysis hours with anomaly (UTC)")
    print(f"  {'-'*12} {'-'*10} {'-'*14}  {'-'*35}")
    for d in sorted(daily):
        hours_str = ", ".join(f"{h:02d}h" for h in sorted(daily[d]["hours"]))
        dstr = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        print(f"  {dstr:<12} {daily[d]['max']:>10.2f} {daily[d]['n_above']:>14}  {hours_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Check SWDDIF2 anomalies in RDPS April 2026 (+6h forecast)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Examples:\n"
               "  python check_swddif2_anomaly.py /ARCV/.../202604\n"
               "  python check_swddif2_anomaly.py /ARCV/.../202604 --start 10 --end 15\n"
               "  python check_swddif2_anomaly.py /ARCV/.../202604 --start 20 --end 20 --hours 0 12\n"
    )
    parser.add_argument("base_dir",    type=str,   help="Path to 202604 directory")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT,
                        help=f"Anomaly threshold (default: {THRESHOLD_DEFAULT})")
    parser.add_argument("--varname",   type=str,   default=VARNAME_DEFAULT,
                        help=f"Variable name (default: {VARNAME_DEFAULT})")
    parser.add_argument("--start",     type=int,   default=1,
                        help="Start day (default: 1)")
    parser.add_argument("--end",       type=int,   default=30,
                        help="End day (default: 30)")
    parser.add_argument("--hours",     type=int,   nargs="+", default=ANAL_HOURS,
                        choices=[0, 6, 12, 18], metavar="{0,6,12,18}",
                        help="Analysis hours to check in UTC (default: 0 6 12 18)")
    args = parser.parse_args()

    base_dir   = Path(args.base_dir)
    threshold  = args.threshold
    varname    = args.varname
    start_day  = args.start
    end_day    = args.end
    anal_hours = sorted(args.hours)

    if not base_dir.exists():
        print(f"[ERROR] Directory not found: {base_dir}")
        sys.exit(1)
    if not (1 <= start_day <= end_day <= 30):
        print(f"[ERROR] Invalid day range: --start {start_day} --end {end_day} (must be 1-30)")
        sys.exit(1)

    print("=" * 70)
    print(f"  SWDDIF2 Anomaly Check  |  threshold >= {threshold}")
    print(f"  Period : 2026-04-{start_day:02d} ~ 2026-04-{end_day:02d}")
    print(f"  Anal.  : {anal_hours} UTC  |  Forecast: +6h")
    print(f"  Dir    : {base_dir}")
    print("=" * 70)

    raw_rows   = []
    light_rows = []
    missing    = []
    errors     = []

    for day in range(start_day, end_day + 1):
        for hh in anal_hours:
            analtim = f"202604{day:02d}{hh:02d}"
            subdir  = base_dir / f"{day:02d}" / f"{hh:02d}"

            raw_path   = subdir / RAW_PATTERN.format(analtim=analtim)
            light_path = subdir / LIGHT_PATTERN.format(analtim=analtim)

            for path, rows in ((raw_path, raw_rows), (light_path, light_rows)):
                r = check_file(path, varname, threshold)
                r["analtim"] = analtim
                if not r["exists"]:
                    missing.append(str(path))
                elif "error" in r:
                    errors.append(f"  {path.name}: {r['error']}"
                                  + (f"  candidates: {r.get('candidates')}" if "candidates" in r else ""))
                else:
                    rows.append(r)

    n_days = end_day - start_day + 1
    total  = n_days * len(anal_hours)
    print(f"\n[File summary]  {total} analysis times ({n_days} days x {len(anal_hours)} hours) x 2 file types")
    print(f"  Raw   files read OK: {len(raw_rows):3d} / missing: {sum(1 for p in missing if 'rdps_pres' in p)}")
    print(f"  Light files read OK: {len(light_rows):3d} / missing: {sum(1 for p in missing if 'r030_v040' in p)}")

    if errors:
        print(f"\n[Read errors] {len(errors)} case(s)")
        for e in errors:
            print(e)

    print_result_table(raw_rows,   threshold, "Raw   file: rdps_pres_r030_h006")
    print_result_table(light_rows, threshold, "Light file: r030_v040_easia_prs.2byte.ft006")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
