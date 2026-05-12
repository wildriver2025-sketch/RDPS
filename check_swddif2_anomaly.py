"""
2026년 4월 WRF 3km(RDPS) 분석시간(00/06/12/18 UTC) +6h 예측자료에서
SWDDIF2(산란일사량) 값이 500 이상인 날짜/시간을 검사하는 스크립트.

디렉터리 구조:
  {base_dir}/DD/HH/rdps_pres_r030_h006.{YYYYMMDDHH}.nc        (원시파일)
  {base_dir}/DD/HH/r030_v040_easia_prs.2byte.ft006.{YYYYMMDDHH}.nc  (경량화파일)

사용법:
    python check_swddif2_anomaly.py /ARCV/NWP/RAWD/MODL/RDPS/NE57/202604
    python check_swddif2_anomaly.py /ARCV/NWP/RAWD/MODL/RDPS/NE57/202604 --threshold 500
    python check_swddif2_anomaly.py /ARCV/NWP/RAWD/MODL/RDPS/NE57/202604 --varname SWDDIF2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


THRESHOLD_DEFAULT = 500.0
VARNAME_DEFAULT   = "SWDDIF2"
ANAL_HOURS        = [0, 6, 12, 18]

# 파일명 패턴
RAW_PATTERN   = "rdps_pres_r030_h006.{analtim}.nc"
LIGHT_PATTERN = "r030_v040_easia_prs.2byte.ft006.{analtim}.nc"


def read_swddif2(fpath, varname):
    """netCDF4로 파일에서 varname 변수를 읽어 통계를 반환."""
    result = {"file": fpath.name, "exists": fpath.exists()}
    if not fpath.exists():
        return result

    try:
        with Dataset(fpath, "r") as nc:
            if varname not in nc.variables:
                candidates = [v for v in nc.variables if "SWDDIF" in v.upper() or "DIF" in v.upper()]
                result["error"] = f"변수 '{varname}' 없음"
                if candidates:
                    result["candidates"] = candidates
                return result

            var    = nc.variables[varname]
            data   = var[:]
            scale  = getattr(var, "scale_factor", 1.0)
            offset = getattr(var, "add_offset",   0.0)
            fill   = getattr(var, "_FillValue",    None)

            data = np.ma.filled(data.astype(np.float32), np.nan)
            if fill is not None:
                data[data == fill] = np.nan
            data = data * scale + offset

            max_val  = float(np.nanmax(data))
            mean_val = float(np.nanmean(data))
            n_above  = int(np.sum(data >= THRESHOLD_DEFAULT))

            result.update({"max": max_val, "mean": mean_val, "n_above": n_above,
                           "shape": data.shape, "units": getattr(var, "units", "-")})
    except Exception as e:
        result["error"] = str(e)

    return result


def check_file(fpath, varname, threshold):
    """파일 읽기 + 임계값 이상 여부 반환."""
    r = read_swddif2(fpath, varname)
    if "n_above" in r:
        r["anomaly"] = r["n_above"] > 0
    return r


def print_result_table(rows, threshold, label):
    """이상값이 있는 행만 테이블로 출력."""
    anomalies = [r for r in rows if r.get("anomaly")]

    print(f"\n{'─'*70}")
    print(f"  [{label}]  임계값 >= {threshold}")
    print(f"{'─'*70}")

    if not anomalies:
        print("  이상값 없음")
        return

    print(f"  {'분석시간(UTC)':<16} {'최대값':>10} {'평균값':>10} {'>={:.0f} 격자수':>14}  파일명".format(threshold))
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*14}  {'-'*35}")

    prev_day = None
    for r in anomalies:
        cur_day = r["analtim"][:8]
        if cur_day != prev_day and prev_day is not None:
            print()
        prev_day = cur_day
        ts = f"{r['analtim'][:4]}-{r['analtim'][4:6]}-{r['analtim'][6:8]} {r['analtim'][8:10]}UTC"
        print(f"  {ts:<16} {r['max']:>10.2f} {r['mean']:>10.4f} {r['n_above']:>14}  {r['file']}")

    # 날짜별 요약
    from collections import defaultdict
    daily = defaultdict(lambda: {"max": 0.0, "n_above": 0, "hours": []})
    for r in anomalies:
        d = r["analtim"][:8]
        daily[d]["max"]    = max(daily[d]["max"], r["max"])
        daily[d]["n_above"] += r["n_above"]
        daily[d]["hours"].append(int(r["analtim"][8:10]))

    print(f"\n  [날짜별 요약]")
    print(f"  {'날짜':<12} {'일최대값':>10} {'누적 이상격자':>14}  이상 발생 분석시간(UTC)")
    print(f"  {'-'*12} {'-'*10} {'-'*14}  {'-'*30}")
    for d in sorted(daily):
        hours_str = ", ".join(f"{h:02d}h" for h in sorted(daily[d]["hours"]))
        dstr = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        print(f"  {dstr:<12} {daily[d]['max']:>10.2f} {daily[d]['n_above']:>14}  {hours_str}")


def main():
    parser = argparse.ArgumentParser(
        description="SWDDIF2 이상값 검사 (2026년 4월, +6h 예측)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="예시:\n"
               "  python check_swddif2_anomaly.py /ARCV/.../202604\n"
               "  python check_swddif2_anomaly.py /ARCV/.../202604 --start 10 --end 15\n"
               "  python check_swddif2_anomaly.py /ARCV/.../202604 --start 20 --end 20 --hours 00 12\n"
    )
    parser.add_argument("base_dir",    type=str, help="202604 디렉터리 경로")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT,
                        help=f"이상값 기준 (기본값: {THRESHOLD_DEFAULT})")
    parser.add_argument("--varname",   type=str,   default=VARNAME_DEFAULT,
                        help=f"변수명 (기본값: {VARNAME_DEFAULT})")
    parser.add_argument("--start",     type=int,   default=1,
                        help="시작 일(day), 기본값: 1")
    parser.add_argument("--end",       type=int,   default=30,
                        help="종료 일(day), 기본값: 30")
    parser.add_argument("--hours",     type=int,   nargs="+", default=ANAL_HOURS,
                        choices=[0, 6, 12, 18], metavar="{0,6,12,18}",
                        help="확인할 분석시간(UTC), 기본값: 0 6 12 18")
    args = parser.parse_args()

    base_dir  = Path(args.base_dir)
    threshold = args.threshold
    varname   = args.varname
    start_day = args.start
    end_day   = args.end
    anal_hours = sorted(args.hours)

    if not base_dir.exists():
        print(f"[ERROR] 디렉터리를 찾을 수 없음: {base_dir}")
        sys.exit(1)
    if not (1 <= start_day <= end_day <= 30):
        print(f"[ERROR] 날짜 범위 오류: --start {start_day} --end {end_day} (1~30 범위)")
        sys.exit(1)

    print("=" * 70)
    print(f"  SWDDIF2 이상값 검사  |  기준: >= {threshold}")
    print(f"  기간: 2026-04-{start_day:02d} ~ 2026-04-{end_day:02d}")
    print(f"  분석시간: {anal_hours} UTC  |  예측시간: +6h")
    print(f"  디렉터리: {base_dir}")
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
                                  + (f"  후보변수: {r.get('candidates')}" if "candidates" in r else ""))
                else:
                    rows.append(r)

    # ---- 통계 ----
    total = len(range(1, 31)) * len(ANAL_HOURS)
    print(f"\n[파일 현황]  분석시간 총 {total}개 (x2 파일유형)")
    print(f"  원시파일    읽기 성공: {len(raw_rows):3d} / 없는 파일: {sum(1 for p in missing if 'rdps_pres' in p)}")
    print(f"  경량화파일  읽기 성공: {len(light_rows):3d} / 없는 파일: {sum(1 for p in missing if 'r030_v040' in p)}")

    if errors:
        print(f"\n[읽기 오류] {len(errors)}건")
        for e in errors:
            print(e)

    # ---- 이상값 테이블 ----
    print_result_table(raw_rows,   threshold, "원시파일   rdps_pres_r030_h006")
    print_result_table(light_rows, threshold, "경량화파일 r030_v040_easia_prs.2byte.ft006")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
