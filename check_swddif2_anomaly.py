"""
2026년 4월 WRF 3km 모델 출력 파일에서 SWDDIF2(산란일사량) 값이
500 이상인 날짜/시간을 검사하는 스크립트.

사용법:
    python check_swddif2_anomaly.py /path/to/wrfout/dir
    python check_swddif2_anomaly.py /path/to/wrfout/dir --threshold 500
    python check_swddif2_anomaly.py /path/to/wrfout/dir --prefix wrfout_d02_
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


THRESHOLD_DEFAULT = 500.0
FILE_PREFIX_DEFAULT = "rdps_fcst_d02_"


def build_april_filepaths(base_dir: Path, prefix: str) -> list[tuple[datetime, Path]]:
    """2026년 4월 01일 00UTC ~ 04월 30일 23UTC 파일 경로 목록 생성."""
    start = datetime(2026, 4, 1, 0)
    end   = datetime(2026, 4, 30, 23)
    dt    = timedelta(hours=1)
    files = []
    t = start
    while t <= end:
        fname = prefix + t.strftime("%Y-%m-%d_%H")
        fpath = base_dir / fname
        files.append((t, fpath))
        t += dt
    return files


def check_file(fpath: Path, threshold: float) -> dict | None:
    """
    단일 WRF 출력 파일에서 SWDDIF2를 읽어 임계값 이상 여부를 반환.
    파일이 없거나 변수가 없으면 None 반환.
    """
    if not fpath.exists():
        return None

    try:
        with Dataset(fpath, "r") as nc:
            if "SWDDIF2" not in nc.variables:
                return {"file": fpath.name, "error": "SWDDIF2 변수 없음"}

            data = nc.variables["SWDDIF2"][:]
            data = np.ma.filled(data, np.nan)

            max_val   = float(np.nanmax(data))
            mean_val  = float(np.nanmean(data))
            n_above   = int(np.sum(data >= threshold))

            return {
                "file"   : fpath.name,
                "max"    : max_val,
                "mean"   : mean_val,
                "n_above": n_above,
                "shape"  : data.shape,
            }
    except Exception as e:
        return {"file": fpath.name, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="SWDDIF2 이상값 검사 (2026년 4월)")
    parser.add_argument("base_dir",  type=str, help="WRF 출력 파일 디렉터리")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT,
                        help=f"이상값 기준 (기본값: {THRESHOLD_DEFAULT})")
    parser.add_argument("--prefix", type=str, default=FILE_PREFIX_DEFAULT,
                        help=f"파일명 접두사 (기본값: {FILE_PREFIX_DEFAULT})")
    args = parser.parse_args()

    base_dir  = Path(args.base_dir)
    threshold = args.threshold
    prefix    = args.prefix

    if not base_dir.exists():
        print(f"[ERROR] 디렉터리를 찾을 수 없음: {base_dir}")
        sys.exit(1)

    file_list = build_april_filepaths(base_dir, prefix)

    print("=" * 70)
    print(f"  SWDDIF2 이상값 검사  |  기준: >= {threshold}  |  기간: 2026년 4월")
    print(f"  디렉터리: {base_dir}")
    print("=" * 70)

    found_files   = 0
    missing_files = 0
    anomaly_times = []
    errors        = []

    for valid_time, fpath in file_list:
        result = check_file(fpath, threshold)

        if result is None:
            missing_files += 1
            continue

        found_files += 1

        if "error" in result:
            errors.append(f"  {result['file']}: {result['error']}")
            continue

        if result["n_above"] > 0:
            anomaly_times.append((valid_time, result))

    # ---- 결과 출력 ----
    print(f"\n[파일 현황]")
    print(f"  발견된 파일: {found_files}개  /  없는 파일: {missing_files}개")

    if errors:
        print(f"\n[읽기 오류] {len(errors)}건")
        for e in errors:
            print(e)

    if not anomaly_times:
        print(f"\n[결과] {threshold} 이상 값이 발견된 날짜/시간 없음.")
    else:
        print(f"\n[결과] {threshold} 이상 값이 발견된 시간: {len(anomaly_times)}건\n")
        print(f"  {'날짜/시간(UTC)':<22} {'최대값':>10} {'평균값':>10} {'>={:.0f} 격자수':>14}  파일명".format(threshold))
        print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*14}  {'-'*30}")

        prev_date = None
        for valid_time, r in sorted(anomaly_times):
            cur_date = valid_time.date()
            if cur_date != prev_date and prev_date is not None:
                print()
            prev_date = cur_date
            print(f"  {valid_time.strftime('%Y-%m-%d %H UTC'):<22} "
                  f"{r['max']:>10.2f} {r['mean']:>10.4f} {r['n_above']:>14}  {r['file']}")

        # 날짜별 요약
        from collections import defaultdict
        daily = defaultdict(lambda: {"max": 0.0, "n_above": 0, "hours": []})
        for valid_time, r in anomaly_times:
            d = valid_time.strftime("%Y-%m-%d")
            daily[d]["max"]     = max(daily[d]["max"], r["max"])
            daily[d]["n_above"] += r["n_above"]
            daily[d]["hours"].append(valid_time.hour)

        print(f"\n[날짜별 요약]")
        print(f"  {'날짜':<14} {'일최대값':>10} {'누적 이상격자':>14}  이상 발생 시간(UTC)")
        print(f"  {'-'*14} {'-'*10} {'-'*14}  {'-'*30}")
        for d in sorted(daily):
            hours_str = ", ".join(f"{h:02d}h" for h in sorted(daily[d]["hours"]))
            print(f"  {d:<14} {daily[d]['max']:>10.2f} {daily[d]['n_above']:>14}  {hours_str}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
