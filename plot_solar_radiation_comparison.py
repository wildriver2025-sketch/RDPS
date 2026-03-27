#!/usr/bin/env python3
"""
plot_solar_radiation_comparison.py

WRF wrfout 파일에서 ACSWDNB(누적일사량)로부터 1시간 평균일사량을 계산하고
SWDOWN2(순간일사량)와 시계열 비교 그래프를 생성합니다.

변수 설명:
  ACSWDNB : Accumulated Downward Shortwave Flux at Bottom (J/m²)
             누적값이므로 연속 시간 차분 후 3600초로 나누면 1시간 평균 W/m²
  SWDOWN2 : Downward Shortwave Flux at Bottom (FARMS 복사, W/m²)
             매 출력 시각의 순간값

사용법:
  python plot_solar_radiation_comparison.py wrfout_d01_YYYY-MM-DD_HH:00:00
  python plot_solar_radiation_comparison.py wrfout_d01_... --lat 37.5 --lon 127.0
  python plot_solar_radiation_comparison.py wrfout_d01_... --ij 100 120
  python plot_solar_radiation_comparison.py wrfout_d01_... --domain_avg
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from netCDF4 import Dataset, num2date
from datetime import datetime


# ──────────────────────────────────────────────────────────────────────────────
# 인자 파싱
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='WRF ACSWDNB vs SWDOWN2 시계열 비교'
    )
    parser.add_argument('wrfout', help='wrfout 파일 경로')
    parser.add_argument('--lat', type=float, default=None,
                        help='위도 지점 (도). --lon과 함께 사용')
    parser.add_argument('--lon', type=float, default=None,
                        help='경도 지점 (도). --lat과 함께 사용')
    parser.add_argument('--ij', type=int, nargs=2, metavar=('I', 'J'),
                        default=None, help='격자 인덱스 (south_north, west_east)')
    parser.add_argument('--domain_avg', action='store_true',
                        help='도메인 전체 평균으로 비교')
    parser.add_argument('--outfile', default='solar_comparison.png',
                        help='출력 이미지 파일명 (기본: solar_comparison.png)')
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 격자 인덱스 탐색 (최근접 lat/lon)
# ──────────────────────────────────────────────────────────────────────────────
def find_nearest_ij(nc, target_lat, target_lon):
    """WRF XLAT, XLONG에서 최근접 격자점 (j, i) 반환"""
    xlat = nc.variables['XLAT'][0]   # (south_north, west_east)
    xlon = nc.variables['XLONG'][0]
    dist = np.sqrt((xlat - target_lat)**2 + (xlon - target_lon)**2)
    j, i = np.unravel_index(np.argmin(dist), dist.shape)
    found_lat = xlat[j, i]
    found_lon = xlon[j, i]
    print(f"  최근접 격자점: (j={j}, i={i})  ->  lat={found_lat:.3f}, lon={found_lon:.3f}")
    return int(j), int(i)


# ──────────────────────────────────────────────────────────────────────────────
# 시각 배열 생성
# ──────────────────────────────────────────────────────────────────────────────
def get_times(nc):
    """wrfout Times 변수에서 datetime 리스트 반환"""
    times_raw = nc.variables['Times'][:]          # (ntimes, 19) bytes
    times = []
    for t in times_raw:
        tstr = b''.join(t).decode('utf-8')         # 'YYYY-MM-DD_HH:MM:SS'
        times.append(datetime.strptime(tstr, '%Y-%m-%d_%H:%M:%S'))
    return times


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print(f"\n[파일 읽기] {args.wrfout}")
    try:
        nc = Dataset(args.wrfout, 'r')
    except FileNotFoundError:
        sys.exit(f"오류: 파일을 찾을 수 없습니다 -> {args.wrfout}")

    # ── 변수 존재 확인 ─────────────────────────────────────────────────────────
    for vname in ('ACSWDNB', 'SWDOWN2'):
        if vname not in nc.variables:
            nc.close()
            sys.exit(f"오류: wrfout 파일에 '{vname}' 변수가 없습니다.")

    # ── 시각 배열 ──────────────────────────────────────────────────────────────
    times = get_times(nc)
    ntimes = len(times)
    print(f"  시간 스텝 수: {ntimes}  ({times[0]} ~ {times[-1]})")

    # ── 변수 읽기 (ntimes, south_north, west_east) ────────────────────────────
    acswdnb_all = nc.variables['ACSWDNB'][:]   # J/m²  (누적)
    swdown2_all  = nc.variables['SWDOWN2'][:]  # W/m²  (순간)
    nc.close()

    # ── 격자점 / 도메인 선택 ───────────────────────────────────────────────────
    if args.domain_avg:
        print("  모드: 도메인 전체 평균")
        acswdnb = acswdnb_all.mean(axis=(1, 2))
        swdown2  = swdown2_all.mean(axis=(1, 2))
        point_label = 'Domain Average'

    elif args.ij is not None:
        j, i = args.ij
        print(f"  모드: 격자 인덱스 (j={j}, i={i})")
        acswdnb = acswdnb_all[:, j, i]
        swdown2  = swdown2_all[:, j, i]
        point_label = f'Grid (j={j}, i={i})'

    elif args.lat is not None and args.lon is not None:
        nc2 = Dataset(args.wrfout, 'r')
        j, i = find_nearest_ij(nc2, args.lat, args.lon)
        nc2.close()
        acswdnb = acswdnb_all[:, j, i]
        swdown2  = swdown2_all[:, j, i]
        point_label = f'lat={args.lat:.2f}, lon={args.lon:.2f} (j={j}, i={i})'

    else:
        # 기본: 도메인 중앙 격자점
        nj = acswdnb_all.shape[1]
        ni = acswdnb_all.shape[2]
        j, i = nj // 2, ni // 2
        print(f"  모드: 도메인 중앙 격자점 (j={j}, i={i})")
        acswdnb = acswdnb_all[:, j, i]
        swdown2  = swdown2_all[:, j, i]
        point_label = f'Center Grid (j={j}, i={i})'

    # ── ACSWDNB → 1시간 평균 W/m² ────────────────────────────────────────────
    #
    #   WRF ACSWDNB 는 시뮬레이션 시작 이후 누적 J/m²
    #   1시간 간격 출력이므로:
    #     avg_sw[t] = (ACSWDNB[t] - ACSWDNB[t-1]) / 3600   (t >= 1)
    #     avg_sw[0] =  ACSWDNB[0] / 3600                    (초기 t=0)
    #
    #   단, t=0 은 모델 초기 시각(야간 등)에 따라 0일 수 있음
    # ─────────────────────────────────────────────────────────────────────────
    dt_sec = 3600.0   # 1시간 간격

    acswdnb_avg = np.empty(ntimes)
    acswdnb_avg[0] = acswdnb[0] / dt_sec               # 첫 시간 (누적 / 3600)
    acswdnb_avg[1:] = np.diff(acswdnb) / dt_sec        # 차분 / 3600

    # 음수 방지 (리셋 또는 수치 오차)
    acswdnb_avg = np.where(acswdnb_avg < 0, 0.0, acswdnb_avg)

    print(f"\n  ACSWDNB 최대 누적값  : {acswdnb.max():.1f} J/m²")
    print(f"  1h 평균 SW 최대      : {acswdnb_avg.max():.1f} W/m²")
    print(f"  SWDOWN2 최대 순간값  : {swdown2.max():.1f} W/m²")

    # ── 시계열 그래프 ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f'WRF Solar Radiation Comparison\n{point_label}',
        fontsize=13, fontweight='bold'
    )

    # --- 상단: 1시간 평균 vs 순간 비교 ----------------------------------------
    ax1 = axes[0]
    ax1.plot(times, acswdnb_avg, color='orangered', linewidth=2,
             marker='o', markersize=4, label='ACSWDNB-derived 1h mean (W/m²)')
    ax1.plot(times, swdown2, color='steelblue', linewidth=2,
             marker='s', markersize=4, linestyle='--',
             label='SWDOWN2 instantaneous (W/m²)')
    ax1.set_ylabel('Shortwave Radiation (W/m²)', fontsize=11)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.4)
    ax1.set_ylim(bottom=0)
    ax1.tick_params(axis='y', labelsize=9)

    # 차이값 표시
    diff = acswdnb_avg - swdown2
    ax1.fill_between(times, acswdnb_avg, swdown2,
                     alpha=0.15, color='gray', label='Difference')

    # --- 하단: 차이 (ACSWDNB 평균 - SWDOWN2) ----------------------------------
    ax2 = axes[1]
    ax2.bar(times, diff, color=np.where(diff >= 0, 'tomato', 'cornflowerblue'),
            width=0.03, alpha=0.8, label='ACSWDNB_avg - SWDOWN2')
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax2.set_ylabel('Difference (W/m²)', fontsize=11)
    ax2.set_xlabel('Forecast Time (UTC)', fontsize=11)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.4)
    ax2.tick_params(axis='both', labelsize=9)

    # x축 포맷
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%HUTC'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center')

    # 통계 텍스트
    bias  = diff.mean()
    rmse  = np.sqrt((diff**2).mean())
    corr  = np.corrcoef(acswdnb_avg, swdown2)[0, 1]
    stats_txt = f'Bias={bias:.1f} W/m²  |  RMSE={rmse:.1f} W/m²  |  Corr={corr:.3f}'
    ax1.set_title(stats_txt, fontsize=9, loc='left', color='dimgray')

    plt.tight_layout()
    plt.savefig(args.outfile, dpi=150, bbox_inches='tight')
    print(f"\n  그래프 저장 완료: {args.outfile}")
    plt.close()

    # ── 텍스트 요약 출력 ──────────────────────────────────────────────────────
    print("\n  [시계열 수치 요약]")
    print(f"  {'Time':>20s}  {'ACSWDNB_1h_avg':>16s}  {'SWDOWN2':>10s}  {'Diff':>8s}")
    print("  " + "-" * 60)
    for t, av, sw in zip(times, acswdnb_avg, swdown2):
        print(f"  {str(t):>20s}  {av:>16.2f}  {sw:>10.2f}  {av-sw:>8.2f}")


if __name__ == '__main__':
    main()
