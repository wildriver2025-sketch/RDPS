
from netCDF4 import Dataset
import numpy as np
import os
from typing import Optional, Dict, Tuple

# Attributes to exclude when copying from source
EXCLUDE_ATTRIBUTES = ['FieldType', 'MemoryOrder', 'stagger']

# =============================================================================
# FIXED PACKING PARAMETERS - 파일 크기 일관성을 위한 고정 스케일/오프셋
# =============================================================================
# 각 변수별로 물리적으로 의미 있는 범위를 기반으로 고정값 설정
# scale_factor: 정밀도 (작을수록 정밀)
# add_offset: 데이터 중심값 (물리적 범위의 중간값)

FIXED_PACKING_PARAMS = {
    # 온도 관련 (K) - 범위: 180~340K
    'T': {'scale': 0.01, 'offset': 260.0},
    'T2': {'scale': 0.01, 'offset': 280.0},
    'TSK': {'scale': 0.01, 'offset': 280.0},
    'TSLB': {'scale': 0.01, 'offset': 280.0},
    'SST': {'scale': 0.01, 'offset': 290.0},

    # 상대습도 (%) - 범위: 0~100%
    'RH': {'scale': 0.01, 'offset': 50.0},
    'RH2': {'scale': 0.01, 'offset': 50.0},

    # 지위고도 (m) - 범위: -500 ~ 20000m
    'GPH': {'scale': 1.0, 'offset': 5000.0},
    'HGT': {'scale': 0.1, 'offset': 500.0},

    # 바람 (m/s) - 범위: -100 ~ 100 m/s
    'U': {'scale': 0.01, 'offset': 0.0},
    'V': {'scale': 0.01, 'offset': 0.0},
    'W': {'scale': 0.0001, 'offset': 0.0},
    'U10': {'scale': 0.01, 'offset': 0.0},
    'V10': {'scale': 0.01, 'offset': 0.0},
    'U80': {'scale': 0.01, 'offset': 0.0},
    'V80': {'scale': 0.01, 'offset': 0.0},
    'U140': {'scale': 0.01, 'offset': 0.0},
    'V140': {'scale': 0.01, 'offset': 0.0},
    'U220': {'scale': 0.01, 'offset': 0.0},
    'V220': {'scale': 0.01, 'offset': 0.0},
    'GUST': {'scale': 0.01, 'offset': 15.0},

    # 기압 (Pa) - 범위: 50000 ~ 110000 Pa
    'PSFC': {'scale': 1.0, 'offset': 100000.0},
    'MSLP': {'scale': 1.0, 'offset': 101325.0},

    # 강수량 (mm) - 범위: 0 ~ 500mm
    'RAIN': {'scale': 0.01, 'offset': 50.0},
    'RAINNC': {'scale': 0.01, 'offset': 50.0},
    'RAINC': {'scale': 0.01, 'offset': 10.0},
    'SNOW': {'scale': 0.01, 'offset': 10.0},
    'GRAUPEL': {'scale': 0.01, 'offset': 5.0},
    'TOTAL_RAIN': {'scale': 0.01, 'offset': 50.0},

    # 혼합비 (kg/kg) - 범위: 0 ~ 0.05 kg/kg
    'QVAPOR': {'scale': 1e-7, 'offset': 0.01},
    'QCLOUD': {'scale': 1e-7, 'offset': 0.0001},
    'QRAIN': {'scale': 1e-7, 'offset': 0.0001},
    'QICE': {'scale': 1e-7, 'offset': 0.0001},
    'QSNOW': {'scale': 1e-7, 'offset': 0.0001},
    'QGRAUP': {'scale': 1e-7, 'offset': 0.0001},

    # 수농도 (#/kg)
    'QNCLOUD': {'scale': 0.001, 'offset': 1e8},
    'QNRAIN': {'scale': 0.001, 'offset': 1e6},
    'QNICE': {'scale': 0.001, 'offset': 1e6},
    'QNSNOW': {'scale': 0.001, 'offset': 1e6},

    # 복사 플럭스 (W/m2) - 범위: 0 ~ 1400 W/m2
    'SWDDIR2': {'scale': 0.1, 'offset': 400.0},
    'SWDDIF2': {'scale': 0.1, 'offset': 200.0},
    'SWDDNI2': {'scale': 0.1, 'offset': 500.0},
    'OLR': {'scale': 0.1, 'offset': 250.0},

    # CAPE/CIN (J/kg)
    'MCAPE': {'scale': 1.0, 'offset': 1000.0},
    'MCIN': {'scale': 0.1, 'offset': -50.0},
    'LCL': {'scale': 1.0, 'offset': 1000.0},
    'LFC': {'scale': 1.0, 'offset': 2000.0},

    # 경계층 높이 (m)
    'PBLH': {'scale': 1.0, 'offset': 1000.0},

    # 시정 (m) - 범위: 0 ~ 50000m
    'VIS': {'scale': 1.0, 'offset': 20000.0},
    'VISB': {'scale': 1.0, 'offset': 20000.0},

    # 토양 수분 (m3/m3)
    'SMOIS': {'scale': 0.0001, 'offset': 0.3},

    # 구름량 (fraction)
    'CLDFRA': {'scale': 0.0001, 'offset': 0.5},
    'CLDFRAC2D': {'scale': 0.0001, 'offset': 0.5},
    'LOW_CLD': {'scale': 0.0001, 'offset': 0.5},
    'MID_CLD': {'scale': 0.0001, 'offset': 0.5},
    'HIGH_CLD': {'scale': 0.0001, 'offset': 0.5},
    'TOTAL_CLD': {'scale': 0.0001, 'offset': 0.5},

    # 오메가 (Pa/s)
    'OMEGA': {'scale': 0.001, 'offset': 0.0},

    # 와도
    'AVO': {'scale': 1e-7, 'offset': 0.0},
    'PVO': {'scale': 1e-9, 'offset': 0.0},

    # 레이더 반사도 (dBZ)
    'DBZ': {'scale': 0.1, 'offset': 20.0},

    # 상당온위 (K)
    'ETH': {'scale': 0.01, 'offset': 320.0},
}

# 기본값 (목록에 없는 변수용)
DEFAULT_SCALE = 0.01
DEFAULT_OFFSET = 0.0


def get_optimal_chunks(dims: tuple, dim_names: tuple) -> tuple:
    """
    차원 크기에 따른 최적 청크 크기 계산

    Parameters:
    -----------
    dims : tuple
        각 차원의 크기 (예: (1, 20, 700, 1000))
    dim_names : tuple
        차원 이름들 (예: ('Time', 'bottom_top', 'south_north', 'west_east'))

    Returns:
    --------
    tuple : 최적화된 청크 크기
    """
    chunks = []
    for dim_size, dim_name in zip(dims, dim_names):
        if 'Time' in dim_name:
            chunks.append(1)
        elif 'bottom_top' in dim_name or 'soil' in dim_name:
            # 연직 레벨은 전체를 한 청크로
            chunks.append(min(dim_size, dim_size))
        elif 'south_north' in dim_name or 'west_east' in dim_name:
            # 수평 방향은 256 또는 차원 크기
            chunks.append(min(256, dim_size))
        else:
            chunks.append(min(64, dim_size))
    return tuple(chunks)


def get_fixed_packing_params(var_name: str, dtype: str = 'i2') -> Tuple[float, float]:
    """
    변수별 고정 packing 파라미터 반환

    Parameters:
    -----------
    var_name : str
        변수 이름 (대문자)
    dtype : str
        패킹 데이터 타입

    Returns:
    --------
    tuple : (scale_factor, add_offset)
    """
    var_upper = var_name.upper()

    if var_upper in FIXED_PACKING_PARAMS:
        params = FIXED_PACKING_PARAMS[var_upper]
        return params['scale'], params['offset']
    else:
        # 기본값 사용
        return DEFAULT_SCALE, DEFAULT_OFFSET


def calculate_packing_params_safe(
    data: np.ndarray,
    var_name: str,
    dtype: str = 'i2',
    use_fixed: bool = True
) -> Tuple[float, float]:
    """
    안전한 packing 파라미터 계산 (고정값 우선, 필요시 동적 계산)

    Parameters:
    -----------
    data : np.ndarray
        패킹할 데이터
    var_name : str
        변수 이름
    dtype : str
        패킹 데이터 타입
    use_fixed : bool
        고정 파라미터 사용 여부 (True 권장)

    Returns:
    --------
    tuple : (scale_factor, add_offset)
    """
    # dtype별 정수 범위
    dtype_ranges = {
        'i1': (-127, 127),
        'i2': (-32767, 32767),
        'i4': (-2147483647, 2147483647),
    }
    n_min, n_max = dtype_ranges.get(dtype, (-32767, 32767))

    if use_fixed:
        scale, offset = get_fixed_packing_params(var_name, dtype)

        # 데이터가 범위 내에 들어가는지 검증
        if np.ma.isMaskedArray(data):
            data_min = np.ma.min(data)
            data_max = np.ma.max(data)
        else:
            data_min = np.nanmin(data)
            data_max = np.nanmax(data)

        if np.isfinite(data_min) and np.isfinite(data_max):
            packed_min = (data_min - offset) / scale
            packed_max = (data_max - offset) / scale

            # 범위 초과시 scale 조정 (offset은 유지)
            if packed_min < n_min or packed_max > n_max:
                data_range = data_max - data_min
                if data_range > 0:
                    # offset은 고정, scale만 조정
                    required_scale = max(
                        abs(data_max - offset) / n_max,
                        abs(data_min - offset) / abs(n_min)
                    )
                    scale = max(scale, required_scale * 1.01)  # 1% 여유

        return scale, offset
    else:
        # 기존 동적 계산 방식 (권장하지 않음)
        return _calculate_dynamic_packing(data, dtype)


def _calculate_dynamic_packing(data: np.ndarray, dtype: str = 'i2') -> Tuple[float, float]:
    """기존 동적 packing 계산 (호환성용)"""
    dtype_ranges = {
        'i1': (-127, 127),
        'i2': (-32767, 32767),
        'i4': (-2147483647, 2147483647),
    }
    n_min, n_max = dtype_ranges.get(dtype, (-32767, 32767))
    n_range = n_max - n_min

    if np.ma.isMaskedArray(data):
        data_min = np.ma.min(data)
        data_max = np.ma.max(data)
    else:
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)

    data_range = data_max - data_min

    if data_range == 0 or not np.isfinite(data_range):
        return 1.0, (data_min if np.isfinite(data_min) else 0.0)

    scale = data_range / (n_range - 2)
    offset = data_min - scale * (n_min + 1)

    return float(scale), float(offset)


def write_variable_with_packing(
    nc_file: Dataset,
    varname: str,
    data: np.ndarray,
    dimensions: tuple,
    attrs: dict,
    comp_opts: dict,
    packing: bool = True,
    packing_dtype: str = 'i2',
    fill_value: int = -32767,
    use_fixed_packing: bool = True
):
    """
    변수를 NetCDF 파일에 기록 (packing 옵션 포함)

    Parameters:
    -----------
    nc_file : Dataset
        NetCDF 파일 객체
    varname : str
        변수 이름
    data : np.ndarray
        기록할 데이터
    dimensions : tuple
        차원 튜플
    attrs : dict
        속성 딕셔너리
    comp_opts : dict
        압축 옵션
    packing : bool
        packing 사용 여부
    packing_dtype : str
        packing 데이터 타입
    fill_value : int
        결측값
    use_fixed_packing : bool
        고정 packing 파라미터 사용 여부
    """
    # 차원 크기 계산
    dim_sizes = tuple(nc_file.dimensions[d].size if nc_file.dimensions[d].size else 1
                      for d in dimensions)

    # 최적 청크 크기 계산
    chunks = get_optimal_chunks(dim_sizes, dimensions)

    # 압축 옵션에 청크 추가
    write_opts = comp_opts.copy()
    write_opts['chunksizes'] = chunks

    if packing:
        scale, offset = calculate_packing_params_safe(
            data, varname, packing_dtype, use_fixed=use_fixed_packing
        )

        var = nc_file.createVariable(
            varname, packing_dtype, dimensions,
            fill_value=fill_value, **write_opts
        )
        var.scale_factor = scale
        var.add_offset = offset
    else:
        var = nc_file.createVariable(
            varname, data.dtype, dimensions, **write_opts
        )

    # 속성 설정 (packing 관련 및 제외 속성 필터링)
    exclude_keys = ["_FillValue", "missing_value", "scale_factor", "add_offset"] + EXCLUDE_ATTRIBUTES
    var.setncatts({k: v for k, v in attrs.items() if k not in exclude_keys})

    # 데이터 기록
    var[:] = data


def create_nout(fin, ncout, fhr, atim, vtim, var_soil, var_2d, var_engy,
                var_post, plev, plev_3d, plev_q, plev_qn,
                compression='deflate',
                deflate_level=4,
                shuffle=True,
                packing=True,
                packing_dtype='i2',
                use_fixed_packing=True,
                single_thread_write=True):
    """
    WRF 출력을 NetCDF로 저장 (최적화 버전)

    Parameters:
    -----------
    fin : Dataset
        입력 WRF 파일
    ncout : str
        출력 파일 경로 (확장자 제외)
    fhr : int/str
        예보 시간
    atim : str
        분석 시간
    vtim : str
        유효 시간
    var_soil, var_2d, var_engy, var_post : dict
        각종 변수 딕셔너리
    plev : list
        기압면 리스트
    plev_3d, plev_q, plev_qn : dict
        3D 변수 딕셔너리
    compression : str
        압축 방식 ('deflate' or None)
    deflate_level : int
        압축 레벨 (1-9)
    shuffle : bool
        shuffle 필터 사용
    packing : bool
        packing 사용
    packing_dtype : str
        packing 데이터 타입
    use_fixed_packing : bool
        고정 packing 파라미터 사용 (True 권장 - 파일 크기 일관성)
    single_thread_write : bool
        파일 쓰기 시 단일 스레드 사용 (True 권장 - HDF5 thread safety)

    Note:
    -----
    HDF5/netCDF4 라이브러리는 기본적으로 thread-safe하지 않습니다.
    OpenMP 환경(OMP_NUM_THREADS > 1)에서 파일 쓰기 시 데이터 손상이나
    파일 크기 불규칙 문제가 발생할 수 있습니다.
    single_thread_write=True로 설정하면 파일 쓰기 동안만 단일 스레드로 전환합니다.
    """

    # ==========================================================================
    # OMP 스레드 관리: 파일 쓰기 전 단일 스레드로 전환
    # ==========================================================================
    original_omp = None
    if single_thread_write:
        original_omp = os.environ.get('OMP_NUM_THREADS', None)
        os.environ['OMP_NUM_THREADS'] = '1'
        print(f"  [OMP] Thread count set to 1 for safe file I/O (was: {original_omp})")

    try:
        # Time 차원 추가
        for key in var_soil:
            var_soil[key] = var_soil[key].squeeze().expand_dims("Time")
        for key in var_2d:
            var_2d[key] = var_2d[key].squeeze().expand_dims("Time")
        for key in var_engy:
            var_engy[key] = var_engy[key].squeeze().expand_dims("Time")
        for key in var_post:
            var_post[key] = var_post[key].squeeze().expand_dims("Time")
        for key in plev_3d:
            plev_3d[key] = plev_3d[key].squeeze().expand_dims("Time")
        for key in plev_q:
            plev_q[key] = plev_q[key].squeeze().expand_dims("Time")
        for key in plev_qn:
            plev_qn[key] = plev_qn[key].squeeze().expand_dims("Time")

        # 압축 옵션 (chunksizes는 변수별로 설정)
        comp_opts = {}
        if compression in ['deflate', 'zlib']:
            comp_opts = {
                'zlib': True,
                'complevel': deflate_level,
                'shuffle': shuffle
            }

        # fill_value
        fill_values = {'i1': -127, 'i2': -32767, 'i4': -2147483647}
        fill_value = fill_values.get(packing_dtype, -32767)

        # 출력 파일 경로
        out_unis = ncout + '_unis_h' + ("%03d" % int(fhr)) + '.' + atim + '.nc'
        out_pres = ncout + '_pres_h' + ("%03d" % int(fhr)) + '.' + atim + '.nc'

        # 그리드 크기 (청크 계산용)
        nx = int(fin.getncattr("WEST-EAST_GRID_DIMENSION")) - 1
        ny = int(fin.getncattr("SOUTH-NORTH_GRID_DIMENSION")) - 1

        # =====================================================================
        # UNIS 파일 (단일면 변수) 먼저 완전히 기록 후 닫기
        # =====================================================================
        print(f"  Writing UNIS file: {out_unis}")
        uout = Dataset(out_unis, "w", format='NETCDF4')

        # 글로벌 속성 복사
        _copy_global_attrs(fin, uout, plev, ncout)

        # 차원 생성
        uout.createDimension("Time", None)
        uout.createDimension("DateStrLen", 19)
        uout.createDimension("west_east", nx)
        uout.createDimension("south_north", ny)
        if len(plev) != 0:
            uout.createDimension("bottom_top", len(plev))

        # 토양층 차원
        sf_physics = fin.getncattr("SF_SURFACE_PHYSICS")
        if sf_physics == 1:
            uout.createDimension("soil_layers_stag", 5)
        elif sf_physics == 2:
            uout.createDimension("soil_layers_stag", 4)

        # 기본 변수 (Times, XLAT, XLONG 등)
        basic_unis = ['Times', 'XLAT', 'XLONG', 'XTIME', 'LANDMASK', 'ZS', 'DZS', 'HGT']
        _write_basic_vars(fin, uout, basic_unis, comp_opts)

        # 2D 변수
        dims_2d = (u'Time', u'south_north', u'west_east')
        for key, data in var_2d.items():
            write_variable_with_packing(
                uout, key.upper(), data.values, dims_2d, dict(data.attrs),
                comp_opts, packing, packing_dtype, fill_value, use_fixed_packing
            )

        # Energy 변수
        for key, data in var_engy.items():
            write_variable_with_packing(
                uout, key.upper(), data.values, dims_2d, dict(data.attrs),
                comp_opts, packing, packing_dtype, fill_value, use_fixed_packing
            )

        # Post 변수
        for key, data in var_post.items():
            write_variable_with_packing(
                uout, key.upper(), data.values, dims_2d, dict(data.attrs),
                comp_opts, packing, packing_dtype, fill_value, use_fixed_packing
            )

        # Soil 변수
        dims_soil = (u'Time', u'soil_layers_stag', u'south_north', u'west_east')
        for key, data in var_soil.items():
            write_variable_with_packing(
                uout, key.upper(), data.values, dims_soil, dict(data.attrs),
                comp_opts, packing, packing_dtype, fill_value, use_fixed_packing
            )

        uout.close()
        print(f"  UNIS file completed")

        # =====================================================================
        # PRES 파일 (기압면 변수) 기록
        # =====================================================================
        print(f"  Writing PRES file: {out_pres}")
        pout = Dataset(out_pres, "w", format='NETCDF4')

        # 글로벌 속성 복사
        _copy_global_attrs(fin, pout, plev, ncout)

        # 차원 생성
        pout.createDimension("Time", None)
        pout.createDimension("DateStrLen", 19)
        pout.createDimension("west_east", nx)
        pout.createDimension("south_north", ny)
        if len(plev) != 0:
            pout.createDimension("bottom_top", len(plev))

        # 기본 변수
        basic_pres = ['Times', 'XLAT', 'XLONG', 'XTIME', 'LANDMASK', 'HGT']
        _write_basic_vars(fin, pout, basic_pres, comp_opts)

        # PLEV 변수
        if len(plev) != 0:
            plev_var = pout.createVariable("PLEV", 'f', (u'bottom_top',), **comp_opts)
            plev_var.setncatts({"description": "Pressure Levels", "units": "hPa"})
            plev_var[:] = plev[:]

        # 3D 변수
        dims_3d = (u'Time', u'bottom_top', u'south_north', u'west_east')
        for key, data in plev_3d.items():
            write_variable_with_packing(
                pout, key.upper(), data.values, dims_3d, dict(data.attrs),
                comp_opts, packing, packing_dtype, fill_value, use_fixed_packing
            )

        # Q 변수 (혼합비)
        for key, data in plev_q.items():
            write_variable_with_packing(
                pout, key.upper(), data.values, dims_3d, dict(data.attrs),
                comp_opts, packing, packing_dtype, fill_value, use_fixed_packing
            )

        # QN 변수 (수농도)
        for key, data in plev_qn.items():
            write_variable_with_packing(
                pout, key.upper(), data.values, dims_3d, dict(data.attrs),
                comp_opts, packing, packing_dtype, fill_value, use_fixed_packing
            )

        pout.close()
        print(f"  PRES file completed")

    finally:
        # =====================================================================
        # OMP 스레드 복구: 원래 설정으로 되돌리기
        # =====================================================================
        if single_thread_write:
            if original_omp is not None:
                os.environ['OMP_NUM_THREADS'] = original_omp
                print(f"  [OMP] Thread count restored to {original_omp}")
            elif 'OMP_NUM_THREADS' in os.environ:
                del os.environ['OMP_NUM_THREADS']
                print(f"  [OMP] Thread count setting removed")


def _copy_global_attrs(fin, fout, plev, ncout):
    """글로벌 속성 복사"""
    for ganame in fin.ncattrs():
        if ganame == "TITLE":
            prefix = ncout.split("/")[-1][:4]
            if prefix == 'r030':
                fout.setncattr(ganame, "OUTPUT FROM KIM-Regional Model")
            elif prefix == 'l010':
                fout.setncattr(ganame, "OUTPUT FROM KIM-Local Model")
            else:
                fout.setncattr(ganame, fin.getncattr(ganame))
        elif ganame == "BOTTOM-TOP_GRID_DIMENSION":
            if len(plev) != 0:
                fout.setncattr(ganame, int(len(plev)))
        elif ganame in ["BOTTOM-TOP_PATCH_START_UNSTAG", "BOTTOM-TOP_PATCH_END_UNSTAG",
                        "BOTTOM-TOP_PATCH_START_STAG", "BOTTOM-TOP_PATCH_END_STAG"]:
            pass
        else:
            fout.setncattr(ganame, fin.getncattr(ganame))


def _write_basic_vars(fin, fout, var_list, comp_opts):
    """기본 변수 (Times, XLAT 등) 기록"""
    for vname, varin in fin.variables.items():
        if vname in var_list:
            # 차원 크기 계산 및 청크 설정
            dim_sizes = tuple(
                fout.dimensions[d].size if d in fout.dimensions and fout.dimensions[d].size
                else varin.shape[varin.dimensions.index(d)]
                for d in varin.dimensions if d in fout.dimensions
            )

            # 사용 가능한 차원만 필터링
            valid_dims = tuple(d for d in varin.dimensions if d in fout.dimensions)

            if len(valid_dims) > 0 and all(d in fout.dimensions for d in valid_dims):
                chunks = get_optimal_chunks(dim_sizes, valid_dims)
                opts = comp_opts.copy()
                opts['chunksizes'] = chunks
            else:
                opts = comp_opts

            var = fout.createVariable(vname, varin.datatype, varin.dimensions, **opts)

            if vname == "Times":
                var.setncatts({"description": "YYYY-MM-DD_hh:mm:ss"})
            else:
                var.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()
                              if k not in EXCLUDE_ATTRIBUTES})
            var[:] = varin[:]


# =============================================================================
# 기존 호환성을 위한 calculate_packing_params 함수 유지
# =============================================================================
def calculate_packing_params(
    data: np.ndarray,
    dtype: str = 'i2',
    scale_factor: Optional[float] = None,
    var_name: Optional[str] = None,
    precision_dict: Optional[Dict[str, float]] = None,
    use_default_precision: bool = True
) -> tuple:
    """
    기존 호환성을 위한 packing 파라미터 계산 함수

    Note: 새 코드에서는 calculate_packing_params_safe() 사용 권장
    """
    # 기존 동적 계산 방식 유지
    return _calculate_dynamic_packing(data, dtype)
