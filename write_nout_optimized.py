
from netCDF4 import Dataset
import numpy as np
import os
from typing import Optional, Dict, Tuple

# Attributes to exclude when copying from source
EXCLUDE_ATTRIBUTES = ['FieldType', 'MemoryOrder', 'stagger']

# =============================================================================
# FIXED PACKING PARAMETERS - Fixed scale/offset for consistent file sizes
# =============================================================================
# Fixed values based on physically meaningful ranges for each variable
# scale_factor: precision (smaller = more precise)
# add_offset: data center value (middle of physical range)

FIXED_PACKING_PARAMS = {
    # ==========================================================================
    # Scale calculation: scale = max_range / 32767 (with safety margin)
    # i2 range: -32767 ~ 32767, so max representable range = scale * 65534
    # ==========================================================================

    # Temperature (K) - physical range: 180~340K (160K span)
    # scale = 160 / 65534 ≈ 0.00244, use 0.005 for margin
    'T': {'scale': 0.005, 'offset': 260.0},      # covers 260 ± 163K
    'T2': {'scale': 0.005, 'offset': 280.0},     # covers 280 ± 163K
    'TSK': {'scale': 0.005, 'offset': 280.0},
    'TSLB': {'scale': 0.005, 'offset': 280.0},
    'SST': {'scale': 0.005, 'offset': 290.0},

    # Relative humidity (%) - physical range: 0~100%
    # scale = 100 / 65534 ≈ 0.00153, use 0.005 for margin
    'RH': {'scale': 0.005, 'offset': 50.0},      # covers 0~100% and beyond
    'RH2': {'scale': 0.005, 'offset': 50.0},

    # Geopotential height (m) - physical range: -500 ~ 20000m
    # scale = 20500 / 65534 ≈ 0.31, use 0.5 for margin
    'GPH': {'scale': 0.5, 'offset': 10000.0},    # covers -22K ~ +42K m
    'HGT': {'scale': 0.1, 'offset': 1000.0},     # terrain: covers -2K ~ +4K m

    # Horizontal wind (m/s) - physical range: -150 ~ +150 m/s (jet stream)
    # scale = 300 / 65534 ≈ 0.0046, use 0.005 for margin
    'U': {'scale': 0.005, 'offset': 0.0},        # covers ±163 m/s
    'V': {'scale': 0.005, 'offset': 0.0},
    'U10': {'scale': 0.005, 'offset': 0.0},
    'V10': {'scale': 0.005, 'offset': 0.0},
    'U80': {'scale': 0.005, 'offset': 0.0},
    'V80': {'scale': 0.005, 'offset': 0.0},
    'U140': {'scale': 0.005, 'offset': 0.0},
    'V140': {'scale': 0.005, 'offset': 0.0},
    'U220': {'scale': 0.005, 'offset': 0.0},
    'V220': {'scale': 0.005, 'offset': 0.0},
    'GUST': {'scale': 0.005, 'offset': 25.0},    # covers 0~50 m/s gusts

    # Vertical velocity (m/s) - physical range: -50 ~ +50 m/s (severe convection)
    # scale = 100 / 65534 ≈ 0.00153, use 0.002 for margin (precision vs range trade-off)
    'W': {'scale': 0.002, 'offset': 0.0},        # covers ±65 m/s

    # Pressure (Pa) - physical range: 30000 ~ 110000 Pa
    # scale = 80000 / 65534 ≈ 1.22, use 2.0 for margin
    'PSFC': {'scale': 2.0, 'offset': 100000.0},  # covers 35K ~ 165K Pa
    'MSLP': {'scale': 2.0, 'offset': 101325.0},

    # Precipitation (mm) - physical range: 0 ~ 1000mm (extreme events)
    # scale = 1000 / 65534 ≈ 0.015, use 0.02 for margin
    'RAIN': {'scale': 0.02, 'offset': 500.0},    # covers 0~1300 mm
    'RAINNC': {'scale': 0.02, 'offset': 500.0},
    'RAINC': {'scale': 0.02, 'offset': 200.0},
    'SNOW': {'scale': 0.02, 'offset': 200.0},
    'GRAUPEL': {'scale': 0.02, 'offset': 100.0},
    'TOTAL_RAIN': {'scale': 0.02, 'offset': 500.0},

    # Mixing ratio (kg/kg) - physical range: 0 ~ 0.05 kg/kg
    # scale = 0.05 / 65534 ≈ 7.6e-7, use 1e-6 for margin
    # Note: Using 1e-6 instead of 1e-7 to ensure full range coverage
    'QVAPOR': {'scale': 1e-6, 'offset': 0.025},  # covers 0~0.09 kg/kg
    'QCLOUD': {'scale': 1e-6, 'offset': 0.005},  # covers 0~0.07 kg/kg
    'QRAIN': {'scale': 1e-6, 'offset': 0.005},
    'QICE': {'scale': 1e-6, 'offset': 0.005},
    'QSNOW': {'scale': 1e-6, 'offset': 0.005},
    'QGRAUP': {'scale': 1e-6, 'offset': 0.005},

    # Number concentration (#/kg) - highly variable, use large range
    # scale chosen to cover typical range with margin
    'QNCLOUD': {'scale': 1e4, 'offset': 1e8},    # covers 0~6.5e8 #/kg
    'QNRAIN': {'scale': 1e2, 'offset': 1e6},     # covers 0~6.5e6 #/kg
    'QNICE': {'scale': 1e2, 'offset': 1e6},
    'QNSNOW': {'scale': 1e2, 'offset': 1e6},

    # Radiation flux (W/m2) - physical range: 0 ~ 1400 W/m2
    # scale = 1400 / 65534 ≈ 0.021, use 0.05 for margin
    'SWDDIR2': {'scale': 0.05, 'offset': 700.0},  # covers 0~1400+ W/m2
    'SWDDIF2': {'scale': 0.05, 'offset': 400.0},
    'SWDDNI2': {'scale': 0.05, 'offset': 700.0},
    'OLR': {'scale': 0.02, 'offset': 250.0},      # covers 0~500 W/m2

    # CAPE/CIN (J/kg) - physical range: 0~6000 J/kg (CAPE), -500~0 (CIN)
    # scale = 6000 / 65534 ≈ 0.092, use 0.2 for margin
    'MCAPE': {'scale': 0.2, 'offset': 3000.0},   # covers 0~6500+ J/kg
    'MCIN': {'scale': 0.02, 'offset': -250.0},   # covers -900~0 J/kg
    'LCL': {'scale': 0.5, 'offset': 5000.0},     # covers 0~20000 m
    'LFC': {'scale': 0.5, 'offset': 5000.0},

    # Boundary layer height (m) - physical range: 0 ~ 5000m
    # scale = 5000 / 65534 ≈ 0.076, use 0.2 for margin
    'PBLH': {'scale': 0.2, 'offset': 2500.0},    # covers 0~6500+ m

    # Visibility (m) - physical range: 0 ~ 100000m
    # scale = 100000 / 65534 ≈ 1.53, use 2.0 for margin
    'VIS': {'scale': 2.0, 'offset': 50000.0},    # covers 0~100K+ m
    'VISB': {'scale': 2.0, 'offset': 50000.0},

    # Soil moisture (m3/m3) - physical range: 0 ~ 0.6
    # scale = 0.6 / 65534 ≈ 9.2e-6, use 2e-5 for margin
    'SMOIS': {'scale': 2e-5, 'offset': 0.3},     # covers 0~0.95 m3/m3

    # Cloud fraction (fraction) - physical range: 0 ~ 1.0
    # scale = 1.0 / 65534 ≈ 1.5e-5, use 5e-5 for margin
    'CLDFRA': {'scale': 5e-5, 'offset': 0.5},    # covers 0~1.0+
    'CLDFRAC2D': {'scale': 5e-5, 'offset': 0.5},
    'LOW_CLD': {'scale': 5e-5, 'offset': 0.5},
    'MID_CLD': {'scale': 5e-5, 'offset': 0.5},
    'HIGH_CLD': {'scale': 5e-5, 'offset': 0.5},
    'TOTAL_CLD': {'scale': 5e-5, 'offset': 0.5},

    # Omega (Pa/s) - physical range: -10 ~ +10 Pa/s (strong convection)
    # scale = 20 / 65534 ≈ 3e-4, use 5e-4 for margin
    'OMEGA': {'scale': 5e-4, 'offset': 0.0},     # covers ±32 Pa/s

    # Vorticity - physical range varies widely
    'AVO': {'scale': 1e-6, 'offset': 0.0},       # covers ±0.03 s-1
    'PVO': {'scale': 1e-7, 'offset': 0.0},       # covers PVU range

    # Radar reflectivity (dBZ) - physical range: -30 ~ 80 dBZ
    # scale = 110 / 65534 ≈ 0.0017, use 0.005 for margin
    'DBZ': {'scale': 0.005, 'offset': 25.0},     # covers -138~188 dBZ

    # Equivalent potential temperature (K) - physical range: 280~400K
    # scale = 120 / 65534 ≈ 0.0018, use 0.005 for margin
    'ETH': {'scale': 0.005, 'offset': 340.0},    # covers 177~503 K
}

# Default values for unlisted variables
DEFAULT_SCALE = 0.01
DEFAULT_OFFSET = 0.0


def get_optimal_chunks(dims: tuple, dim_names: tuple) -> tuple:
    """
    Calculate optimal chunk sizes based on dimension sizes.

    Chunking strategy:
    - Time dimension: 1 (single timestep per chunk)
    - Vertical levels (bottom_top, soil): 1 (single level per chunk for efficient level-wise access)
    - Horizontal dimensions: full grid size (efficient for whole-field operations)

    Parameters:
    -----------
    dims : tuple
        Size of each dimension (e.g., (1, 20, 700, 1000))
    dim_names : tuple
        Dimension names (e.g., ('Time', 'bottom_top', 'south_north', 'west_east'))

    Returns:
    --------
    tuple : Optimized chunk sizes
    """
    chunks = []
    for dim_size, dim_name in zip(dims, dim_names):
        if 'Time' in dim_name:
            # Single timestep per chunk
            chunks.append(1)
        elif 'bottom_top' in dim_name or 'soil' in dim_name:
            # Single vertical level per chunk for efficient pressure-level access
            chunks.append(1)
        else:
            # Full horizontal grid size for whole-field operations
            chunks.append(dim_size)
    return tuple(chunks)


def get_fixed_packing_params(var_name: str, dtype: str = 'i2') -> Tuple[float, float]:
    """
    Get fixed packing parameters for a variable.

    Parameters:
    -----------
    var_name : str
        Variable name (uppercase)
    dtype : str
        Packing data type

    Returns:
    --------
    tuple : (scale_factor, add_offset)
    """
    var_upper = var_name.upper()

    if var_upper in FIXED_PACKING_PARAMS:
        params = FIXED_PACKING_PARAMS[var_upper]
        return params['scale'], params['offset']
    else:
        # Use default values
        return DEFAULT_SCALE, DEFAULT_OFFSET


def calculate_packing_params_safe(
    data: np.ndarray,
    var_name: str,
    dtype: str = 'i2',
    use_fixed: bool = True
) -> Tuple[float, float]:
    """
    Calculate packing parameters safely (fixed values preferred, dynamic fallback).

    Parameters:
    -----------
    data : np.ndarray
        Data to be packed
    var_name : str
        Variable name
    dtype : str
        Packing data type
    use_fixed : bool
        Whether to use fixed parameters (True recommended)

    Returns:
    --------
    tuple : (scale_factor, add_offset)
    """
    # Integer range for each dtype
    dtype_ranges = {
        'i1': (-127, 127),
        'i2': (-32767, 32767),
        'i4': (-2147483647, 2147483647),
    }
    n_min, n_max = dtype_ranges.get(dtype, (-32767, 32767))

    if use_fixed:
        scale, offset = get_fixed_packing_params(var_name, dtype)

        # Validate that data fits within range
        if np.ma.isMaskedArray(data):
            data_min = np.ma.min(data)
            data_max = np.ma.max(data)
        else:
            data_min = np.nanmin(data)
            data_max = np.nanmax(data)

        if np.isfinite(data_min) and np.isfinite(data_max):
            packed_min = (data_min - offset) / scale
            packed_max = (data_max - offset) / scale

            # Adjust scale if data exceeds range (keep offset fixed)
            if packed_min < n_min or packed_max > n_max:
                data_range = data_max - data_min
                if data_range > 0:
                    # Keep offset fixed, only adjust scale
                    required_scale = max(
                        abs(data_max - offset) / n_max,
                        abs(data_min - offset) / abs(n_min)
                    )
                    scale = max(scale, required_scale * 1.01)  # 1% margin

        return scale, offset
    else:
        # Legacy dynamic calculation (not recommended)
        return _calculate_dynamic_packing(data, dtype)


def _calculate_dynamic_packing(data: np.ndarray, dtype: str = 'i2') -> Tuple[float, float]:
    """Legacy dynamic packing calculation (for backward compatibility)."""
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
    Write a variable to NetCDF file with packing options.

    Parameters:
    -----------
    nc_file : Dataset
        NetCDF file object
    varname : str
        Variable name
    data : np.ndarray
        Data to write
    dimensions : tuple
        Dimension tuple
    attrs : dict
        Attribute dictionary
    comp_opts : dict
        Compression options
    packing : bool
        Whether to use packing
    packing_dtype : str
        Packing data type
    fill_value : int
        Missing value
    use_fixed_packing : bool
        Whether to use fixed packing parameters
    """
    # Calculate dimension sizes
    dim_sizes = tuple(nc_file.dimensions[d].size if nc_file.dimensions[d].size else 1
                      for d in dimensions)

    # Calculate optimal chunk sizes
    chunks = get_optimal_chunks(dim_sizes, dimensions)

    # Add chunks to compression options
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

    # Set attributes (filter out packing-related and excluded attributes)
    exclude_keys = ["_FillValue", "missing_value", "scale_factor", "add_offset"] + EXCLUDE_ATTRIBUTES
    var.setncatts({k: v for k, v in attrs.items() if k not in exclude_keys})

    # Write data
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
    Save WRF output to NetCDF (optimized version).

    Parameters:
    -----------
    fin : Dataset
        Input WRF file
    ncout : str
        Output file path (without extension)
    fhr : int/str
        Forecast hour
    atim : str
        Analysis time
    vtim : str
        Valid time
    var_soil, var_2d, var_engy, var_post : dict
        Variable dictionaries
    plev : list
        Pressure level list
    plev_3d, plev_q, plev_qn : dict
        3D variable dictionaries
    compression : str
        Compression method ('deflate' or None)
    deflate_level : int
        Compression level (1-9)
    shuffle : bool
        Use shuffle filter
    packing : bool
        Use packing
    packing_dtype : str
        Packing data type
    use_fixed_packing : bool
        Use fixed packing parameters (True recommended for consistent file sizes)
    single_thread_write : bool
        Use single thread for file writing (True recommended for HDF5 thread safety)

    Note:
    -----
    HDF5/netCDF4 library is not thread-safe by default.
    In OpenMP environments (OMP_NUM_THREADS > 1), file writing may cause
    data corruption or inconsistent file sizes.
    Setting single_thread_write=True switches to single thread during file I/O.
    """

    # ==========================================================================
    # OMP thread management: switch to single thread before file writing
    # ==========================================================================
    original_omp = None
    if single_thread_write:
        original_omp = os.environ.get('OMP_NUM_THREADS', None)
        os.environ['OMP_NUM_THREADS'] = '1'
        print(f"  [OMP] Thread count set to 1 for safe file I/O (was: {original_omp})")

    try:
        # Add Time dimension to all variables
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

        # Compression options (chunksizes set per variable)
        comp_opts = {}
        if compression in ['deflate', 'zlib']:
            comp_opts = {
                'zlib': True,
                'complevel': deflate_level,
                'shuffle': shuffle
            }

        # Fill value for packing dtype
        fill_values = {'i1': -127, 'i2': -32767, 'i4': -2147483647}
        fill_value = fill_values.get(packing_dtype, -32767)

        # Output file paths
        out_unis = ncout + '_unis_h' + ("%03d" % int(fhr)) + '.' + atim + '.nc'
        out_pres = ncout + '_pres_h' + ("%03d" % int(fhr)) + '.' + atim + '.nc'

        # Grid size for chunk calculation
        nx = int(fin.getncattr("WEST-EAST_GRID_DIMENSION")) - 1
        ny = int(fin.getncattr("SOUTH-NORTH_GRID_DIMENSION")) - 1

        # =====================================================================
        # UNIS file (single-level variables) - write completely then close
        # =====================================================================
        print(f"  Writing UNIS file: {out_unis}")
        uout = Dataset(out_unis, "w", format='NETCDF4')

        # Copy global attributes
        _copy_global_attrs(fin, uout, plev, ncout)

        # Create dimensions
        uout.createDimension("Time", None)
        uout.createDimension("DateStrLen", 19)
        uout.createDimension("west_east", nx)
        uout.createDimension("south_north", ny)
        if len(plev) != 0:
            uout.createDimension("bottom_top", len(plev))

        # Soil layer dimension
        sf_physics = fin.getncattr("SF_SURFACE_PHYSICS")
        if sf_physics == 1:
            uout.createDimension("soil_layers_stag", 5)
        elif sf_physics == 2:
            uout.createDimension("soil_layers_stag", 4)

        # Basic variables (Times, XLAT, XLONG, etc.)
        basic_unis = ['Times', 'XLAT', 'XLONG', 'XTIME', 'LANDMASK', 'ZS', 'DZS', 'HGT']
        _write_basic_vars(fin, uout, basic_unis, comp_opts)

        # 2D variables
        dims_2d = (u'Time', u'south_north', u'west_east')
        for key, data in var_2d.items():
            write_variable_with_packing(
                uout, key.upper(), data.values, dims_2d, dict(data.attrs),
                comp_opts, packing, packing_dtype, fill_value, use_fixed_packing
            )

        # Energy variables
        for key, data in var_engy.items():
            write_variable_with_packing(
                uout, key.upper(), data.values, dims_2d, dict(data.attrs),
                comp_opts, packing, packing_dtype, fill_value, use_fixed_packing
            )

        # Post-processed variables
        for key, data in var_post.items():
            write_variable_with_packing(
                uout, key.upper(), data.values, dims_2d, dict(data.attrs),
                comp_opts, packing, packing_dtype, fill_value, use_fixed_packing
            )

        # Soil variables
        dims_soil = (u'Time', u'soil_layers_stag', u'south_north', u'west_east')
        for key, data in var_soil.items():
            write_variable_with_packing(
                uout, key.upper(), data.values, dims_soil, dict(data.attrs),
                comp_opts, packing, packing_dtype, fill_value, use_fixed_packing
            )

        uout.close()
        print(f"  UNIS file completed")

        # =====================================================================
        # PRES file (pressure-level variables)
        # =====================================================================
        print(f"  Writing PRES file: {out_pres}")
        pout = Dataset(out_pres, "w", format='NETCDF4')

        # Copy global attributes
        _copy_global_attrs(fin, pout, plev, ncout)

        # Create dimensions
        pout.createDimension("Time", None)
        pout.createDimension("DateStrLen", 19)
        pout.createDimension("west_east", nx)
        pout.createDimension("south_north", ny)
        if len(plev) != 0:
            pout.createDimension("bottom_top", len(plev))

        # Basic variables
        basic_pres = ['Times', 'XLAT', 'XLONG', 'XTIME', 'LANDMASK', 'HGT']
        _write_basic_vars(fin, pout, basic_pres, comp_opts)

        # PLEV variable
        if len(plev) != 0:
            plev_var = pout.createVariable("PLEV", 'f', (u'bottom_top',), **comp_opts)
            plev_var.setncatts({"description": "Pressure Levels", "units": "hPa"})
            plev_var[:] = plev[:]

        # 3D variables
        dims_3d = (u'Time', u'bottom_top', u'south_north', u'west_east')
        for key, data in plev_3d.items():
            write_variable_with_packing(
                pout, key.upper(), data.values, dims_3d, dict(data.attrs),
                comp_opts, packing, packing_dtype, fill_value, use_fixed_packing
            )

        # Moisture mixing ratio variables
        for key, data in plev_q.items():
            write_variable_with_packing(
                pout, key.upper(), data.values, dims_3d, dict(data.attrs),
                comp_opts, packing, packing_dtype, fill_value, use_fixed_packing
            )

        # Number concentration variables
        for key, data in plev_qn.items():
            write_variable_with_packing(
                pout, key.upper(), data.values, dims_3d, dict(data.attrs),
                comp_opts, packing, packing_dtype, fill_value, use_fixed_packing
            )

        pout.close()
        print(f"  PRES file completed")

    finally:
        # =====================================================================
        # OMP thread restoration: restore original settings
        # =====================================================================
        if single_thread_write:
            if original_omp is not None:
                os.environ['OMP_NUM_THREADS'] = original_omp
                print(f"  [OMP] Thread count restored to {original_omp}")
            elif 'OMP_NUM_THREADS' in os.environ:
                del os.environ['OMP_NUM_THREADS']
                print(f"  [OMP] Thread count setting removed")


def _copy_global_attrs(fin, fout, plev, ncout):
    """Copy global attributes from input to output file."""
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
    """Write basic variables (Times, XLAT, etc.) to output file."""
    for vname, varin in fin.variables.items():
        if vname in var_list:
            # Calculate dimension sizes and set chunks
            dim_sizes = tuple(
                fout.dimensions[d].size if d in fout.dimensions and fout.dimensions[d].size
                else varin.shape[varin.dimensions.index(d)]
                for d in varin.dimensions if d in fout.dimensions
            )

            # Filter to valid dimensions only
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
# Legacy function for backward compatibility
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
    Legacy packing parameter calculation function for backward compatibility.

    Note: For new code, use calculate_packing_params_safe() instead.
    """
    # Use legacy dynamic calculation
    return _calculate_dynamic_packing(data, dtype)
