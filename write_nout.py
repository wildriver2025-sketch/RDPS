
from netCDF4 import Dataset
import numpy as np
from typing import Optional, Dict

# Attributes to exclude when copying from source
EXCLUDE_ATTRIBUTES = ['FieldType', 'MemoryOrder', 'stagger']


def create_nout(fin, ncout, fhr, atim, vtim, var_soil, var_2d, var_engy,
                var_post, plev, plev_3d, plev_q, plev_qn,
                compression='deflate',
                deflate_level=4,
                shuffle=True,
                packing=True,
                packing_dtype='i2'):

    # Add time dimension
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

    # Compression options
    comp_opts = {}
    if compression in ['deflate', 'zlib']:
        comp_opts = {
            'zlib': True,
            'complevel': deflate_level,
            'shuffle': shuffle
        }

    # Get _FillValue for packing dtype
    fill_values = {
        'i1': -127,
        'i2': -32767,
        'i4': -2147483647,
    }
    fill_value = fill_values.get(packing_dtype, -32767)

    # OPEN OUTFILE
    out_unis = ncout + '_unis_h' + ("%03d" % int(fhr)) + '.' + atim + '.nc'
    out_pres = ncout + '_pres_h' + ("%03d" % int(fhr)) + '.' + atim + '.nc'

    uout = Dataset(out_unis, "w", format='NETCDF4')
    pout = Dataset(out_pres, "w", format='NETCDF4')

    # GET NETCDF HEADER from WRFOUT
    #required_attrs = [ "TITLE", "START_DATE", "SIMULATION_START_DATE",
    #                   "WEST-EAST_GRID_DIMENSION", "SOUTH-NORTH_GRID_DIMENSION", "BOTTOM-TOP_GRID_DIMENSION",
    #                   "DX","DY"]
    #for ganame in required_attrs:
    for ganame in fin.ncattrs():
        if len(plev) != 0:
            if ganame == "TITLE":
                if ncout.split("/")[-1][:4] == 'r030':
                    uout.setncattr(ganame, "OUTPUT FROM KIM-Regional Model")
                    pout.setncattr(ganame, "OUTPUT FROM KIM-Regional Model")
                if ncout.split("/")[-1][:4] == 'l010':
                    uout.setncattr(ganame, "OUTPUT FROM KIM-Local Model")
                    pout.setncattr(ganame, "OUTPUT FROM KIM-Local Model")
            elif ganame == "BOTTOM-TOP_GRID_DIMENSION":
                uout.setncattr(ganame, int(len(plev)))
                pout.setncattr(ganame, int(len(plev)))
            elif ganame == "BOTTOM-TOP_PATCH_START_UNSTAG":
                pass
            elif ganame == "BOTTOM-TOP_PATCH_END_UNSTAG":
                pass
            elif ganame == "BOTTOM-TOP_PATCH_START_STAG":
                pass
            elif ganame == "BOTTOM-TOP_PATCH_END_STAG":
                pass
            else:
              uout.setncattr(ganame, fin.getncattr(ganame))
              pout.setncattr(ganame, fin.getncattr(ganame))
              continue
        else:
            if ganame == "TITLE":
                uout.setncattr(ganame, fin.getncattr(ganame) + " - ON SINGLE LEVEL")
                pout.setncattr(ganame, fin.getncattr(ganame) + " - ON SINGLE LEVEL")
            elif ganame == "BOTTOM-TOP_GRID_DIMENSION":
                pass
            elif ganame == "BOTTOM-TOP_PATCH_START_UNSTAG":
                pass
            elif ganame == "BOTTOM-TOP_PATCH_END_UNSTAG":
                pass
            elif ganame == "BOTTOM-TOP_PATCH_START_STAG":
                pass
            elif ganame == "BOTTOM-TOP_PATCH_END_STAG":
                pass
            else:
                uout.setncattr(ganame, fin.getncattr(ganame))
                pout.setncattr(ganame, fin.getncattr(ganame))

    # CREATE DIMENSION
    uout.createDimension("Time", None)
    uout.createDimension("DateStrLen", 19)
    uout.createDimension("west_east", int(fin.getncattr("WEST-EAST_GRID_DIMENSION")) - 1)
    uout.createDimension("south_north", int(fin.getncattr("SOUTH-NORTH_GRID_DIMENSION")) - 1)
    pout.createDimension("Time", None)
    pout.createDimension("DateStrLen", 19)
    pout.createDimension("west_east", int(fin.getncattr("WEST-EAST_GRID_DIMENSION")) - 1)
    pout.createDimension("south_north", int(fin.getncattr("SOUTH-NORTH_GRID_DIMENSION")) - 1)

    if len(plev) != 0:
        uout.createDimension("bottom_top", int(len(plev)))
        pout.createDimension("bottom_top", int(len(plev)))
    
    # Soil layer dimension for surface physics scheme
    if fin.getncattr("SF_SURFACE_PHYSICS") == 1:
        uout.createDimension("soil_layers_stag", 5)
        pout.createDimension("soil_layers_stag", 5)
    elif fin.getncattr("SF_SURFACE_PHYSICS") == 2:
        uout.createDimension("soil_layers_stag", 4)
        pout.createDimension("soil_layers_stag", 4)
    else:
        print("[CHECK] SURFACE LAYER PHYSICS for soil_layer_stag")

    # GET VARIABLE from WRFOUT - unis
    basic = ['Times', 'XLAT', 'XLONG', 'XTIME', 'LANDMASK', 'ZS', 'DZS', 'HGT']
    for vname, varin in fin.variables.items():
        for key in basic:
            if vname == key:
                ubase = uout.createVariable(vname,
                                           varin.datatype,
                                           varin.dimensions,
                                           **comp_opts)
                if vname == "Times":
                    ubase.setncatts({"description": "YYYY-MM-DD_hh:mm:ss"})
                else:
                    ubase.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()
                if k not in EXCLUDE_ATTRIBUTES})
                ubase[:] = varin[:]

    # GET VARIABLE from WRFOUT - pres
    basic = ['Times', 'XLAT', 'XLONG', 'XTIME', 'LANDMASK', 'HGT']
    for vname, varin in fin.variables.items():
        for key in basic:
            if vname == key:
                pbase = pout.createVariable(vname,
                                           varin.datatype,
                                           varin.dimensions,
                                           **comp_opts)
                if vname == "Times":
                    pbase.setncatts({"description": "YYYY-MM-DD_hh:mm:ss"})
                else:
                    pbase.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()
                if k not in EXCLUDE_ATTRIBUTES})
                pbase[:] = varin[:]

    # NEW VARIABLE
    if len(plev) != 0:
        var = pout.createVariable("PLEV", 'f', (u'bottom_top'), **comp_opts)
        var.setncatts({"description": "Pressure Levels"})
        var.setncatts({"units": "hPa"})
        var[:] = plev[:]

    # Write 2D variables
    for key in var_2d:
        varname = key.upper()
        
        if packing:
            var_data = var_2d[key].values  # Get numpy array
            scale, offset = calculate_packing_params(var_data, packing_dtype, var_name=varname)
            
            # Create variable with fill_value parameter
            var = uout.createVariable(varname, packing_dtype,
                                     (u'Time', u'south_north', u'west_east'),
                                     fill_value=fill_value,
                                     **comp_opts)
            
            # Set packing attributes BEFORE writing data
            var.scale_factor = scale
            var.add_offset = offset
            
            # Set other attributes (excluding packing-related ones)
            var.setncatts({k: l for k, l in var_2d[key].attrs.items()
                          if k not in ["_FillValue", "missing_value", "scale_factor", "add_offset"] + EXCLUDE_ATTRIBUTES})
            
            # Write data - NetCDF will automatically pack it
            var[:] = var_data
        else:
            var = uout.createVariable(varname, var_2d[key].dtype,
                                     (u'Time', u'south_north', u'west_east'),
                                     **comp_opts)
            var.setncatts({k: l for k, l in var_2d[key].attrs.items()
                          if k not in ["_FillValue", "missing_value"] + EXCLUDE_ATTRIBUTES})
            var[:] = var_2d[key][:]

    # Write engy variables
    for key in var_engy:
        varname = key.upper()
        print(varname)
        
        if packing:
            var_data = var_engy[key].values
            scale, offset = calculate_packing_params(var_data, packing_dtype, var_name=varname)
            
            var = uout.createVariable(varname, packing_dtype,
                                     (u'Time', u'south_north', u'west_east'),
                                     fill_value=fill_value,
                                     **comp_opts)
            var.scale_factor = scale
            var.add_offset = offset
            var.setncatts({k: l for k, l in var_engy[key].attrs.items()
                          if k not in ["_FillValue", "missing_value", "scale_factor", "add_offset"] + EXCLUDE_ATTRIBUTES})
            var[:] = var_data
        else:
            var = uout.createVariable(varname, var_engy[key].dtype,
                                     (u'Time', u'south_north', u'west_east'),
                                     **comp_opts)
            var.setncatts({k: l for k, l in var_engy[key].attrs.items()
                          if k not in ["_FillValue", "missing_value"] + EXCLUDE_ATTRIBUTES})
            var[:] = var_engy[key][:]

    # Write post variables
    for key in var_post:
        varname = key.upper()
        
        if packing:
            var_data = var_post[key].values
            scale, offset = calculate_packing_params(var_data, packing_dtype, var_name=varname)
            
            var = uout.createVariable(varname, packing_dtype,
                                     (u'Time', u'south_north', u'west_east'),
                                     fill_value=fill_value,
                                     **comp_opts)
            var.scale_factor = scale
            var.add_offset = offset
            var.setncatts({k: l for k, l in var_post[key].attrs.items()
                          if k not in ["_FillValue", "missing_value", "scale_factor", "add_offset"] + EXCLUDE_ATTRIBUTES})
            var[:] = var_data
        else:
            var = uout.createVariable(varname, var_post[key].dtype,
                                     (u'Time', u'south_north', u'west_east'),
                                     **comp_opts)
            var.setncatts({k: l for k, l in var_post[key].attrs.items()
                          if k not in ["_FillValue", "missing_value"] + EXCLUDE_ATTRIBUTES})
            var[:] = var_post[key][:]

    # Write soil variables
    for key in var_soil:
        varname = key.upper()
        
        if packing:
            var_data = var_soil[key].values
            scale, offset = calculate_packing_params(var_data, packing_dtype, var_name=varname)
            
            var = uout.createVariable(varname, packing_dtype,
                                     (u'Time', u'soil_layers_stag', u'south_north', u'west_east'),
                                     fill_value=fill_value,
                                     **comp_opts)
            var.scale_factor = scale
            var.add_offset = offset
            var.setncatts({k: l for k, l in var_soil[key].attrs.items()
                          if k not in ["_FillValue", "missing_value", "scale_factor", "add_offset"] + EXCLUDE_ATTRIBUTES})
            var[:] = var_data
        else:
            var = uout.createVariable(varname, var_soil[key].dtype,
                                     (u'Time', u'soil_layers_stag', u'south_north', u'west_east'),
                                     **comp_opts)
            var.setncatts({k: l for k, l in var_soil[key].attrs.items()
                          if k not in ["_FillValue", "missing_value"] + EXCLUDE_ATTRIBUTES})
            var[:] = var_soil[key][:]

    # Write 3D variables
    for key in plev_3d:
        varname = key.upper()
        
        if packing:
            var_data = plev_3d[key].values
            scale, offset = calculate_packing_params(var_data, packing_dtype, var_name=varname)
            
            var = pout.createVariable(varname, packing_dtype,
                                     (u'Time', u'bottom_top', u'south_north', u'west_east'),
                                     fill_value=fill_value,
                                     **comp_opts)
            var.scale_factor = scale
            var.add_offset = offset
            var.setncatts({k: l for k, l in plev_3d[key].attrs.items()
                          if k not in ["_FillValue", "missing_value", "scale_factor", "add_offset"] + EXCLUDE_ATTRIBUTES})
            var[:] = var_data
        else:
            var = pout.createVariable(varname, plev_3d[key].dtype,
                                     (u'Time', u'bottom_top', u'south_north', u'west_east'),
                                     **comp_opts)
            var.setncatts({k: l for k, l in plev_3d[key].attrs.items()
                          if k not in ["_FillValue", "missing_value"] + EXCLUDE_ATTRIBUTES})
            var[:] = plev_3d[key][:]

    # Write moisture variables
    for key in plev_q:
        varname = key.upper()
        
        if packing:
            var_data = plev_q[key].values
            scale, offset = calculate_packing_params(var_data, packing_dtype, var_name=varname)
            
            var = pout.createVariable(varname, packing_dtype,
                                     (u'Time', u'bottom_top', u'south_north', u'west_east'),
                                     fill_value=fill_value,
                                     **comp_opts)
            var.scale_factor = scale
            var.add_offset = offset
            var.setncatts({k: l for k, l in plev_q[key].attrs.items()
                          if k not in ["_FillValue", "missing_value", "scale_factor", "add_offset"] + EXCLUDE_ATTRIBUTES})
            var[:] = var_data
        else:
            var = pout.createVariable(varname, plev_q[key].dtype,
                                     (u'Time', u'bottom_top', u'south_north', u'west_east'),
                                     **comp_opts)
            var.setncatts({k: l for k, l in plev_q[key].attrs.items()
                          if k not in ["_FillValue", "missing_value"] + EXCLUDE_ATTRIBUTES})
            var[:] = plev_q[key][:]

    # Write moisture number variables
    for key in plev_qn:
        varname = key.upper()
        
        if packing:
            var_data = plev_qn[key].values
            scale, offset = calculate_packing_params(var_data, packing_dtype, var_name=varname)
            
            var = pout.createVariable(varname, packing_dtype,
                                     (u'Time', u'bottom_top', u'south_north', u'west_east'),
                                     fill_value=fill_value,
                                     **comp_opts)
            var.scale_factor = scale
            var.add_offset = offset
            var.setncatts({k: l for k, l in plev_qn[key].attrs.items()
                          if k not in ["_FillValue", "missing_value", "scale_factor", "add_offset"] + EXCLUDE_ATTRIBUTES})
            var[:] = var_data
        else:
            var = pout.createVariable(varname, plev_qn[key].dtype,
                                     (u'Time', u'bottom_top', u'south_north', u'west_east'),
                                     **comp_opts)
            var.setncatts({k: l for k, l in plev_qn[key].attrs.items()
                          if k not in ["_FillValue", "missing_value"] + EXCLUDE_ATTRIBUTES})
            var[:] = plev_qn[key][:]

    uout.close()
    pout.close()

DEFAULT_PRECISION = {
    # Mixing ratios (7 decimal places)
    'QVAPOR': 1e-7,
    'QCLOUD': 1e-7,
    'QRAIN': 1e-7,
    'QICE': 1e-7,
    'QSNOW': 1e-7,
    'QGRAUP': 1e-7,

    # Fluxes (3 decimal places)
    'QNCLOUD': 0.001,
    'QNRAIN': 0.001,
    'SWDDIR2': 0.001,
    'SWDDIF2': 0.001,
    'SWDDNI2': 0.001,
    'MCAPE': 0.001,
    'MCIN': 0.001,

    # Vertical velocity (4 decimal places)
    'W': 0.0001,
}

DEFAULT_SCALE_FACTOR = 0.01  # 2 decimal places for unlisted variables

def calculate_packing_params(
    data: np.ndarray,
    dtype: str = 'i2',
    scale_factor: Optional[float] = None,
    var_name: Optional[str] = None,
    precision_dict: Optional[Dict[str, float]] = None,
    use_default_precision: bool = True
) -> tuple:

    """
    Calculate scale_factor and add_offset for NetCDF data packing

    Parameters:
    -----------
    data : numpy.ndarray
        Floating-point data to be packed
    dtype : str
        Target integer type ('i1'=int8, 'i2'=int16, 'i4'=int32)
    scale_factor : float, optional
        Explicit scale factor (highest priority)
    var_name : str, optional
        Variable name for precision lookup
    precision_dict : dict, optional
        Custom precision {var_name: scale_factor}
        Merged with DEFAULT_PRECISION if use_default_precision=True
    use_default_precision : bool
        Use DEFAULT_PRECISION and DEFAULT_SCALE_FACTOR (default: True)

    Returns:
    --------
    scale_factor : float
    add_offset : float

    Priority Order:
    ---------------
    1. scale_factor parameter (explicit)
    2. precision_dict[var_name] (custom)
    3. DEFAULT_PRECISION[var_name] (built-in special cases)
    4. DEFAULT_SCALE_FACTOR (0.01, default for unlisted vars)
    5. Auto-calculate from data range (fallback)
    """
    # Get data range (handle masked arrays)
    if np.ma.isMaskedArray(data):
        data_min = np.ma.min(data)
        data_max = np.ma.max(data)
    else:
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
    
    data_range = data_max - data_min

    # Handle constant or invalid data
    if data_range == 0 or not np.isfinite(data_range):
        return 1.0, (data_min if np.isfinite(data_min) else 0.0)

    # Get integer type range
    dtype_ranges = {
        'i1': (-127, 127),
        'i2': (-32767, 32767),
        'i4': (-2147483647, 2147483647),
    }

    if dtype not in dtype_ranges:
        raise ValueError(f"Unsupported dtype: {dtype}. Use 'i1', 'i2', or 'i4'")

    n_min, n_max = dtype_ranges[dtype]
    n_range = n_max - n_min

    # Determine scale_factor
    determined_scale = None
    scale_from_config = False  # Track if from config/dict

    # Priority 1: Explicit scale_factor
    if scale_factor is not None:
        determined_scale = scale_factor

    # Priority 2-4: Lookup from dictionaries
    elif var_name is not None and use_default_precision:
        # Merge custom with defaults (custom overrides)
        merged = {**DEFAULT_PRECISION, **(precision_dict or {})}

        if var_name in merged:
            determined_scale = merged[var_name]
            scale_from_config = True
        else:
            # Use default scale factor for unlisted variables
            determined_scale = DEFAULT_SCALE_FACTOR
            scale_from_config = True

    # Priority: Only custom precision (no defaults)
    elif var_name is not None and not use_default_precision and precision_dict:
        if var_name in precision_dict:
            determined_scale = precision_dict[var_name]
            scale_from_config = True

    # Priority 5: Auto-calculate
    if determined_scale is None:
        determined_scale = data_range / n_range

    # Calculate add_offset (center of data range)
    offset = (data_max + data_min) / 2.0

    # Verify data fits within integer range
    packed_min = (data_min - offset) / determined_scale
    packed_max = (data_max - offset) / determined_scale

    # Only adjust if auto-calculated (not from config/explicit)
    if packed_min < n_min or packed_max > n_max:
        if not scale_from_config and scale_factor is None:
            determined_scale = data_range / (n_range - 2)
            offset = data_min - determined_scale * (n_min + 1)

    # Safety checks
    if determined_scale == 0 or not np.isfinite(determined_scale):
        determined_scale = 1.0
    if not np.isfinite(offset):
        offset = 0.0
    
    # Round add_offset to match scale_factor precision
    # e.g., scale=0.01 → 2 decimal places, scale=0.001 → 3 decimal places
    if determined_scale > 0 and np.isfinite(determined_scale):
        decimal_places = max(0, -int(np.floor(np.log10(abs(determined_scale)))))
        offset = np.round(offset, decimal_places)

    return float(determined_scale), float(offset)
