
from wrf import (getvar, extract_vars)
from wrf import (omp_get_num_procs, omp_set_num_threads, omp_enabled)
import numpy as np
import xarray as xr
import os


def build_var_cache(wrfin, timeidx=None, omp=False):
    if omp :
        num_omp = os.environ['OMP_NUM_THREADS']
        omp_enabled=False
        omp_set_num_threads(int(num_omp))
        #print( '   Using OpenMP (Num of procs :', num_omp,')' )
    else :
        #print( '   Do not use OpenMP' )
        pass

    vars = ["T", "HGT", "QVAPOR", "PSFC", "P", "PB", "PH", "PHB"]
    my_cache = extract_vars(wrfin,
                            timeidx = timeidx,
                            varnames = vars)

    return my_cache


def read_wrf_var(wrfin, var, ftim, timeidx=None, cache=None, omp=False):
    if omp :
        num_omp = os.environ['OMP_NUM_THREADS']
        omp_enabled=False
        omp_set_num_threads(int(num_omp))
        #print( '   Using OpenMP (Num of procs :', num_omp,')' )
    else :
        #print( '   Do not use OpenMP' )
        pass

    if var == 't':		# Temperature
        value = getvar(wrfin,
                       varname = "temp",
                       units   = "K",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 130
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 0
        value.attrs["GRIB_parameterNumber"] = 0
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'eth':		# Equivalent potential temperature
        value = getvar(wrfin,
                       varname = "eth",
                       units   = "K",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 3014
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 0
        value.attrs["GRIB_parameterNumber"] = 3
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'rh':		# Relative humidity
        value = getvar(wrfin,
                       varname = "rh",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 157
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 1
        value.attrs["GRIB_parameterNumber"] = 1
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'u':		# u-component wind
        value = getvar(wrfin,
                       varname = "ua",
                       units   = "ms-1",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 131
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 2
        value.attrs["GRIB_parameterNumber"] = 2
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'v':		# v-component wind
        value = getvar(wrfin,
                       varname = "va",
                       units   = "ms-1",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 132
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 2
        value.attrs["GRIB_parameterNumber"] = 3
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'w':		# w-component wind
        value = getvar(wrfin,
                       varname = "wa",
                       units   = "ms-1",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 260238
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 2
        value.attrs["GRIB_parameterNumber"] = 9
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'gph':		# Geopotential height
        value = getvar(wrfin,
                       varname = "height",
                       msl     = True,
                       units   = "m",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 156
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 3
        value.attrs["GRIB_parameterNumber"] = 5
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'cldfra':	# Cloud fraction
        value = getvar(wrfin,
                       varname = "CLDFRA",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 248
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 6
        value.attrs["GRIB_parameterNumber"] = 32
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'avo':		# Absolute vorticity
        value = getvar(wrfin,
                       varname = "avo",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 3041
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 2
        value.attrs["GRIB_parameterNumber"] = 10
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'pvo':		# Potential vorticity
        value = getvar(wrfin,
                       varname = "pvo",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 60
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 2
        value.attrs["GRIB_parameterNumber"] = 14
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'dbz':		# Reflectivity
        value = getvar(wrfin,
                       varname = "dbz",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 260389
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 16
        value.attrs["GRIB_parameterNumber"] = 4
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'omega':	# Omega
        value = getvar(wrfin,
                       varname = "omg",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 135
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 2
        value.attrs["GRIB_parameterNumber"] = 8
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 't2':		# T at 2m
        value = getvar(wrfin,
                       varname = "T2",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 167
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 0
        value.attrs["GRIB_parameterNumber"] = 0
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 103
        value.attrs["GRIB_level"] = 2
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'rh2':		# RH at 2m
        value = getvar(wrfin,
                       varname = "rh2",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 260242
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 1
        value.attrs["GRIB_parameterNumber"] = 1
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 103
        value.attrs["GRIB_level"] = 2
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'u10':		# U at 10m
        value = getvar(wrfin,
                       varname = "U10",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 165
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 2
        value.attrs["GRIB_parameterNumber"] = 2
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 103
        value.attrs["GRIB_level"] = 10
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'u10':		# U at 10m
        value = getvar(wrfin,
                       varname = "U10",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 165
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 2
        value.attrs["GRIB_parameterNumber"] = 2
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 103
        value.attrs["GRIB_level"] = 10
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'u80':		# U at 80m
        value = getvar(wrfin,
                       varname = "U80",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 165
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 2
        value.attrs["GRIB_parameterNumber"] = 2
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 103
        value.attrs["GRIB_level"] = 80
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'u140':		# U at 140m
        value = getvar(wrfin,
                       varname = "U140",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 165
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 2
        value.attrs["GRIB_parameterNumber"] = 2
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 103
        value.attrs["GRIB_level"] = 140
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'u220':		# U at 220m
        value = getvar(wrfin,
                       varname = "U220",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 165
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 2
        value.attrs["GRIB_parameterNumber"] = 2
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 103
        value.attrs["GRIB_level"] = 220
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'v10':		# V at 10m
        value = getvar(wrfin,
                       varname = "V10",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 166
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 2
        value.attrs["GRIB_parameterNumber"] = 3
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 103
        value.attrs["GRIB_level"] = 10
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'v80':		# V at 80m
        value = getvar(wrfin,
                       varname = "V80",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 165
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 2
        value.attrs["GRIB_parameterNumber"] = 3
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 103
        value.attrs["GRIB_level"] = 80
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'v140':		# V at 140m
        value = getvar(wrfin,
                       varname = "V140",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 165
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 2
        value.attrs["GRIB_parameterNumber"] = 3
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 103
        value.attrs["GRIB_level"] = 140
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'v220':		# V at 220m
        value = getvar(wrfin,
                       varname = "V220",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 165
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 2
        value.attrs["GRIB_parameterNumber"] = 3
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 103
        value.attrs["GRIB_level"] = 220
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'mslp':		# Sea level pressure
        value = getvar(wrfin,
                       varname = "slp",
                       units   = "Pa",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 151
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 3
        value.attrs["GRIB_parameterNumber"] = 0
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 101
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'rain':		# Precipitation (convetive + grid scale)
        RAIN_EXP = getvar(wrfin,
                          varname = "RAINNC",
                          timeidx = timeidx,
                          cache   = cache)
        RAIN_CON = getvar(wrfin,
                          varname = "RAINC",
                          timeidx = timeidx,
                          cache   = cache)
        value = RAIN_EXP + RAIN_CON
        value = value.rename("total_rain")
        value.attrs["FieldType"] = RAIN_CON.attrs["FieldType"]
        value.attrs["MemoryOrder"] = RAIN_CON.attrs["MemoryOrder"]
        value.attrs["description"] = "ACCUMULATED TOTAL (CUMULUS + GRID SCALE) PRECIPITATION"
        value.attrs["units"] = RAIN_CON.attrs["units"]
        value.attrs["stagger"] = RAIN_CON.attrs["stagger"]
        value.attrs["coordinates"] = RAIN_CON.attrs["coordinates"]
        value.attrs["projection"] = RAIN_CON.attrs["projection"]
        #value.attrs["GRIB_paramId"] = 260138
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 1
        value.attrs["GRIB_parameterNumber"] = 8
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 1
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'rainnc':		# Precipitation (grid scale)
        value = getvar(wrfin,
                       varname = "RAINNC",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 260009
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 1
        value.attrs["GRIB_parameterNumber"] = 9
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 1
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'rainc':		# Precipitation (convetive scale)
        value = getvar(wrfin,
                       varname = "RAINC",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 3063
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 1
        value.attrs["GRIB_parameterNumber"] = 10
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 1
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'snow':		# SNOW (grid scale)
        value = getvar(wrfin,
                       varname = "SNOWNC",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 260012
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 1
        value.attrs["GRIB_parameterNumber"] = 15
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 1
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'graupel':	# GRAUPEL (grid scale)
        value = getvar(wrfin,
                       varname = "GRAUPELNC",
                       timeidx = timeidx,
                       cache   = cache)
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 1
        value.attrs["GRIB_parameterNumber"] = 74
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 1
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'hail':		# HAIL (grid scale)
        value = getvar(wrfin,
                       varname = "HAILNC",
                       timeidx = timeidx,
                       cache   = cache)
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 1
        value.attrs["GRIB_parameterNumber"] = 72
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 1
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'psfc':		# P at surface
        value = getvar(wrfin,
                       varname = "PSFC",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 134
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 3
        value.attrs["GRIB_parameterNumber"] = 0
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 1
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'tskin':	# T at skin surface
        value = getvar(wrfin,
                       varname = "TSK",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 235
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 0
        value.attrs["GRIB_parameterNumber"] = 17
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 1
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'soilm':	# Soil moisture
        value = getvar(wrfin,
                       varname = "SMOIS",
                       timeidx = timeidx,
                       cache   = cache)
        value.attrs["GRIB_discipline"] = 2
        value.attrs["GRIB_parameterCategory"] = 3
        value.attrs["GRIB_parameterNumber"] = 19
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 106
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'soilt':	# Soil temperature
        value = getvar(wrfin,
                       varname = "TSLB",
                       timeidx = timeidx,
                       cache   = cache)
        value.attrs["GRIB_discipline"] = 2
        value.attrs["GRIB_parameterCategory"] = 3
        value.attrs["GRIB_parameterNumber"] = 18
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 106
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'mcape':         # Maximum CAPE
        value = getvar(wrfin,
                       varname = "cape_2d",
                       timeidx = timeidx,
                       cache   = cache)[0,:]
        value = value.rename("mcape")
        value.attrs["description"] = "Maximum CAPE"
        value.attrs["units"] = "J kg-1"
        del value['mcape_mcin_lcl_lfc']
        #value.attrs["GRIB_paramId"] = 59
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 7
        value.attrs["GRIB_parameterNumber"] = 6
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 1
        value.attrs["GRIB_typeOfSecondFixedSurface"] = 8
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'mcin':         # Maximum CIN
        value = getvar(wrfin,
                       varname = "cape_2d",
                       timeidx = timeidx,
                       cache   = cache)[1,:]
        value = value.rename("mcin")
        value.attrs["description"] = "Maximum CIN"
        value.attrs["units"] = "J kg-1"
        del value['mcape_mcin_lcl_lfc']
        #value.attrs["GRIB_paramId"] = 228001
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 7
        value.attrs["GRIB_parameterNumber"] = 7
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 1
        value.attrs["GRIB_typeOfSecondFixedSurface"] = 8
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'lcl':         # LCL
        value = getvar(wrfin,
                       varname = "cape_2d",
                       timeidx = timeidx,
                       cache   = cache)[2,:]
        value = value.rename("lcl")
        value.attrs["description"] = "LCL"
        value.attrs["units"] = "m"
        del value['mcape_mcin_lcl_lfc']
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 3
        value.attrs["GRIB_parameterNumber"] = 6
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 5
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'lfc':         # LFC
        value = getvar(wrfin,
                       varname = "cape_2d",
                       timeidx = timeidx,
                       cache   = cache)[3,:]
        value = value.rename("lfc")
        value.attrs["description"] = "LFC"
        value.attrs["units"] = "m"
        del value['mcape_mcin_lcl_lfc']
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 3
        value.attrs["GRIB_parameterNumber"] = 6
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 14
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'qvapor':	# Water vapor mixing ratio
        value = getvar(wrfin,
                       varname = "QVAPOR",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 133
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 1
        value.attrs["GRIB_parameterNumber"] = 0
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'qcloud':	# Cloud water mixing ratio
        value = getvar(wrfin,
                       varname = "QCLOUD",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 260018
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 1
        value.attrs["GRIB_parameterNumber"] = 22
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'qrain':	# Rain water mixing ratio
        value = getvar(wrfin,
                       varname = "QRAIN",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 260020
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 1
        value.attrs["GRIB_parameterNumber"] = 24
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'qice':		# Ice mixing ratio
        value = getvar(wrfin,
                       varname = "QICE",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 260019
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 1
        value.attrs["GRIB_parameterNumber"] = 23
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'qsnow':	# Snow mixing ratio
        value = getvar(wrfin,
                       varname = "QSNOW",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 260021
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 1
        value.attrs["GRIB_parameterNumber"] = 25
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'qgraup':	# Graupel mixing ratio
        value = getvar(wrfin,
                       varname = "QGRAUP",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 260028
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 1
        value.attrs["GRIB_parameterNumber"] = 32
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'qhail':	# Hail mixing ratio
        value = getvar(wrfin,
                       varname = "QHAIL",
                       timeidx = timeidx,
                       cache   = cache)
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 1
        value.attrs["GRIB_parameterNumber"] = 71
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'qnccn':	# CCN number concentration
        value = getvar(wrfin,
                       varname = "QNCCN",
                       timeidx = timeidx,
                       cache   = cache)

    elif var == 'qncloud':	# Cloud water number concentration
        value = getvar(wrfin,
                       varname = "QNCLOUD",
                       timeidx = timeidx,
                       cache   = cache)
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 6
        value.attrs["GRIB_parameterNumber"] = 28
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'qnrain':	# Rain number concentration
        value = getvar(wrfin,
                       varname = "QNRAIN",
                       timeidx = timeidx,
                       cache   = cache)
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 1
        value.attrs["GRIB_parameterNumber"] = 100
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'qnice':	# Ice number concentration
        value = getvar(wrfin,
                       varname = "QNICE",
                       timeidx = timeidx,
                       cache   = cache)
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 6
        value.attrs["GRIB_parameterNumber"] = 29
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'qnsnow':	# Snow number concentration
        value = getvar(wrfin,
                       varname = "QNSNOW",
                       timeidx = timeidx,
                       cache   = cache)
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 1
        value.attrs["GRIB_parameterNumber"] = 101
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'qngraup':	# Graupel number concentration
        value = getvar(wrfin,
                       varname = "QNGRAUPEL",
                       timeidx = timeidx,
                       cache   = cache)
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 1
        value.attrs["GRIB_parameterNumber"] = 102
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 105
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'pblh':	        #
        value = getvar(wrfin,
                       varname = "PBLH",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 260083
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 3
        value.attrs["GRIB_parameterNumber"] = 18
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 1
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'sst':	        #
        value = getvar(wrfin,
                       varname = "SST",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 34
        value.attrs["GRIB_discipline"] = 10
        value.attrs["GRIB_parameterCategory"] = 3
        value.attrs["GRIB_parameterNumber"] = 0
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 1
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'olr':	        #
        value = getvar(wrfin,
                       varname = "OLR",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 260096
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 5
        value.attrs["GRIB_parameterNumber"] = 1
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 8
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'swddif':	        #
        value = getvar(wrfin,
                       varname = "SWDDIF",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 260263
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 4
        value.attrs["GRIB_parameterNumber"] = 14
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 1
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'swddir':	        #
        value = getvar(wrfin,
                       varname = "SWDDIR",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 260262
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 4
        value.attrs["GRIB_parameterNumber"] = 13
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 1
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'swddif2':	        #
        value = getvar(wrfin,
                       varname = "SWDDIF2",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 260263
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 4
        value.attrs["GRIB_parameterNumber"] = 14
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 1
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'swddir2':	        #
        value = getvar(wrfin,
                       varname = "SWDDIR2",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 260262
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 4
        value.attrs["GRIB_parameterNumber"] = 13
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 1
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'swddni2':              # Shortwave surface downward direct normal irradiance from FARMS.20250421.jslee
        value = getvar(wrfin,
                       varname = "SWDDNI2",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 260262
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 4
        value.attrs["GRIB_parameterNumber"] = 54
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 1
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'cldfrac2d':	        #
        value = getvar(wrfin,
                       varname = "CLDFRAC2D",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 248
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 6
        value.attrs["GRIB_parameterNumber"] = 32
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 1
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'visb':            # Visibility.20250421.jslee
        value = getvar(wrfin,
                       varname = "VISB",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 248
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 19
        value.attrs["GRIB_parameterNumber"] = 0 
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 1
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'ph':            # perturbation geopotential.20250709.syjeong
        value = getvar(wrfin,
                       varname = "PH",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 248
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 19
        value.attrs["GRIB_parameterNumber"] = 0
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 1
        value.attrs['GRIB_stepRange'] = int(ftim)

    elif var == 'phb':            # base-state geopotential.20250709.syjeong
        value = getvar(wrfin,
                       varname = "PHB",
                       timeidx = timeidx,
                       cache   = cache)
        #value.attrs["GRIB_paramId"] = 248
        value.attrs["GRIB_discipline"] = 0
        value.attrs["GRIB_parameterCategory"] = 19
        value.attrs["GRIB_parameterNumber"] = 0
        value.attrs["GRIB_typeOfFirstFixedSurface"] = 1
        value.attrs['GRIB_stepRange'] = int(ftim)

    else :
        print( ' Check Parameter: ', var )
        pass

    value.attrs.pop('projection')

    return value
