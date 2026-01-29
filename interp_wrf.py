
from wrf import vinterp
from wrf import (omp_get_num_procs, omp_set_num_threads, omp_enabled)
import numpy as np
import os


def to_agl(wrfin, agl, var, field, timeidx=None, cache=None, omp=False):
    if omp :
        num_omp = os.environ['OMP_NUM_THREADS']
        omp_enabled=False
        omp_set_num_threads(int(num_omp))
        #print( '   Using OpenMP (Num of procs :', num_omp,')' )
    else :
        #print( '   Do not use OpenMP' )
        pass

def to_isobar(wrfin, plev, var, field, extrapolate=False, field_type=None, 
              log_p=True, timeidx=None, cache=None, omp=False):
    if omp :
        num_omp = os.environ['OMP_NUM_THREADS']
        omp_enabled=False
        omp_set_num_threads(int(num_omp))
        #print( '   Using OpenMP (Num of procs :', num_omp,')' )
    else :
        #print( '   Do not use OpenMP' )
        pass

    if var == 't':	# Temperature
        value = vinterp(wrfin, field = field, 
                        vert_coord    = "pressure",
                        interp_levels = plev,
                        extrapolate   = extrapolate,
                        field_type    = "tk",
                        timeidx       = timeidx,
                        log_p         = log_p,
                        cache         = cache)

    elif var == 'gph':	# Geopotential height
        value = vinterp(wrfin, field = field, 
                        vert_coord    = "pressure",
                        interp_levels = plev,
                        extrapolate   = extrapolate,
                        field_type    = "ght",
                        timeidx       = timeidx,
                        log_p         = log_p,
                        cache         = cache)

    else :		# other variable
        value = vinterp(wrfin, field = field, 
                        vert_coord    = "pressure",
                        interp_levels = plev,
                        extrapolate   = extrapolate,
                        field_type    = field_type,
                        timeidx       = timeidx,
                        log_p         = log_p,
                        cache         = cache)


    return value


