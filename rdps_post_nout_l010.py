
# SET FOR CONFIG (NAMELIST)
class ConfigRead():
    def __init__(self, conf_file):
        self.loadConfig(conf_file)
        self.flag = True
    
    def loadConfig(self, conf_file):
        if os.path.exists(conf_file) == False :
            raise Exception("%s file does not exist. \n" % conf_file)
        
        config = configparser.ConfigParser()
        config.read(conf_file)
        
        out = 'namelist.out_nout'
        os.system('rm -f ./'+out)
        f = open('./'+out, 'w')
        
        self.dict = {}
        for s in config.sections() :
            f.write(' &' + s +'\n')
            self.dict[s] = dict(config.items(s))
            for d in self.dict[s] :
                f.write('' + d + '\t\t = ' + self.dict[s][d] +'\n')
            f.write(' /\n\n')
        
        f.close()
        print( '----- CONFIGURE ['+conf_file+' => '+out+']' )


# MAIN CODE: WRFOUT TO PLEV 
def main(conf_file):
    import time
    from datetime import (datetime, timedelta)
    
    from netCDF4 import Dataset
    import numpy as np
    import xarray as xr
    
    from wrf2plev import (readnc_wrf_var, build_var_cache,
                          nc_calc_vis, nc_calc_gust,
                          to_isobar, to_agl,
                          nc_interpolate_and_compute)
    from write_nout_optimized import create_nout


    print( '\n========== WRF to PLEV ==========' )
    total_st = time.time()
    
    cr = ConfigRead(conf_file).dict
    
    analtim = cr['control']['anltim']
    ftim    = cr['control']['fcst']
    wrfpath = cr['control']['inpath']
    print( ' ANAL TIME : ', analtim )
    print( ' WRF OUTPUT: ', wrfpath )
    print( ' FCST TIME : ', '+'+ftim+'h' )
    
    anal_yy = int(analtim[0:4])
    anal_mm = int(analtim[4:6])
    anal_dd = int(analtim[6:8])
    anal_hh = int(analtim[8:10])
    anal_time = datetime(anal_yy,anal_mm,anal_dd,anal_hh)
    valid_time = anal_time + timedelta(hours=int(ftim))
    valid_time = valid_time.strftime("%Y-%m-%d_%H")
    
    wrffile = "rdps_fcst_d02_" + valid_time
    print( ' FILE NAME : ', wrffile )
    
    my_omp = 1 if cr['control']['omp'] == 'True' else 0
    if my_omp :
        try :
            print( '  OMP (Num): ', os.environ['OMP_NUM_THREADS'] )
        except :
            print( "  ERROR    :  Check 'OMP_NUM_THREADS'" )
        else :
            pass
    else :
        print( '  OMP (Num):  Not Uesd' )
    timeid = None


    # READ WRFOUT NETCDF FILE
    print( '----- READ WRFOUT' )
    wrfin = Dataset(wrfpath+'/'+wrffile)
    
    build_cache = 1 if cr['control']['cache'] == 'True' else 0
    if build_cache :
        st = time.time()
        my_cache = build_var_cache(wrfin, timeidx=timeid, omp=my_omp)
        ed = time.time()
        print( '   -> Build cache (', ed-st, 's)' )
    else :
        my_cache = None
        print( '   -> Do not build cache' )

    var_3d = {}
    for key in cr['3d_variable'].keys():
        bool = 1 if cr['3d_variable'][key] == 'True' else 0
        if bool :
            var_3d[key]   = readnc_wrf_var(wrfin, key, ftim,
                                         timeidx=timeid, cache=my_cache, omp=my_omp)

    var_q = {}
    for key in cr['mixing_ratio'].keys():
        bool = 1 if cr['mixing_ratio'][key] == 'True' else 0
        if bool :
            var_q[key]    = readnc_wrf_var(wrfin, key, ftim,
                                         timeidx=timeid, cache=my_cache, omp=my_omp)

    var_qn = {}
    for key in cr['number_concentration'].keys():
        bool = 1 if cr['number_concentration'][key] == 'True' else 0
        if bool :
            var_qn[key]   = readnc_wrf_var(wrfin, key, ftim,
                                         timeidx=timeid, cache=my_cache, omp=my_omp)

    var_2d = {}
    for key in cr['2d_variable'].keys():
        bool = 1 if cr['2d_variable'][key] == 'True' else 0
        if bool :
            var_2d[key]   = readnc_wrf_var(wrfin, key, ftim,
                                         timeidx=timeid, cache=my_cache, omp=my_omp)
    var_soil = {}
    for key in cr['soil_variable'].keys():
        bool = 1 if cr['soil_variable'][key] == 'True' else 0
        if bool :
            var_soil[key] = readnc_wrf_var(wrfin, key, ftim,
                                         timeidx=timeid, cache=my_cache, omp=my_omp)

    ed = time.time()
    print( '   (', ed-st, 's)' )


    # POST-PROCESS
    print( '----- NC POST PROCESS' )
    st = time.time()
    var_post = {}
    for key in cr['post'].keys():
        bool = 1 if cr['post'][key] == 'True' else 0
        if bool :
            if key == 'vis':
                full_p = my_cache['P'] + my_cache['PB']
                var_post[key] = nc_calc_vis(var_q['qvapor'][0,:], var_q['qcloud'][0,:],
                                         var_q['qrain'][0,:],  var_q['qice'][0,:],
                                         var_q['qsnow'][0,:],  var_3d['t'][0,:],
                                         full_p[0,:], var_2d['t2'])
            elif key == 'gust':
                terrain = wrfin['HGT'][0,:]
                var_post[key] = nc_calc_gust(var_3d['gph'], terrain, var_2d['pblh'],
                                          var_2d['u10'], var_2d['v10'],
                                          var_3d['u'],   var_3d['v'])
    ed = time.time()
    print( '   (', ed-st, 's)' )


    # INTERPOLATION for ISOBARIC
    isobar = 1 if cr['interpolation']['isobar'] == 'True' else 0
    if isobar :
        print( '----- INTERPOLATE to PLEV ' )
        st = time.time()
        
        plev = [int(p.strip()) for p in cr['interpolation']['plev'].split(',')]
        print( ' PLEV (hPa):', plev )
        
        plev_3d = {}
        for key in var_3d :
            plev_3d[key] = to_isobar(wrfin, plev, key, var_3d[key],
                                     extrapolate=True, timeidx=timeid, cache=my_cache, omp=my_omp)
        plev_q = {}
        for key in var_q :
            plev_q[key]  = to_isobar(wrfin, plev, key, var_q[key],
                                     extrapolate=True, timeidx=timeid, cache=my_cache, omp=my_omp)
        plev_qn = {}
        for key in var_qn :
            plev_qn[key] = to_isobar(wrfin, plev, key, var_qn[key],
                                     extrapolate=True, timeidx=timeid, cache=my_cache, omp=my_omp)
        ed = time.time()
        print( '   (', ed-st, 's)' )
    else :
        print( '----- Do not INTERPOLATE to PLEV ' )
        plev    = []
        plev_3d = {}
        plev_q  = {}
        plev_qn = {}

    # ENGY & CLOUD COVER FILE
    var_uv_sw = {}
    var_cld = {}
    var_engy = {}
    for key in cr['engy_variable'].keys():
        bool = 1 if cr['engy_variable'][key] == 'True' else 0
        if bool :
          if key == 'cld':
              var_cld = nc_interpolate_and_compute(wrfin, plev, ftim, timeid, my_cache, my_omp, decorr_length=2500.0)
          else:
            var_uv_sw[key]   = readnc_wrf_var(wrfin, key, ftim,
                                         timeidx=timeid, cache=my_cache, omp=my_omp)
        var_engy = xr.merge([var_uv_sw, var_cld])

    # WRITE TO NETCDF FILE
        print( '----- WRITE PLEV to netCDF ' )
        st = time.time()

        outpath = cr['control']['outpath']
        oheader = cr['control']['outheader']
        os.system('mkdir -p '+outpath)
        outname = oheader+'_h'+("%03d"%int(ftim))+'.'+analtim+'.nc'
        print( ' PLEV  OUT :', outpath )
        print( '  -> ', outname )
        outfile = outpath +'/'+ oheader

        create_nout(wrfin, outfile, ftim, analtim, valid_time, var_soil, var_2d, var_engy,
                       var_post, plev, plev_3d, plev_q, plev_qn,
                       compression='deflate', deflate_level=4, shuffle=True, packing=True,
                       packing_dtype='i2',
                       use_fixed_packing=True,      # Fixed scale/offset for consistent file sizes
                       single_thread_write=False)   # Keep OMP threads for performance
#
        ed = time.time()
        print( '   (', ed-st, 's)' )

    total_ed = time.time()
    print( ' Total :', total_ed-total_st, 's' )
    print( '=================================' )

    wrfin.close()
    exit()


if __name__ == "__main__":
    import os
    import sys
    import configparser
    
    if len(sys.argv) != 2:
        conf_file = 'config.ini'
    elif len(sys.argv) == 2:
        conf_file = sys.argv[1]
    
    main(conf_file)
