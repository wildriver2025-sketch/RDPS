
from netCDF4 import Dataset
import numpy as np


def create_nc_plev(fin, ncout, var_wrf, var_soil, var_2d, var_engy,
                   var_post, plev, plev_3d, plev_q, plev_qn) :

    # Add time dimension
    for key in var_soil :
        var_soil[key] = var_soil[key].squeeze().expand_dims("Time")
    for key in var_2d :
        var_2d[key]   = var_2d[key].squeeze().expand_dims("Time")
    for key in var_engy :
        var_engy[key]   = var_engy[key].squeeze().expand_dims("Time")
    for key in var_post :
        var_post[key] = var_post[key].squeeze().expand_dims("Time")
    for key in plev_3d :
        plev_3d[key]  = plev_3d[key].squeeze().expand_dims("Time")
    for key in plev_q :
        plev_q[key]   = plev_q[key].squeeze().expand_dims("Time")
    for key in plev_qn :
        plev_qn[key]  = plev_qn[key].squeeze().expand_dims("Time")

    # OPEN OUTFILE
#    fout = Dataset(ncout, "w", format=fin.file_format)
    fout = Dataset(ncout, "w", format='NETCDF4')

    # GET NETCDF HEADER from WRFOUT
    for ganame in fin.ncattrs() :
        if len(plev) != 0 :
            if ganame == "TITLE" :
                fout.setncattr(ganame, fin.getncattr(ganame)+" - ON PRESSURE LEVEL")
            elif ganame == "BOTTOM-TOP_GRID_DIMENSION" :
                fout.setncattr(ganame, int(len(plev)))
            elif ganame == "BOTTOM-TOP_PATCH_START_UNSTAG" : pass
            elif ganame == "BOTTOM-TOP_PATCH_END_UNSTAG" : pass
            elif ganame == "BOTTOM-TOP_PATCH_START_STAG" : pass
            elif ganame == "BOTTOM-TOP_PATCH_END_STAG" : pass
            else : fout.setncattr(ganame, fin.getncattr(ganame))
        else :
            if ganame == "TITLE" :
                fout.setncattr(ganame, fin.getncattr(ganame)+" - ON SINGLE LEVEL")
            elif ganame == "BOTTOM-TOP_GRID_DIMENSION" : pass
            elif ganame == "BOTTOM-TOP_PATCH_START_UNSTAG" : pass
            elif ganame == "BOTTOM-TOP_PATCH_END_UNSTAG" : pass
            elif ganame == "BOTTOM-TOP_PATCH_START_STAG" : pass
            elif ganame == "BOTTOM-TOP_PATCH_END_STAG" : pass
            else : fout.setncattr(ganame, fin.getncattr(ganame))

    # CREATE DIMENSION
    fout.createDimension("Time", None)
    fout.createDimension("DateStrLen", 19)
    fout.createDimension("west_east", int(fin.getncattr("WEST-EAST_GRID_DIMENSION"))-1)
    fout.createDimension("south_north", int(fin.getncattr("SOUTH-NORTH_GRID_DIMENSION"))-1)
    if len(plev) != 0 :
        fout.createDimension("bottom_top", int(len(plev)))
    # Soil layer dimension for surface physics scheme
    if fin.getncattr("SF_SURFACE_PHYSICS") == 1 :
        fout.createDimension("soil_layers_stag", 5)
    elif fin.getncattr("SF_SURFACE_PHYSICS") == 2 :
        fout.createDimension("soil_layers_stag", 4)
    else : print( "[CHECK] SURFACE LAYER PHYSCIS for soil_layer_stag" )

    # GET VARIABLE from WRFOUT
    basic = ['Times', 'XLAT', 'XLONG', 'XTIME', 'LANDMASK', 'ZS', 'DZS', 'HGT']
    basic.extend(var_wrf)
    for vname, varin in fin.variables.items():
        for key in basic :
            if vname == key :
                vbase = fout.createVariable(vname, 
                                            varin.datatype, 
                                            varin.dimensions)
                if vname == "Times" :
                    vbase.setncatts({"description":"YYYY-MM-DD_hh:mm:ss"})
                else : 
                    vbase.setncatts({k:varin.getncattr(k) for k in varin.ncattrs()})
                vbase[:] = varin[:]

    # NEW VARIABLE 
    if len(plev) != 0 :
        var = fout.createVariable("PLEV", 'f', (u'bottom_top'))
        var.setncatts({"description":"Pressure Levels"})
        var.setncatts({"units":"hPa"})
        var[:] = plev[:]

    for key in var_soil :
        varname = key.upper()
        var = fout.createVariable(varname, var_soil[key].dtype,
                                  (u'Time', u'soil_layers_stag', u'south_north', u'west_east'))
        var.setncatts({k:l for k, l in var_soil[key].attrs.items() 
                      if k!="_FillValue" and k!="missing_value"})
        var[:] = var_soil[key][:]

    for key in var_2d :
        varname = key.upper()
        var = fout.createVariable(varname, var_2d[key].dtype,
                                  (u'Time', u'south_north', u'west_east'))
        var.setncatts({k:l for k, l in var_2d[key].attrs.items() 
                      if k!="_FillValue" and k!="missing_value"})
        var[:] = var_2d[key][:]

    for key in var_engy :
        varname = key.upper()
        var = fout.createVariable(varname, var_engy[key].dtype,
                                  (u'Time', u'south_north', u'west_east'))
        var.setncatts({k:l for k, l in var_engy[key].attrs.items()
                      if k!="_FillValue" and k!="missing_value"})
        var[:] = var_engy[key][:]

    for key in var_post :
        varname = key.upper()
        var = fout.createVariable(varname, var_post[key].dtype,
                                  (u'Time', u'south_north', u'west_east'))
        var.setncatts({k:l for k, l in var_post[key].attrs.items() 
                      if k!="_FillValue" and k!="missing_value"})
        var[:] = var_post[key][:]

    for key in plev_3d :
        varname = key.upper()
        var = fout.createVariable(varname, plev_3d[key].dtype,
                                  (u'Time', u'bottom_top', u'south_north', u'west_east'))
        var.setncatts({k:l for k, l in plev_3d[key].attrs.items() 
                      if k!="_FillValue" and k!="missing_value"})
        var[:] = plev_3d[key][:]

    for key in plev_q :
        varname = key.upper()
        var = fout.createVariable(varname, plev_q[key].dtype,
                                  (u'Time', u'bottom_top', u'south_north', u'west_east'))
        var.setncatts({k:l for k, l in plev_q[key].attrs.items() 
                      if k!="_FillValue" and k!="missing_value"})
        var[:] = plev_q[key][:]

    for key in plev_qn :
        varname = key.upper()
        var = fout.createVariable(varname, plev_qn[key].dtype,
                                  (u'Time', u'bottom_top', u'south_north', u'west_east'))
        var.setncatts({k:l for k, l in plev_qn[key].attrs.items() 
                      if k!="_FillValue" and k!="missing_value"})
        var[:] = plev_qn[key][:]


    fout.close()


