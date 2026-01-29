
import numpy as np
import xarray as xr
import os


def nc_calc_vis(qv, qc, qr, qi, qs, t, p, t2):
    '''
!
!   This routine computes horizontal visibility at the
!   surface or lowest model layer, from qc, qr, qi, and qs.
!   qv--water vapor mixing ratio (kg/kg)
!   qc--cloud water mixing ratio (kg/kg)
!   qr--rain water mixing ratio  (kg/kg)
!   qi--cloud ice mixing ratio   (kg/kg)
!   qs--snow mixing ratio        (kg/kg)
!   tt--temperature              (k)
!   pp--pressure                 (Pa)
!
!   If iice=0:
!      qprc=qr     qrain=qr and qclw=qc if T>0C
!      qcld=qc          =0          =0  if T<0C
!                  qsnow=qs and qclice=qc  if T<0C
!                       =0            =0   if T>0C
!   If iice=1:
!      qprc=qr+qs   qrain=qr and qclw=qc
!      qcld=qc+qi   qsnow=qs and qclice=qi
!
!   Independent of the above definitions, the scheme can use different
!   assumptions of the state of hydrometeors:
!        meth='d': qprc is all frozen if T<0, liquid if T>0
!        meth='b': Bocchieri scheme used to determine whether qprc
!           is rain or snow. A temperature assumption is used to
!           determine whether qcld is liquid or frozen.
!        meth='r': Uses the four mixing ratios qrain, qsnow, qclw,
!           and qclice
!
!   The routine uses the following
!   expressions for extinction coefficient, beta (in km**-1),
!   with C being the mass concentration (in g/m**3):
!
!      cloud water:  beta = 144.7 * C ** (0.8800)
!      rain water:   beta =  2.24 * C ** (0.7500)
!      cloud ice:    beta = 327.8 * C ** (1.0000)
!      snow:         beta = 10.36 * C ** (0.7776)
!
!   These expressions were obtained from the following sources:
!
!      for cloud water: from Kunkel (1984)
!      for rainwater: from M-P dist'n, with No=8e6 m**-4 and
!         rho_w=1000 kg/m**3
!      for cloud ice: assume randomly oriented plates which follow
!         mass-diameter relationship from Rutledge and Hobbs (1983)
!      for snow: from Stallabrass (1985), assuming beta = -ln(.02)/vis
!
!   The extinction coefficient for each water species present is
!   calculated, and then all applicable betas are summed to yield
!   a single beta. Then the following relationship is used to
!   determine visibility (in km), where epsilon is the threshhold
!   of contrast, usually taken to be .02:
!
!      vis = -ln(epsilon)/beta      [found in Kunkel (1984)]
!
    '''

    H1   = 1.0
    D608 = 0.608
    RD   = 287.04

    CELKEL  = 273.15
    TICE    = CELKEL-10.
    COEFLC  = 144.7
    COEFLP  = 2.24
    COEFFC  = 327.8
    COEFFP  = 10.36
    EXPONLC = 0.8800
    EXPONLP = 0.7500
    EXPONFC = 1.0000
    EXPONFP = 0.7776
    CONST1  = -np.log(0.02)
    RHOICE  = 917.
    RHOWAT  = 1000.

    QPRC    = qr + qs
    QRAIN   = qr
    QSNOW   = qs
    QCLD    = qc + qi
    QCLW    = qc
    QCLICE  = qi

    TV = t * (H1 + D608 * qv)
    RHOAIR  = p / (RD * TV)

    VOVERMD = (1.0 + qv) / RHOAIR + (QCLW+QRAIN) / RHOWAT + (QCLICE + QSNOW) / RHOICE

    CONCLC=np.maximum(0., QCLW / VOVERMD *1000.)
    CONCLP=np.maximum(0., QRAIN / VOVERMD *1000.)
    CONCFC=np.maximum(0., QCLICE / VOVERMD *1000.)
    CONCFP=np.maximum(0., QSNOW / VOVERMD *1000.)

    BETAV = COEFFC * CONCFC**EXPONFC + COEFFP * CONCFP**EXPONFP + \
            COEFLC * CONCLC**EXPONLC + COEFLP * CONCLP**EXPONLP + \
            1.E-10

    #VIS = 1.E3 * np.minimum(20., CONST1 / BETAV)
    VIS = 1.E3 * np.minimum(50., CONST1 / BETAV)

    VIS = VIS.rename("vis")
    VIS.attrs["FieldType"] = t2.attrs["FieldType"]
    VIS.attrs["MemoryOrder"] = t2.attrs["MemoryOrder"]
    VIS.attrs["description"] = "Visibility"
    VIS.attrs["units"] = "m"
    VIS.attrs["stagger"] = t2.attrs["stagger"]
    VIS.attrs["coordinates"] = t2.attrs["coordinates"]

    return VIS


def nc_calc_gust(gph, ter, pbl, u10, v10, u, v):

    uppwind = np.sqrt(np.power(u,2) + np.power(v,2))
    sfcwind = np.sqrt(np.power(u10,2) + np.power(v10,2))

    pblter = pbl + ter
    hdiff = np.abs(gph - pblter)
    minloc = hdiff.argmin(axis=0)

    gt = np.squeeze(np.array( [gph[minloc[j,i],j,i]-pblter[j,i] 
                               for i in [range(0, minloc.shape[1])] 
                               for j in [range(0, minloc.shape[0])]] ))
    ll = xr.DataArray( np.where(gt>0, np.maximum(minloc-1, 0), minloc) ,
                                      dims=["south_north","west_east"] )

    wind = uppwind - sfcwind
    dz = gph - ter
    delwind = wind * (1. - np.minimum(0.5, dz/2000.))

    wind3d = np.maximum(delwind + sfcwind, sfcwind)
    max_wind = xr.DataArray(np.maximum.accumulate(np.array(wind3d), axis=0),
                            dims=["bottom_top","south_north","west_east"] )

    GUST = [max_wind[ll[j,i],j,i] for i in [range(0, ll.shape[1])] 
                                  for j in [range(0, ll.shape[0])]]
    GUST = xr.DataArray( np.squeeze(np.array(GUST)), dims=["south_north","west_east"] )

    GUST = GUST.rename("gust")
    GUST.attrs["FieldType"] = u10.attrs["FieldType"]
    GUST.attrs["MemoryOrder"] = u10.attrs["MemoryOrder"]
    GUST.attrs["description"] = "Wind speed(gust)"
    GUST.attrs["units"] = u10.attrs["units"]
    GUST.attrs["stagger"] = u10.attrs["stagger"]
    GUST.attrs["coordinates"] = u10.attrs["coordinates"]

    return GUST
