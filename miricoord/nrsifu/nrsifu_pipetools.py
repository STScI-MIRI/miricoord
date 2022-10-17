#
"""
Useful python tools for working with the NIRSpec IFU WCS.

This uses the JWST pipeline implementation and relies upon having
input data that has gone through assign_wcs since the distortion
solution change because of the grating wheel non-repeatability.

This means that per-slice solutions are required, and must be called
iteratively and rejected where nan.  This front-end obscures all of that
and allows things to be called in a MIRI-like way as much as possible.

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
17-Oct-2019  Written by David Law (dlaw@stsci.edu)
"""
# NOTES
#from jwst import datamodels
#im=datamodels.ImageModel('sim_with_small_ptsrcinj_data_dith_assign_wcs.fits')
# See WCS metadata
#im.meta.wcsinfo.instance

# detector pixel to v2,v3,lam for a given slice
#s3=nirspec.nrs_wcs_set_input(im,3)
#temp=s3.get_transform('detector','v2v3')
#temp(745,1278)

from jwst import datamodels
from jwst.assign_wcs import nirspec
import pdb as pdb
from astropy.io import fits
import numpy as np

#############################

# Convert x,y pixel values to v2,v3,lam for a given input file
def xytov2v3l(x,y,file):
    im=datamodels.ImageModel(file)

    nslice=30
    # Big structure to save all the returned values
    v2all=np.zeros([len(x),nslice])
    v3all=np.zeros([len(x),nslice])
    lamall=np.zeros([len(x),nslice])
    slall=np.zeros([len(x),nslice])
    for ii in range(0,nslice):
        xform=(nirspec.nrs_wcs_set_input(im,ii)).get_transform('detector','v2v3')
        v2all[:,ii],v3all[:,ii],lamall[:,ii]=xform(x,y)
        slall[:,ii]=ii

    # slice all is nan where v2all is nan
    v2_1d=v2all.reshape(-1)
    sl_1d=slall.reshape(-1)
    finite=(np.isfinite(v2_1d))
    indx=(np.where(finite == False))[0]
    if (len(indx) > 0):
        sl_1d[indx]=np.nan
        
    # Element 1330000 should be x=848,y=649, in slice 6 (0-ind)

    v2=np.nanmedian(v2all,axis=1)
    v3=np.nanmedian(v3all,axis=1)
    lam=np.nanmedian(lamall,axis=1)
    sl=np.nanmedian(slall,axis=1)
    
    return v2,v3,lam,sl

#############################

# Convert x,y pixel values to v2,v3,lam for a given input file
def v2v3ltoxy(v2,v3,lam,file):
    im=datamodels.ImageModel(file)

    nslice=30
    # Big structure to save all the returned values
    xall=np.zeros([len(v2),nslice])
    yall=np.zeros([len(v2),nslice])
    slall=np.zeros([len(v2),nslice])
    for ii in range(0,nslice):
        xform=(nirspec.nrs_wcs_set_input(im,ii)).get_transform('v2v3','detector')
        xall[:,ii],yall[:,ii]=xform(v2,v3,lam)
        slall[:,ii]=ii

    # slice all is nan where xall is nan
    x_1d=xall.reshape(-1)
    sl_1d=slall.reshape(-1)
    finite=(np.isfinite(x_1d))
    indx=(np.where(finite == False))[0]
    if (len(indx) > 0):
        sl_1d[indx]=np.nan

    x=np.nanmedian(xall,axis=1)
    y=np.nanmedian(yall,axis=1)
    sl=np.nanmedian(slall,axis=1)
    
    return x,y,sl
    
#############################

# Make a slice number map
def sliceimage(file):
    # Define 0-indexed base x and y pixel number (2048x2048 grid)
    basex,basey = np.meshgrid(np.arange(2048),np.arange(2048))
    # Convert to 1d vectors
    basex=basex.reshape(-1)
    basey=basey.reshape(-1)
    v2,v3,lam,sl=xytov2v3l(basex,basey,file)

    mockimg=np.zeros([2048,2048])
    mock1d=mockimg.reshape(-1)
    mock1d[:]=sl

    finite=(np.isfinite(mock1d))
    indx=(np.where(finite == False))[0]
    if (len(indx) > 0):
        mock1d[indx]=-1
    
    return mockimg

#############################

# Make a set of maps
def mapset(file):
    # Define 0-indexed base x and y pixel number (2048x2048 grid)
    basex,basey = np.meshgrid(np.arange(2048),np.arange(2048))
    # Convert to 1d vectors
    basex=basex.reshape(-1)
    basey=basey.reshape(-1)
    v2,v3,lam,sl=xytov2v3l(basex,basey,file)

    slmap=np.zeros([2048,2048])
    slmap1d=slmap.reshape(-1)
    slmap1d[:]=sl
    
    v2map=np.zeros([2048,2048])
    v2map1d=v2map.reshape(-1)
    v2map1d[:]=v2

    v3map=np.zeros([2048,2048])
    v3map1d=v3map.reshape(-1)
    v3map1d[:]=v3

    lmap=np.zeros([2048,2048])
    lmap1d=lmap.reshape(-1)
    lmap1d[:]=lam
    
    finite=(np.isfinite(slmap1d))
    indx=(np.where(finite == False))[0]
    if (len(indx) > 0):
        slmap1d[indx]=-1
        v2map1d[indx]=-1
        v3map1d[indx]=-1
        lmap1d[indx]=-1
        
    return v2map,v3map,lmap,slmap
