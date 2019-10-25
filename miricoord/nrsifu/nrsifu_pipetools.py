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
    slall=np.zeros(nslice)
    for ii in range(0,nslice):
        print(ii)
        xform=(nirspec.nrs_wcs_set_input(im,ii)).get_transform('detector','v2v3')
        v2all[:,ii],v3all[:,ii],lamall[:,ii]=xform(x,y)
        slall[ii]=ii

    pdb.set_trace()
    return 3
        
    
