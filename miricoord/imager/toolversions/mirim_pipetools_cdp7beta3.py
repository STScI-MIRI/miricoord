#
"""
Useful python tools for working with the MIRI Imager.
This contains cdp7beta3 specific code.

This version of the tools hooks into the JWST Calibration
Pipeline code to do the heavy lifting.  Note that this
means performance may be affected by what version of
the pipeline you are running!!  It does, however, use
offline versions of the CRDS reference files contained
within this github repository.

Convert JWST v2,v3 locations (in arcsec) to MIRI Imager SCA x,y pixel locations.
Note that the pipeline uses a 0-indexed detector pixel (1032x1024) convention while
SIAF uses a 1-indexed detector pixel convention.  The CDP files define
the origin such that (0,0) is the middle of the lower-left light sensitive pixel
(1024x1024),therefore also need to transform between this science frame and detector frame.

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
10-Oct-2018  Written by David Law (dlaw@stsci.edu)
"""

import os as os
import numpy as np
from astropy.modeling import models
from asdf import AsdfFile
from jwst import datamodels
from jwst.assign_wcs import miri
import pdb

#############################

# Set the relevant FITS distortion file
def get_fitsreffile():
    rootdir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    distfile=os.path.join(rootdir,'data/crds/jwst_miri_distortion_0023.asdf')
    offfile=os.path.join(rootdir,'data/crds/jwst_miri_filteroffset_0003.asdf')

    refs = {"distortion": distfile, "filteroffset": offfile}
   
    return refs

#############################

# Function to convert input X,Y pixel coordinates
# to V2V3 given a filter and the most recent reference files
def xytov2v3model(filter,**kwargs):
    # Construct the reference data model in general JWST imager type
    input_model = datamodels.ImageModel()
    # Set the filter in the data model meta header
    input_model.meta.instrument.filter = filter

    # If passed input refs keyword, unpack and use these reference files
    if ('refs' in kwargs):
        therefs=kwargs['refs']
    # Otherwise use default reference files
    else:
        therefs=get_fitsreffile()

    # Call the pipeline code to make a distortion object given these inputs
    distortion = miri.imaging_distortion(input_model, therefs)
    # Return the distortion object that can then be queried
    return distortion

#############################

# Function to return test data about x,y,v2,v3 locations
# in a variety of filters
def testdata():
    # F770W tests
    v2v3_770=np.array([[-453.363,-374.069],[-480.,-348.],[-450.,-348.],[-420.,-348.],[-390.,-348.],[-480.,-378.],[-450.,-378.],[-420.,-378.],[-390.,-378.],[-480.,-408.],[-450.,-408.],[-420.,-408.],[-390.,-408.]])
    xy_770=np.array([[688.5,511.5],[946.30, 728.95],[677.07,749.13],[408.79,769.19],[138.52,789.59],[924.80,457.09],[655.68,478.39],[387.55,499.49],[117.42,520.46],[904.81,185.52],[635.59,207.87],[367.03,229.95],[96.08,251.45]],dtype=np.float) + [4,0]
    filt_770=['F770W' for i in range(0,v2v3_770.shape[0])]

    # F1800W tests
    v2v3_1800=np.array([[-453.363,-374.069],[-480.,-348.],[-450.,-348.],[-420.,-348.],[-390.,-348.],[-480.,-378.],[-450.,-378.],[-420.,-378.],[-390.,-378.],[-480.,-408.],[-450.,-408.],[-420.,-408.],[-390.,-408.]])
    xy_1800=np.array([[688.10606,512.22995],[945.91,729.68],[676.68,749.86],[408.40,769.92],[138.13,790.32],[924.41,457.82],[655.29,479.12],[387.16,500.22],[117.03,521.19],[904.42,186.25],[635.20,208.60],[366.64,230.68],[95.69,252.18]],dtype=np.float)+[4,0]
    filt_1800=['1800W' for i in range(0,v2v3_1800.shape[0])]

    x=[xy_770[:,0],xy_1800[:,0]]
    y=[xy_770[:,1],xy_1800[:,1]]
    v2=[v2v3_770[:,0],v2v3_1800[:,0]]
    v3=[v2v3_770[:,1],v2v3_1800[:,1]]
    filter=['F770W','F1800W']

    return x,y,v2,v3,filter
