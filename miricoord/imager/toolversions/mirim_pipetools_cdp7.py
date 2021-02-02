#
"""
Useful python tools for working with the MIRI Imager.
This contains cdp7 specific code.

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
02-Dec-2018  Written by David Law (dlaw@stsci.edu)
"""

import os as os
import numpy as np
from astropy.modeling import models
from asdf import AsdfFile
from jwst import datamodels
from jwst.assign_wcs import miri
import pdb

#############################

# Return the tools version
def version():
    return 'cdp7'

#############################

# Set the relevant FITS distortion file
def get_fitsreffile():
    base_dist='data/crds/jwst_miri_distortion_0028.asdf'
    base_off='data/crds/jwst_miri_filteroffset_0004.asdf'
    
    # Try looking for the file in the expected location
    rootdir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    distfile=os.path.join(rootdir,base_dist)
    offfile=os.path.join(rootdir,base_off)
    refs = {"distortion": distfile, "filteroffset": offfile}
    if os.path.exists(distfile):
        return refs
    
    # If that didn't work, look in the system path
    rootdir=sys.prefix
    distfile=os.path.join(rootdir,base_dist)
    offfile=os.path.join(rootdir,base_off)
    refs = {"distortion": distfile, "filteroffset": offfile}
    if os.path.exists(distfile):
        return refs

    # If that didn't work either, just return what we've got
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
    v2v3_770=np.array([[-415.069,-400.576],[-453.559,-373.814],[-434.083,-375.388],[-480.,-348.],[-450.,-348.],[-420.,-348.],[-390.,-348.],[-480.,-378.],[-450.,-378.],[-420,-378.],[-390.,-378.],[-480.,-408.],[-450.,-408.],[-420.,-408.],[-390.,-408.]])
    xy_770=np.array([[321.13,299.7],[688.5,511.5],[511.5,511.5],[948.18,724.94],[676.75,745.67],[404.81,767.77],[132.65,791.34],[923.52,455.40],[653.11,476.53],[382.37,498.57],[111.34,521.66],[899.64,184.81],[629.88,206.95],[360.00,229.12],[89.77,251.55]],dtype=np.float) + [4,0]
    # Note that we had to add 4 to Alistair's x locations because he uses science pixels, not detector pixels
    filt_770=['F770W' for i in range(0,v2v3_770.shape[0])]

    # F1800W tests
    v2v3_1800=np.array([[-480.,-348.],[-450.,-348.],[-420.,-348.],[-390.,-348.],[-480.,-378.],[-450.,-378.],[-420,-378.],[-390.,-378.],[-480.,-408.],[-450.,-408.],[-420.,-408.],[-390.,-408.]])
    xy_1800=np.array([[947.79,725.67],[676.36,746.40],[404.42,768.50],[132.26,792.07],[923.13,456.13],[652.72,477.26],[381.98,499.30],[110.95,522.39],[899.25,185.54],[629.49,207.68],[359.61,229.85],[89.38,252.28]],dtype=np.float) + [4,0]
    filt_1800=['F1800W' for i in range(0,v2v3_1800.shape[0])]

    x=[xy_770[:,0],xy_1800[:,0]]
    y=[xy_770[:,1],xy_1800[:,1]]
    v2=[v2v3_770[:,0],v2v3_1800[:,0]]
    v3=[v2v3_770[:,1],v2v3_1800[:,1]]
    filter=['F770W','F1800W']

    return x,y,v2,v3,filter

