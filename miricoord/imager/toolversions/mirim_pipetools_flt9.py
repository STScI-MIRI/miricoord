#
"""
Useful python tools for working with the MIRI Imager.
This contains flt9 specific code; it ONLY works with imager, not coronagraphs.

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
16-Nov-2022  Port to FLT-2 (D. Law)
16-May-2023  Add FLT-3 coron boresights (D. Law)
23-Jul-2025  Add FLT-9 (D. Law)
"""

import os as os
import numpy as np
import sys
from astropy.modeling import models
from asdf import AsdfFile
from jwst import datamodels
from jwst.assign_wcs import miri
import pdb

#############################

# Return the tools version
def version():
    return 'flt9'

#############################

# Set the relevant FITS distortion file
def get_fitsreffile():
    base_dist='data/crds/flt9/jwst_miri_imager_distortion_flt9.asdf'
    base_off='data/crds/flt9/jwst_miri_filteroffset_full_flt9.asdf'
    
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
# in two filter based on hand computations by DRL.
def testdata():
    # F770W tests, xy are 0-indexed detector pixels
    x_770=np.array([692.5 , 511.5 , 948.18, 676.75, 404.81, 132.65, 923.52, 653.11,
       382.37, 111.34, 899.64, 629.88, 360.  ,  89.77])
    y_770=np.array([511.5 , 511.5 , 724.94, 745.67, 767.77, 791.34, 455.4 , 476.54,
       498.57, 521.66, 184.81, 206.95, 229.12, 251.55])
    v2_770=np.array([-453.34829012, -433.41691881, -479.35702157, -449.34943073,
        -419.33092578, -389.32606367, -479.3617599 , -449.34596406,
        -419.32677586, -389.34044485, -479.37280538, -449.35190468,
        -419.34256534, -389.38958253])
    v3_770=np.array([-373.6679493 , -375.27358152, -347.87742698, -347.86857266,
        -347.86671455, -347.85671539, -377.89957418, -377.89105908,
        -377.88356193, -377.85860274, -407.91748198, -407.90462395,
        -407.88815064, -407.84818552])
    filt_770=['F770W' for i in range(0,len(v2_770))]

    # F1800W tests
    x_1800=np.array([692.5 , 511.5 , 948.18, 676.75, 404.81, 132.65, 923.52, 653.11,
       382.37, 111.34, 899.64, 629.88, 360.  ,  89.77])
    y_1800=np.array([511.5 , 511.5 , 724.94, 745.67, 767.77, 791.34, 455.4 , 476.54,
       498.57, 521.66, 184.81, 206.95, 229.12, 251.55])
    v2_1800=np.array([-453.39832496, -433.46669506, -479.4072696 , -449.3992657 ,
        -419.38029999, -389.37506905, -479.41204734, -449.39596267,
        -419.37637275, -389.38966299, -479.42281681, -449.40188008,
        -419.39232507, -389.43908253])
    v3_1800=np.array([-373.74551164, -375.35105926, -347.95484426, -347.94583891,
        -347.94369818, -347.93337295, -377.97704079, -377.96859868,
        -377.96093636, -377.93565983, -407.99380009, -407.98134963,
        -407.96493408, -407.92476492])
    filt_1800=['F1800W' for i in range(0,len(v2_1800))]

    x=[x_770,x_1800]
    y=[y_770,y_1800]
    v2=[v2_770,v2_1800]
    v3=[v3_770,v3_1800]
    filter=['F770W','F1800W']

    return x,y,v2,v3,filter
