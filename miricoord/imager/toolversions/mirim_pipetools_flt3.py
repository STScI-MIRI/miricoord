#
"""
Useful python tools for working with the MIRI Imager.
This contains flt3 specific code; it ONLY works with imager, not coronagraphs.

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
    return 'flt3'

#############################

# Set the relevant FITS distortion file
def get_fitsreffile():
    base_dist='data/crds/flt1/jwst_miri_imager_distortion_flt1.asdf'
    base_off='data/crds/flt3/jwst_miri_filteroffset_full_flt3.asdf'
    
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
    v2_770=np.array([-453.37849012, -433.44711881, -479.38722157, -449.37963073,
       -419.36112578, -389.35626367, -479.3919599 , -449.37616406,
       -419.35697586, -389.37064485, -479.40300538, -449.38210468,
       -419.37276534, -389.41978253])
    v3_770=np.array([-373.8105493 , -375.41618152, -348.02002698, -348.01117266,
       -348.00931455, -347.99931539, -378.04217418, -378.03365908,
       -378.02616193, -378.00120274, -408.06008198, -408.04722395,
       -408.03075064, -407.99078552])
    filt_770=['F770W' for i in range(0,len(v2_770))]

    # F1800W tests
    x_1800=np.array([692.5 , 511.5 , 948.18, 676.75, 404.81, 132.65, 923.52, 653.11,
       382.37, 111.34, 899.64, 629.88, 360.  ,  89.77])
    y_1800=np.array([511.5 , 511.5 , 724.94, 745.67, 767.77, 791.34, 455.4 , 476.54,
       498.57, 521.66, 184.81, 206.95, 229.12, 251.55])
    v2_1800=np.array([-453.42852496, -433.49689506, -479.4374696 , -449.4294657 ,
       -419.41049999, -389.40526905, -479.44224734, -449.42616267,
       -419.40657275, -389.41986299, -479.45301681, -449.43208008,
       -419.42252507, -389.46928253])
    v3_1800=np.array([-373.88811164, -375.49365926, -348.09744426, -348.08843891,
       -348.08629818, -348.07597295, -378.11964079, -378.11119868,
       -378.10353636, -378.07825983, -408.13640009, -408.12394963,
       -408.10753408, -408.06736492])
    filt_1800=['F1800W' for i in range(0,len(v2_1800))]

    x=[x_770,x_1800]
    y=[y_770,y_1800]
    v2=[v2_770,v2_1800]
    v3=[v3_770,v3_1800]
    filter=['F770W','F1800W']

    return x,y,v2,v3,filter
