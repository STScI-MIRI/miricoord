#
"""
Useful python tools for working with the MIRI Imager; calls a specific version
of the tools specified below.

This version of the tools hooks into the JWST Calibration
Pipeline code to do the heavy lifting.  Note that this
means performance may be affected by what version of
the pipeline you are running!!  It does, however, use
offline versions of the CRDS reference files contained
within this github repository.

This is mostly useful for testing the pipeline rather than
for creating reference files.

Convert JWST v2,v3 locations (in arcsec) to MIRI Imager SCA x,y pixel locations.
Note that the pipeline uses a 0-indexed pixel convention
while SIAF uses 1-indexed pixels.

By default, calling a function in here will use the default version of the linked
CDP-specific tools.  This can be overridden by calling set_toolversion(version).

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
10-Oct-2018  Written by David Law (dlaw@stsci.edu)
02-Dec-2018  Revise version handling using globals (D. Law)
06-Feb-2019  Set default to CDP-7 (D. Law)
26-May-2022  Add FLT-1 (D. Law)
16-Nov-2022  Add FLT-2 (D. Law)
"""

import os as os
import sys
import numpy as np
from astropy.modeling import models
from asdf import AsdfFile
from jwst import datamodels
from jwst.assign_wcs import miri
from numpy.testing import assert_allclose
import pdb

#############################

# Set the tools version.  Default is FLT-2
def set_toolversion(version):
    # If the toolversion global was already set, delete it
    try:
        del globals()['tv']
    except:
        pass

    # Define toolversion as global scope within mirim_tools
    global tv
    # Import appropriate version
    if (version == 'default'):
        import miricoord.imager.toolversions.mirim_pipetools_flt2 as tv
    elif (version == 'cdp7b'):
        import miricoord.imager.toolversions.mirim_pipetools_cdp7beta3 as tv
    elif (version == 'cdp7'):
        import miricoord.imager.toolversions.mirim_pipetools_cdp7 as tv
    elif (version == 'flt1'):
        import miricoord.imager.toolversions.mirim_pipetools_flt1 as tv
    elif (version == 'flt2'):
        import miricoord.imager.toolversions.mirim_pipetools_flt2 as tv
    else:
        print('Invalid tool version specified!')
        
    return

#############################

# Return the tools version
def version():
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')
        
    return tv.version()

#############################

# Return a model for the detector pixel to v2,v3 distortion
# Note that filter must be a single string
def xytov2v3model(filter,**kwargs):
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')
        
    model=tv.xytov2v3model(filter,**kwargs)

    return model

#############################

# Convert 0-indexed detector pixels to v2,v3 in arcsec using the model
# Note that filter must be a single string
def xytov2v3(x,y,filter,**kwargs):
    model=xytov2v3model(filter,**kwargs)

    v2,v3=model(x,y)

    return v2,v3

#############################

# Convert v2,v3 in arcsec to 0-indexed detector pixels using the model
# Note that filter must be a single string
def v2v3toxy(v2,v3,filter,**kwargs):
    model=xytov2v3model(filter,**kwargs)
    
    x,y=model.inverse(v2,v3)

    return x,y

#############################

# Test the forward and reverse transforms
def testtransform(**kwargs):
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')
        
    # Get test data from a generating function
    x,y,v2,v3,filter=tv.testdata()

    nfilter=len(filter)
    # Loop over the different filters of test data
    for i in range(0,nfilter):
        thisx,thisy,thisv2,thisv3,thisfilt=x[i],y[i],v2[i],v3[i],filter[i]
        v2new,v3new=xytov2v3(thisx,thisy,thisfilt,**kwargs)
        xnew,ynew=v2v3toxy(thisv2,thisv3,thisfilt,**kwargs)
        # Assert that reference values and newly-created values are close
        assert_allclose(thisx,xnew,atol=0.05)
        assert_allclose(thisy,ynew,atol=0.05)
        assert_allclose(thisv2,v2new,atol=0.05)
        assert_allclose(thisv3,v3new,atol=0.05)
                    
    return
