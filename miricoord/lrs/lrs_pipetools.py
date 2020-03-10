#
"""
Useful python tools for working with the MIRI LRS; calls a specific version
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
17-Dec-2018  Written by David Law (dlaw@stsci.edu)
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

# Set the tools version.  Default is CDP-7 (there is no CDP-7b)
def set_toolversion(version):
    # If the toolversion global was already set, delete it
    try:
        del globals()['tv']
    except:
        pass

    # Define toolversion as global scope within lrs_tools
    global tv
    # Import appropriate version
    if (version == 'default'):
        import miricoord.lrs.toolversions.lrs_pipetools_cdp7 as tv
    elif (version == 'cdp7'):
        import miricoord.lrs.toolversions.lrs_pipetools_cdp7 as tv
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

# Return a model for the detector pixel to v2,v3,lambda distortion
# Note that stype must be a single string (slit or slitless)
def xytov2v3lam_model(stype,**kwargs):
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')
        
    model=tv.xytov2v3lam_model(stype,**kwargs)

    return model

#############################

# Convert 0-indexed subarray pixels to v2,v3 in arcsec using the model
# Note that stype must be a single string (slit or slitless)
def xytov2v3lam(x,y,stype,**kwargs):
    model=xytov2v3lam_model(stype,**kwargs)

    v2,v3,lam=model(x,y)

    return v2,v3,lam

#############################

# Convert v2,v3,lambda in arcsec to 0-indexed subarray pixels using the model
# Note that stype must be a single string (slit or slitless)
def v2v3lamtoxy(v2,v3,lam,stype,**kwargs):
    model=xytov2v3lam_model(stype,**kwargs)
    
    x,y=model.inverse(v2,v3,lam)

    return x,y

#############################

# Test the forward and reverse transforms
def testtransform():
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')

    # Get test data from a generating function
    x,y,v2,v3,lam,stype=tv.testdata()

    ntype=len(stype)
    # Loop over the slit and slitless varieties of test data
    for i in range(0,ntype):
        thisx,thisy,thisv2,thisv3,thislam,thisstype=x[i],y[i],v2[i],v3[i],lam[i],stype[i]
        v2new,v3new,lamnew=xytov2v3lam(thisx,thisy,thisstype)
        xnew,ynew=v2v3lamtoxy(thisv2,thisv3,thislam,thisstype)

        # Assert that reference values and newly-created values are close
        assert_allclose(thisx,xnew,atol=0.05)
        assert_allclose(thisy,ynew,atol=0.05)
        assert_allclose(thisv2,v2new,atol=0.05)
        assert_allclose(thisv3,v3new,atol=0.05)
        assert_allclose(thislam,lamnew,atol=0.05)
    
    return
