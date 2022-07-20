#
"""
Useful python tools for working with the MIRI LRS; calls a specific version
of the tools specific below.

This version of the tools uses a standalone implementation
of the distortion solution to do the transformations,
and hooks into offline versions of the FITS reference
files contained within this github repository.  When working
with the SIAF files it hooks into pysiaf.

Convert JWST v2,v3 locations (in arcsec) to MIRI Imager SCA x,y pixel locations.
Note that the pipeline uses a 0-indexed pixel convention
while SIAF uses 1-indexed pixels.

By default, calling a function in here will use the default version of the linked
CDP-specific tools.  This can be overridden by calling set_toolversion(version).

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
17-Dec-2018  Written by David Law (dlaw@stsci.edu)
06-Feb-2019  Change default to CDP-7 (D. Law)
19-Jul-2022  Add FLT-1 (D. Law)
"""

import os as os
import sys
import math
import numpy as np
from numpy.testing import assert_allclose
import pdb

#############################

# Set the tools version.  Default is FLT-1
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
        import miricoord.lrs.toolversions.lrs_tools_flt1 as tv
    elif (version == 'flt1'):
        import miricoord.lrs.toolversions.lrs_tools_flt1 as tv
    elif (version == 'cdp7'):
        import miricoord.lrs.toolversions.lrs_tools_cdp7 as tv
    elif (version == 'cdp7beta3'):
        import miricoord.lrs.toolversions.lrs_tools_cdp7beta3 as tv
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

# Get the relevant FITS wavelengths file
def get_fitsreffile():
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')
        
    reffile=tv.get_fitsreffile()
    return reffile

#############################

# Convert 0-indexed subarray pixels to v2,v3,lambda in arcsec and microns
# stype is 'slit' or 'slitless'
def xytov2v3lam(x,y,stype):
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')

    v2,v3,lam=tv.xytov2v3lam(x,y,stype)

    return v2,v3,lam

#############################

# Convert v2,v3 in arcsec to 0-indexed detector pixels
# stype is 'slit' or 'slitless'
def v2v3lamtoxy(v2,v3,lam,stype):
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')
        
    x,y=tv.v2v3lamtoxy(v2,v3,lam,stype)

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
