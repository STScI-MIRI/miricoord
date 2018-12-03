#
"""
Useful python tools for working with the MIRI Imager; calls a specific version
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
10-Oct-2018  Written by David Law (dlaw@stsci.edu)
02-Dec-2018  Revise version handling using globals (D. Law)
"""

import os as os
import sys
import math
import numpy as np
from numpy.testing import assert_allclose
import pdb

#############################

# Set the tools version.  Default is CDP-7b3
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
        import miricoord.miricoord.imager.toolversions.mirim_tools_cdp7beta3 as tv
    elif (version == 'cdp7b'):
        import miricoord.miricoord.imager.toolversions.mirim_tools_cdp7beta3 as tv
    elif (version == 'cdp7'):
        import miricoord.miricoord.imager.toolversions.mirim_tools_cdp7 as tv
    else:
        print('Invalid tool version specified!')
        
    return

#############################

# Get the relevant FITS distortion file
def get_fitsreffile():
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')
        
    reffile=tv.get_fitsreffile()
    return reffile

#############################

# Convert v2,v3 in arcsec to xan,yan in arcmin
def v2v3_to_xanyan(v2,v3):
    xan=v2/60.
    yan=-(v3+7.8*60.)/60.
    return xan,yan

#############################

# Convert xan,yan in arcmin to v2,v3 in arcsec
def xanyan_to_v2v3(xan,yan):
    v2=xan*60.
    v3=(-yan-7.8)*60.
    return v2,v3

#############################

# Convert 0-indexed detector pixels to v2,v3 in arcsec
def xytov2v3(x,y,filter):
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')

    v2,v3=tv.xytov2v3(x,y,filter)

    return v2,v3

#############################

# Convert v2,v3 in arcsec to 0-indexed detector pixels
def v2v3toxy(v2,v3,filter):
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')
        
    x,y=tv.v2v3toxy(v2,v3,filter)

    return x,y

#############################

# Convert from Ideal coordinates to v2,v3 coordinates in arcsec
# for a given SIAF aperture using pysiaf
def Idealtov2v3(XIdl,YIdl,apername):
    import pysiaf

    siaf = pysiaf.Siaf('MIRI')
    thisentry=siaf[apername]

    v2ref,v3ref=thisentry.V2Ref,thisentry.V3Ref
    angle,parity=thisentry.V3IdlYAngle,thisentry.VIdlParity
    
    # Per Colin Cox:
    # V2 = V2Ref + VIdlParity*XIdl*cos(a) + YIdl*sin(a)
    # V3 = V3Ref - VIdlParity*XIdl*sin(a) + YIdl*cos(a)

    # Do the math
    rpd=math.pi/180. # Radians per degree
    v2 = v2ref + parity*XIdl*math.cos(angle*rpd) + YIdl*math.sin(angle*rpd)
    v3 = v3ref - parity*XIdl*math.sin(angle*rpd) + YIdl*math.cos(angle*rpd)
    
    return v2,v3

#############################

# Convert from v2,v3 coordinates in arcsec to Ideal coordinates for
# a given SIAF aperture using pysiaf
def v2v3toIdeal(v2,v3,apername,**kwargs):
    import pysiaf

    siaf = pysiaf.Siaf('MIRI')
    thisentry=siaf[apername]

    v2ref,v3ref=thisentry.V2Ref,thisentry.V3Ref
    angle,parity=thisentry.V3IdlYAngle,thisentry.VIdlParity
    
    # Inverting the above equations we get
    # XIdl = VIdlParity*(V2-V2Ref)*cos(a) - VIdlParity*(V3-V3REF)*sin(a)
    # YIdl = (V2-V2Ref)*sin(a) + (V3-V3Ref)*cos(a)

    # Do the math
    rpd=math.pi/180. # Radians per degree
    XIdl = parity*(v2-v2ref)*math.cos(angle*rpd) - parity*(v3-v3ref)*math.sin(angle*rpd)
    YIdl = (v2-v2ref)*math.sin(angle*rpd) + (v3-v3ref)*math.cos(angle*rpd)

    return XIdl,YIdl

#############################

# Test the forward and reverse transforms
def testtransform():
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
        v2new,v3new=xytov2v3(thisx,thisy,thisfilt)
        xnew,ynew=v2v3toxy(thisv2,thisv3,thisfilt)
        # Assert that reference values and newly-created values are close
        assert_allclose(thisx,xnew,atol=0.05)
        assert_allclose(thisy,ynew,atol=0.05)
        assert_allclose(thisv2,v2new,atol=0.05)
        assert_allclose(thisv3,v3new,atol=0.05)
                    
    return
