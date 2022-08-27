#
"""
Useful python tools for working with the MIRI MRS.

This version of the tools uses the JWST pipeline implementation
of the distortion solution to do the transformations,
and hooks into offline versions of the CRDS reference
files contained within this github repository.

Convert JWST v2,v3 locations (in arcsec) to MIRI MRS SCA x,y pixel locations.
Note that the pipeline uses a 0-indexed detector pixel (1032x1024) convention while
SIAF uses a 1-indexed detector pixel convention.

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
10-Oct-2018  Written by David Law (dlaw@stsci.edu)
27-May-2022  Add FLT-1 (D. Law)
14-Jun-2022  Add FLT-2 (D. Law)
01-Aug-2022  Add FLT-3 (D. Law)
27-Aug-2022  Add FLT-4 (D. Law)
"""

import os as os
import numpy as np
import sys
import pdb as pdb
from astropy.modeling import models
from asdf import AsdfFile
from jwst import datamodels
from jwst.assign_wcs import miri
from numpy.testing import assert_allclose

#############################

# Set the tools version.  Default is FLT-4
def set_toolversion(version):
    # If the toolversion global was already set, delete it
    try:
        del globals()['tv']
    except:
        pass

    # Define toolversion as global scope within mrs_pipetools
    global tv
    # Import appropriate version
    if (version == 'default'):
        import miricoord.mrs.toolversions.mrs_pipetools_flt4 as tv
    elif (version == 'cdp6'):
        import miricoord.mrs.toolversions.mrs_pipetools_cdp6 as tv
    elif (version == 'cdp8b'):
        import miricoord.mrs.toolversions.mrs_pipetools_cdp8b as tv
    elif (version == 'flt1'):
        import miricoord.mrs.toolversions.mrs_pipetools_flt1 as tv
    elif (version == 'flt2'):
        import miricoord.mrs.toolversions.mrs_pipetools_flt2 as tv
    elif (version == 'flt3'):
        import miricoord.mrs.toolversions.mrs_pipetools_flt3 as tv
    elif (version == 'flt4'):
        import miricoord.mrs.toolversions.mrs_pipetools_flt4 as tv
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

# Convenience function to turn '1A' type name into '12' and 'SHORT' type names
def bandchan(channel):
    # Channel should be of the form (e.g.) '1A', '3C', etc
    if ((channel == '1A')or(channel == '2A')):
       newband='SHORT'
       newchannel='12'
    elif ((channel == '3A')or(channel == '4A')):
       newband='SHORT'
       newchannel='34'
    elif ((channel == '1B')or(channel == '2B')):
       newband='MEDIUM'
       newchannel='12'
    elif ((channel == '3B')or(channel == '4B')):
       newband='MEDIUM'
       newchannel='34'
    elif ((channel == '1C')or(channel == '2C')):
       newband='LONG'
       newchannel='12'
    elif ((channel == '3C')or(channel == '4C')):
       newband='LONG'
       newchannel='34'
    else:
       newband='FAIL'
       newchannel='FAIL'

    return newband,newchannel

#############################

# Convenience function to turn '12A' type name into '1A' and '2A' type names
def channel(detband):
    if (detband == '12A'):
       ch1='1A'
       ch2='2A'
    elif (detband == '12B'):
       ch1='1B'
       ch2='2B'
    elif (detband == '12C'):
       ch1='1C'
       ch2='2C'
    elif (detband == '34A'):
       ch1='3A'
       ch2='4A'
    elif (detband == '34B'):
       ch1='3B'
       ch2='4B'
    elif (detband == '34C'):
       ch1='3C'
       ch2='4C'
    else:
       ch1='FAIL'
       ch2='FAIL'

    return ch1,ch2

#############################

# Convenience function to read a header and return detband (e.g., '12A')
def hdr_to_detband(hdr):
    channel = hdr['CHANNEL']
    band = hdr['BAND']

    if ((channel == '12')&(band == 'SHORT')):
        detband = '12A'
    if ((channel == '12')&(band == 'MEDIUM')):
        detband = '12B'        
    if ((channel == '12')&(band == 'LONG')):
        detband = '12C'

    if ((channel == '34')&(band == 'SHORT')):
        detband = '34A'
    if ((channel == '34')&(band == 'MEDIUM')):
        detband = '34B'        
    if ((channel == '34')&(band == 'LONG')):
        detband = '34C'

    return detband
        
#############################

# Convenience function to return the rough middle wavelength of a given channel
# Note that this ISNT exact, just some valid value
def midwave(channel):
    if (channel == '1A'):
       thewave=5.32
    elif (channel == '1B'):
       thewave=6.145
    elif (channel == '1C'):
       thewave=7.09
    elif (channel == '2A'):
       thewave=8.135
    elif (channel == '2B'):
       thewave=9.395
    elif (channel == '2C'):
       thewave=10.85
    elif (channel == '3A'):
       thewave=12.505
    elif (channel == '3B'):
       thewave=14.5
    elif (channel == '3C'):
       thewave=16.745
    elif (channel == '4A'):
       thewave=19.29
    elif (channel == '4B'):
       thewave=22.47
    elif (channel == '4C'):
       thewave=26.2

    return thewave

#############################

# Convenience function to return model distortion object
# for the x,y to alpha,beta,lam transform
def xytoablmodel(channel,**kwargs):
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')

    model=tv.xytoablmodel(channel,**kwargs)

    return model

#############################

# Convert x,y pixel values to alpha,beta,lam for a given channel.
# Channel must be a single string, e.g. '1A'
# Return alpha,beta,lambda
def xytoabl(x,y,channel,**kwargs):
    model=xytoablmodel(channel,**kwargs)

    alpha,beta,lam=model(x,y)

    return alpha,beta,lam

#############################

# Convert alpha,beta,lam to x,y pixel values for a given channel.
# Channel must be a single string, e.g. '1A'
# Return x,y
def abltoxy(alpha,beta,lam,channel,**kwargs):
    model=xytoablmodel(channel,**kwargs)

    x,y=model.inverse(alpha,beta,lam)

    return x,y

#############################

# Convenience function to return model distortion object
# for the alpha,beta to v2,v3 transform
def abtov2v3model(channel,**kwargs):
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')
    
    model=tv.abtov2v3model(channel,**kwargs)

    return model

#############################

# Convert from alpha,beta to JWST v2,v3 coordinates
# all coordinates are in arcsec
def abtov2v3(alpha,beta,channel,**kwargs):
    model=abtov2v3model(channel,**kwargs)
    
    v2,v3=model(alpha,beta)

    return v2,v3

#############################

# Convert from JWST v2,v3 coordinates to alpha,beta
# all coordinates are in arcsec
def v2v3toab(v2,v3,channel,**kwargs):
    model=abtov2v3model(channel,**kwargs)

    alpha,beta=model.inverse(v2,v3)

    return alpha,beta

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

# Test the transforms
def testtransform(**kwargs):
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')

    # Get test data
    refdata=tv.mrs_ref_data

    # If passed input channels keyword, test only those channels
    if ('testchannel' in kwargs):
        channel=kwargs['testchannel']
    # Otherwise test all channels
    else:
        channel=['1A','1B','1C','2A','2B','2C','3A','3B','3C','4A','4B','4C']

    # Set up a loop over channels
    nchan=len(channel)
    for i in range(0,nchan):
        print('Testing channel '+channel[i])
        data=refdata[channel[i]]
        thisx,thisy,thisal,thisbe,thislam=data['x'],data['y'],data['alpha'],data['beta'],data['lam']
        if (tv.version() == 'cdp6'):
            thisxan,thisyan=data['xan'],data['yan']
            thisv2,thisv3=xanyan_to_v2v3(thisxan,thisyan)
        else:
            thisv2,thisv3=data['v2'],data['v3']
        
        # Forward transform
        newal,newbe,newlam=xytoabl(thisx,thisy,channel[i],**kwargs)
        newv2,newv3=abtov2v3(newal,newbe,channel[i],**kwargs)
        # Test equality
        assert_allclose(thisal,newal,atol=0.05)
        assert_allclose(thisbe,newbe,atol=0.05)
        assert_allclose(thislam,newlam,atol=0.05)
        assert_allclose(thisv2,newv2,atol=0.05)
        assert_allclose(thisv3,newv3,atol=0.05)

        # Backward transform
        newal2,newbe2=v2v3toab(newv2,newv3,channel[i],**kwargs)
        newx,newy=abltoxy(newal2,newbe2,newlam,channel[i],**kwargs)
        # Test equality
        assert_allclose(thisal,newal2,atol=0.05)
        assert_allclose(thisbe,newbe2,atol=0.05)
        assert_allclose(thisx,newx,atol=0.08)
        assert_allclose(thisy,newy,atol=0.08)

    return
