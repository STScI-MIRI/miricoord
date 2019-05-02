#
"""
Useful python tools for working with the MIRI MRS.  Note that this is the only code
that should be calling thing in toolversions!

This version of the tools uses a standalone implementation
of the distortion solution to do the transformations,
and hooks into offline versions of the FITS reference
files contained within this github repository.

Convert JWST v2,v3 locations (in arcsec) to MIRI MRS SCA x,y pixel locations.
Note that the pipeline uses a 0-indexed detector pixel (1032x1024) convention while
SIAF uses a 1-indexed detector pixel convention.

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
10-Oct-2018  Written by David Law (dlaw@stsci.edu)
07-Dec-2018  Revise version handling using globals (D. Law)
18-Apr-2019  Add slice width and pixel size (D. Law)
"""

import os as os
import sys
import math
import numpy as np
from astropy.io import fits
from numpy.testing import assert_allclose
import miricoord.miricoord.mrs.makesiaf.makesiaf_mrs as makesiaf

import pdb


#############################

# Set the tools version.  Default is CDP-6
def set_toolversion(version):
    # If the toolversion global was already set, delete it
    try:
        del globals()['tv']
    except:
        pass

    # Define toolversion as global scope within mrs_tools
    global tv
    # Import appropriate version
    if (version == 'default'):
        import miricoord.miricoord.mrs.toolversions.mrs_tools_cdp6 as tv
    elif (version == 'cdp6'):
        import miricoord.miricoord.mrs.toolversions.mrs_tools_cdp6 as tv
    elif (version == 'cdp8b'):
        import miricoord.miricoord.mrs.toolversions.mrs_tools_cdp8b as tv
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

# Return the FITS reference file name
def get_fitsreffile(channel):
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')

    value=tv.get_fitsreffile(channel)

    return value


#############################

# Return the average slice width (beta) for a given channel

def slicewidth(channel):
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')

    value=tv.slicewidth(channel)

    return value

#############################

# Return the average pixel size (alpha) for a given channel

def pixsize(channel):
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')

    value=tv.pixsize(channel)

    return value

#############################

# Convert x,y pixels values to alpha,beta,lambda for a given channel.
# Channel must be a single string, e.g. '1A'
# Return is a dictionary with elements
# x,y,alpha,beta,lam,slicenum,slicename
# where x,y are possibly trimmed input values if the trim=0 keyword was set.
# (Trimming is to only detector pixels corresponding to a slice)
# E.g., values=xytoabl([30,40,50],[50,60,70],'1A',trim=1)
def xytoabl(x,y,channel,**kwargs):
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')

    values=tv.xytoabl(x,y,channel,**kwargs)

    return values

#############################

# Convert alpha,beta,lam to x,y pixel values for a given channel.
# Channel must be a single string, e.g. '1A'
# Return is a dictionary with elements
# x,y,alpha,beta,lam,slicenum,slicename,slicephase,pixelphase
# where alpha,beta,lam are possibly trimmed input values if the trim=0 keyword was set.
# (Trimming is to only detector pixels corresponding to a slice)
# E.g., values=xytoabl([30,40,50],[50,60,70],'1A',trim=1)
def abltoxy(alpha,beta,lam,channel,**kwargs):
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')
        
    values=tv.abltoxy(alpha,beta,lam,channel,**kwargs)

    return values

#############################

# Convert from alpha,beta to JWST v2,v3 coordinates
# all coordinates are in arcsec
def abtov2v3(alpha,beta,channel):
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')
        
    v2,v3=tv.abtov2v3(alpha,beta,channel)

    return v2,v3

#############################

# Convert from JWST v2,v3 coordinates to alpha,beta
# all coordinates are in arcsec
def v2v3toab(v2,v3,channel):
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')
    
    alpha,beta=tv.v2v3toab(v2,v3,channel)

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

# Convert v2,v3 locations to xIdeal, yIdeal coordinates
# Note that xIdeal, yIdeal are defined such that xIdeal=-v2 yIdeal=+V3
# with origin at the Ch1A reference point.
def v2v3_to_xyideal(v2,v3):
    values=makesiaf.create_siaf_oneband('1A')
    v2ref,v3ref=values['inscr_v2ref'],values['inscr_v3ref']

    xidl = -(v2-v2ref)
    yidl = v3-v3ref

    return xidl,yidl

#############################

# Test the transforms
def testtransform():
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')

    # Get test data
    refdata=tv.mrs_ref_data

    # Set up a loop over all channels
    channel=['1A','1B','1C','2A','2B','2C','3A','3B','3C','4A','4B','4C']
    nchan=len(channel)
    for i in range(0,nchan):
        print('Testing channel '+channel[i])
        data=refdata[channel[i]]
        thisx,thisy,thisal,thisbe,thislam=data['x'],data['y'],data['alpha'],data['beta'],data['lam']
        if (tv.version() is 'cdp6'):
            thisxan,thisyan=data['xan'],data['yan']
            thisv2,thisv3=xanyan_to_v2v3(thisxan,thisyan)
        else:
            thisv2,thisv3=data['v2'],data['v3']
            
        # Forward transform
        values=xytoabl(thisx,thisy,channel[i])
        newal,newbe,newlam=values['alpha'],values['beta'],values['lam']
        newv2,newv3=abtov2v3(newal,newbe,channel[i])
        # Test equality
        assert_allclose(thisal,newal,atol=0.05)
        assert_allclose(thisbe,newbe,atol=0.05)
        assert_allclose(thislam,newlam,atol=0.05)
        assert_allclose(thisv2,newv2,atol=0.05)
        assert_allclose(thisv3,newv3,atol=0.05)

        # Backward transform
        newal2,newbe2=v2v3toab(newv2,newv3,channel[i])
        values=abltoxy(newal2,newbe2,newlam,channel[i])
        newx,newy=values['x'],values['y']
        # Test equality
        assert_allclose(thisal,newal2,atol=0.05)
        assert_allclose(thisbe,newbe2,atol=0.05)
        assert_allclose(thisx,newx,atol=0.08)
        assert_allclose(thisy,newy,atol=0.08)
    return
