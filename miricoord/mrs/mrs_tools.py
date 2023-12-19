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
04-Dec-2020  Add reverse transform from ideal to v2v3 (D. Law)
04-Apr-2022  Start adding flt1 model (D. Law)
27-May-2022  Add FLT-1 (D. Law)
14-Jun-2022  Add FLT-2 (D. Law)
01-Aug-2022  Add FLT-3 (D. Law)
25-Aug-2022  Add FLT-4 (D. Law)
13-Mar-2023  Add FLT-5 (D. Law)
31-May-2023  Add FLT-6 (D. Law)
30-Aug-2023  Add FLT-7 (D. Law)
06-Dec-2023  Add FLT-8 (D. Law)
"""

import os as os
import sys
import math
import numpy as np
from astropy.io import fits
from numpy.testing import assert_allclose

import spherical_geometry.polygon as poly

import pdb


#############################

# Set the tools version.  Default is flt8
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
        import miricoord.mrs.toolversions.mrs_tools_flt8 as tv
    elif (version == 'flt8'):
        import miricoord.mrs.toolversions.mrs_tools_flt8 as tv
    elif (version == 'flt7'):
        import miricoord.mrs.toolversions.mrs_tools_flt7 as tv
    elif (version == 'flt6'):
        import miricoord.mrs.toolversions.mrs_tools_flt6 as tv
    elif (version == 'flt5'):
        import miricoord.mrs.toolversions.mrs_tools_flt5 as tv
    elif (version == 'flt4'):
        import miricoord.mrs.toolversions.mrs_tools_flt4 as tv
    elif (version == 'flt3'):
        import miricoord.mrs.toolversions.mrs_tools_flt3 as tv
    elif (version == 'flt2'):
        import miricoord.mrs.toolversions.mrs_tools_flt2 as tv
    elif (version == 'flt1'):
        import miricoord.mrs.toolversions.mrs_tools_flt1 as tv
    elif (version == 'cdp6'):
        import miricoord.mrs.toolversions.mrs_tools_cdp6 as tv
    elif (version == 'cdp8b'):
        import miricoord.mrs.toolversions.mrs_tools_cdp8b as tv
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

# Return the alpha min/max for all slices in given channel

def alphafov(channel):
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')

    value=tv.alphafov(channel)

    return value

#############################

# Create an image in a given channel where all pixels have
# the value of their wavelength.  Specifying loc='cen','lo', or 'hi'
# gives values at center, or low/high values for pixel

def waveimage(channel,loc='cen', **kwargs):
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')

    # Define 0-indexed base x and y pixel number (1032x1024 grid)
    basex,basey = np.meshgrid(np.arange(1032),np.arange(1024))

    # Account for the fact that lambda increases up for Ch1/2, and down for Ch 3/4
    if (loc == 'cen'):
        usey=basey
    if (loc == 'lo'):
        if ((channel == '1A')or(channel == '1B')or(channel == '1C')):
            usey=basey-0.4999
        if ((channel == '2A')or(channel == '2B')or(channel == '2C')):
            usey=basey-0.4999
        if ((channel == '3A')or(channel == '3B')or(channel == '3C')):
            usey=basey+0.4999
        if ((channel == '4A')or(channel == '4B')or(channel == '4C')):
            usey=basey+0.4999
    if (loc == 'hi'):
        if ((channel == '1A')or(channel == '1B')or(channel == '1C')):
            usey=basey+0.4999
        if ((channel == '2A')or(channel == '2B')or(channel == '2C')):
            usey=basey+0.4999
        if ((channel == '3A')or(channel == '3B')or(channel == '3C')):
            usey=basey-0.4999
        if ((channel == '4A')or(channel == '4B')or(channel == '4C')):
            usey=basey-0.4999
    
    # Convert to 1d vectors
    basex=basex.reshape(-1)
    basey=basey.reshape(-1)
    usey=usey.reshape(-1)
    
    # Convert to base alpha,beta,lambda at pixel center
    values=xytoabl(basex,usey,channel,**kwargs)
    basealpha,basebeta=values['alpha'],values['beta']
    baselambda,slicenum=values['lam'],values['slicenum']

    # Crop to only pixels on a real slice for this channel
    index0=np.where(basealpha > -50)
    
    basex,basey,usey=basex[index0],basey[index0],usey[index0]
    basealpha,basebeta=basealpha[index0],basebeta[index0]
    baselambda,slicenum=baselambda[index0],slicenum[index0]

    mockimg=np.zeros([1024,1032])
    npix=len(baselambda)

    for jj in range(0,npix):
        mockimg[basey[jj],basex[jj]]=baselambda[jj]

    return mockimg

#############################

# Create an image in a given channel where all pixels have
# the value of their alpha.  Specifying loc='cen','lo', or 'hi'
# gives values at center, or low/high values for pixel

def alphaimage(channel,loc='cen'):
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')

    # Define 0-indexed base x and y pixel number (1032x1024 grid)
    basex,basey = np.meshgrid(np.arange(1032),np.arange(1024))

    # Account for the fact that alpha increases right for Ch1/4, and left for Ch 2/3
    if (loc == 'cen'):
        usex=basex
    if (loc == 'lo'):
        if ((channel == '1A')or(channel == '1B')or(channel == '1C')):
            usex=basex-0.4999
        if ((channel == '2A')or(channel == '2B')or(channel == '2C')):
            usex=basex+0.4999
        if ((channel == '3A')or(channel == '3B')or(channel == '3C')):
            usex=basex+0.4999
        if ((channel == '4A')or(channel == '4B')or(channel == '4C')):
            usex=basex-0.4999
    if (loc == 'hi'):
        if ((channel == '1A')or(channel == '1B')or(channel == '1C')):
            usex=basex+0.4999
        if ((channel == '2A')or(channel == '2B')or(channel == '2C')):
            usex=basex-0.4999
        if ((channel == '3A')or(channel == '3B')or(channel == '3C')):
            usex=basex-0.4999
        if ((channel == '4A')or(channel == '4B')or(channel == '4C')):
            usex=basex+0.4999

    # Convert to 1d vectors
    basex=basex.reshape(-1)
    basey=basey.reshape(-1)
    usex=usex.reshape(-1)
    
    # Convert to base alpha,beta,lambda at pixel center
    values=xytoabl(usex,basey,channel)
    basealpha,basebeta=values['alpha'],values['beta']
    baselambda,slicenum=values['lam'],values['slicenum']

    # Crop to only pixels on a real slice for this channel
    index0=np.where(basealpha > -50)
    basex,basey,usex=basex[index0],basey[index0],usex[index0]
    basealpha,basebeta=basealpha[index0],basebeta[index0]
    baselambda,slicenum=baselambda[index0],slicenum[index0]

    mockimg=np.zeros([1024,1032])
    mockimg[:,:]=-100
    npix=len(baselambda)
    for jj in range(0,npix):
        mockimg[basey[jj],basex[jj]]=basealpha[jj]
        
    return mockimg

#############################

# Create an image in a given channel where all pixels have
# the value of their beta.

def betaimage(channel):
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')

    # Define 0-indexed base x and y pixel number (1032x1024 grid)
    basex,basey = np.meshgrid(np.arange(1032),np.arange(1024))
    
    # Convert to 1d vectors
    basex=basex.reshape(-1)
    basey=basey.reshape(-1)
    
    # Convert to base alpha,beta,lambda at pixel center
    values=xytoabl(basex,basey,channel)
    basealpha,basebeta=values['alpha'],values['beta']
    baselambda,slicenum=values['lam'],values['slicenum']

    # Crop to only pixels on a real slice for this channel
    index0=np.where(basealpha > -50)
    basex,basey=basex[index0],basey[index0]
    basealpha,basebeta=basealpha[index0],basebeta[index0]
    baselambda,slicenum=baselambda[index0],slicenum[index0]

    mockimg=np.zeros([1024,1032])
    mockimg[:,:]=-100
    npix=len(baselambda)
    for jj in range(0,npix):
        mockimg[basey[jj],basex[jj]]=basebeta[jj]
        
    return mockimg

#############################

# Create an image in a given channel of the pixel area in arcsec.
# Either in alpha-beta or v2-v3 frame.

def pixarea(band,frame='ab', **kwargs):
    # Determine whether the CDP toolversion has been set.  If not, set to default.
    try:
        sys.getrefcount(tv)
    except:
        set_toolversion('default')

    # Define 0-indexed base x and y pixel number (1032x1024 grid)
    basex,basey = np.meshgrid(np.arange(1032),np.arange(1024))
    
    # Convert to 1d vectors
    basex=basex.reshape(-1)
    basey=basey.reshape(-1)
    
    # Convert to base alpha,beta,lambda at pixel center
    values=xytoabl(basex,basey,band,**kwargs)
    alpha,beta=values['alpha'],values['beta']
    lam,slicenum=values['lam'],values['slicenum']

    # Crop to only pixels on a real slice for this channel
    index0=np.where(alpha > -50)
    basex,basey=basex[index0],basey[index0]
    alpha,beta=alpha[index0],beta[index0]
    lam,slicenum=lam[index0],slicenum[index0]

    # Define the slice width
    swidth=slicewidth(band)
    
    # Convert to  alpha,beta,lambda at pixel lower-left
    valuesll=xytoabl(basex-0.4999,basey-0.4999,band,**kwargs)
    alphall,betall=valuesll['alpha'],valuesll['beta']-swidth/2.
    # Convert to  alpha,beta,lambda at pixel upper-right
    valuesur=xytoabl(basex+0.4999,basey+0.4999,band,**kwargs)
    alphaur,betaur=valuesur['alpha'],valuesur['beta']+swidth/2.
    # Convert to  alpha,beta,lambda at pixel upper-left
    valuesul=xytoabl(basex-0.4999,basey+0.4999,band,**kwargs)
    alphaul,betaul=valuesul['alpha'],valuesul['beta']+swidth/2.
    # Convert to  alpha,beta,lambda at pixel lower-right
    valueslr=xytoabl(basex+0.4999,basey-0.4999,band,**kwargs)
    alphalr,betalr=valueslr['alpha'],valueslr['beta']-swidth/2.

    # Convert to v2,v3 at pixel corners
    v2ll,v3ll=abtov2v3(alphall,betall,band)
    v2lr,v3lr=abtov2v3(alphalr,betalr,band)    
    v2ur,v3ur=abtov2v3(alphaur,betaur,band)
    v2ul,v3ul=abtov2v3(alphaul,betaul,band)
     
    npix=len(basex)
    pixarea=np.zeros(npix)
    arcsec_per_sterad=(180/np.pi)*(180/np.pi)*3600*3600
    for ii in range(0,npix):
        if (np.mod(ii,np.round(npix/20)) == 0):
            print('Working: ',int(ii/npix*100),'% complete')
        if (frame == 'ab'):
            vector1=np.array([alphall[ii],alphalr[ii],alphaur[ii],alphaul[ii],alphall[ii]])/3600.
            vector2=np.array([betall[ii],betalr[ii],betaur[ii],betaul[ii],betall[ii]])/3600.
        if (frame == 'v2v3'):
            vector1=np.array([v2ll[ii],v2lr[ii],v2ur[ii],v2ul[ii],v2ll[ii]])/3600.
            vector2=np.array([v3ll[ii],v3lr[ii],v3ur[ii],v3ul[ii],v3ll[ii]])/3600.
            
        thepix=poly.SphericalPolygon.from_radec(vector1,vector2,degrees=True)
        pixarea[ii]=thepix.area()*arcsec_per_sterad
        
    mockimg=np.zeros([1024,1032])
    for jj in range(0,npix):
        mockimg[basey[jj],basex[jj]]=pixarea[jj]
        
    return mockimg 

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
#
# Allow a forced override of a given SIAF structure.  This saves a ton
# of time if making many calls to this code.  It should ALWAYS be
# passed the 1A structure though!

def v2v3_to_xyideal(v2,v3,**kwargs):
    if ('siaf1A' in kwargs):
        siaf1A=kwargs['siaf1A']
    else:
        # Import locally to this function so that pysiaf isn't required for everything in mrs_tools
        import miricoord.mrs.makesiaf.makesiaf_mrs as makesiaf
        siaf1A=makesiaf.create_siaf_oneband('1A')
        
    v2ref,v3ref=siaf1A['inscr_v2ref'],siaf1A['inscr_v3ref']

    xidl = -(v2-v2ref)
    yidl = v3-v3ref

    return xidl,yidl

#############################

# Convert xIdeal, yIdeal coordinates to v2,v3
# Note that xIdeal, yIdeal are defined such that xIdeal=-v2 yIdeal=+V3
# with origin at the Ch1A reference point.
#
# Allow a forced override of a given SIAF structure.  This saves a ton
# of time if making many calls to this code.  It should ALWAYS be
# passed the 1A structure though!

def xyideal_to_v2v3(xi,yi,**kwargs):
    if ('siaf1A' in kwargs):
        siaf1A=kwargs['siaf1A']
    else:
        # Import locally to this function so that pysiaf isn't required for everything in mrs_tools
        import miricoord.mrs.makesiaf.makesiaf_mrs as makesiaf
        siaf1A=makesiaf.create_siaf_oneband('1A')
        
    v2ref,v3ref=siaf1A['inscr_v2ref'],siaf1A['inscr_v3ref']

    v2 = -(xi - v2ref)
    v3 = yi + v3ref
    
    return v2,v3

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
        thisx,thisy=data['x'],data['y']
        thissl,thisal,thisbe,thislam=data['s'],data['alpha'],data['beta'],data['lam']
        if (tv.version() == 'cdp6'):
            thisxan,thisyan=data['xan'],data['yan']
            thisv2,thisv3=xanyan_to_v2v3(thisxan,thisyan)
        else:
            thisv2,thisv3=data['v2'],data['v3']
            
        # Forward transform
        values=xytoabl(thisx,thisy,channel[i])
        newsl,newal,newbe,newlam=values['slicenum'],values['alpha'],values['beta'],values['lam']
        newv2,newv3=abtov2v3(newal,newbe,channel[i])
        # Test equality
        assert_allclose(thissl,newsl,atol=0.05)
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
