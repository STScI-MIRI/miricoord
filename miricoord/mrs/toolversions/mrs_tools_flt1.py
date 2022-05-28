#
"""
Useful python tools for working with the MIRI MRS.
This contains flt1 specific code.

This version of the tools uses a standalone implementation
of the distortion solution to do the transformations,
and hooks into offline versions of the FITS reference
files contained within this github repository.

Convert JWST v2,v3 locations (in arcsec) to MIRI MRS SCA x,y pixel locations.
Note that the pipeline uses a 0-indexed detector pixel (1032x1024) convention while
SIAF uses a 1-indexed detector pixel convention.  The CDP files define
the origin such that (0,0) is the middle of the lower-left pixel
(1032x1024)- note that this is a CHANGE of convention from earlier CDP!

Since not all detector pixels actually map to alpha-beta (since some pixels are between slices)
these have alpha=beta=lambda=-999 and can be trimmed using 'trim=1'

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
26-Apr-2019  Written by David Law (dlaw@stsci.edu)
26-Mar-2020  Add computation of wavelength pixel phase (D. Law)
04-Apr-2022  Initial work for flight distortion model (D. Law)
27-May-2022  Real flight distortion model FLT-1 (D. Law)
"""

import os as os
import math
import sys
import numpy as np
from numpy import matlib as mb
from astropy.io import fits
import pdb

#############################

# Return the tools version
def version():
    return 'flt1'

#############################

# Set the relevant FITS distortion file based on channel (e.g., '1A')
def get_fitsreffile(channel):
    if (channel == '1A'):
        file='MIRI_FM_MIRIFUSHORT_12SHORT_DISTORTION_9B.05.00.fits'
    elif (channel == '1B'):
        file='MIRI_FM_MIRIFUSHORT_12MEDIUM_DISTORTION_9B.05.00.fits'
    elif (channel == '1C'):
        file='MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_9B.05.00.fits'
    elif (channel == '2A'):
        file='MIRI_FM_MIRIFUSHORT_12SHORT_DISTORTION_9B.05.00.fits'
    elif (channel == '2B'):
        file='MIRI_FM_MIRIFUSHORT_12MEDIUM_DISTORTION_9B.05.00.fits'
    elif (channel == '2C'):
        file='MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_9B.05.00.fits'
    elif (channel == '3A'):
        file='MIRI_FM_MIRIFULONG_34SHORT_DISTORTION_9B.05.00.fits'
    elif (channel == '3B'):
        file='MIRI_FM_MIRIFULONG_34MEDIUM_DISTORTION_9B.05.00.fits'
    elif (channel == '3C'):
        file='MIRI_FM_MIRIFULONG_34LONG_DISTORTION_9B.05.00.fits'
    elif (channel == '4A'):
        file='MIRI_FM_MIRIFULONG_34SHORT_DISTORTION_9B.05.00.fits'
    elif (channel == '4B'):
        file='MIRI_FM_MIRIFULONG_34MEDIUM_DISTORTION_9B.05.00.fits'
    elif (channel == '4C'):
        file='MIRI_FM_MIRIFULONG_34LONG_DISTORTION_9B.05.00.fits'

    # Try looking for the file in the expected location
    rootdir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    rootdir=os.path.join(rootdir,'data/fits/flt1/')
    reffile=os.path.join(rootdir,file)

    if os.path.exists(reffile):
        return reffile
    
    # If that didn't work, look in the system path
    rootdir=sys.prefix
    rootdir=os.path.join(rootdir,'data/fits/flt1/')
    reffile=os.path.join(rootdir,file)
    if os.path.exists(reffile):
        return reffile    

    # If that didn't work either, just return what we've got
    return reffile

#############################

# Return the average slice width (beta) for a given channel

def slicewidth(channel):
    # Open relevant distortion file
    distfile=fits.open(get_fitsreffile(channel))
    
    # Read global header
    hdr=distfile[0].header

    if ((channel == '1A')or(channel == '1B')or(channel == '1C')):
        value=hdr['B_DEL1']
    if ((channel == '2A')or(channel == '2B')or(channel == '2C')):
        value=hdr['B_DEL2']
    if ((channel == '3A')or(channel == '3B')or(channel == '3C')):
        value=hdr['B_DEL3']
    if ((channel == '4A')or(channel == '4B')or(channel == '4C')):
        value=hdr['B_DEL4']
    
    return value

#############################

# Return the average pixel size (alpha) for a given channel.  Do
# this by computing dalpha for every pixel in the channel.

def pixsize(channel):
    # Define where this channel is on the detector
    ymin,ymax=0,1024
    if ((channel == '1A')or(channel == '1B')or(channel == '1C')):
        xmin,xmax=0,512
    if ((channel == '2A')or(channel == '2B')or(channel == '2C')):
        xmin,xmax=513,1031
    if ((channel == '3A')or(channel == '3B')or(channel == '3C')):
        xmin,xmax=513,1031
    if ((channel == '4A')or(channel == '4B')or(channel == '4C')):
        xmin,xmax=0,492

    xrow=np.mgrid[xmin:xmax]
    yrow=np.mgrid[ymin:ymax]
    xall=mb.repmat(xrow,yrow.size,1)*1.
    yall=mb.repmat(yrow,xrow.size,1)*1.
    yall=np.transpose(yall)
    # Recast as 1d arrays
    xall=xall.reshape(-1)
    yall=yall.reshape(-1)

    # Convert a list of all pixels, and a list of all pixels offset
    # by one in x to alpha/beta.
    values1=xytoabl(xall,yall,channel)
    values2=xytoabl(xall+1,yall,channel)
    # Look for where both pixel and offset pixels had defined alpha
    indx=(np.where((values1['slicenum'] > 0) & (values2['slicenum'] > 0)))[0]
    # Crop to these indices and look at the difference in alpha
    alpha1=values1['alpha']
    alpha2=values2['alpha']
    alpha1=alpha1[indx]
    alpha2=alpha2[indx]
    da=np.abs(alpha1-alpha2)

    return np.median(da)

#############################

# Compute the alpha min/max values for all slices in a given band
# Do this by computing alpha at left/right pixel edges

def alphafov(channel):
    # Define where this channel is on the detector
    ymin,ymax=0,1024
    if ((channel == '1A')or(channel == '1B')or(channel == '1C')):
        xmin,xmax=0,512
    if ((channel == '2A')or(channel == '2B')or(channel == '2C')):
        xmin,xmax=513,1031
    if ((channel == '3A')or(channel == '3B')or(channel == '3C')):
        xmin,xmax=513,1031
    if ((channel == '4A')or(channel == '4B')or(channel == '4C')):
        xmin,xmax=0,492

    xrow=np.mgrid[xmin:xmax]
    yrow=np.mgrid[ymin:ymax]
    xall=mb.repmat(xrow,yrow.size,1)*1.
    yall=mb.repmat(yrow,xrow.size,1)*1.
    yall=np.transpose(yall)
    # Recast as 1d arrays
    xall=xall.reshape(-1)
    yall=yall.reshape(-1)

    # Convert a list of all pixel left/right edges to alpha/beta.
    values1=xytoabl(xall-0.4999,yall,channel)
    values2=xytoabl(xall+0.4999,yall,channel)
    # Look for where both had defined alpha
    indx=(np.where((values1['slicenum'] > 0) & (values2['slicenum'] > 0)))[0]
    # Crop to these indices and look at the difference in alpha
    alpha1=values1['alpha']
    alpha2=values2['alpha']
    yvec=values1['y']
    xvec=values1['x']
    slice=values1['slicenum']
    alpha1=alpha1[indx]
    alpha2=alpha2[indx]
    slice=slice[indx]
    yvec=yvec[indx]
    xvec=xvec[indx]
    nslice=np.max(slice)# Number of slices

    slice_out=np.zeros(nslice)
    amin_out=np.zeros(nslice)
    amax_out=np.zeros(nslice)
    
    # Loop over slices
    for ii in range(1,nslice+1):
        indx=(np.where(slice == ii))[0]
        slice_out[ii-1]=ii
        ytemp=yvec[indx]
        # Keep in mind that sometimes alpha increase l->r, sometimes r->l depending on channel
        # Therefore map from left-edge and right-edge alpha to max and min alpha in a pixel
        if (alpha1[indx[0]] < alpha2[indx[0]]):
            alow=alpha1[indx]
            ahigh=alpha2[indx]
        else:
            alow=alpha2[indx]
            ahigh=alpha1[indx]
        amin,amax=np.zeros(ymax),np.zeros(ymax)
        # Loop over rows
        for jj in range(0,ymax):
            indx=(np.where(ytemp == jj))[0]
            amin[jj]=np.min(alow[indx])
            amax[jj]=np.max(ahigh[indx])
        amin_out[ii-1]=np.median(amin)
        amax_out[ii-1]=np.median(amax)

    # Dictionary for return values
    values=dict();
    values['slice']=slice_out
    values['amin']=amin_out
    values['amax']=amax_out
    
    return values


#############################

# Convert from detector x,y pixel (0-indexed detector pixels) to alpha,beta,lambda
# Return is a dictionary with elements
# x,y,alpha,beta,lam,slicenum,slicename
# where x,y are possibly trimmed input values if the trim=0 keyword was set.
# (Trimming is to only detector pixels corresponding to a slice)
#
# slicenum is the slice number (e.g., 11)
# slicename is the slice name (e.g., 211A for ch2, slice 11, sub-band A)
def xytoabl(xin,yin,channel,**kwargs):
    # Split input channel into components, e.g.
    # if channel='1A' then ch=1 and sband=A
    ch=channel[0]
    sband=channel[1]

    trimx=np.array(xin)
    trimy=np.array(yin)
    
    # Ensure we're not using integer inputs
    # Also handle possible 1-element or multi-element input
    try:
        numpoints=len(xin)
        x=np.array(xin)*1.0
        y=np.array(yin)*1.0
    except:
        numpoints=1
        x=np.array([xin])*1.0
        y=np.array([yin])*1.0

    # Open relevant distortion file
    distfile=fits.open(get_fitsreffile(channel))
    
    # Read global header
    hdr=distfile[0].header
 
    # Get beta zeropoint and spacing from header
    beta0=hdr['B_ZERO'+ch]
    dbeta=hdr['B_DEL'+ch]

    # Alpha matrix
    d2c_alpha=distfile['Alpha_CH'+ch].data
    # Lambda matrix
    d2c_lambda=distfile['Lambda_CH'+ch].data
    # Slice map
    d2c_slice_all=distfile['Slice_Number'].data
    # Unless otherwise specified, use the 80% throughput slice map
    if 'mapplane' in kwargs:
        d2c_slice=d2c_slice_all[kwargs['mapplane'],:,:]
    else:
        d2c_slice=d2c_slice_all[7,:,:]

    # Define slice for these pixels
    slicenum=np.zeros(x.size, int)
    slicename=np.array(['JUNK' for i in range(0,x.size)])

    for i in range(0,x.size):
        slicenum[i]=int(d2c_slice[int(round(y[i])),int(round(x[i]))])-int(ch)*100
        slicename[i]=str(int(d2c_slice[int(round(y[i])),int(round(x[i]))]))+sband

    # Eliminate slice numbers on the wrong half of the detector
    bad=np.where((slicenum < 0) | (slicenum > 50))
    slicenum[bad]=-100
    slicename[bad]='NA'
        
    # Define index0 where the slice number is physical
    # (i.e., not between slices).  The [0] seems necessary to get
    # actual values rather than a single list object
    index0=(np.where((slicenum > 0) & (slicenum < 50)))[0]
    nindex0=len(index0)

    # Initialize a,b,l to -999.
    # (they will be set to something else if the pixel lands on a valid slice)
    al=np.zeros(x.size)-999.
    be=np.zeros(x.size)-999.
    lam=np.zeros(x.size)-999.

    # Define beta for these pixels
    if (nindex0 > 0):
        be[index0]=beta0+(slicenum[index0]-1.)*dbeta

    # Get the alpha,lambda coefficients for all of the valid pixels
    alphacoeff=d2c_alpha[slicenum[index0]-1]
    lamcoeff=d2c_lambda[slicenum[index0]-1]
    # Build big matrices of the x,y inputs combined with the corresponding coefficients
    thealphamatrix=np.zeros([nindex0,26])
    thelammatrix=np.zeros([nindex0,26])
    # Python hates loops, so instead of looping over individual entries
    # loop over columns in the big matrices instead
    for i in range(0,5):
        for j in range(0,5):
            coind=1+(i*5)+j
            thealphamatrix[:,coind]=alphacoeff.field(coind)*(((x[index0]-alphacoeff.field(0))**j)*(y[index0]**i))
            thelammatrix[:,coind]=lamcoeff.field(coind)*(((x[index0]-lamcoeff.field(0))**j)*(y[index0]**i))

    # Sum the contributions from each column in the big matrices
    al[index0]=np.sum(thealphamatrix,axis=1)    
    lam[index0]=np.sum(thelammatrix,axis=1)   
    
    distfile.close()
    
    # If the 'trim' keyword was set, trim the outputs to only pixels on valid slices
    if ('trim' in kwargs):
        trimx=trimx[index0]
        trimy=trimy[index0]
        al=al[index0]
        be=be[index0]
        lam=lam[index0]
        slicenum=slicenum[index0]
        slicename=slicename[index0]

    # Return a dictionary of results
    values=dict();
    values['x']=trimx
    values['y']=trimy
    values['alpha']=al
    values['beta']=be
    values['lam']=lam
    values['slicenum']=slicenum
    values['slicename']=slicename

    return values

#############################

# Convert from alpha,beta,lambda to detector x,y pixel (0-indexed detector pixels)
# Return is a dictionary with elements
# x,y,alpha,beta,lam,slicenum,slicename
# where x,y are possibly trimmed input values if the trim=0 keyword was set.
# (Trimming is to only detector pixels corresponding to a slice)
#
# slicenum is the slice number (e.g., 11)
# slicename is the slice name (e.g., 211A for ch2, slice 11, sub-band A)
def abltoxy(alin,bein,lamin,channel,**kwargs):
    # Split input channel into components, e.g.
    # if channel='1A' then ch=1 and sband=A
    ch=channel[0]
    sband=channel[1]
    
    # Handle possible 1-element or multi-element input
    try:
        numpoints=len(np.array(alin))
        trimal=np.array(alin)*1.
        trimbe=np.array(bein)*1.
        trimlam=np.array(lamin)*1.
    except:
        numpoints=1    
        trimal=np.array([alin])*1.
        trimbe=np.array([bein])*1.
        trimlam=np.array([lamin])*1.
    
    # Open relevant distortion file
    distfile=fits.open(get_fitsreffile(channel))
    
    # Read global header
    hdr=distfile[0].header
 
    # Get wavelength ranges from header
    wmin=hdr['L_MIN'+ch]
    wmax=hdr['L_MAX'+ch]
    # Replace any negative wavelengths with the midpoint wavelength
    # (this lets us use this code to compute a typical midpoint location)
    negval=(np.where(trimlam < 0))[0]
    if (len(negval) > 0):
        trimlam[negval]=(wmin+wmax)/2.

    # Get beta zeropoint and spacing from header
    beta0=hdr['B_ZERO'+ch]
    dbeta=hdr['B_DEL'+ch]

    # X matrix
    c2d_x=distfile['X_CH'+ch].data
    # Y matrix
    c2d_y=distfile['Y_CH'+ch].data

    # Determine slices
    slicefloat=np.array((trimbe-beta0)/dbeta+1)
    slicenum=(np.round(slicefloat)).astype(int)
    slicename=np.array(['JUNK' for i in range(0,trimal.size)])
    for i in range(0,trimal.size):
        slicename[i]=str(slicenum[i]+int(ch)*100)+sband

    # Read the Slice Map
    c2d_slice_all=distfile['Slice_Number'].data
    # Use the 80% throughput slice map based on OTIS data
    c2d_slicefull=c2d_slice_all[7,:,:]

    # Crop to the correct half of the detector
    if (int(ch) == 1):
        c2d_slicefull=c2d_slicefull-100
        c2d_slice=c2d_slicefull[:,0:507]
    elif (int(ch) == 2):
        c2d_slicefull=c2d_slicefull-200
        c2d_slice=c2d_slicefull[:,508:]
    elif (int(ch) == 4):
        c2d_slicefull=c2d_slicefull-400
        c2d_slice=c2d_slicefull[:,0:492]
    elif (int(ch) == 3):
        c2d_slicefull=c2d_slicefull-300
        c2d_slice=c2d_slicefull[:,493:]

    # Find maximum and minimum slice numbers allowed from the slice map
    temp=(np.where(c2d_slice >= 0))
    smax=int(max(c2d_slice[temp]))
    smin=int(min(c2d_slice[temp]))

    # Determine where slice numbers computed from beta are unphysical
    badval=(np.where(np.logical_or(slicenum < smin, slicenum > smax)))[0]
    if len(badval) > 0:
        slicenum[badval]=-999
        slicename[badval]='-999'
        slicefloat[badval]=-999.

    # Define index0 where the slice number is physical
    # (i.e., not between slices).  The [0] seems necessary to get
    # actual values rather than a single list object
    index0=(np.where((slicenum > 0) & (slicenum < 99)))[0]
    nindex0=len(index0)
    
    # Initialize x,y to -999.
    # (they will only be set to something else if the pixel
    # lands on a valid slice at valid alpha)
    x=np.zeros(trimal.size)-999.
    y=np.zeros(trimal.size)-999.

    # Get the x,y coefficients for all of the valid pixels
    xcoeff=c2d_x[slicenum[index0]-1]
    ycoeff=c2d_y[slicenum[index0]-1]
    # Build big matrices of the alpha,lam inputs combined with the corresponding coefficients
    thexmatrix=np.zeros([nindex0,26])
    theymatrix=np.zeros([nindex0,26])
    # Python hates loops, so instead of looping over individual entries
    # loop over columns in the big matrices instead
    for i in range(0,5):
        for j in range(0,5):
            coind=1+(i*5)+j
            thexmatrix[:,coind]=xcoeff.field(coind)*(((trimlam[index0]-xcoeff.field(0))**i)*(trimal[index0]**j))
            theymatrix[:,coind]=ycoeff.field(coind)*(((trimlam[index0]-ycoeff.field(0))**i)*(trimal[index0]**j))
    # Sum the contributions from each column in the big matrices
    x[index0]=np.sum(thexmatrix,axis=1)    
    y[index0]=np.sum(theymatrix,axis=1)

    # Check that computed x,y land within array boundaries and on real slice pixels (i.e., not beyond edges)
    badval=np.where((x > c2d_slicefull.shape[1]) | (x < 0) | (y > c2d_slicefull.shape[0]) | (y < 0))
    nbad=len(badval)
    if (nbad > 0):
        # Set these to zero so that array indexing doesn't fail
        x[badval]=0
        y[badval]=0
        slicenum[badval]=-999
        slicename[badval]='-999'    
    sliceact=(c2d_slicefull[(np.round(y)).astype(int),(np.round(x)).astype(int)]).astype(int)
    # If something was not on the slice pixels, it's bad
    badval=np.where(sliceact != slicenum)
    nbad=len(badval)
    if (nbad > 0):
        x[badval]=-999
        y[badval]=-999
        slicenum[badval]=-999
        slicename[badval]='-999'
    
    # Look for wherever x,y != -999, those are our good cases
    index0=(np.where(np.logical_and(x > -999, y > -999)))[0]
    nindex0=len(index0)

    # Determine slice, pixel, and wavelength phases
    # 0 is in the middle of a sample, -0.5 at the bottom edge, 0.5 at the
    # top edge
    slicephase = np.zeros(trimal.size) - 999.
    pixelphase = np.zeros(trimal.size) - 999.
    wavephase = np.zeros(trimal.size) - 999.
    slicephase[index0] = slicefloat[index0] - np.round(slicefloat[index0])
    pixelphase[index0] = x[index0] - np.round(x[index0])
    wavephase[index0] = y[index0] - np.round(y[index0])

    distfile.close()
    
    # If the 'trim' keyword was set, trim the outputs to only pixels on valid slices
    if ('trim' in kwargs):
        x=x[index0]
        y=y[index0]
        trimal=trimal[index0]
        trimbe=trimbe[index0]
        trimlam=trimlam[index0]
        slicenum=slicenum[index0]
        slicename=slicename[index0]
        slicephase=slicephase[index0]
        pixelphase=pixelphase[index0]
        wavephase = wavephase[index0]

    # Return a dictionary of results
    values=dict();
    values['x']=x
    values['y']=y
    values['alpha']=trimal
    values['beta']=trimbe
    values['lam']=trimlam
    values['slicenum']=slicenum
    values['slicename']=slicename
    values['slicephase']=slicephase
    values['pixelphase']=pixelphase
    values['wavephase'] = wavephase

    return values

#############################

# Convert from alpha,beta to JWST v2,v3 coordinates
# all coordinates are in arcsec
def abtov2v3(alin,bein,channel):
    # Split input channel into components, e.g.
    # if channel='1A' then ch=1 and sband=A
    ch=channel[0]
    sband=channel[1]

    alpha=np.array(alin)*1.
    beta=np.array(bein)*1.

    # Open relevant distortion file
    distfile=fits.open(get_fitsreffile(channel))

    # Read the distortion table
    convtable=distfile['albe_to_V2V3'].data
    # Determine which rows we need
    v2index=(np.where(convtable['Label'] == 'T_CH'+channel+'_V2'))[0][0]
    v3index=(np.where(convtable['Label'] == 'T_CH'+channel+'_V3'))[0][0]

    if (np.logical_or(v2index < 0, v3index < 0)):
        print('Bad channel specification!')

    conv_v2=convtable[v2index]
    conv_v3=convtable[v3index]

    # Apply transform to V2,V3
    v2=conv_v2[1]+conv_v2[2]*alpha+conv_v2[3]*alpha*alpha+ \
       conv_v2[4]*beta+conv_v2[5]*beta*alpha+conv_v2[6]*beta*alpha*alpha+ \
       conv_v2[7]*beta*beta+conv_v2[8]*beta*beta*alpha+conv_v2[9]*beta*beta*alpha*alpha
    v3=conv_v3[1]+conv_v3[2]*alpha+conv_v3[3]*alpha*alpha+ \
       conv_v3[4]*beta+conv_v3[5]*beta*alpha+conv_v3[6]*beta*alpha*alpha+ \
       conv_v3[7]*beta*beta+conv_v3[8]*beta*beta*alpha+conv_v3[9]*beta*beta*alpha*alpha

    distfile.close()

    return v2,v3

#############################

# Convert from JWST v2,v3 coordinates to alpha,beta
# all coordinates are in arcsec
def v2v3toab(v2in,v3in,channel):
    # Split input channel into components, e.g.
    # if channel='1A' then ch=1 and sband=A
    ch=channel[0]
    sband=channel[1]

    v2=np.array(v2in)*1.
    v3=np.array(v3in)*1.

    # Open relevant distortion file
    distfile=fits.open(get_fitsreffile(channel))

    # Read the distortion table
    convtable=distfile['V2V3_to_albe'].data
    # Determine which rows we need
    alindex=(np.where(convtable['Label'] == 'U_CH'+channel+'_al'))[0][0]
    beindex=(np.where(convtable['Label'] == 'U_CH'+channel+'_be'))[0][0]

    if (np.logical_or(alindex < 0, beindex < 0)):
        print('Bad channel specification!')

    conv_al=convtable[alindex]
    conv_be=convtable[beindex]

    # Apply transform to alpha,beta
    alpha=conv_al[1]+conv_al[2]*v2+conv_al[3]*v2*v2+ \
       conv_al[4]*v3+conv_al[5]*v3*v2+conv_al[6]*v3*v2*v2+ \
       conv_al[7]*v3*v3+conv_al[8]*v3*v3*v2+conv_al[9]*v3*v3*v2*v2

    beta=conv_be[1]+conv_be[2]*v2+conv_be[3]*v2*v2+ \
       conv_be[4]*v3+conv_be[5]*v3*v2+conv_be[6]*v3*v2*v2+ \
       conv_be[7]*v3*v3+conv_be[8]*v3*v3*v2+conv_be[9]*v3*v3*v2*v2

    distfile.close()

    return alpha,beta

#############################

# MRS test reference data
# Created by DRL 5/27/22
mrs_ref_data = {
    '1A': {'x': np.array([ 76., 354.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([ 0.05765538, -0.01703262]),
           'beta': np.array([-0.17721014, -1.24047101]),
           'lam': np.array([5.34854658, 5.51364206]),
           'v2': np.array([-503.24467276, -503.22055087]),
           'v3': np.array([-318.85930417, -317.78711049]),
           },
    '1B': {'x': np.array([ 76., 355.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([-0.01299074,  0.10766448]),
           'beta': np.array([-0.17720418, -1.24042924]),
           'lam': np.array([6.1683104 , 6.35800764]),
           'v2': np.array([-503.24759111, -503.00533651]),
           'v3': np.array([-318.96012176, -317.92370507]),
           },
    '1C': {'x': np.array([ 78., 356.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([ 0.02871804, -0.02831582]),
           'beta': np.array([-0.17720219, -1.24041531]),
           'lam': np.array([7.00660816, 7.21845515]),
           'v2': np.array([-503.18994409, -503.12704291]),
           'v3': np.array([-318.69505738, -317.62464996]),
           },
    '2A': {'x': np.array([574., 719.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([0.02286234, 0.02410476]),
           'beta': np.array([ 0.27971819, -1.39859093]),
           'lam': np.array([8.1394638 , 8.42387972]),
           'v2': np.array([-503.48552164, -503.26240257]),
           'v3': np.array([-319.57793882, -317.91753328]),
           },
    '2B': {'x': np.array([570., 715.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([-0.04101483, -0.02196444]),
           'beta': np.array([ 0.27972605, -1.39863026]),
           'lam': np.array([9.49091779, 9.8261122 ]),
           'v2': np.array([-503.63714024, -503.38064949]),
           'v3': np.array([-319.84152926, -318.19364938]),
           },
    '2C': {'x': np.array([573., 718.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([-0.0806554 , -0.07196316]),
           'beta': np.array([ 0.27972212, -1.3986106 ]),
           'lam': np.array([10.90955839, 11.29265821]),
           'v2': np.array([-503.47509601, -503.24943244]),
           'v3': np.array([-319.77378269, -318.12791192]),
           },
    '3A': {'x': np.array([918., 827.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([-0.14902641, -0.13941115]),
           'beta': np.array([ 0.58472067, -1.754162  ]),
           'lam': np.array([12.58608529, 12.17180378]),
           'v2': np.array([-504.13326367, -504.01896234]),
           'v3': np.array([-319.56514265, -317.2602021]),
           },
    '3B': {'x': np.array([919., 827.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([-0.18610617,  0.05223449]),
           'beta': np.array([ 0.58472067, -1.754162  ]),
           'lam': np.array([14.60074102, 14.12035326]),
           'v2': np.array([-504.24725567, -503.83572757]),
           'v3': np.array([-319.72339118, -317.49857585]),
           },    
    '3C': {'x': np.array([917., 826.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([-0.08917305, -0.09924684]),
           'beta': np.array([ 0.58472067, -1.754162  ]),
           'lam': np.array([16.86061623, 16.30564805]),
           'v2': np.array([-504.0412986 , -503.88521261]),
           'v3': np.array([-319.71572779, -317.46972039]),
           },    
    '4A': {'x': np.array([195., 232.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([-0.18281232, -0.10820927]),
           'beta': np.array([ 2.29613309, -1.64009507]),
           'lam': np.array([19.42967253, 18.7337858 ]),
           'v2': np.array([-503.31946874, -502.5823108]),
           'v3': np.array([-321.83103724, -318.15509964]),
           },    
    '4B': {'x': np.array([192., 229.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([-0.03596952, -0.10259403]),
           'beta': np.array([ 2.29613654, -1.64009753]),
           'lam': np.array([22.47574269, 21.67074831]),
           'v2': np.array([-503.29034462, -502.66132575]),
           'v3': np.array([-321.66332479, -318.09363384]),
           },
    '4C': {'x': np.array([194., 231.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([-0.06619308, -0.01176626]),
           'beta': np.array([ 2.29611932, -1.64008523]),
           'lam': np.array([26.29237924, 25.35069458]),
           'v2': np.array([-503.28627393, -502.59623788]),
           'v3': np.array([-321.91663845, -318.09196097]),
           }
}
