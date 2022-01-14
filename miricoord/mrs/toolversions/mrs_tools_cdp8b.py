#
"""
Useful python tools for working with the MIRI MRS.
This contains cdp8b specific code.

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
    return 'cdp8b'

#############################

# Set the relevant FITS distortion file based on channel (e.g., '1A')
def get_fitsreffile(channel):
    if (channel == '1A'):
        file='MIRI_FM_MIRIFUSHORT_12SHORT_DISTORTION_8B.05.01.fits'
    elif (channel == '1B'):
        file='MIRI_FM_MIRIFUSHORT_12MEDIUM_DISTORTION_8B.05.01.fits'
    elif (channel == '1C'):
        file='MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_8B.05.01.fits'
    elif (channel == '2A'):
        file='MIRI_FM_MIRIFUSHORT_12SHORT_DISTORTION_8B.05.01.fits'
    elif (channel == '2B'):
        file='MIRI_FM_MIRIFUSHORT_12MEDIUM_DISTORTION_8B.05.01.fits'
    elif (channel == '2C'):
        file='MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_8B.05.01.fits'
    elif (channel == '3A'):
        file='MIRI_FM_MIRIFULONG_34SHORT_DISTORTION_8B.05.01.fits'
    elif (channel == '3B'):
        file='MIRI_FM_MIRIFULONG_34MEDIUM_DISTORTION_8B.05.01.fits'
    elif (channel == '3C'):
        file='MIRI_FM_MIRIFULONG_34LONG_DISTORTION_8B.05.01.fits'
    elif (channel == '4A'):
        file='MIRI_FM_MIRIFULONG_34SHORT_DISTORTION_8B.05.01.fits'
    elif (channel == '4B'):
        file='MIRI_FM_MIRIFULONG_34MEDIUM_DISTORTION_8B.05.01.fits'
    elif (channel == '4C'):
        file='MIRI_FM_MIRIFULONG_34LONG_DISTORTION_8B.05.01.fits'

    # Try looking for the file in the expected location
    rootdir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    rootdir=os.path.join(rootdir,'data/fits/cdp8b/')
    reffile=os.path.join(rootdir,file)
    if os.path.exists(reffile):
        return reffile
    
    # If that didn't work, look in the system path
    rootdir=sys.prefix
    rootdir=os.path.join(rootdir,'data/fits/cdp8b/')
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
    # Use the 80% throughput slice map, based on OTIS data
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
# Provided by Polychronis 5/9/19
mrs_ref_data = {
    '1A': {'x': np.array([76.0,354.0]),
           'y': np.array([512.0,700.0]),
           's': np.array([10,4]),
           'alpha': np.array([0.05765538365149925,-0.017032619150995743]),
           'beta': np.array([-0.17721014379699995,-1.240471006579]),
           'lam': np.array([5.348546577257886,5.5136420569934925]),
           'v2': np.array([-503.57285226785064,-503.4979806620663]),
           'v3': np.array([-318.5749892859028,-317.5090073056335]),
           },
    '1B': {'x': np.array([76.0,355.0]),
           'y': np.array([512.0,700.0]),
           's': np.array([10,4]),
           'alpha': np.array([-0.012990737471741731,0.10766447914943456]),
           'beta': np.array([-0.17720417669099997,-1.240429236837]),
           'lam': np.array([6.168310398808807,6.358007642348213]),
           'v2': np.array([-503.643100332753,-503.37069816112813]),
           'v3': np.array([-318.72773306477103,-317.6938248759762]),
           },
    '1C': {'x': np.array([78.0,356.0]),
           'y': np.array([512.0,700.0]),
           's': np.array([10,4]),
           'alpha': np.array([0.02871804339196271,-0.028315822861031847]),
           'beta': np.array([-0.17720218765499984,-1.240415313585]),
           'lam': np.array([7.006608159574103,7.218455147089075]),
           'v2': np.array([-503.5598371896608,-503.45975848303885]),
           'v3': np.array([-318.4367657801553,-317.3779485524358]),
           },
    '2A': {'x': np.array([574.0,719.0]),
           'y': np.array([512.0,700.0]),
           's': np.array([10,4]),
           'alpha': np.array([0.022862344416012093,0.024104763006107532]),
           'beta': np.array([0.27971818633699996,-1.3985909316610001]),
           'lam': np.array([8.139463800053713, 8.423879719165456]),
           'v2': np.array([-503.65782416704644, -503.3907046961389]),
           'v3': np.array([-319.3709764579651, -317.71318662530217]),
           },
    '2B': {'x': np.array([570.0,715.0]),
           'y': np.array([512.0,700.0]),
           's': np.array([10,4]),
           'alpha': np.array([-0.04101483043351095,-0.021964438108625473]),
           'beta': np.array([0.27972605223,-1.39863026115]),
           'lam': np.array([9.49091778668766, 9.826112199836349]),
           'v2': np.array([-503.872441161987, -503.58468453126545]),
           'v3': np.array([-319.6066193816802, -317.9526192173689]),
           },
    '2C': {'x': np.array([573.0,718.0]),
           'y': np.array([512.0,700.0]),
           's': np.array([10,4]),
           'alpha': np.array([-0.08065540123411097,-0.07196315905207484]),
           'beta': np.array([0.2797221192789996, -1.3986105964070001]),
           'lam': np.array([10.909558387414732,11.292658213110698]),
           'v2': np.array([-503.7062367371822, -503.4292038385116]),
           'v3': np.array([-319.54349206004053, -317.8886490566051]),
           },
    '3A': {'x': np.array([918.0,827.0]),
           'y': np.array([512.0,700.0]),
           's': np.array([10,4]),
           'alpha': np.array([-0.14902640584477922, -0.1394111481404252]),
           'beta': np.array([0.5847206674920002, -1.7541620024759998]),
           'lam': np.array([12.586085291551054, 12.171803779467552]),
           'v2': np.array([-504.57532179184557, -504.3428404141017]),
           'v3': np.array([-319.3596209726561, -317.0363338552647]),
           },
    '3B': {'x': np.array([919.0, 827.0]),
           'y': np.array([512.0, 700.0]),
           's': np.array([10,4]),
           'alpha': np.array([-0.18610616903060873, 0.05223448620927229]),
           'beta': np.array([0.5847206674920002, -1.7541620024759998]),
           'lam': np.array([14.60074101845329, 14.120353260795175]),
           'v2': np.array([-504.29128783278026, -503.81513623681207]),
           'v3': np.array([-319.5977726217362, -317.30169796071453]),
           },
    '3C': {'x': np.array([917.0,826.0]),
           'y': np.array([512.0,700.0]),
           's': np.array([10,4]),
           'alpha': np.array([-0.08917305254544772, -0.09924683542340063]),
           'beta': np.array([0.5847206674920002, -1.7541620024759998]),
           'lam': np.array([16.860616228418674, 16.305648049347006]),
           'v2': np.array([-504.29179372150304, -504.06099473540036]),
           'v3': np.array([-319.5864222556306, -317.26146053061063]),
           },
    '4A': {'x': np.array([195.0, 232.0]),
           'y': np.array([512.0, 700.0]),
           's': np.array([10,4]),
           'alpha': np.array([-0.18281231856817595, -0.10820926727846612]),
           'beta': np.array([2.2961330928359995, -1.640095066308]),
           'lam': np.array([19.42967253041467, 18.733785802367724]),
           'v2': np.array([-503.73916258138155, -502.9287085654886]),
           'v3': np.array([-321.7198475574414, -317.8596067111157]),
           },
    '4B': {'x': np.array([192.0, 229.0]),
           'y': np.array([512.0, 700.0]),
           's': np.array([10,4]),
           'alpha': np.array([-0.03596952007447607, -0.10259402857181654]),
           'beta': np.array([2.2961365363689996, -1.640097525977]),
           'lam': np.array([22.47574268879503, 21.67074830984225]),
           'v2': np.array([-503.7051048327475, -502.9891450100565]),
           'v3': np.array([-321.6637327196876, -317.78403487305536]),
           },
    '4C': {'x': np.array([194.0, 231.0]),
           'y': np.array([512.0, 700.0]),
           's': np.array([10,4]),
           'alpha': np.array([-0.0661930805678849, -0.01176625661012924]),
           'beta': np.array([2.296119318687, -1.640085227631]),
           'lam': np.array([26.292379242285914, 25.350694577065074]),
           'v2': np.array([-503.7171854824459, -502.9282547181127]),
           'v3': np.array([-321.57006077329663, -317.7252303132135]),
           }
    
}
