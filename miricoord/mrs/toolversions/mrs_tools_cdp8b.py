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
the origin such that (0,0) is the middle of the lower-left light sensitive pixel
(1024x1024), therefore we also need to transform between this science frame and detector frame.

Since not all detector pixels actually map to alpha-beta (since some pixels are between slices)
these have alpha=beta=lambda=-999 and can be trimmed using 'trim=1'

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
11-Apr-2019  Written by David Law (dlaw@stsci.edu)
"""

import os as os
import math
import numpy as np
from astropy.io import fits
import pdb

#############################

# Return the tools version
def version():
    return 'cdp8b'

#############################

# Set the relevant FITS distortion file based on channel (e.g., '1A')
def get_fitsreffile(channel):
    rootdir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    if (channel == '1A'):
        file='MIRI_FM_MIRIFUSHORT_12SHORT_DISTORTION_8B.05.00.fits'
    elif (channel == '1B'):
        file='MIRI_FM_MIRIFUSHORT_12MEDIUM_DISTORTION_06.04.00.fits'
    elif (channel == '1C'):
        file='MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_06.04.00.fits'
    elif (channel == '2A'):
        file='MIRI_FM_MIRIFUSHORT_12SHORT_DISTORTION_06.04.00.fits'
    elif (channel == '2B'):
        file='MIRI_FM_MIRIFUSHORT_12MEDIUM_DISTORTION_06.04.00.fits'
    elif (channel == '2C'):
        file='MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_06.04.00.fits'
    elif (channel == '3A'):
        file='MIRI_FM_MIRIFULONG_34SHORT_DISTORTION_06.04.00.fits'
    elif (channel == '3B'):
        file='MIRI_FM_MIRIFULONG_34MEDIUM_DISTORTION_06.04.00.fits'
    elif (channel == '3C'):
        file='MIRI_FM_MIRIFULONG_34LONG_DISTORTION_06.04.00.fits'
    elif (channel == '4A'):
        file='MIRI_FM_MIRIFULONG_34SHORT_DISTORTION_06.04.00.fits'
    elif (channel == '4B'):
        file='MIRI_FM_MIRIFULONG_34MEDIUM_DISTORTION_06.04.00.fits'
    elif (channel == '4C'):
        file='MIRI_FM_MIRIFULONG_34LONG_DISTORTION_06.04.00.fits'
        
    rootdir=os.path.join(rootdir,'data/fits/cdp8b/')
    reffile=os.path.join(rootdir,file)
   
    return reffile

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
    
    # Transform from 0-indexed to 1-indexed pixels CDP assumes and ensure we're not using integer inputs
    # Also handle possible 1-element or multi-element input
    try:
        numpoints=len(xin)
        x=np.array(xin)+1.
        y=np.array(yin)+1.
    except:
        numpoints=1
        x=np.array([xin])+1.
        y=np.array([yin])+1.

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
    # Use the 80% throughput slice map
    d2c_slice=d2c_slice_all[7,:,:]

    # Define slice for these pixels
    slicenum=np.zeros(x.size, int)
    slicename=np.array(['JUNK' for i in range(0,x.size)])
 
    for i in range(0,x.size):
        # Note that since x,y here index into an array we need to subtract 1 again!
        slicenum[i]=int(d2c_slice[int(round(y[i]))-1,int(round(x[i]))-1])-int(ch)*100
        slicename[i]=str(int(d2c_slice[int(round(y[i]))-1,int(round(x[i]))-1]))+sband

    # Define index0 where the slice number is physical
    # (i.e., not between slices).  The [0] seems necessary to get
    # actual values rather than a single list object
    index0=(np.where((slicenum > 0) & (slicenum < 99)))[0]
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
    # If the input lambda was a single negative value, replace it with the midpoint wavelength
    # (this lets us use this code to compute a typical midpoint location)
    if ((len(trimlam) == 1)and(trimlam[0] < 0)):
        trimlam[:]=(wmin+wmax)/2.

    # Get beta zeropoint and spacing from header
    beta0=hdr['B_ZERO'+ch]
    dbeta=hdr['B_DEL'+ch]

    # X matrix
    c2d_x=distfile['X_CH'+ch].data
    # Y matrix
    c2d_y=distfile['Y_CH'+ch].data
    # Fov matrix
    fovalpha=distfile['FoV_CH'+ch].data

    # Determine slices
    slicefloat=np.array((trimbe-beta0)/dbeta+1)
    slicenum=(np.round(slicefloat)).astype(int)
    slicename=np.array(['JUNK' for i in range(0,trimal.size)])
    for i in range(0,trimal.size):
        slicename[i]=str(slicenum[i]+int(ch)*100)+sband

    # Read the Slice Map
    c2d_slice_all=distfile['Slice_Number'].data
    # Use the 80% throughput slice map
    c2d_slice=c2d_slice_all[7,:,:]

    # Crop to the correct half of the detector
    if (int(ch) == 1):
        c2d_slice=c2d_slice[:,0:507]-100
    elif (int(ch) == 2):
        c2d_slice=c2d_slice[:,508:]-200
    elif (int(ch) == 4):
        c2d_slice=c2d_slice[:,0:492]-400
    elif (int(ch) == 3):
        c2d_slice=c2d_slice[:,493:]-300

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
    # Check for cases where the input alpha is beyond episilon of the field boundary for the slice
    # and use this to modify index0
    eps=0.02
    goodalpha=(np.where(((trimal[index0]+eps) >= fovalpha[slicenum[index0]-1]['alpha_min']) & ((trimal[index0]-eps) <= fovalpha[slicenum[index0]-1]['alpha_max'])))[0]
    index0=index0[goodalpha]
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
    # Look for wherever x,y != -999, those are our good cases
    index0=(np.where(np.logical_and(x > -999, y > -999)))[0]
    nindex0=len(index0)
 
    # Transform from 1-indexed to 0-indexed values
    x[index0]=x[index0]-1
    y[index0]=y[index0]-1

    # Determine slice and pixel phase
    # 0 is in the middle of a sample, -0.5 at the bottom edge, 0.5 at the
    # top edge
    slicephase=np.zeros(trimal.size)-999.
    pixelphase=np.zeros(trimal.size)-999.
    slicephase[index0]=slicefloat[index0]-np.round(slicefloat[index0])
    pixelphase[index0]=x[index0]-np.round(x[index0])

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
# Convert all x,y values to 0-indexed
mrs_ref_data = {
    '1A': {'x': np.array([28.310396, 475.02154, 493.9777, 41.282537, 58.998266])-1,
           'y': np.array([512., 10, 100, 900, 1014])-1,
           's': np.array([11, 1, 1, 21, 21]),
           'alpha': np.array([0, -1.66946, 1.65180, -1.70573, 1.70244]),
           'beta': np.array([0, -1.77210, -1.77210, 1.77210, 1.77210]),
           'lam': np.array([5.34437, 4.86642, 4.95325, 5.65296, 5.74349]),
           'xan': np.array([-8.39424, -8.41746, -8.36306, -8.42653, -8.37026]),
           'yan': np.array([-2.48763, -2.52081, -2.51311, -2.46269, -2.45395]),
           },
    '1B': {'x': np.array([28.648221, 475.07259, 493.98157, 41.559386, 59.738296])-1,
           'y': np.array([512., 10, 100, 900, 1014])-1,
           's': np.array([11, 1, 1, 21, 21]),
           'alpha': np.array([0., -1.70796, 1.60161, -1.70854, 1.78261]),
           'beta': np.array([0., -1.77204, -1.77204, 1.77204, 1.77204]),
           'lam': np.array([6.17572, 5.62345, 5.72380, 6.53231, 6.63698]),
           'xan': np.array([-8.39426, -8.41808, -8.36368, -8.42682, -8.36899]),
           'yan': np.array([-2.48492, -2.51808, -2.51040, -2.46001, -2.45126])
           },
    '1C': {'x': np.array([30.461871, 477.23742, 495.96228, 43.905314, 60.995683])-1,
           'y': np.array([512., 10, 100, 900, 1014])-1,
           's': np.array([11, 1, 1, 21, 21]),
           'alpha': np.array([0., -1.60587, 1.67276, -1.60766, 1.68720]),
           'beta': np.array([0., -1.77202, -1.77202, 1.77202, 1.77202]),
           'lam': np.array([7.04951, 6.42424, 6.53753, 7.45360, 7.57167]),
           'xan': np.array([-8.39357, -8.41570, -8.36165, -8.42457, -8.36996]),
           'yan': np.array([-2.48987, -2.52271, -2.51525, -2.46467, -2.45649])
           },
    '2A': {'x': np.array([992.158, 545.38386, 525.76143, 969.29711, 944.19303])-1,
           'y': np.array([512., 10, 100, 900, 1014])-1,
           's': np.array([9, 1, 1, 17, 17]),
           'alpha': np.array([0., -2.11250, 2.10676, -2.17239, 2.10447]),
           'beta': np.array([0., -2.23775, -2.23775, 2.23775, 2.23775]),
           'lam': np.array([8.20797, 7.52144, 7.64907, 8.68677, 8.83051]),
           'xan': np.array([-8.39393, -8.42259, -8.35355, -8.43583, -8.36499]),
           'yan': np.array([-2.48181, -2.52375, -2.51357, -2.44987, -2.44022])
           },
    '2B': {'x': np.array([988.39977, 541.23447, 521.60207, 964.91753, 940.10325])-1,
           'y': np.array([512., 10, 100, 900, 1014])-1,
           's': np.array([9, 1, 1, 17, 17]),
           'alpha': np.array([0., -2.10593, 2.10015, -2.08817, 2.10422]),
           'beta': np.array([0., -2.23781, -2.23781, 2.23781, 2.23781]),
           'lam': np.array([9.44205, 8.65341, 8.79991, 9.99257, 10.15795]),
           'xan': np.array([-8.39645, -8.42502, -8.35603, -8.43716, -8.36742]),
           'yan': np.array([-2.47773, -2.51972, -2.50938, -2.44554, -2.43626])
           },
    '2C': {'x': np.array([990.89693, 543.82344, 524.34514, 967.98318, 942.77564])-1,
           'y': np.array([512., 10, 100, 900, 1014])-1,
           's': np.array([9, 1, 1, 17, 17]),
           'alpha': np.array([0., -2.07490, 2.11234, -2.14704, 2.14196]),
           'beta': np.array([0., -2.23778, -2.23778, 2.23778, 2.23778]),
           'lam': np.array([10.90225, 9.99162, 10.16079, 11.53780, 11.72887]),
           'xan': np.array([-8.39303, -8.42129, -8.35221, -8.43454, -8.36352]),
           'yan': np.array([-2.47869, -2.52052, -2.51036, -2.44668, -2.43712])
           },
    '3A': {'x': np.array([574.80828, 1001.0602, 984.6387, 547.27479, 518.89992])-1,
           'y': np.array([512., 10, 100, 900, 1014])-1,
           's': np.array([8, 1, 1, 16, 16]),
           'alpha': np.array([0., -2.86745, 3.20982, -3.01230, 2.96643]),
           'beta': np.array([-0.19491, -2.92360, -2.92360, 2.92360, 2.92360]),
           'lam': np.array([12.5335, 13.49968, 13.33846, 11.77148, 11.52350]),
           'xan': np.array([-8.40590, -8.44849, -8.34906, -8.46070, -8.36174]),
           'yan': np.array([-2.48992, -2.54104, -2.52854, -2.44547, -2.43112])
           },
    '3B': {'x': np.array([574.26012, 1001.7349, 985.30166, 548.016, 519.98])-1,
           'y': np.array([512., 10., 100, 900, 1014])-1,
           's': np.array([8, 1, 1, 16, 16]),
           'alpha': np.array([0, -3.17728, 2.92434, -3.29402, 2.60797]),
           'beta': np.array([-0.19491, -2.92360, -2.92360, 2.92360, 2.92360]),
           'lam': np.array([14.53997, 15.66039, 15.47355, 13.65622, 13.36833]),
           'xan': np.array([-8.40044, -8.44785, -8.34786, -8.46088, -8.36211]),
           'yan': np.array([-2.48588, -2.53771, -2.52512, -2.44219, -2.42776])
           },
    '3C': {'x': np.array([573.25446, 1000.21721, 983.92918, 546.00285, 518.2782])-1,
           'y': np.array([512., 10, 100, 900, 1014])-1,
           's': np.array([8, 1, 1, 16, 16]),
           'alpha': np.array([0., -2.94573, 3.09057, -3.07810, 2.73161]),
           'beta': np.array([-0.19491, -2.92360, -2.92360, 2.92360, 2.92360]),
           'lam': np.array([16.79017, 18.08441, 17.86845, 15.76948, 15.43724]),
           'xan': np.array([-8.40205, -8.44574, -8.34664, -8.45859, -8.36196]),
           'yan': np.array([-2.48627, -2.53761, -2.52502, -2.44221, -2.42787]),
           },
    '4A': {'x': np.array([80.987181, 434.34987, 461.90855, 26.322503, 53.674656])-1,
           'y': np.array([512., 10, 100, 900, 1014])-1,
           's': np.array([6, 1, 1, 12, 12]),
           'alpha': np.array([0., -3.74625, 3.72621, -3.94261, 3.62762]),
           'beta': np.array([-0.32802, -3.60821, -3.60821, 3.60821, 3.60821]),
           'lam': np.array([19.34914, 20.93078, 20.6464, 18.07975, 17.67221]),
           'xan': np.array([-8.38446, -8.43506, -8.31378, -8.46256, -8.33609]),
           'yan': np.array([-2.48058, -2.5444, -2.52426, -2.42449, -2.40839])
           },
    '4B': {'x': np.array([77.625553, 431.57061, 458.86869, 23.559111, 50.632416])-1,
           'y': np.array([512., 10, 100, 900, 1014])-1,
           's': np.array([6, 1, 1, 12, 12]),
           'alpha': np.array([0., -3.64817, 3.73313, -3.73558, 3.74096]),
           'beta': np.array([-0.32802, -3.60821, -3.60821, 3.60821, 3.60821]),
           'lam': np.array([22.38267, 24.21212, 23.88327, 20.91426, 20.44279]),
           'xan': np.array([-8.38581, -8.43443, -8.3141, -8.46152, -8.33604]),
           'yan': np.array([-2.48185, -2.54526, -2.52568, -2.42513, -2.40959])
           },
    '4C': {'x': np.array([79.662059, 433.73384, 460.75026, 25.820431, 52.412219])-1,
           'y': np.array([512., 10, 100, 900, 1014])-1,
           's': np.array([6, 1, 1, 12, 12]),
           'alpha': np.array([0., -3.61682, 3.69713, -3.66259, 3.69888]),
           'beta': np.array([-0.32802, -3.60819, -3.60819, 3.60819, 3.60819]),
           'lam': np.array([26.18343, 28.32354, 27.93894, 24.46574, 23.91417]),
           'xan': np.array([-8.38603, -8.43509, -8.31524, -8.45888, -8.33707]),
           'yan': np.array([-2.48315, -2.54647, -2.52661, -2.42721, -2.41060])
           }
}
