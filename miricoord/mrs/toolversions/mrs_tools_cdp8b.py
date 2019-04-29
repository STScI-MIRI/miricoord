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
        file='MIRI_FM_MIRIFUSHORT_12MEDIUM_DISTORTION_8B.05.00.fits'
    elif (channel == '1C'):
        file='MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_8B.05.00.fits'
    elif (channel == '2A'):
        file='MIRI_FM_MIRIFUSHORT_12SHORT_DISTORTION_8B.05.00.fits'
    elif (channel == '2B'):
        file='MIRI_FM_MIRIFUSHORT_12MEDIUM_DISTORTION_8B.05.00.fits'
    elif (channel == '2C'):
        file='MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_8B.05.00.fits'
    elif (channel == '3A'):
        file='MIRI_FM_MIRIFULONG_34SHORT_DISTORTION_8B.05.00.fits'
    elif (channel == '3B'):
        file='MIRI_FM_MIRIFULONG_34MEDIUM_DISTORTION_8B.05.00.fits'
    elif (channel == '3C'):
        file='MIRI_FM_MIRIFULONG_34LONG_DISTORTION_8B.05.00.fits'
    elif (channel == '4A'):
        file='MIRI_FM_MIRIFULONG_34SHORT_DISTORTION_8B.05.00.fits'
    elif (channel == '4B'):
        file='MIRI_FM_MIRIFULONG_34MEDIUM_DISTORTION_8B.05.00.fits'
    elif (channel == '4C'):
        file='MIRI_FM_MIRIFULONG_34LONG_DISTORTION_8B.05.00.fits'
        
    rootdir=os.path.join(rootdir,'data/fits/cdp8b/')
    reffile=os.path.join(rootdir,file)
   
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
        x=np.array(xin)*1.
        y=np.array(yin)*1.
    except:
        numpoints=1
        x=np.array([xin])*1.
        y=np.array([yin])*1.

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
    # Use the 50% throughput slice map, based on OTIS data
    d2c_slice=d2c_slice_all[4,:,:]

    # Define slice for these pixels
    slicenum=np.zeros(x.size, int)
    slicename=np.array(['JUNK' for i in range(0,x.size)])
 
    for i in range(0,x.size):
        slicenum[i]=int(d2c_slice[int(round(y[i])),int(round(x[i]))])-int(ch)*100
        slicename[i]=str(int(d2c_slice[int(round(y[i])),int(round(x[i]))]))+sband
        
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

    # Determine slices
    slicefloat=np.array((trimbe-beta0)/dbeta+1)
    slicenum=(np.round(slicefloat)).astype(int)
    slicename=np.array(['JUNK' for i in range(0,trimal.size)])
    for i in range(0,trimal.size):
        slicename[i]=str(slicenum[i]+int(ch)*100)+sband

    # Read the Slice Map
    c2d_slice_all=distfile['Slice_Number'].data
    # Use the 50% throughput slice map based on OTIS data
    c2d_slicefull=c2d_slice_all[4,:,:]

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
# These values are all CDP-8b placeholders mocked up by DRL!
mrs_ref_data = {
    '1A': {'x': np.array([123, 468]),
           'y': np.array([245, 1000]),
           's': np.array([9, 12]),
           'alpha': np.array([0.10731135, 1.12529977]),
           'beta': np.array([-0.35442029,  0.17721014]),
           'lam': np.array([5.11499695, 5.74016832]),
           'v2': np.array([-503.49916454, -502.56838429]),
           'v3': np.array([-318.40628359, -319.08393052]),
           },
    '1B': {'x': np.array([51, 244]),
           'y': np.array([1016,  476]),
           's': np.array([21, 17]),
           'alpha': np.array([0.38199789, 0.63723143]),
           'beta': np.array([1.77204177, 1.06322506]),
           'lam': np.array([6.62421656, 6.12990972]),
           'v2': np.array([-503.5315322 , -503.17667842]),
           'v3': np.array([-320.71381511, -320.05014591]),
           },
    '1C': {'x': np.array([127, 394]),
           'y': np.array([747, 111]),
           's': np.array([9, 3]),
           'alpha': np.array([0.37487296, -0.87620923]),
           'beta': np.array([-0.35440438, -1.4176175]),
           'lam': np.array([7.26440645, 6.52961571]),
           'v2': np.array([-503.19062471, -504.27276937]),
           'v3': np.array([-318.31059564, -317.08599443]),
           },
    '2A': {'x': np.array([574, 913]),
           'y': np.array([578, 163]),
           's': np.array([10, 16]),
           'alpha': np.array([0.02652122, -1.44523112]),
           'beta': np.array([0.27971819, 1.9580273]),
           'lam': np.array([8.22398597, 7.66495464]),
           'v2': np.array([-503.65420691, -505.38172957]),
           'v3': np.array([-319.37148692, -320.82933868]),
           },
    '2B': {'x': np.array([634, 955]),
           'y': np.array([749,  12]),
           's': np.array([11, 17]),
           'alpha': np.array([-1.31986085, -1.66029886]),
           'beta': np.array([0.5594521 , 2.23780842]),
           'lam': np.array([9.85535403, 8.65341739]),
           'v2': np.array([-505.18703764, -505.80250684]),
           'v3': np.array([-319.7057936 , -321.32425399]),
           },
    '2C': {'x': np.array([530, 884]),
           'y': np.array([965, 346]),
           's': np.array([1, 7]),
           'alpha': np.array([1.17219936, -0.13199122]),
           'beta': np.array([-2.23777695, -0.55944424]),
           'lam': np.array([11.68798183, 10.65732315]),
           'v2': np.array([-502.0634552 , -503.62291245]),
           'v3': np.array([-317.2417194 , -318.70820411]),
           },
    '3A': {'x': np.array([573, 913]),
           'y': np.array([851, 323]),
           's': np.array([8, 10]),
           'alpha': np.array([-1.0181757 ,  0.65295329]),
           'beta': np.array([-0.19490689,  0.58472067]),
           'lam': np.array([11.84245153, 12.96396074]),
           'v2': np.array([-505.35888594, -503.7824966]),
           'v3': np.array([-318.46913272, -319.4685406]),
           },
    '3B': {'x': np.array([606, 861]),
           'y': np.array([926, 366]),
           's': np.array([15, 11]),
           'alpha': np.array([-1.5124193 , -0.79361415]),
           'beta': np.array([2.53378956, 0.97453445]),
           'lam': np.array([13.60306079, 14.94878428]),
           'v2': np.array([-505.82191056, -504.9372123]),
           'v3': np.array([-321.34413558, -319.90108102]),
           },
    '3C': {'x': np.array([663, 852]),
           'y': np.array([822,  86]),
           's': np.array([14, 11]),
           'alpha': np.array([0.83845626, -1.00005387]),
           'beta': np.array([2.14397578, 0.97453445]),
           'lam': np.array([16.01468948, 17.97678143]),
           'v2': np.array([-503.52817761, -505.23700039]),
           'v3': np.array([-321.27004219, -319.84577337]),
           },
    '4A': {'x': np.array([448, 409]),
           'y': np.array([873,  49]),
           's': np.array([1, 7]),
           'alpha': np.array([-0.45466621, -1.07614592]),
           'beta': np.array([-3.60820915,  0.32801901]),
           'lam': np.array([18.05366191, 20.88016154]),
           'v2': np.array([-502.89806847, -504.25439193]),
           'v3': np.array([-315.86847223, -319.65622713]),
           },
    '4B': {'x': np.array([380, 260]),
           'y': np.array([926, 325]),
           's': np.array([2, 9]),
           'alpha': np.array([1.64217386, -1.70062938]),
           'beta': np.array([-2.95217555,  1.64009753]),
           'lam': np.array([20.69573674, 23.17990504]),
           'v2': np.array([-501.01720495, -505.23791555]),
           'v3': np.array([-316.76598039, -320.79546159]),
           },
    '4C': {'x': np.array([309, 114]),
           'y': np.array([941, 196]),
           's': np.array([3, 11]),
           'alpha': np.array([1.65440228, -0.87408042]),
           'beta': np.array([-2.29611932,  2.95215341]),
           'lam': np.array([24.17180582, 27.63402178]),
           'v2': np.array([-501.1647203 , -504.64107203]),
           'v3': np.array([-317.34628   , -322.10088837]),
           }
    
}
