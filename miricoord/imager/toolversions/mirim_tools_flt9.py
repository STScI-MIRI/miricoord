#
"""
Useful python tools for working with the MIRI Imager.
This contains flt9 specific code.

FLT-9 updates the MIRI Imaging SIAF boresight to fix an offset relative to FGS.
Version number advances substantially from the previous FLT-3 to bring it into agreement
with the MRS, which is being given an identical shift to maintain same relative position
to the imager.

This version of the tools uses a standalone implementation
of the distortion solution to do the transformations,
and hooks into offline versions of the FITS reference
files contained within this github repository.

Convert JWST v2,v3 locations (in arcsec) to MIRI Imager SCA x,y pixel locations.
Note that the pipeline uses a 0-indexed detector pixel (1032x1024) convention while
SIAF uses a 1-indexed detector pixel convention.  The CDP files define
the origin such that (0,0) is the middle of the lower-left light sensitive pixel
(1024x1024),therefore also need to transform between this science frame and detector frame.

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
19-Oct-2018  Written by David Law (dlaw@stsci.edu)
26-May-2022  Port to FLT-1 (dlaw@stsci.edu)
17-Oct-2022  Add coron-specific boresights for FLT-2 (dlaw@stsci.edu)
16-May-2023  Coron-specific boresights for FLT-3 (dlaw@stsci.edu)
05-Jun-2025  Shift boresight for FLT-9 (dlaw@stsci.edu)
"""

import os as os
import math
import numpy as np
from astropy.io import fits
import pdb
import sys

#############################

# Return the tools version
def version():
    return 'flt9'

#############################

# Set the relevant FITS distortion file
def get_fitsreffile():
    basefile='data/fits/flt9/MIRI_FM_MIRIMAGE_DISTORTION_SS.09.00.fits'

    # Try looking for the file in the expected location
    rootdir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    reffile=os.path.join(rootdir,basefile)
    if os.path.exists(reffile):
        return reffile
    
    # If that didn't work, look in the system path
    rootdir=sys.prefix
    reffile=os.path.join(rootdir,basefile)
    if os.path.exists(reffile):
        return reffile    

    # If that didn't work either, just return what we've got
    return reffile

#############################

def xytov2v3(xin,yin,filter,**kwargs):
    # Open relevant distortion file
    distfile=fits.open(get_fitsreffile())
    
    # Convert input x,y vectors to numpy arrays
    x=np.array(xin)
    y=np.array(yin)

    # Global header
    hdr=distfile[0].header

    # AI-matrix
    AI=distfile['AI matrix'].data
    # BI-matrix
    BI=distfile['BI matrix'].data
    # TI-matrix
    TI=distfile['TI matrix'].data
    # MI-matrix
    MI=distfile['MI matrix'].data
    
    # Boresight offsets default to using the main extension; use coron-specific boresight
    # offsets if keyword is set
    if ('coron' in kwargs):
        if (kwargs['coron'] == 'CORON1065'):
            boresight=distfile['BoresightCORON1065'].data
        elif (kwargs['coron'] == 'CORON1140'):
            boresight=distfile['BoresightCORON1140'].data
        elif (kwargs['coron'] == 'CORON1550'):
            boresight=distfile['BoresightCORON1550'].data
        elif (kwargs['coron'] == 'CORONLYOT'):
            boresight=distfile['BoresightCORONLYOT'].data
        else:
            boresight=distfile['Boresight offsets'].data
    else:
        boresight=distfile['Boresight offsets'].data

    # Convert the boresight names to a string array
    boresight_filters=['' for boresight_filters in range(boresight.size)]
    for i in range(0,boresight.size):
        boresight_filters[i]=str(boresight['filter'][i])

    # How many input points?
    if (x.size != y.size):
        print('Input xpixel,ypixel array size mismatch!')

    UnitVec=np.ones(x.size)

    # Find the right boresight entry
    indx=boresight_filters.index(filter)

    # Transform to SCA pixel position without boresight offsets,
    # also shifting from detector frame to science frame
    PFP=np.zeros((3,x.size))
    PFP[0,:]=x-boresight[indx]['col_offset']-4
    PFP[1,:]=y-boresight[indx]['row_offset']
    PFP[2,:]=UnitVec

    # Transform to detector focal plane coordinates
    DFP=np.matmul(MI,PFP)
    DFP_X = DFP[0,:]
    DFP_Y = DFP[1,:]

    # Transform to Rotated Focal Plane coordinates
    Xout=np.array([UnitVec,DFP_X,DFP_X**2,DFP_X**3,DFP_X**4])
    Yout=np.array([UnitVec,DFP_Y,DFP_Y**2,DFP_Y**3,DFP_Y**4])
    Xin=np.zeros(x.size)
    Yin=np.zeros(x.size)
    for i in range(0,x.size):
        Xin[i]=np.matmul(Yout[:,i],np.matmul(AI,Xout[:,i]))
        Yin[i]=np.matmul(Yout[:,i],np.matmul(BI,Xout[:,i]))
    RFP=np.array([Xin,Yin,UnitVec])

    # Transform to V2,V3
    JWST=np.matmul(TI,RFP)

    # Output vectors in units of arcsec
    v2=JWST[0,:]
    v3=JWST[1,:]
    
    distfile.close()
    
    return v2,v3

#############################

def v2v3toxy(v2in,v3in,filter,**kwargs):
    distfile=fits.open(get_fitsreffile())
    
    # Convert input x,y vectors to numpy arrays
    v2=np.array(v2in)
    v3=np.array(v3in)

    # Global header
    hdr=distfile[0].header

    # A-matrix
    A=distfile['A matrix'].data
    # B-matrix
    B=distfile['B matrix'].data
    # T-matrix
    T=distfile['T matrix'].data
    # M-matrix
    M=distfile['M matrix'].data
    
    # Boresight offsets default to using the main extension; use coron-specific boresight
    # offsets if keyword is set
    if ('coron' in kwargs):
        if (kwargs['coron'] == 'CORON1065'):
            boresight=distfile['BoresightCORON1065'].data
        elif (kwargs['coron'] == 'CORON1140'):
            boresight=distfile['BoresightCORON1140'].data
        elif (kwargs['coron'] == 'CORON1550'):
            boresight=distfile['BoresightCORON1550'].data
        elif (kwargs['coron'] == 'CORONLYOT'):
            boresight=distfile['BoresightCORONLYOT'].data
        else:
            boresight=distfile['Boresight offsets'].data
    else:
        boresight=distfile['Boresight offsets'].data

    # Convert the boresight names to a string array
    boresight_filters=['' for boresight_filters in range(boresight.size)]
    for i in range(0,boresight.size):
        boresight_filters[i]=str(boresight['filter'][i])

    # How many input points?
    if (v2.size != v3.size):
        print('Input xpixel,ypixel array size mismatch!')

    UnitVec=np.ones(v2.size)

    # Set up JWST V2,V3 arrays in units of arcsec
    JWST=np.zeros((3,v2.size))
    JWST[0,:]=v2
    JWST[1,:]=v3
    JWST[2,:]=UnitVec

    # Compute MIRI Rotated Focal Plane coordinates
    RFP=np.matmul(T,JWST)
    # Components
    RFP_X=RFP[0,:]
    RFP_Y=RFP[1,:]

    # Transform to MIRI Detector Focal Plane
    Xin = np.array([UnitVec,RFP_X,RFP_X**2,RFP_X**3,RFP_X**4])
    Yin = np.array([UnitVec,RFP_Y,RFP_Y**2,RFP_Y**3,RFP_Y**4])
    Xout=np.zeros(v2.size)
    Yout=np.zeros(v2.size)
    for i in range(0,v2.size):
        Xout[i]=np.matmul(Yin[:,i],np.matmul(A,Xin[:,i]))
        Yout[i]=np.matmul(Yin[:,i],np.matmul(B,Xin[:,i]))
    DFP=np.array([Xout,Yout,UnitVec])

    # Transform to Pixel Focal Plane
    PFP=np.matmul(M,DFP)

    # Add boresight offset
    # Find the right boresight entry

    indx=boresight_filters.index(filter)
    x=PFP[0,:]+4+boresight[indx]['col_offset']
    y=PFP[1,:]+boresight[indx]['row_offset']

    distfile.close()
    
    return x,y

#############################

# Function to return test data about x,y,v2,v3 locations
# in two filter based on hand computations by DRL.
def testdata():
    # F770W tests, xy are 0-indexed detector pixels
    x_770=np.array([692.5 , 511.5 , 948.18, 676.75, 404.81, 132.65, 923.52, 653.11,
       382.37, 111.34, 899.64, 629.88, 360.  ,  89.77])
    y_770=np.array([511.5 , 511.5 , 724.94, 745.67, 767.77, 791.34, 455.4 , 476.54,
       498.57, 521.66, 184.81, 206.95, 229.12, 251.55])
    v2_770=np.array([-453.34829012, -433.41691881, -479.35702157, -449.34943073,
        -419.33092578, -389.32606367, -479.3617599 , -449.34596406,
        -419.32677586, -389.34044485, -479.37280538, -449.35190468,
        -419.34256534, -389.38958253])
    v3_770=np.array([-373.6679493 , -375.27358152, -347.87742698, -347.86857266,
        -347.86671455, -347.85671539, -377.89957418, -377.89105908,
        -377.88356193, -377.85860274, -407.91748198, -407.90462395,
        -407.88815064, -407.84818552])
    filt_770=['F770W' for i in range(0,len(v2_770))]

    # F1800W tests
    x_1800=np.array([692.5 , 511.5 , 948.18, 676.75, 404.81, 132.65, 923.52, 653.11,
       382.37, 111.34, 899.64, 629.88, 360.  ,  89.77])
    y_1800=np.array([511.5 , 511.5 , 724.94, 745.67, 767.77, 791.34, 455.4 , 476.54,
       498.57, 521.66, 184.81, 206.95, 229.12, 251.55])
    v2_1800=np.array([-453.39832496, -433.46669506, -479.4072696 , -449.3992657 ,
        -419.38029999, -389.37506905, -479.41204734, -449.39596267,
        -419.37637275, -389.38966299, -479.42281681, -449.40188008,
        -419.39232507, -389.43908253])
    v3_1800=np.array([-373.74551164, -375.35105926, -347.95484426, -347.94583891,
        -347.94369818, -347.93337295, -377.97704079, -377.96859868,
        -377.96093636, -377.93565983, -407.99380009, -407.98134963,
        -407.96493408, -407.92476492])
    filt_1800=['F1800W' for i in range(0,len(v2_1800))]

    x=[x_770,x_1800]
    y=[y_770,y_1800]
    v2=[v2_770,v2_1800]
    v3=[v3_770,v3_1800]
    filter=['F770W','F1800W']

    return x,y,v2,v3,filter
