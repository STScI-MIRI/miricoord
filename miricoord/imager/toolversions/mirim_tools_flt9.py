#
"""
Useful python tools for working with the MIRI Imager.
This contains flt4 specific code.

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
    v2_770=np.array([-453.37849012, -433.44711881, -479.38722157, -449.37963073,
       -419.36112578, -389.35626367, -479.3919599 , -449.37616406,
       -419.35697586, -389.37064485, -479.40300538, -449.38210468,
       -419.37276534, -389.41978253])
    v3_770=np.array([-373.8105493 , -375.41618152, -348.02002698, -348.01117266,
       -348.00931455, -347.99931539, -378.04217418, -378.03365908,
       -378.02616193, -378.00120274, -408.06008198, -408.04722395,
       -408.03075064, -407.99078552])
    filt_770=['F770W' for i in range(0,len(v2_770))]

    # F1800W tests
    x_1800=np.array([692.5 , 511.5 , 948.18, 676.75, 404.81, 132.65, 923.52, 653.11,
       382.37, 111.34, 899.64, 629.88, 360.  ,  89.77])
    y_1800=np.array([511.5 , 511.5 , 724.94, 745.67, 767.77, 791.34, 455.4 , 476.54,
       498.57, 521.66, 184.81, 206.95, 229.12, 251.55])
    v2_1800=np.array([-453.42852496, -433.49689506, -479.4374696 , -449.4294657 ,
       -419.41049999, -389.40526905, -479.44224734, -449.42616267,
       -419.40657275, -389.41986299, -479.45301681, -449.43208008,
       -419.42252507, -389.46928253])
    v3_1800=np.array([-373.88811164, -375.49365926, -348.09744426, -348.08843891,
       -348.08629818, -348.07597295, -378.11964079, -378.11119868,
       -378.10353636, -378.07825983, -408.13640009, -408.12394963,
       -408.10753408, -408.06736492])
    filt_1800=['F1800W' for i in range(0,len(v2_1800))]

    x=[x_770,x_1800]
    y=[y_770,y_1800]
    v2=[v2_770,v2_1800]
    v3=[v3_770,v3_1800]
    filter=['F770W','F1800W']

    return x,y,v2,v3,filter
