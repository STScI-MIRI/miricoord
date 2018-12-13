#
"""
Useful python tools for working with the MIRI Imager.
This contains cdp7 specific code.

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
"""

import os as os
import math
import numpy as np
from astropy.io import fits
import pdb

#############################

# Return the tools version
def version():
    return 'cdp7'

#############################

# Set the relevant FITS distortion file
def get_fitsreffile():
    rootdir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    reffile=os.path.join(rootdir,'data/fits/cdp7/MIRI_FM_MIRIMAGE_DISTORTION_07.04.00.fits')
   
    return reffile

#############################

def xytov2v3(xin,yin,filter):
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
    # Boresight offsets
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

def v2v3toxy(v2in,v3in,filter):
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
    # Boresight offsets
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
# in a variety of filters
def testdata():
    # F770W tests
    v2v3_770=np.array([[-480.,-348.],[-450.,-348.],[-420.,-348.],[-390.,-348.],[-480.,-378.],[-450.,-378.],[-420,-378.],[-390.,-378.],[-480.,-408.],[-450.,-408.],[-420.,-408.],[-390.,-408.]])
    xy_770=np.array([[948.18,724.94],[676.75,745.67],[404.81,767.77],[132.65,791.34],[923.52,455.40],[653.11,476.53],[382.37,498.57],[111.34,521.66],[899.64,184.81],[629.88,206.95],[360.00,229.12],[89.77,251.55]],dtype=np.float) + [4,0]
    # Note that we had to add 4 to Alistair's x locations because he uses science pixels, not detector pixels
    filt_770=['F770W' for i in range(0,v2v3_770.shape[0])]

    # F1800W tests
    v2v3_1800=np.array([[-480.,-348.],[-450.,-348.],[-420.,-348.],[-390.,-348.],[-480.,-378.],[-450.,-378.],[-420,-378.],[-390.,-378.],[-480.,-408.],[-450.,-408.],[-420.,-408.],[-390.,-408.]])
    xy_1800=np.array([[947.79,725.67],[676.36,746.40],[404.42,768.50],[132.26,792.07],[923.13,456.13],[652.72,477.26],[381.98,499.30],[110.95,522.39],[899.25,185.54],[629.49,207.68],[359.61,229.85],[89.38,252.28]],dtype=np.float) + [4,0]
    filt_1800=['F1800W' for i in range(0,v2v3_1800.shape[0])]

    x=[xy_770[:,0],xy_1800[:,0]]
    y=[xy_770[:,1],xy_1800[:,1]]
    v2=[v2v3_770[:,0],v2v3_1800[:,0]]
    v3=[v2v3_770[:,1],v2v3_1800[:,1]]
    filter=['F770W','F1800W']

    return x,y,v2,v3,filter
