#
"""
Useful python tools for working with the MIRI Imager.
This contains cdp7beta3 specific code.

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
10-Oct-2018  Written by David Law (dlaw@stsci.edu)
"""

import os as os
import math
import pdb
import sys
import numpy as np
from astropy.io import fits

#############################

# Return the tools version
def version():
    return 'poly6'

#############################

# Set the relevant FITS distortion file
def get_fitsreffile():
    basefile='data/fits/poly6/MIRI_FM_MIRIMAGE_DISTORTION_Ord4.fits'

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
    SCA=np.zeros((3,x.size))
    SCA[0,:]=x-boresight[indx]['col_offset']-4
    SCA[1,:]=y-boresight[indx]['row_offset']
    SCA[2,:]=UnitVec

    # Transform to detector focal plane coordinates
    DFP=np.matmul(MI,SCA)
    DFP_X = DFP[0,:]
    DFP_Y = DFP[1,:]

    # Transform to Entrance Focal Plane coordinates
    Xout=np.array([UnitVec,DFP_X,DFP_X**2,DFP_X**3,DFP_X**4])#,DFP_X**5,DFP_X**6])
    Yout=np.array([UnitVec,DFP_Y,DFP_Y**2,DFP_Y**3,DFP_Y**4])#,DFP_Y**5,DFP_Y**6])
    Xin=np.zeros(x.size)
    Yin=np.zeros(x.size)
    for i in range(0,x.size):
        Xin[i]=np.matmul(Yout[:,i],np.matmul(AI,Xout[:,i]))
        Yin[i]=np.matmul(Yout[:,i],np.matmul(BI,Xout[:,i]))
    EFP=np.array([Xin,Yin,UnitVec])
    #pdb.set_trace()

    # Transform to XAN, YAN
    JWST_XYAN = np.matmul(TI,EFP)

    # Transform to V2,V3
    JWST=np.zeros((3,x.size))
    JWST[0,:]=JWST_XYAN[0,:]
    JWST[1,:]=-JWST_XYAN[1,:]-7.8
    JWST[2,:]=JWST_XYAN[2,:]

    # Output vectors in units of arcsec
    v2=JWST[0,:]*60.
    v3=JWST[1,:]*60.
    xan=JWST_XYAN[0,:]*60.
    yan=JWST_XYAN[1,:]*60.

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

    # Set up JWST V2,V3 arrays in units of arcmin
    JWST=np.zeros((3,v2.size))
    JWST[0,:]=v2/60.
    JWST[1,:]=v3/60.
    JWST[2,:]=UnitVec

    # Convert to XAN, YAN
    JWST_XYAN=np.zeros((3,v2.size))
    JWST_XYAN[0,:]=JWST[0,:]
    JWST_XYAN[1,:]=-JWST[1,:]-7.8
    JWST_XYAN[2,:]=JWST[2,:]

    # Compute MIRI Entrace Focal Plane coordinates
    EFP=np.matmul(T,JWST_XYAN)
    # Components
    EFP_X=EFP[0,:]
    EFP_Y=EFP[1,:]

    # Transform to MIRI Detector Focal Plane
    Xin = np.array([UnitVec,EFP_X,EFP_X**2,EFP_X**3,EFP_X**4])#,EFP_X**5,EFP_X**6])
    Yin = np.array([UnitVec,EFP_Y,EFP_Y**2,EFP_Y**3,EFP_Y**4])#,EFP_Y**5,EFP_Y**6])
    Xout=np.zeros(v2.size)
    Yout=np.zeros(v2.size)
    for i in range(0,v2.size):
        Xout[i]=np.matmul(Yin[:,i],np.matmul(A,Xin[:,i]))
        Yout[i]=np.matmul(Yin[:,i],np.matmul(B,Xin[:,i]))
    DFP=np.array([Xout,Yout,UnitVec])

    # Transform to SCA pixel position
    SCA=np.matmul(M,DFP)

    # Add boresight offset
    # Find the right boresight entry
    indx=boresight_filters.index(filter)
    x=SCA[0,:]+4+boresight[indx]['col_offset']
    y=SCA[1,:]+boresight[indx]['row_offset']
    
    distfile.close()
    
    return x,y

#############################

# Function to return test data about x,y,v2,v3 locations
# in a variety of filters
def testdata():
    # F770W tests
    v2v3_770=np.array([[-453.363,-374.069],[-480.,-348.],[-450.,-348.],[-420.,-348.],[-390.,-348.],[-480.,-378.],[-450.,-378.],[-420.,-378.],[-390.,-378.],[-480.,-408.],[-450.,-408.],[-420.,-408.],[-390.,-408.]])
    xy_770=np.array([[688.5,511.5],[946.30, 728.95],[677.07,749.13],[408.79,769.19],[138.52,789.59],[924.80,457.09],[655.68,478.39],[387.55,499.49],[117.42,520.46],[904.81,185.52],[635.59,207.87],[367.03,229.95],[96.08,251.45]],dtype=np.float) + [4,0]
    filt_770=['F770W' for i in range(0,v2v3_770.shape[0])]

    # F1800W tests
    v2v3_1800=np.array([[-453.363,-374.069],[-480.,-348.],[-450.,-348.],[-420.,-348.],[-390.,-348.],[-480.,-378.],[-450.,-378.],[-420.,-378.],[-390.,-378.],[-480.,-408.],[-450.,-408.],[-420.,-408.],[-390.,-408.]])
    xy_1800=np.array([[688.10606,512.22995],[945.91,729.68],[676.68,749.86],[408.40,769.92],[138.13,790.32],[924.41,457.82],[655.29,479.12],[387.16,500.22],[117.03,521.19],[904.42,186.25],[635.20,208.60],[366.64,230.68],[95.69,252.18]],dtype=np.float)+[4,0]
    filt_1800=['1800W' for i in range(0,v2v3_1800.shape[0])]

    x=[xy_770[:,0],xy_1800[:,0]]
    y=[xy_770[:,1],xy_1800[:,1]]
    v2=[v2v3_770[:,0],v2v3_1800[:,0]]
    v3=[v2v3_770[:,1],v2v3_1800[:,1]]
    filter=['F770W','F1800W']

    return x,y,v2,v3,filter
