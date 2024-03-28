#
"""
Useful python tools for working with the MIRI LRS.
This contains FLT-2 specific code.
For LRS, FLT-2 wavelength solution uses the FLT-1 Imager distortion,
and different wavelength calibration reference files for the SLIT
and SLITLESS modes.

This version of the tools uses a standalone implementation
of the distortion solution to do the transformations,
and hooks into offline versions of the FITS reference
files contained within this github repository.  It is implemented in an almost
identical manner to the pipeline, but doesn't depend on the pipeline.

Convert JWST v2,v3 locations (in arcsec) to MIRI Imager SCA x,y pixel locations.
Note that the pipeline uses a 0-indexed detector pixel (1032x1024) convention while
SIAF uses a 1-indexed detector pixel convention.  The CDP files define
the origin such that (0,0) is the middle of the lower-left light sensitive pixel
(1024x1024),therefore also need to transform between this science frame and detector frame.

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
17-Dec-2018  Written by David Law (dlaw@stsci.edu)
19-Jul-2022  Adapt to FLT-1 (dlaw@stsci.edu)
03-Oct-2022  Adapt to FLT-2 (dlaw@stsci.edu)
"""

import os as os
import math
import sys
import numpy as np
from numpy import matlib as mb
from astropy.modeling import models
from astropy.io import fits
from scipy.interpolate import UnivariateSpline
import pdb

# Import the corresponding FLT-1 Imager distortion code
import miricoord.imager.toolversions.mirim_tools_flt1 as mt

#############################

# Return the tools version
def version():
    return 'flt2'

#############################

# Set the relevant FITS wavelengths file
def get_fitsreffile(stype):
    if (stype.lower() == 'slit'):
        basefile='data/fits/flt2/MIRI_FM_MIRIMAGE_P750L_FULL_DISTORTION_9B.00.07.fits'
    elif (stype.lower() == 'slitless'):
        basefile='data/fits/flt2/MIRI_FM_MIRIMAGE_P750L_SLITLESSPRISM_DISTORTION_9B.00.08.fits'
    else:
        print('Invalid file type: specify either slit or slitless')

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

# Convert from subarray x,y location to v2,v3,lambda
# 'stype' is 'slit', or 'slitless'
def xytov2v3lam(xin,yin,stype):
    # Open relevant distortion file
    specfile=fits.open(get_fitsreffile(stype))

    # Convert input x,y vectors to numpy arrays
    x=np.array(xin)
    y=np.array(yin)
    
    # Global header
    hdr=specfile[0].header
    # File data
    lrsdata = np.array([l for l in specfile[1].data])
    # Define zero points (in subarray frame), subarray location, subarray size
    if (stype.lower() == 'slit'):
        xsub0,ysub0=0,0
        xsub1,ysub1=1031,1023
        zero_point = hdr['imx']-1, hdr['imy']-1
    elif (stype.lower() == 'slitless'):
        xsub0,ysub0=0,528
        xsub1,ysub1=71,415
        zero_point = hdr['imxsltl']-1-xsub0, hdr['imysltl']-1-ysub0
    else:
        print('Invalid operation type: specify either slit or slitless')
        
    # In the lrsdata reference table, X_center,y_center,wavelength describe the location of the
    # centroid trace along the detector in pixels relative to nominal location.
    # The box corners for this wavelength are x0,y0(ul) x1,y1 (ur) x2,y2(lr) x3,y3(ll)
    # Use these to create the bounding box for all valid detector locations in units of subarray pixels
    xcen = lrsdata[:, 0]
    ycen = lrsdata[:, 1]
    wavetab=lrsdata[:,2]
    x0 = lrsdata[:, 3]
    y0 = lrsdata[:, 4]
    x1 = lrsdata[:, 5]
    y2 = lrsdata[:, 8]
    bb = ((x0.min() - 0.5 + zero_point[0], x1.max() + 0.5 + zero_point[0]),
          (y2.min() - 0.5 + zero_point[1], y0.max() + 0.5 + zero_point[1]))
        
    # Find the ROW of the zero point
    row_zero_point = zero_point[1]
    # Make a vector of x,y locations for every pixel in the reference row
    yrow, xrow = np.mgrid[row_zero_point:row_zero_point + 1, 0:xsub1+1]
    # And compute the v2,v3 coordinates of pixels in this reference row
    v2refrow,v3refrow=mt.xytov2v3(xrow+xsub0,yrow+ysub0,'F770W')

    # Now repeat the v2,v3, matrix from the central row so that it is copied to all of the other valid rows too
    v2_full = mb.repmat(v2refrow, int(bb[1][1]) + 1 - int(bb[1][0]), 1)
    v3_full = mb.repmat(v3refrow, int(bb[1][1]) + 1 - int(bb[1][0]), 1)
    # v2_full and v3_full now have shape (e.g. for slitless) 391x72

    # Now take these matrices and put them into tabular models that can be interpolated to find v2,v3 for arbitrary
    # x,y pixel values in the valid region.
    v2_t2d = models.Tabular2D(lookup_table=v2_full, name='v2table',
                              bounds_error=False, fill_value=np.nan)
    v3_t2d = models.Tabular2D(lookup_table=v3_full, name='v3table',
                              bounds_error=False, fill_value=np.nan)

    # Now deal with the fact that the spectral trace isn't perfectly up and down along detector.
    # This information is contained in the xcenter/ycenter values in the CDP table, but we'll handle it
    # as a simple rotation using a linear fit to this relation provided by the CDP.
    z=np.polyfit(xcen,ycen,1)
    slope=1./z[0]
    traceangle = np.arctan(slope)*180./np.pi # trace angle in degrees
    rot = models.Rotation2D(traceangle) # Rotation model   
    
    # Now include this rotation in our overall transform
    # First shift to a frame relative to the trace zeropoint, then apply the rotation
    # to correct for the curved trace.  End in a rotated frame relative to zero at the reference point
    # and where yrot is aligned with the spectral trace)
    xysubtoxyrot=models.Shift(-zero_point[0]) & models.Shift(-zero_point[1]) | rot
    # Next shift back to the subarray frame, and then map to v2v3
    xyrottov2v3=models.Shift(zero_point[0]) & models.Shift(zero_point[1]) | models.Mapping((1, 0, 1, 0)) | v2_t2d & v3_t2d

    # Compute the rotated x,y points for our inputs
    xrot,yrot=xysubtoxyrot(x,y)
    # Compute the v2,v3 points for our inputs
    v2,v3=xyrottov2v3(xrot,yrot)

    # Work out the spectral component of the transform
    # First compute the reference trace in the rotated-Y frame
    xcenrot,ycenrot=rot(xcen,ycen)
    # The input table of wavelengths isn't perfect, and the delta-wavelength steps show some unphysical behaviour
    # Therefore fit with a spline for the ycenrot->wavelength transform
    # Reverse vectors so that yinv is increasing (needed for spline fitting function)
    wavetab=lrsdata[:,2]
    yrev=ycenrot[::-1]
    wrev=wavetab[::-1]
    # Spline fit with enforced smoothness
    spl=UnivariateSpline(yrev,wrev,s=0.002)
    # Evaluate the fit at the rotated-y reference points
    waves=spl(yrot)

    return v2,v3,waves

#############################

# Convert from v2,v3,lambda to subarray x,y location
# 'stype' is 'slit', or 'slitless'
def v2v3lamtoxy(v2in,v3in,lamin,stype):
    # Open relevant distortion file
    specfile=fits.open(get_fitsreffile(stype))

    # Convert input vectors to numpy arrays
    v2=np.array(v2in)
    v3=np.array(v3in)
    lam=np.array(lamin)
    
    # Global header
    hdr=specfile[0].header
    # File data
    lrsdata = np.array([l for l in specfile[1].data])
    # Define zero points (in subarray frame), subarray location, subarray size
    if (stype.lower() == 'slit'):
        xsub0,ysub0=0,0
        xsub1,ysub1=1031,1023
        zero_point = hdr['imx']-1, hdr['imy']-1
    elif (stype.lower() == 'slitless'):
        xsub0,ysub0=0,528
        xsub1,ysub1=71,415
        zero_point = hdr['imxsltl']-1-xsub0, hdr['imysltl']-1-ysub0
    else:
        print('Invalid operation type: specify either slit or slitless')
        
    # In the lrsdata reference table, X_center,y_center,wavelength describe the location of the
    # centroid trace along the detector in pixels relative to nominal location.
    # The box corners for this wavelength are x0,y0(ul) x1,y1 (ur) x2,y2(lr) x3,y3(ll)
    # Use these to create the bounding box for all valid detector locations in units of subarray pixels
    xcen = lrsdata[:, 0]
    ycen = lrsdata[:, 1]
    wavetab=lrsdata[:,2]
    x0 = lrsdata[:, 3]
    y0 = lrsdata[:, 4]
    x1 = lrsdata[:, 5]
    y2 = lrsdata[:, 8]
    bb = ((x0.min() - 0.5 + zero_point[0], x1.max() + 0.5 + zero_point[0]),
          (y2.min() - 0.5 + zero_point[1], y0.max() + 0.5 + zero_point[1]))
        
    # Now deal with the fact that the spectral trace isn't perfectly up and down along detector.
    # This information is contained in the xcenter/ycenter values in the CDP table, but we'll handle it
    # as a simple rotation using a linear fit to this relation provided by the CDP.
    z=np.polyfit(xcen,ycen,1)
    slope=1./z[0]
    traceangle = np.arctan(slope)*180./np.pi # trace angle in degrees
    rot = models.Rotation2D(traceangle) # Rotation model   
    
    # Now include this rotation in our overall transform
    # First shift to a frame relative to the trace zeropoint, then apply the rotation
    # to correct for the curved trace.  End in a rotated frame relative to zero at the reference point
    # and where yrot is aligned with the spectral trace)
    xysubtoxyrot=models.Shift(-zero_point[0]) & models.Shift(-zero_point[1]) | rot

    # Work out the spectral component of the transform
    # First compute the reference trace in the rotated-Y frame
    xcenrot,ycenrot=rot(xcen,ycen)
    # The input table of wavelengths isn't perfect, and the delta-wavelength steps show some unphysical behaviour
    # Therefore fit with a spline for the ycenrot->wavelength transform
    # Reverse vectors so that yinv is increasing (needed for spline fitting function)
    wavetab=lrsdata[:,2]
    yrev=ycenrot[::-1]
    wrev=wavetab[::-1]
    # Spline fit with enforced smoothness
    spl=UnivariateSpline(yrev,wrev,s=0.002)
    # Evaluate the fit at the rotated-y reference points
    wavereference=spl(yrev)
    # wavereference now contains the wavelengths corresponding to regularly-sampled ycenrot, create the model
    wavemodel = models.Tabular1D(lookup_table=wavereference, points=yrev,name='waveref',bounds_error=False, fill_value=np.nan)

    # Now construct the inverse spectral transform.
    # First we need to create a spline going from wavereference -> ycenrot
    spl2=UnivariateSpline(wavereference[::-1],ycenrot,s=0.002)
    
    # Compute the nominal x,y pixels in subarray frame for this v2,v3
    xnom,ynom=mt.v2v3toxy(v2,v3,'F770W')
    xnom=xnom-xsub0
    ynom=ynom-ysub0
    # Compute this in the rotated frame
    xrot,_=xysubtoxyrot(xnom,ynom)
    # Get the real yrot from the wavelength
    yrot=spl2(lam)

    # Convert rotated x,y to subarray x,y
    xsub,ysub=xysubtoxyrot.inverse(xrot,yrot)

    return xsub,ysub

#############################

# Function to return test data about x,y,v2,v3,lam locations
# for slit and slitless cases
def testdata():
    # Slit tests
    xy_slit=np.array([[310,20],[320,350],[335,300]])
    v2v3_slit=np.array([[-413.3763212850157, -400.6917936017185],[-414.2829339963274, -400.6171534046023],[-415.96552912187417, -400.4786909939293]])
    lam_slit=np.array([13.83647336,6.56232407,8.33895567])
    stype_slit=['slit' for i in range(0,v2v3_slit.shape[0])]
    
    # Slitless tests
    xy_slitless=np.array([[12,373],[70,105],[36,257]])
    v2v3_slitless=np.array([[-375.813403412429, -345.14939927527473],[-382.2585404264871, -344.5683084799784],[-378.48233212862783, -344.9083932543407]])
    lam_slitless=np.array([5.84529359,12.58603723,9.7000971])
    stype_slitless=['slitless' for i in range(0,v2v3_slitless.shape[0])]
    
    x=[xy_slit[:,0],xy_slitless[:,0]]
    y=[xy_slit[:,1],xy_slitless[:,1]]
    v2=[v2v3_slit[:,0],v2v3_slitless[:,0]]
    v3=[v2v3_slit[:,1],v2v3_slitless[:,1]]
    lam=[lam_slit[:],lam_slitless[:]]
    stype=['slit','slitless']

    return x,y,v2,v3,lam,stype
