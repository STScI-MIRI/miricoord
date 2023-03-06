#
"""
Python tools for creating the MIRI MRS dither sequences for a given
set of distortion files.  These functions will be called from the
associated notebook front-end.

Beta dithers: Ch1 long offset is 5.5 times the Ch1 width because that will be
half-integer for all other channels.  Ch 2/3/4 are odd multiples of the 5.5 times
slice width offset so that it will similarly be half-integer for all channels.
Short beta dithers are 1.5x integer in the local channel because we're trying to
optimize for this channel and using larger than 0.5x offset will significantly
help simultaneous imaging by having a larger throw.

Alpha dithers both long and short offsets are defined related to the channel-specific
pixel size because we're optimizing for that channel.

However, alpha/beta axes in Ch1 aren't quite aligned with other channels, and alpha
sampling changes discontinuously between slices, so attemps to do half-integer alpha
while changing slices aren't perfect.  Assess performance using dedicated simulation
code.

Add additional tweak-ups so that that pattern is centered for a given channel.

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
15-Apr-2019  Adapt from old IDL routines (D. Law; dlaw@stsci.edu)
05-Aug-2020  Remove dither flips for mirisim per ticket MIRI-677 (D. Law)
08-Jun-2021  Fold JDox figure creation into these functions instead of notebook (D. Law)
27-May-2022  Update for FLT-1 distortion (D. Law)
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

import os as os
import math
import numpy as np
import datetime
from astropy.io import fits
import csv
import getpass
import socket

import pdb

import miricoord.mrs.mrs_tools as mrst
import miricoord.mrs.makesiaf.makesiaf_mrs as makesiaf

#############################

# Global record of pixel sizes so that we only need to calculate once

def setsizes(**kwargs):
    global pixsize,slicewidth
    slicewidth=np.array([mrst.slicewidth('1A'),mrst.slicewidth('2A'),mrst.slicewidth('3A'),mrst.slicewidth('4A')])
    pixsize=np.array([mrst.pixsize('1A'),mrst.pixsize('2A'),mrst.pixsize('3A'),mrst.pixsize('4A')])

    if ('verbose' in kwargs):
        print('Slice widths: ',slicewidth)
        print('Pixel sizes: ',pixsize)
    
    return

#############################

# Generate the 8-position point source pattern
# given a set of input long and short offsets
# in the alpha and beta directions.  Note that positions 3 and 4 (plus 7 and 8)
# are swapped relative to the IDT original pattern.

def makepattern_generic(astart,along,ashort,bstart,blong,bshort):
    pattern_alpha=np.array([0,along,ashort,along+ashort,along+ashort,ashort,along,0])+astart
    pattern_beta=np.array([blong+bshort,bshort,blong,0,blong+bshort,bshort,blong,0])-blong-bshort+bstart
    
    return pattern_alpha,pattern_beta

#############################

# A routine to return rough maximum PSF FWHM in arcsec for a given channel

def maxfwhm(channel):
    # Maximum wavelength in microns for the channel
    wave=0.
    if (channel == 1): wave=8.0
    if (channel == 2): wave=12.0
    if (channel == 3): wave=18.0
    if (channel == 4): wave=29.0

    return 0.31*wave/8.0
    
#############################

# A routine to recenter a given dither pattern within a particular channel FOV
# Must be passed SIAF structures for all 3 bands within a channel

def recenterFOV(pat_v2,pat_v3,siafA,siafB,siafC): 
    # Average the three mean field positions
    v2_fieldmean=(siafA['inscr_v2ref']+siafB['inscr_v2ref']+siafC['inscr_v2ref'])/3.
    v3_fieldmean=(siafA['inscr_v3ref']+siafB['inscr_v3ref']+siafC['inscr_v3ref'])/3.
    v2_mean=np.mean(pat_v2)
    v3_mean=np.mean(pat_v3)

    newpat_v2 = pat_v2 - v2_mean + v2_fieldmean
    newpat_v3 = pat_v3 - v3_mean + v3_fieldmean

    return newpat_v2,newpat_v3

#############################

# A routine to recenter a given dither pattern with respect to a given channel reference point
# (which is not quite the same thing as centering wrt the FOV)
# Must be passed SIAF structure for a given band

def recenterRP(pat_v2,pat_v3,siaf):
    v2ref,v3ref=siaf['inscr_v2ref'],siaf['inscr_v3ref']
    
    v2_mean=np.mean(pat_v2)
    v3_mean=np.mean(pat_v3)

    newpat_v2 = pat_v2 - v2_mean + v2ref
    newpat_v3 = pat_v3 - v3_mean + v3ref

    return newpat_v2,newpat_v3

#############################

# Generate the commissioning Ch1 point-source patterns
# SIAF structures for the 3 bands must be passed in

def makepattern_ch1(siafA,siafB,siafC):
    # See if the pixel and slice sizes have already been calculated
    try:
        slicewidth
        pixsize
    # If not, calculate them
    except:
        setsizes()
        
    pixsiz1=pixsize[0]# Ch1
    pixsiz=pixsize[0]# Ch1
    slicesiz1=slicewidth[0]# Ch1
    slicesiz=slicewidth[0]# Ch1

    along=10.5*pixsiz
    ashort=1.5*pixsiz
    astart=0
    blong=5.5*slicesiz1
    bshort=1.5*slicesiz
    bstart=0

    pat_a,pat_b=makepattern_generic(astart,along,ashort,bstart,blong,bshort)
    # Transform assuming input in Ch1A alpha-beta
    pat_v2,pat_v3=mrst.abtov2v3(pat_a,pat_b,'1A')
    
    # Recenter the pattern within the field
    pat_v2,pat_v3=recenterFOV(pat_v2,pat_v3,siafA,siafB,siafC)

    return pat_v2,pat_v3

#############################

# Generate the commissioning Ch2 point-source patterns
# SIAF structures for the 3 bands must be passed in

def makepattern_ch2(siafA,siafB,siafC):
    # See if the pixel and slice sizes have already been calculated
    try:
        slicewidth
        pixsize
    # If not, calculate them
    except:
        print('Recalculating pixel/slice sizes')
        setsizes()
        
    pixsiz1=pixsize[0]# Ch1
    pixsiz=pixsize[1]# Ch2
    slicesiz1=slicewidth[0]# Ch1
    slicesiz=slicewidth[1]# Ch2

    along=10.5*pixsiz
    ashort=1.5*pixsiz
    astart=0
    blong=8.5*slicesiz
    bshort=1.5*slicesiz
    bstart=0

    pat_a,pat_b=makepattern_generic(astart,along,ashort,bstart,blong,bshort)
    # Transform assuming input pattern in Ch2A alpha-beta
    pat_v2,pat_v3=mrst.abtov2v3(pat_a,pat_b,'2A')

    # Recenter the pattern within the field
    pat_v2,pat_v3=recenterFOV(pat_v2,pat_v3,siafA,siafB,siafC)

    return pat_v2,pat_v3

#############################

# Generate the commissioning Ch3 point-source patterns
# SIAF structures for the 3 bands must be passed in

def makepattern_ch3(siafA,siafB,siafC):
    # See if the pixel and slice sizes have already been calculated
    try:
        slicewidth
        pixsize
    # If not, calculate them
    except:
        setsizes()
        
    pixsiz1=pixsize[0]# Ch1
    pixsiz=pixsize[2]# Ch3
    slicesiz1=slicewidth[0]# Ch1
    slicesiz=slicewidth[2]# Ch3

    along=10.5*pixsiz
    ashort=1.5*pixsiz
    astart=0
    blong=16.5*slicesiz1
    bshort=1.5*slicesiz
    bstart=0

    pat_a,pat_b=makepattern_generic(astart,along,ashort,bstart,blong,bshort)
    # Transform assuming input in Ch3A alpha-beta
    pat_v2,pat_v3=mrst.abtov2v3(pat_a,pat_b,'3A')

    # Recenter the pattern within the field
    pat_v2,pat_v3=recenterFOV(pat_v2,pat_v3,siafA,siafB,siafC)
    
    return pat_v2,pat_v3

#############################

# Generate the commissioning Ch4 point-source patterns
# SIAF structures for the 3 bands must be passed in

def makepattern_ch4(siafA,siafB,siafC):
    # See if the pixel and slice sizes have already been calculated
    try:
        slicewidth
        pixsize
    # If not, calculate them
    except:
        setsizes()
        
    pixsiz1=pixsize[0]# Ch1
    pixsiz=pixsize[3]# Ch4
    slicesiz1=slicewidth[0]# Ch1
    slicesiz=slicewidth[3]# Ch4

    along=12.5*pixsiz
    ashort=1.5*pixsiz
    astart=0
    blong=5.5*slicesiz
    bshort=1.5*slicesiz
    bstart=0

    pat_a,pat_b=makepattern_generic(astart,along,ashort,bstart,blong,bshort)
    # Transform assuming input in Ch4A alpha-beta
    pat_v2,pat_v3=mrst.abtov2v3(pat_a,pat_b,'4A')
    # Recenter the pattern within the field
    pat_v2,pat_v3=recenterFOV(pat_v2,pat_v3,siafA,siafB,siafC)

    return pat_v2,pat_v3

#############################

# Routine to generate the extended-source pattern optimized for ALL channels

def makepattern_ext_all(siafA,siafB,siafC):
    # See if the pixel and slice sizes have already been calculated
    try:
        slicewidth
        pixsize
    # If not, calculate them
    except:
        setsizes()

    # Ch3 and Ch4 are well-sampled in the pixel direction already, so optimize the along-slice
    # offset to be half-integer in Ch1 and Ch2
    da=pixsize[0]*3/2.# Ch1
    # Use the mathematically related slice widths in each channel to construct a half-integer
    # offset for all channels
    db=slicewidth[0]*5.5# Ch1

    pat_a=np.array([-da/2.,da/2.,da/2.,-da/2.])
    pat_b=np.array([db/2.,-db/2.,db/2.,-db/2.])

    # Transform assuming input in Ch1A alpha-beta
    pat_v2,pat_v3=mrst.abtov2v3(pat_a,pat_b,'1A')

    # Recenter the pattern within the field
    pat_v2,pat_v3=recenterFOV(pat_v2,pat_v3,siafA,siafB,siafC)
    
    return pat_v2,pat_v3

#############################

# Routine to generate the extended-source pattern optimized for a given channel

def makepattern_ext_ChX(ptpat_v2,ptpat_v3,siaf):
    # First dither pair; pull out short-dithers with parity = 1
    temp1_v2=ptpat_v2[[0,2]]
    temp1_v3=ptpat_v3[[0,2]]
    # Recenter within field
    temp1_v2,temp1_v3=recenterRP(temp1_v2,temp1_v3,siaf)

    # Second dither pair; pull out short-dithers with parity = -1
    temp2_v2=ptpat_v2[[4,6]]
    temp2_v3=ptpat_v3[[4,6]]
    # Recenter within field
    temp2_v2,temp2_v3=recenterRP(temp2_v2,temp2_v3,siaf)

    # Combine the dither pairs
    pat_v2,pat_v3=np.append(temp1_v2,temp2_v2),np.append(temp1_v3,temp2_v3)
    # And recenter the combined dithers
    pat_v2,pat_v3=recenterRP(pat_v2,pat_v3,siaf)
    
    return pat_v2,pat_v3

#############################

# Routine to convert fixed v2,v3 dither points into actual xideal,yideal offsets
# relative to the fiducial reference point for a given channel
# Must be passed the siafRP structure containing the reference point to be used,
# and optionally the siaf1A structure used to define Ideal coordinates

def compute_dxdyideal(pat_v2,pat_v3,siaf,**kwargs):
    v2ref,v3ref=siaf['inscr_v2ref'],siaf['inscr_v3ref']
    # Ideal coordinate of the dither position
    x,y=mrst.v2v3_to_xyideal(pat_v2,pat_v3,**kwargs)
    # Ideal coordinate of the fiducial (undithered) point
    xref,yref=mrst.v2v3_to_xyideal(v2ref,v3ref,**kwargs)
    # Delta offsets
    dxidl=x-xref
    dyidl=y-yref

    return dxidl,dyidl

#############################

# Routine to write results to a file formatted for the PRD

def writeresults_prd(dxidl,dyidl,outdir=''):
    # Set the default output data directory if it was not provided
    if (outdir == ''):
        data_dir=os.path.expandvars('$MIRICOORD_DATA_DIR')
        relfile='dithers/temp/MiriMrsDithers.txt'
        outfile=os.path.join(data_dir,relfile)
    else:
        relfile='MiriMrsDithers.txt'
        outfile=os.path.join(outdir,relfile)

    now=datetime.datetime.now()
    thisfile=__file__
    _,thisfile=os.path.split(thisfile)

    # No header information is allowed, and specific names must be given for each set of points
    # which makes the file quite fragile

    print('CHANNEL1-POINT_SOURCE-NEGATIVE',file=open(outfile,"w"))
    for ii in range(0,4):
        print("{0:<3}{1:>10.6f}       {2:>10.6f}".format(ii+1,dxidl[ii],dyidl[ii]),file=open(outfile,"a"))
    print('',file=open(outfile,"a"))

    print('CHANNEL1-POINT_SOURCE-POSITIVE',file=open(outfile,"a"))
    for ii in range(4,8):
        print("{0:<3}{1:>10.6f}       {2:>10.6f}".format(ii-3,dxidl[ii],dyidl[ii]),file=open(outfile,"a"))
    print('',file=open(outfile,"a"))

    print('CHANNEL2-POINT_SOURCE-NEGATIVE',file=open(outfile,"a"))
    for ii in range(8,12):
        print("{0:<3}{1:>10.6f}       {2:>10.6f}".format(ii-7,dxidl[ii],dyidl[ii]),file=open(outfile,"a"))
    print('',file=open(outfile,"a"))

    print('CHANNEL2-POINT_SOURCE-POSITIVE',file=open(outfile,"a"))
    for ii in range(12,16):
        print("{0:<3}{1:>10.6f}       {2:>10.6f}".format(ii-11,dxidl[ii],dyidl[ii]),file=open(outfile,"a"))
    print('',file=open(outfile,"a"))

    print('CHANNEL3-POINT_SOURCE-NEGATIVE',file=open(outfile,"a"))
    for ii in range(16,20):
        print("{0:<3}{1:>10.6f}       {2:>10.6f}".format(ii-15,dxidl[ii],dyidl[ii]),file=open(outfile,"a"))
    print('',file=open(outfile,"a"))

    print('CHANNEL3-POINT_SOURCE-POSITIVE',file=open(outfile,"a"))
    for ii in range(20,24):
        print("{0:<3}{1:>10.6f}       {2:>10.6f}".format(ii-19,dxidl[ii],dyidl[ii]),file=open(outfile,"a"))
    print('',file=open(outfile,"a"))

    print('CHANNEL4-POINT_SOURCE-NEGATIVE',file=open(outfile,"a"))
    for ii in range(24,28):
        print("{0:<3}{1:>10.6f}       {2:>10.6f}".format(ii-23,dxidl[ii],dyidl[ii]),file=open(outfile,"a"))
    print('',file=open(outfile,"a"))

    print('CHANNEL4-POINT_SOURCE-POSITIVE',file=open(outfile,"a"))
    for ii in range(28,32):
        print("{0:<3}{1:>10.6f}       {2:>10.6f}".format(ii-27,dxidl[ii],dyidl[ii]),file=open(outfile,"a"))
    print('',file=open(outfile,"a"))

    print('ALL-EXTENDED_SOURCE',file=open(outfile,"a"))
    for ii in range(32,36):
        print("{0:<3}{1:>10.6f}       {2:>10.6f}".format(ii-31,dxidl[ii],dyidl[ii]),file=open(outfile,"a"))
    print('',file=open(outfile,"a"))

    print('CHANNEL1-EXTENDED_SOURCE',file=open(outfile,"a"))
    for ii in range(36,40):
        print("{0:<3}{1:>10.6f}       {2:>10.6f}".format(ii-35,dxidl[ii],dyidl[ii]),file=open(outfile,"a"))
    print('',file=open(outfile,"a"))
   
    print('CHANNEL2-EXTENDED_SOURCE',file=open(outfile,"a"))
    for ii in range(40,44):
        print("{0:<3}{1:>10.6f}       {2:>10.6f}".format(ii-39,dxidl[ii],dyidl[ii]),file=open(outfile,"a"))
    print('',file=open(outfile,"a"))

    print('CHANNEL3-EXTENDED_SOURCE',file=open(outfile,"a"))
    for ii in range(44,48):
        print("{0:<3}{1:>10.6f}       {2:>10.6f}".format(ii-43,dxidl[ii],dyidl[ii]),file=open(outfile,"a"))
    print('',file=open(outfile,"a"))

    print('CHANNEL4-EXTENDED_SOURCE',file=open(outfile,"a"))
    for ii in range(48,52):
        print("{0:<3}{1:>10.6f}       {2:>10.6f}".format(ii-47,dxidl[ii],dyidl[ii]),file=open(outfile,"a"))
    print('',file=open(outfile,"a"))

    print('DEDICATED-BG',file=open(outfile,"a"))
    for ii in range(52,56):
        print("{0:<3}{1:>10.6f}       {2:>10.6f}".format(ii-51,dxidl[ii],dyidl[ii]),file=open(outfile,"a"))
    print('',file=open(outfile,"a"))
    
    print('SCAN-CALIBRATION',file=open(outfile,"a"))
    for ii in range(56,len(dxidl)):
        print("{0:<3}{1:>10.6f}       {2:>10.6f}".format(ii-55,dxidl[ii],dyidl[ii]),file=open(outfile,"a"))
    print('',file=open(outfile,"a"))

    # Log the operation is complete
    print('Successfully wrote PRD dithers file to ',relfile)
    
#############################

# Routine to write results to a file formatted for mirisim

def writeresults_mirisim(ch,v2,v3,outdir=''):
    # Set the default output data directory if it was not provided
    if (outdir == ''):
        data_dir=os.path.expandvars('$MIRICOORD_DATA_DIR')
        relfile='dithers/temp/mrs_recommended_dither.dat'
        outfile=os.path.join(data_dir,relfile)
    else:
        relfile='mrs_recommended_dither.dat'
        outfile=os.path.join(outdir,relfile)

    now=datetime.datetime.now()
    thisfile=__file__
    _,thisfile=os.path.split(thisfile)

    # Mirisim specifies dithers in alpha-beta, so we need to convert to that frame from v2-v3
    # It looks like we should always use the 1A reference point and coordinate system,
    # even though this is strictly ill-defed outside 1A footprint
    ndither=len(v2)
    dalpha,dbeta=np.zeros(ndither),np.zeros(ndither)
    band=["" for ii in range(ndither)]
    for ii in range(0,ndither):
        thisch=ch[ii]
        # Reference point of this channel
        if (thisch == 1): band[ii]='1A'
        if (thisch == 2): band[ii]='1A'
        if (thisch == 3): band[ii]='1A'
        if (thisch == 4): band[ii]='1A'
        dalpha[ii],dbeta[ii]=mrst.v2v3toab(v2[ii],v3[ii],band[ii])

    # As of mirisim 2.3.0 (ticket MIRI-677) we no longer need to invert dithers
    # to work properly with mirisim
        
    # Write header information to the output text file
    print('# Default MIRISim dither pattern for MRS.',file=open(outfile,"w"))
    print('#',file=open(outfile,"a"))
    print('# Created ',now.isoformat(),file=open(outfile,"a"))
    print('# Using program miricoord.',thisfile,file=open(outfile,"a"))
    print('#',file=open(outfile,"a"))
    print('# Offsets are defined in the MRS 1A channel-band field-of-view,',file=open(outfile,"a"))
    print('# and are tabulated as (alpha, beta) coordinates (in units of arcsec)',file=open(outfile,"a"))
    print('# relative to initial pointing center at (0, 0).',file=open(outfile,"a"))
    print('#',file=open(outfile,"a"))
    print('',file=open(outfile,"a"))
    
    # Mirisim doesn't use column names, they are taken as a given
    # It also add comments before each set, so we'll need to break them up
    # Note that this makes the code less robust against changes!

    print('# Optimized for channel 1 point sources.',file=open(outfile,"a"))
    for ii in range(0,8):
        print("{0:>7.4f}, {1:>7.4f}".format(dalpha[ii],dbeta[ii]),file=open(outfile,"a"))
    print('',file=open(outfile,"a"))
    
    print('# Optimized for channel 2 point sources.',file=open(outfile,"a"))
    for ii in range(8,16):
        print("{0:>7.4f}, {1:>7.4f}".format(dalpha[ii],dbeta[ii]),file=open(outfile,"a"))  
    print('',file=open(outfile,"a"))
    
    print('# Optimized for channel 3 point sources.',file=open(outfile,"a"))
    for ii in range(16,24):
        print("{0:>7.4f}, {1:>7.4f}".format(dalpha[ii],dbeta[ii]),file=open(outfile,"a"))  
    print('',file=open(outfile,"a"))
    
    print('# Optimized for channel 4 point sources.',file=open(outfile,"a"))
    for ii in range(24,32):
        print("{0:>7.4f}, {1:>7.4f}".format(dalpha[ii],dbeta[ii]),file=open(outfile,"a"))  
    print('',file=open(outfile,"a"))
    
    print('# Optimized for ALL channel extended sources.',file=open(outfile,"a"))
    for ii in range(32,36):
        print("{0:>7.4f}, {1:>7.4f}".format(dalpha[ii],dbeta[ii]),file=open(outfile,"a"))  
    print('',file=open(outfile,"a"))
    
    print('# Optimized for channel 1 extended sources.',file=open(outfile,"a"))
    for ii in range(36,40):
        print("{0:>7.4f}, {1:>7.4f}".format(dalpha[ii],dbeta[ii]),file=open(outfile,"a"))  
    print('',file=open(outfile,"a"))
    
    print('# Optimized for channel 2 extended sources.',file=open(outfile,"a"))
    for ii in range(40,44):
        print("{0:>7.4f}, {1:>7.4f}".format(dalpha[ii],dbeta[ii]),file=open(outfile,"a"))
    print('',file=open(outfile,"a"))
    
    print('# Optimized for channel 3 extended sources.',file=open(outfile,"a"))
    for ii in range(44,48):
        print("{0:>7.4f}, {1:>7.4f}".format(dalpha[ii],dbeta[ii]),file=open(outfile,"a"))
    print('',file=open(outfile,"a"))
    
    print('# Optimized for channel 4 extended sources.',file=open(outfile,"a"))
    for ii in range(48,52):
        print("{0:>7.4f}, {1:>7.4f}".format(dalpha[ii],dbeta[ii]),file=open(outfile,"a"))  

    # Log the operation is complete
    print('Successfully wrote mirisim dithers file to ',relfile)
        
#############################

# Routine to write full results to a file

def writeresults_full(index,ch,v2,v3,dxidl,dyidl,outdir=''):
    # Set the default output data directory if it was not provided
    if (outdir == ''):
        data_dir=os.path.expandvars('$MIRICOORD_DATA_DIR')
        relfile='dithers/temp/mrs_dithers.txt'
        outfile=os.path.join(data_dir,relfile)
    else:
        relfile='mrs_dithers.txt'
        outfile=os.path.join(outdir,relfile)

    now=datetime.datetime.now()
    thisfile=__file__
    _,thisfile=os.path.split(thisfile)

    # Save info in alpha-beta too
    ndither=len(v2)
    dalpha,dbeta=np.zeros(ndither),np.zeros(ndither)
    band=["" for ii in range(ndither)]
    for ii in range(0,ndither):
        thisch=ch[ii]
        # Reference point of this channel
        if (thisch == 1): band[ii]='1A'
        if (thisch == 2): band[ii]='2A'
        if (thisch == 3): band[ii]='3A'
        if (thisch == 4): band[ii]='4A'
        dalpha[ii],dbeta[ii]=mrst.v2v3toab(v2[ii],v3[ii],band[ii])
    
    # Write header information to the output text file
    print('# Created ',now.isoformat(),file=open(outfile,"w"))
    print('# Using program',thisfile,file=open(outfile,"a"))
    
    # Column names
    print("{:<10} {:<8} {:<10} {:<10} {:<10} {:<10} {:<15} {:<15}".format('PosnIndex','Band','alpha','beta','V2','V3','dXIdeal','dYIdeal'),file=open(outfile,"a"))
    
    for i in range(0,len(index)):
        # Write information to a text file
        print("{0:<10} {1:<8} {2:<10.5f} {3:<10.5f} {4:<10.5f} {5:<10.5f} {6:<15.5f} {7:<15.5f}".format(index[i],band[i],dalpha[i],dbeta[i],v2[i],v3[i],dxidl[i],dyidl[i]),file=open(outfile,"a"))

    # Log the operation is complete
    print('Successfully wrote full dithers file to ',relfile)

#############################

# Make assorted plots for JDox
def make_jdox(v2_all,v3_all,dx_all,dy_all,allsiaf,vertxt,outdir='./'):
    qaplot_ptsourceloc(v2_all,v3_all,allsiaf,vertxt,outdir)
    qaplot_extsourceloc(v2_all,v3_all,allsiaf,vertxt,outdir)
    qaplot_ps4all(v2_all,v3_all,dx_all,dy_all,allsiaf,vertxt,outdir)
    qaplot_ps2ch4(v2_all,v3_all,dx_all,dy_all,allsiaf,vertxt,outdir)
    qaplot_ext2all(v2_all,v3_all,dx_all,dy_all,allsiaf,vertxt,outdir)
    qaplot_ext4all(v2_all,v3_all,dx_all,dy_all,allsiaf,vertxt,outdir)
    qaplot_ext2ch3(v2_all,v3_all,dx_all,dy_all,allsiaf,vertxt,outdir)
    qaplot_ext4ch3(v2_all,v3_all,dx_all,dy_all,allsiaf,vertxt,outdir)

#############################

# Plot showing the location of the point-source dithers

def qaplot_ptsourceloc(v2,v3,allsiaf,vertxt,outdir=''):
    # Set the default output data directory if it was not provided
    if (outdir == ''):
        data_dir=os.path.expandvars('$MIRICOORD_DATA_DIR')
        outdir=os.path.join(data_dir,'dithers/temp/')
    # Set the output filename
    filename=os.path.join(outdir,'dithers_pt.pdf')

    now=datetime.datetime.now()
    nowstring=now.ctime()
    nowstring=nowstring[4:8]+nowstring[20:24]
    
    # Field locations
    siaf1A=allsiaf[0]
    siaf1B=allsiaf[1]
    siaf1C=allsiaf[2]
    siaf2A=allsiaf[3]
    siaf2B=allsiaf[4]
    siaf2C=allsiaf[5]
    siaf3A=allsiaf[6]
    siaf3B=allsiaf[7]
    siaf3C=allsiaf[8]
    siaf4A=allsiaf[9]
    siaf4B=allsiaf[10]
    siaf4C=allsiaf[11]
    
    # Plot thickness
    mpl.rcParams['axes.linewidth'] = 1.5

    plt.figure(figsize=(5,5),dpi=150)
    plt.tick_params(axis='both',direction='in',which='both',top=True,right=True,width=2.)
    plt.minorticks_on()
    plt.xlim(-497.4,-509.4)
    plt.ylim(-325.8,-313.8)

    plt.plot(siaf1A['inscr_v2_corners'],siaf1A['inscr_v3_corners'],color='b',linewidth=1.5,label='Ch1')
    plt.plot(siaf1B['inscr_v2_corners'],siaf1B['inscr_v3_corners'],color='b',linewidth=1.5,linestyle='--')
    plt.plot(siaf1C['inscr_v2_corners'],siaf1C['inscr_v3_corners'],color='b',linewidth=1.5,linestyle=':')
    plt.plot(siaf2A['inscr_v2_corners'],siaf2A['inscr_v3_corners'],color='g',linewidth=1.5,label='Ch2')
    plt.plot(siaf2B['inscr_v2_corners'],siaf2B['inscr_v3_corners'],color='g',linewidth=1.5,linestyle='--')
    plt.plot(siaf2C['inscr_v2_corners'],siaf2C['inscr_v3_corners'],color='g',linewidth=1.5,linestyle=':')
    plt.plot(siaf3A['inscr_v2_corners'],siaf3A['inscr_v3_corners'],color='gold',linewidth=1.5,label='Ch3')
    plt.plot(siaf3B['inscr_v2_corners'],siaf3B['inscr_v3_corners'],color='gold',linewidth=1.5,linestyle='--')
    plt.plot(siaf3C['inscr_v2_corners'],siaf3C['inscr_v3_corners'],color='gold',linewidth=1.5,linestyle=':')
    plt.plot(siaf4A['inscr_v2_corners'],siaf4A['inscr_v3_corners'],color='r',linewidth=1.5,label='Ch4')
    plt.plot(siaf4B['inscr_v2_corners'],siaf4B['inscr_v3_corners'],color='r',linewidth=1.5,linestyle='--')
    plt.plot(siaf4C['inscr_v2_corners'],siaf4C['inscr_v3_corners'],color='r',linewidth=1.5,linestyle=':')

    plt.plot(v2[0:8],v3[0:8],'+',color='b',linewidth=1.5)
    plt.plot(v2[8:16],v3[8:16],'+',color='g')
    plt.plot(v2[16:24],v3[16:24],'+',color='gold')
    plt.plot(v2[24:32],v3[24:32],'+',color='r')
            
    plt.xlabel('V2 (arcsec)')
    plt.ylabel('V3 (arcsec)')
    plt.title('MRS Dithers: Flight '+vertxt+' ('+nowstring+')')
    plt.legend()
    
    plt.savefig(filename)
    #plt.show()
    plt.close()

#############################

# Plot showing the location of the extended-source dithers

def qaplot_extsourceloc(v2,v3,allsiaf,vertxt,outdir=''):
    # Set the default output data directory if it was not provided
    if (outdir == ''):
        data_dir=os.path.expandvars('$MIRICOORD_DATA_DIR')
        outdir=os.path.join(data_dir,'dithers/temp/')
    # Set the output filename
    filename=os.path.join(outdir,'dithers_ext.pdf')

    now=datetime.datetime.now()
    nowstring=now.ctime()
    nowstring=nowstring[4:8]+nowstring[20:24]
    
    # Field locations
    siaf1A=allsiaf[0]
    siaf1B=allsiaf[1]
    siaf1C=allsiaf[2]
    siaf2A=allsiaf[3]
    siaf2B=allsiaf[4]
    siaf2C=allsiaf[5]
    siaf3A=allsiaf[6]
    siaf3B=allsiaf[7]
    siaf3C=allsiaf[8]
    siaf4A=allsiaf[9]
    siaf4B=allsiaf[10]
    siaf4C=allsiaf[11]
    
    # Plot thickness
    mpl.rcParams['axes.linewidth'] = 1.5

    plt.figure(figsize=(5,5),dpi=150)
    plt.tick_params(axis='both',direction='in',which='both',top=True,right=True,width=2.)
    plt.minorticks_on()
    plt.xlim(-497.4,-509.4)
    plt.ylim(-325.8,-313.8)

    plt.plot(siaf1A['inscr_v2_corners'],siaf1A['inscr_v3_corners'],color='b',linewidth=1.5,label='Ch1')
    plt.plot(siaf1B['inscr_v2_corners'],siaf1B['inscr_v3_corners'],color='b',linewidth=1.5,linestyle='--')
    plt.plot(siaf1C['inscr_v2_corners'],siaf1C['inscr_v3_corners'],color='b',linewidth=1.5,linestyle=':')
    plt.plot(siaf2A['inscr_v2_corners'],siaf2A['inscr_v3_corners'],color='g',linewidth=1.5,label='Ch2')
    plt.plot(siaf2B['inscr_v2_corners'],siaf2B['inscr_v3_corners'],color='g',linewidth=1.5,linestyle='--')
    plt.plot(siaf2C['inscr_v2_corners'],siaf2C['inscr_v3_corners'],color='g',linewidth=1.5,linestyle=':')
    plt.plot(siaf3A['inscr_v2_corners'],siaf3A['inscr_v3_corners'],color='gold',linewidth=1.5,label='Ch3')
    plt.plot(siaf3B['inscr_v2_corners'],siaf3B['inscr_v3_corners'],color='gold',linewidth=1.5,linestyle='--')
    plt.plot(siaf3C['inscr_v2_corners'],siaf3C['inscr_v3_corners'],color='gold',linewidth=1.5,linestyle=':')
    plt.plot(siaf4A['inscr_v2_corners'],siaf4A['inscr_v3_corners'],color='r',linewidth=1.5,label='Ch4')
    plt.plot(siaf4B['inscr_v2_corners'],siaf4B['inscr_v3_corners'],color='r',linewidth=1.5,linestyle='--')
    plt.plot(siaf4C['inscr_v2_corners'],siaf4C['inscr_v3_corners'],color='r',linewidth=1.5,linestyle=':')

    plt.plot(v2[32:36],v3[32:36],'+',color='black',linewidth=1.5)
    plt.plot(v2[36:40],v3[36:40],'+',color='b',linewidth=1.5)
    plt.plot(v2[40:44],v3[40:44],'+',color='g')
    plt.plot(v2[44:48],v3[44:48],'+',color='gold')
    plt.plot(v2[48:52],v3[48:52],'+',color='r')
            
    plt.xlabel('V2 (arcsec)')
    plt.ylabel('V3 (arcsec)')
    plt.title('MRS Dithers: Flight '+vertxt+' ('+nowstring+')')
    plt.legend()
    
    plt.savefig(filename)
    #plt.show()
    plt.close()
    
#############################

# Plot showing field coverage of a 4-pt ALL point-source dither

def qaplot_ps4all(v2,v3,dx,dy,allsiaf,vertxt,outdir=''):
    # Set the default output data directory if it was not provided
    if (outdir == ''):
        data_dir=os.path.expandvars('$MIRICOORD_DATA_DIR')
        outdir=os.path.join(data_dir,'dithers/temp/')
    # Set the output filename
    filename=os.path.join(outdir,'dithers_ps4ALL.pdf')

    now=datetime.datetime.now()
    nowstring=now.ctime()
    nowstring=nowstring[4:8]+nowstring[20:24]
    
    # Field locations
    siaf1A=allsiaf[0]
    siaf1B=allsiaf[1]
    siaf1C=allsiaf[2]
    siaf2A=allsiaf[3]
    siaf2B=allsiaf[4]
    siaf2C=allsiaf[5]
    siaf3A=allsiaf[6]
    siaf3B=allsiaf[7]
    siaf3C=allsiaf[8]
    siaf4A=allsiaf[9]
    siaf4B=allsiaf[10]
    siaf4C=allsiaf[11]

    # Recenter everything to be based around zero
    v2ref,v3ref=siaf1A['inscr_v2ref'],siaf1A['inscr_v3ref']
    v2corn_1A=siaf1A['inscr_v2_corners']-v2ref
    v3corn_1A=siaf1A['inscr_v3_corners']-v3ref
    v2corn_2A=siaf2A['inscr_v2_corners']-v2ref
    v3corn_2A=siaf2A['inscr_v3_corners']-v3ref
    v2corn_3A=siaf3A['inscr_v2_corners']-v2ref
    v3corn_3A=siaf3A['inscr_v3_corners']-v3ref 
    v2corn_4A=siaf4A['inscr_v2_corners']-v2ref
    v3corn_4A=siaf4A['inscr_v3_corners']-v3ref

    # Plot thickness
    mpl.rcParams['axes.linewidth'] = 1.5

    plt.figure(figsize=(5,5),dpi=150)
    ax = plt.gca()
    plt.tick_params(axis='both',direction='in',which='both',top=True,right=True,width=2.)
    plt.minorticks_on()
    plt.xlim(6,-6)
    plt.ylim(-6,6)

    plt.plot(v2corn_1A+dx[0],v3corn_1A-dy[0],color='b',linewidth=1.2,label='Ch1')
    plt.plot(v2corn_1A+dx[1],v3corn_1A-dy[1],color='b',linewidth=1.2)
    plt.plot(v2corn_1A+dx[2],v3corn_1A-dy[2],color='b',linewidth=1.2)
    plt.plot(v2corn_1A+dx[3],v3corn_1A-dy[3],color='b',linewidth=1.2)

    plt.plot(v2corn_2A+dx[0],v3corn_2A-dy[0],color='g',linewidth=1.2,label='Ch2')
    plt.plot(v2corn_2A+dx[1],v3corn_2A-dy[1],color='g',linewidth=1.2)
    plt.plot(v2corn_2A+dx[2],v3corn_2A-dy[2],color='g',linewidth=1.2)
    plt.plot(v2corn_2A+dx[3],v3corn_2A-dy[3],color='g',linewidth=1.2)

    plt.plot(v2corn_3A+dx[0],v3corn_3A-dy[0],color='gold',linewidth=1.2,label='Ch3')
    plt.plot(v2corn_3A+dx[1],v3corn_3A-dy[1],color='gold',linewidth=1.2)
    plt.plot(v2corn_3A+dx[2],v3corn_3A-dy[2],color='gold',linewidth=1.2)
    plt.plot(v2corn_3A+dx[3],v3corn_3A-dy[3],color='gold',linewidth=1.2)
    
    plt.plot(v2corn_4A+dx[0],v3corn_4A-dy[0],color='r',linewidth=1.2,label='Ch4')
    plt.plot(v2corn_4A+dx[1],v3corn_4A-dy[1],color='r',linewidth=1.2)
    plt.plot(v2corn_4A+dx[2],v3corn_4A-dy[2],color='r',linewidth=1.2)
    plt.plot(v2corn_4A+dx[3],v3corn_4A-dy[3],color='r',linewidth=1.2)

    plt.plot(0,0,'x',linewidth=1.5,color='black')
    circle1 = mpl.patches.Circle((0., 0.), maxfwhm(1),linewidth=1,edgecolor='b', facecolor=(0, 0, 0, .0125))
    ax.add_artist(circle1)
    circle1 = mpl.patches.Circle((0., 0.), maxfwhm(4),linewidth=1,edgecolor='r', facecolor=(0, 0, 0, .0125))
    ax.add_artist(circle1)
                                 
    plt.plot(v2[0:4]-v2ref,v3[0:4]-v3ref,'+',color='b',linewidth=1.5)
    
    plt.xlabel('$\Delta$ R.A. (arcsec)')
    plt.ylabel('$\Delta$ Decl. (arcsec)')
    plt.title('MRS Dithers: Flight '+vertxt+' ('+nowstring+')')
    plt.text(1,5,'ALL, 4-PT, point source')
    plt.legend()

    plt.savefig(filename)
    #plt.show()
    plt.close()

#############################

# Plot showing field coverage of a 2-pt Ch4 point-source dither

def qaplot_ps2ch4(v2,v3,dx,dy,allsiaf,vertxt,outdir=''):
    # Set the default output data directory if it was not provided
    if (outdir == ''):
        data_dir=os.path.expandvars('$MIRICOORD_DATA_DIR')
        outdir=os.path.join(data_dir,'dithers/temp/')
    # Set the output filename
    filename=os.path.join(outdir,'dithers_ps2ch4.pdf')

    now=datetime.datetime.now()
    nowstring=now.ctime()
    nowstring=nowstring[4:8]+nowstring[20:24]
    
    # Field locations
    siaf1A=allsiaf[0]
    siaf1B=allsiaf[1]
    siaf1C=allsiaf[2]
    siaf2A=allsiaf[3]
    siaf2B=allsiaf[4]
    siaf2C=allsiaf[5]
    siaf3A=allsiaf[6]
    siaf3B=allsiaf[7]
    siaf3C=allsiaf[8]
    siaf4A=allsiaf[9]
    siaf4B=allsiaf[10]
    siaf4C=allsiaf[11]

    # Recenter everything to be based around zero
    v2ref,v3ref=siaf4A['inscr_v2ref'],siaf4A['inscr_v3ref']
    v2corn_1A=siaf1A['inscr_v2_corners']-v2ref
    v3corn_1A=siaf1A['inscr_v3_corners']-v3ref
    v2corn_2A=siaf2A['inscr_v2_corners']-v2ref
    v3corn_2A=siaf2A['inscr_v3_corners']-v3ref
    v2corn_3A=siaf3A['inscr_v2_corners']-v2ref
    v3corn_3A=siaf3A['inscr_v3_corners']-v3ref  
    v2corn_4A=siaf4A['inscr_v2_corners']-v2ref
    v3corn_4A=siaf4A['inscr_v3_corners']-v3ref

    # Plot thickness
    mpl.rcParams['axes.linewidth'] = 1.5

    plt.figure(figsize=(5,5),dpi=150)
    ax = plt.gca()
    plt.tick_params(axis='both',direction='in',which='both',top=True,right=True,width=2.)
    plt.minorticks_on()
    plt.xlim(8,-8)
    plt.ylim(-8,8)

    plt.plot(v2corn_1A+dx[24],v3corn_1A-dy[24],color='b',linewidth=1.2,label='Ch1')
    plt.plot(v2corn_1A+dx[25],v3corn_1A-dy[25],color='b',linewidth=1.2)

    plt.plot(v2corn_2A+dx[24],v3corn_2A-dy[24],color='g',linewidth=1.2,label='Ch2')
    plt.plot(v2corn_2A+dx[25],v3corn_2A-dy[25],color='g',linewidth=1.2)

    plt.plot(v2corn_3A+dx[24],v3corn_3A-dy[24],color='gold',linewidth=1.2,label='Ch3')
    plt.plot(v2corn_3A+dx[25],v3corn_3A-dy[25],color='gold',linewidth=1.2)
    
    plt.plot(v2corn_4A+dx[24],v3corn_4A-dy[24],color='r',linewidth=1.2,label='Ch4')
    plt.plot(v2corn_4A+dx[25],v3corn_4A-dy[25],color='r',linewidth=1.2)

    plt.plot(0,0,'x',linewidth=1.5,color='black')
    circle1 = mpl.patches.Circle((0., 0.), maxfwhm(1),linewidth=1,edgecolor='b', facecolor=(0, 0, 0, .0125))
    ax.add_artist(circle1)
    circle1 = mpl.patches.Circle((0., 0.), maxfwhm(4),linewidth=1,edgecolor='r', facecolor=(0, 0, 0, .0125))
    ax.add_artist(circle1)
                                 
    plt.plot(v2[24:26]-v2ref,v3[24:26]-v3ref,'+',color='r',linewidth=1.5)
    
    plt.xlabel('$\Delta$ R.A. (arcsec)')
    plt.ylabel('$\Delta$ Decl. (arcsec)')
    plt.title('MRS Dithers: Flight '+vertxt+' ('+nowstring+')')
    plt.text(7,7,'Ch4, 2-PT, point source')
    plt.legend()

    plt.savefig(filename)
    #plt.show()
    plt.close()

#############################

# Plot showing field coverage of a 2-pt ALL extended-source dither

def qaplot_ext2all(v2,v3,dx,dy,allsiaf,vertxt,outdir=''):
    # Set the default output data directory if it was not provided
    if (outdir == ''):
        data_dir=os.path.expandvars('$MIRICOORD_DATA_DIR')
        outdir=os.path.join(data_dir,'dithers/temp/')
    # Set the output filename
    filename=os.path.join(outdir,'dithers_ext2all.pdf')

    now=datetime.datetime.now()
    nowstring=now.ctime()
    nowstring=nowstring[4:8]+nowstring[20:24]
    
    # Field locations
    siaf1A=allsiaf[0]
    siaf1B=allsiaf[1]
    siaf1C=allsiaf[2]
    siaf2A=allsiaf[3]
    siaf2B=allsiaf[4]
    siaf2C=allsiaf[5]
    siaf3A=allsiaf[6]
    siaf3B=allsiaf[7]
    siaf3C=allsiaf[8]
    siaf4A=allsiaf[9]
    siaf4B=allsiaf[10]
    siaf4C=allsiaf[11]

    # Recenter everything to be based around zero
    v2ref,v3ref=siaf1A['inscr_v2ref'],siaf1A['inscr_v3ref']
    v2corn_1A=siaf1A['inscr_v2_corners']-v2ref
    v3corn_1A=siaf1A['inscr_v3_corners']-v3ref
    v2corn_2A=siaf2A['inscr_v2_corners']-v2ref
    v3corn_2A=siaf2A['inscr_v3_corners']-v3ref
    v2corn_3A=siaf3A['inscr_v2_corners']-v2ref
    v3corn_3A=siaf3A['inscr_v3_corners']-v3ref  
    v2corn_4A=siaf4A['inscr_v2_corners']-v2ref
    v3corn_4A=siaf4A['inscr_v3_corners']-v3ref
    
    # Plot thickness
    mpl.rcParams['axes.linewidth'] = 1.5

    plt.figure(figsize=(5,5),dpi=150)
    ax = plt.gca()
    plt.tick_params(axis='both',direction='in',which='both',top=True,right=True,width=2.)
    plt.minorticks_on()
    plt.xlim(6,-6)
    plt.ylim(-6,6)

    plt.plot(v2corn_1A+dx[32],v3corn_1A-dy[32],color='b',linewidth=1.2,label='Ch1')
    plt.plot(v2corn_1A+dx[33],v3corn_1A-dy[33],color='b',linewidth=1.2)

    plt.plot(v2corn_2A+dx[32],v3corn_2A-dy[32],color='g',linewidth=1.2,label='Ch2')
    plt.plot(v2corn_2A+dx[33],v3corn_2A-dy[33],color='g',linewidth=1.2)

    plt.plot(v2corn_3A+dx[32],v3corn_3A-dy[32],color='gold',linewidth=1.2,label='Ch3')
    plt.plot(v2corn_3A+dx[33],v3corn_3A-dy[33],color='gold',linewidth=1.2)
    
    plt.plot(v2corn_4A+dx[32],v3corn_4A-dy[32],color='r',linewidth=1.2,label='Ch4')
    plt.plot(v2corn_4A+dx[33],v3corn_4A-dy[33],color='r',linewidth=1.2)

    plt.plot(0,0,'x',linewidth=1.5,color='black')
    circle1 = mpl.patches.Circle((0., 0.), maxfwhm(1),linewidth=1,edgecolor='b', facecolor=(0, 0, 0, .0125))
    ax.add_artist(circle1)
    circle1 = mpl.patches.Circle((0., 0.), maxfwhm(4),linewidth=1,edgecolor='r', facecolor=(0, 0, 0, .0125))
    ax.add_artist(circle1)
                                 
    plt.plot(v2[32:34]-v2ref,v3[32:34]-v3ref,'+',color='b',linewidth=1.5)
    
    plt.xlabel('$\Delta$ R.A. (arcsec)')
    plt.ylabel('$\Delta$ Decl. (arcsec)')
    plt.title('MRS Dithers: Flight '+vertxt+' ('+nowstring+')')
    plt.text(5,5,'ALL, 2-PT, extended source')
    plt.legend()

    plt.savefig(filename)
    #plt.show()
    plt.close()

#############################

# Plot showing field coverage of a 4-pt ALL extended-source dither

def qaplot_ext4all(v2,v3,dx,dy,allsiaf,vertxt,outdir=''):
    # Set the default output data directory if it was not provided
    if (outdir == ''):
        data_dir=os.path.expandvars('$MIRICOORD_DATA_DIR')
        outdir=os.path.join(data_dir,'dithers/temp/')
    # Set the output filename
    filename=os.path.join(outdir,'dithers_ext4all.pdf')

    now=datetime.datetime.now()
    nowstring=now.ctime()
    nowstring=nowstring[4:8]+nowstring[20:24]
    
    # Field locations
    siaf1A=allsiaf[0]
    siaf1B=allsiaf[1]
    siaf1C=allsiaf[2]
    siaf2A=allsiaf[3]
    siaf2B=allsiaf[4]
    siaf2C=allsiaf[5]
    siaf3A=allsiaf[6]
    siaf3B=allsiaf[7]
    siaf3C=allsiaf[8]
    siaf4A=allsiaf[9]
    siaf4B=allsiaf[10]
    siaf4C=allsiaf[11]

    # Recenter everything to be based around zero
    v2ref,v3ref=siaf1A['inscr_v2ref'],siaf1A['inscr_v3ref']
    v2corn_1A=siaf1A['inscr_v2_corners']-v2ref
    v3corn_1A=siaf1A['inscr_v3_corners']-v3ref
    v2corn_2A=siaf2A['inscr_v2_corners']-v2ref
    v3corn_2A=siaf2A['inscr_v3_corners']-v3ref
    v2corn_3A=siaf3A['inscr_v2_corners']-v2ref
    v3corn_3A=siaf3A['inscr_v3_corners']-v3ref  
    v2corn_4A=siaf4A['inscr_v2_corners']-v2ref
    v3corn_4A=siaf4A['inscr_v3_corners']-v3ref
    
    # Plot thickness
    mpl.rcParams['axes.linewidth'] = 1.5

    plt.figure(figsize=(5,5),dpi=150)
    ax = plt.gca()
    plt.tick_params(axis='both',direction='in',which='both',top=True,right=True,width=2.)
    plt.minorticks_on()
    plt.xlim(6,-6)
    plt.ylim(-6,6)

    plt.plot(v2corn_1A+dx[32],v3corn_1A-dy[32],color='b',linewidth=1.2,label='Ch1')
    plt.plot(v2corn_1A+dx[33],v3corn_1A-dy[33],color='b',linewidth=1.2)
    plt.plot(v2corn_1A+dx[34],v3corn_1A-dy[34],color='b',linewidth=1.2)
    plt.plot(v2corn_1A+dx[35],v3corn_1A-dy[35],color='b',linewidth=1.2)

    plt.plot(v2corn_2A+dx[32],v3corn_2A-dy[32],color='g',linewidth=1.2,label='Ch2')
    plt.plot(v2corn_2A+dx[33],v3corn_2A-dy[33],color='g',linewidth=1.2)
    plt.plot(v2corn_2A+dx[34],v3corn_2A-dy[34],color='g',linewidth=1.2)
    plt.plot(v2corn_2A+dx[35],v3corn_2A-dy[35],color='g',linewidth=1.2)

    plt.plot(v2corn_3A+dx[32],v3corn_3A-dy[32],color='gold',linewidth=1.2,label='Ch3')
    plt.plot(v2corn_3A+dx[33],v3corn_3A-dy[33],color='gold',linewidth=1.2)
    plt.plot(v2corn_3A+dx[34],v3corn_3A-dy[34],color='gold',linewidth=1.2)
    plt.plot(v2corn_3A+dx[35],v3corn_3A-dy[35],color='gold',linewidth=1.2)
    
    plt.plot(v2corn_4A+dx[32],v3corn_4A-dy[32],color='r',linewidth=1.2,label='Ch4')
    plt.plot(v2corn_4A+dx[33],v3corn_4A-dy[33],color='r',linewidth=1.2)
    plt.plot(v2corn_4A+dx[34],v3corn_4A-dy[34],color='r',linewidth=1.2)
    plt.plot(v2corn_4A+dx[35],v3corn_4A-dy[35],color='r',linewidth=1.2)
        
    plt.plot(0,0,'x',linewidth=1.5,color='black')
    circle1 = mpl.patches.Circle((0., 0.), maxfwhm(1),linewidth=1,edgecolor='b', facecolor=(0, 0, 0, .0125))
    ax.add_artist(circle1)
    circle1 = mpl.patches.Circle((0., 0.), maxfwhm(4),linewidth=1,edgecolor='r', facecolor=(0, 0, 0, .0125))
    ax.add_artist(circle1)
                                 
    plt.plot(v2[32:36]-v2ref,v3[32:36]-v3ref,'+',color='b',linewidth=1.5)
    
    plt.xlabel('$\Delta$ R.A. (arcsec)')
    plt.ylabel('$\Delta$ Decl. (arcsec)')
    plt.title('MRS Dithers: Flight '+vertxt+' ('+nowstring+')')
    plt.text(5,5,'ALL, 4-PT, extended source')
    plt.legend()

    plt.savefig(filename)
    #plt.show()
    plt.close()
    
#############################

# Plot showing field coverage of a 2-pt Ch3 extended-source dither

def qaplot_ext2ch3(v2,v3,dx,dy,allsiaf,vertxt,outdir=''):
    # Set the default output data directory if it was not provided
    if (outdir == ''):
        data_dir=os.path.expandvars('$MIRICOORD_DATA_DIR')
        outdir=os.path.join(data_dir,'dithers/temp/')
    # Set the output filename
    filename=os.path.join(outdir,'dithers_ext2ch3.pdf')

    now=datetime.datetime.now()
    nowstring=now.ctime()
    nowstring=nowstring[4:8]+nowstring[20:24]
    
    # Field locations
    siaf1A=allsiaf[0]
    siaf1B=allsiaf[1]
    siaf1C=allsiaf[2]
    siaf2A=allsiaf[3]
    siaf2B=allsiaf[4]
    siaf2C=allsiaf[5]
    siaf3A=allsiaf[6]
    siaf3B=allsiaf[7]
    siaf3C=allsiaf[8]
    siaf4A=allsiaf[9]
    siaf4B=allsiaf[10]
    siaf4C=allsiaf[11]

    # Recenter everything to be based around zero
    v2ref,v3ref=siaf3A['inscr_v2ref'],siaf3A['inscr_v3ref']
    v2corn_1A=siaf1A['inscr_v2_corners']-v2ref
    v3corn_1A=siaf1A['inscr_v3_corners']-v3ref
    v2corn_2A=siaf2A['inscr_v2_corners']-v2ref
    v3corn_2A=siaf2A['inscr_v3_corners']-v3ref
    v2corn_3A=siaf3A['inscr_v2_corners']-v2ref
    v3corn_3A=siaf3A['inscr_v3_corners']-v3ref 
    v2corn_4A=siaf4A['inscr_v2_corners']-v2ref
    v3corn_4A=siaf4A['inscr_v3_corners']-v3ref

    # Plot thickness
    mpl.rcParams['axes.linewidth'] = 1.5

    plt.figure(figsize=(5,5),dpi=150)
    ax = plt.gca()
    plt.tick_params(axis='both',direction='in',which='both',top=True,right=True,width=2.)
    plt.minorticks_on()
    plt.xlim(6,-6)
    plt.ylim(-6,6)

    plt.plot(v2corn_1A+dx[44],v3corn_1A-dy[44],color='b',linewidth=1.2,label='Ch1')
    plt.plot(v2corn_1A+dx[45],v3corn_1A-dy[45],color='b',linewidth=1.2)

    plt.plot(v2corn_2A+dx[44],v3corn_2A-dy[44],color='g',linewidth=1.2,label='Ch2')
    plt.plot(v2corn_2A+dx[45],v3corn_2A-dy[45],color='g',linewidth=1.2)
    
    plt.plot(v2corn_3A+dx[44],v3corn_3A-dy[44],color='gold',linewidth=1.2,label='Ch3')
    plt.plot(v2corn_3A+dx[45],v3corn_3A-dy[45],color='gold',linewidth=1.2)
    
    plt.plot(v2corn_4A+dx[44],v3corn_4A-dy[44],color='r',linewidth=1.2,label='Ch4')
    plt.plot(v2corn_4A+dx[45],v3corn_4A-dy[45],color='r',linewidth=1.2)

    plt.plot(0,0,'x',linewidth=1.5,color='black')
    circle1 = mpl.patches.Circle((0., 0.), maxfwhm(1),linewidth=1,edgecolor='b', facecolor=(0, 0, 0, .0125))
    ax.add_artist(circle1)
    circle1 = mpl.patches.Circle((0., 0.), maxfwhm(4),linewidth=1,edgecolor='r', facecolor=(0, 0, 0, .0125))
    ax.add_artist(circle1)
                                 
    plt.plot(v2[44:46]-v2ref,v3[44:46]-v3ref,'+',color='gold',linewidth=1.5)
    
    plt.xlabel('$\Delta$ R.A. (arcsec)')
    plt.ylabel('$\Delta$ Decl. (arcsec)')
    plt.title('MRS Dithers: Flight '+vertxt+' ('+nowstring+')')
    plt.text(5,5,'Ch3, 2-PT, extended source')
    plt.legend()

    plt.savefig(filename)
    #plt.show()
    plt.close()
        
#############################

# Plot showing field coverage of a 4-pt Ch3 extended-source dither

def qaplot_ext4ch3(v2,v3,dx,dy,allsiaf,vertxt,outdir=''):
    # Set the default output data directory if it was not provided
    if (outdir == ''):
        data_dir=os.path.expandvars('$MIRICOORD_DATA_DIR')
        outdir=os.path.join(data_dir,'dithers/temp/')
    # Set the output filename
    filename=os.path.join(outdir,'dithers_ext4ch3.pdf')

    now=datetime.datetime.now()
    nowstring=now.ctime()
    nowstring=nowstring[4:8]+nowstring[20:24]
    
    # Field locations
    siaf1A=allsiaf[0]
    siaf1B=allsiaf[1]
    siaf1C=allsiaf[2]
    siaf2A=allsiaf[3]
    siaf2B=allsiaf[4]
    siaf2C=allsiaf[5]
    siaf3A=allsiaf[6]
    siaf3B=allsiaf[7]
    siaf3C=allsiaf[8]
    siaf4A=allsiaf[9]
    siaf4B=allsiaf[10]
    siaf4C=allsiaf[11]

    # Recenter everything to be based around zero
    v2ref,v3ref=siaf3A['inscr_v2ref'],siaf3A['inscr_v3ref']
    v2corn_1A=siaf1A['inscr_v2_corners']-v2ref
    v3corn_1A=siaf1A['inscr_v3_corners']-v3ref
    v2corn_2A=siaf2A['inscr_v2_corners']-v2ref
    v3corn_2A=siaf2A['inscr_v3_corners']-v3ref
    v2corn_3A=siaf3A['inscr_v2_corners']-v2ref
    v3corn_3A=siaf3A['inscr_v3_corners']-v3ref 
    v2corn_4A=siaf4A['inscr_v2_corners']-v2ref
    v3corn_4A=siaf4A['inscr_v3_corners']-v3ref

    # Plot thickness
    mpl.rcParams['axes.linewidth'] = 1.5

    plt.figure(figsize=(5,5),dpi=150)
    ax = plt.gca()
    plt.tick_params(axis='both',direction='in',which='both',top=True,right=True,width=2.)
    plt.minorticks_on()
    plt.xlim(6,-6)
    plt.ylim(-6,6)

    plt.plot(v2corn_1A+dx[44],v3corn_1A-dy[44],color='b',linewidth=1.2,label='Ch1')
    plt.plot(v2corn_1A+dx[45],v3corn_1A-dy[45],color='b',linewidth=1.2)
    plt.plot(v2corn_1A+dx[46],v3corn_1A-dy[46],color='b',linewidth=1.2)
    plt.plot(v2corn_1A+dx[47],v3corn_1A-dy[47],color='b',linewidth=1.2)

    plt.plot(v2corn_2A+dx[44],v3corn_2A-dy[44],color='g',linewidth=1.2,label='Ch2')
    plt.plot(v2corn_2A+dx[45],v3corn_2A-dy[45],color='g',linewidth=1.2)
    plt.plot(v2corn_2A+dx[46],v3corn_2A-dy[46],color='g',linewidth=1.2)
    plt.plot(v2corn_2A+dx[47],v3corn_2A-dy[47],color='g',linewidth=1.2)
    
    plt.plot(v2corn_3A+dx[44],v3corn_3A-dy[44],color='gold',linewidth=1.2,label='Ch3')
    plt.plot(v2corn_3A+dx[45],v3corn_3A-dy[45],color='gold',linewidth=1.2)
    plt.plot(v2corn_3A+dx[46],v3corn_3A-dy[46],color='gold',linewidth=1.2)
    plt.plot(v2corn_3A+dx[47],v3corn_3A-dy[47],color='gold',linewidth=1.2)
           
    plt.plot(v2corn_4A+dx[44],v3corn_4A-dy[44],color='r',linewidth=1.2,label='Ch4')
    plt.plot(v2corn_4A+dx[45],v3corn_4A-dy[45],color='r',linewidth=1.2)
    plt.plot(v2corn_4A+dx[46],v3corn_4A-dy[46],color='r',linewidth=1.2)
    plt.plot(v2corn_4A+dx[47],v3corn_4A-dy[47],color='r',linewidth=1.2)
        
    plt.plot(0,0,'x',linewidth=1.5,color='black')
    circle1 = mpl.patches.Circle((0., 0.), maxfwhm(1),linewidth=1,edgecolor='b', facecolor=(0, 0, 0, .0125))
    ax.add_artist(circle1)
    circle1 = mpl.patches.Circle((0., 0.), maxfwhm(4),linewidth=1,edgecolor='r', facecolor=(0, 0, 0, .0125))
    ax.add_artist(circle1)
                                 
    plt.plot(v2[44:48]-v2ref,v3[44:48]-v3ref,'+',color='gold',linewidth=1.5)
    
    plt.xlabel('$\Delta$ R.A. (arcsec)')
    plt.ylabel('$\Delta$ Decl. (arcsec)')
    plt.title('MRS Dithers: Flight '+vertxt+' ('+nowstring+')')
    plt.text(5,5,'Ch3, 4-PT, extended source')
    plt.legend()

    plt.savefig(filename)
    #plt.show()
    plt.close()

#############################




# Plot the pt-source points for a given channel with field bound in ab space

# Plot points in v2/v3 space
