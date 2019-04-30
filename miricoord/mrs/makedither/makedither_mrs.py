#
"""
Python tools for creating the MIRI MRS dither sequences for a given
set of distortion files.

Beta dithers: Ch1 long offset is 5.5 times the Ch1 width because that will be
half-integer for all other channels.  Ch 2/3/4 are odd multiples of the 5.5 times
slice width offset so that it will similarly be half-integer for all channels.
Short beta dithers are half-integer in the local channel because we're trying to
optimize for this channel.

Alpha dithers both long and short offsets are define related to the channel-specific
pixel size because we're optimizing for that channel.

However, alpha/beta axes in Ch1 aren't quite aligned with other channels...  second
order effect?

Other thoughts: originally dithers were all defined using Ch1 alpha/beta even if they
strictly shouldn't be.  However- using the Ch1 xforms to map them to v2/v3 requires
using the transforms beyond the area for which they are defined!  They may not be
entirely reliable outside the Ch1 footprint.

So implement in here both the CDP-6 approach and a commissioning-like approach?
Commissioning should do alpha-beta to v2,v3 with local transform
and center around that channel's centroid

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
15-Apr-2019  Adapt from old IDL routines (D. Law; dlaw@stsci.edu)
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

import miricoord.miricoord.mrs.mrs_tools as mrst
import miricoord.miricoord.mrs.makesiaf.makesiaf_mrs as makesiaf

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
# in the alpha and beta directions.

def makepattern_generic(astart,along,ashort,bstart,blong,bshort):
    pattern_alpha=np.array([0,along,along+ashort,ashort,along+ashort,ashort,0,along])+astart
    pattern_beta=np.array([blong+bshort,bshort,0,blong,blong+bshort,bshort,0,blong])-blong-bshort+bstart

    # Still need to flip the coordinate pairs
    
    return pattern_alpha,pattern_beta

#############################

# A routine to return rough maximum PSF FWHM in arcsec for a given channel

def maxfwhm(channel):
    # Maximum wavelength in microns for the channel
    wave=0.
    if (channel is 1): wave=8.0
    if (channel is 2): wave=12.0
    if (channel is 3): wave=18.0
    if (channel is 4): wave=29.0

    return 0.31*wave/8.0
    
#############################

# Generate the CDP6 Ch1 point-source patterns

def makepattern_cdp6_ch1():
    pixsiz1=0.196
    pixsiz=0.196
    slicesiz1=0.176
    slicesiz=0.176

    along=10.5*pixsiz
    ashort=0.5*pixsiz
    astart=-5.5*pixsiz1
    blong=5.5*slicesiz1
    bshort=0.5*slicesiz
    bstart=3*slicesiz1

    pat_a,pat_b=makepattern_generic(astart,along,ashort,bstart,blong,bshort)
    # Transform assuming input in Ch1A alpha-beta
    pat_v2,pat_v3=mrst.abtov2v3(pat_a,pat_b,'1A')

    return pat_v2,pat_v3
    
#############################

# Generate the CDP6 Ch2 point-source patterns

def makepattern_cdp6_ch2():
    pixsiz1=0.196
    pixsiz=0.196
    slicesiz1=0.176
    slicesiz=0.277

    along=10.5*pixsiz
    ashort=0.5*pixsiz
    astart=-5.5*pixsiz1
    blong=16.5*slicesiz1
    bshort=0.5*slicesiz
    bstart=9*slicesiz1

    pat_a,pat_b=makepattern_generic(astart,along,ashort,bstart,blong,bshort)
    # Transform assuming input in Ch1A alpha-beta
    pat_v2,pat_v3=mrst.abtov2v3(pat_a,pat_b,'1A')

    return pat_v2,pat_v3

#############################

# Generate the CDP6 Ch3 point-source patterns

def makepattern_cdp6_ch3():
    pixsiz1=0.196
    pixsiz=0.244
    slicesiz1=0.176
    slicesiz=0.387

    along=16.5*pixsiz
    ashort=0.5*pixsiz
    astart=-12.5*pixsiz1
    blong=16.5*slicesiz1
    bshort=0.5*slicesiz
    bstart=12.6*slicesiz1

    pat_a,pat_b=makepattern_generic(astart,along,ashort,bstart,blong,bshort)
    # Transform assuming input in Ch1A alpha-beta
    pat_v2,pat_v3=mrst.abtov2v3(pat_a,pat_b,'1A')

    return pat_v2,pat_v3

#############################

# Generate the CDP6 Ch4 point-source patterns

def makepattern_cdp6_ch4():
    pixsiz1=0.196
    pixsiz=0.273
    slicesiz1=0.176
    slicesiz=0.645

    along=17.5*pixsiz
    ashort=0.5*pixsiz
    astart=-10*pixsiz1
    blong=27.5*slicesiz1
    bshort=0.5*slicesiz
    bstart=18*slicesiz1

    pat_a,pat_b=makepattern_generic(astart,along,ashort,bstart,blong,bshort)
    # Transform assuming input in Ch1A alpha-beta
    pat_v2,pat_v3=mrst.abtov2v3(pat_a,pat_b,'1A')

    return pat_v2,pat_v3

#############################

# Generate the commissioning Ch1 point-source patterns

def makepattern_ch1():
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
    ashort=0.5*pixsiz
    astart=0
    blong=5.5*slicesiz1
    bshort=0.5*slicesiz
    bstart=0

    pat_a,pat_b=makepattern_generic(astart,along,ashort,bstart,blong,bshort)
    # Transform assuming input in Ch1A alpha-beta
    pat_v2,pat_v3=mrst.abtov2v3(pat_a,pat_b,'1A')

    # Get the Ch1 field boundaries
    values1A=makesiaf.create_siaf_oneband('1A')
    values1B=makesiaf.create_siaf_oneband('1B')
    values1C=makesiaf.create_siaf_oneband('1C')
    
    # Recenter the pattern
    v2_fieldmean=(values1A['inscr_v2ref']+values1B['inscr_v2ref']+values1C['inscr_v2ref'])/3.
    v3_fieldmean=(values1A['inscr_v3ref']+values1B['inscr_v3ref']+values1C['inscr_v3ref'])/3.
    v2_mean=np.mean(pat_v2)
    v3_mean=np.mean(pat_v3)

    pat_v2 = pat_v2 - v2_mean + v2_fieldmean
    pat_v3 = pat_v3 - v3_mean + v3_fieldmean

    return pat_v2,pat_v3

#############################

# Generate the commissioning Ch2 point-source patterns

def makepattern_ch2():
    # See if the pixel and slice sizes have already been calculated
    try:
        slicewidth
        pixsize
    # If not, calculate them
    except:
        setsizes()
        
    pixsiz1=pixsize[0]# Ch1
    pixsiz=pixsize[1]# Ch2
    slicesiz1=slicewidth[0]# Ch1
    slicesiz=slicewidth[1]# Ch2

    along=10.5*pixsiz
    ashort=0.5*pixsiz
    astart=0
    blong=16.5*slicesiz1
    bshort=0.5*slicesiz
    bstart=0

    pat_a,pat_b=makepattern_generic(astart,along,ashort,bstart,blong,bshort)
    # Transform assuming input pattern in Ch2A alpha-beta
    pat_v2,pat_v3=mrst.abtov2v3(pat_a,pat_b,'2A')

    # Get the Ch2 field boundaries
    values2A=makesiaf.create_siaf_oneband('2A')
    values2B=makesiaf.create_siaf_oneband('2B')
    values2C=makesiaf.create_siaf_oneband('2C')
    
    # Recenter the pattern
    v2_fieldmean=(values2A['inscr_v2ref']+values2B['inscr_v2ref']+values2C['inscr_v2ref'])/3.
    v3_fieldmean=(values2A['inscr_v3ref']+values2B['inscr_v3ref']+values2C['inscr_v3ref'])/3.
    v2_mean=np.mean(pat_v2)
    v3_mean=np.mean(pat_v3)

    pat_v2 = pat_v2 - v2_mean + v2_fieldmean
    pat_v3 = pat_v3 - v3_mean + v3_fieldmean
    
    return pat_v2,pat_v3

#############################

# Generate the commissioning Ch3 point-source patterns

def makepattern_ch3():
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

    along=16.5*pixsiz
    ashort=0.5*pixsiz
    astart=0
    blong=16.5*slicesiz1
    bshort=0.5*slicesiz
    bstart=0

    pat_a,pat_b=makepattern_generic(astart,along,ashort,bstart,blong,bshort)
    # Transform assuming input in Ch3A alpha-beta
    pat_v2,pat_v3=mrst.abtov2v3(pat_a,pat_b,'3A')

    # Get the Ch3 field boundaries
    values3A=makesiaf.create_siaf_oneband('3A')
    values3B=makesiaf.create_siaf_oneband('3B')
    values3C=makesiaf.create_siaf_oneband('3C')
    
    # Recenter the pattern
    v2_fieldmean=(values3A['inscr_v2ref']+values3B['inscr_v2ref']+values3C['inscr_v2ref'])/3.
    v3_fieldmean=(values3A['inscr_v3ref']+values3B['inscr_v3ref']+values3C['inscr_v3ref'])/3.
    v2_mean=np.mean(pat_v2)
    v3_mean=np.mean(pat_v3)

    pat_v2 = pat_v2 - v2_mean + v2_fieldmean
    pat_v3 = pat_v3 - v3_mean + v3_fieldmean
    
    return pat_v2,pat_v3

#############################

# Generate the commissioning Ch4 point-source patterns

def makepattern_ch4():
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

    along=17.5*pixsiz
    ashort=0.5*pixsiz
    astart=0
    blong=27.5*slicesiz1
    bshort=0.5*slicesiz
    bstart=0

    pat_a,pat_b=makepattern_generic(astart,along,ashort,bstart,blong,bshort)
    # Transform assuming input in Ch4A alpha-beta
    pat_v2,pat_v3=mrst.abtov2v3(pat_a,pat_b,'4A')

    # Get the Ch4 field boundaries
    values4A=makesiaf.create_siaf_oneband('4A')
    values4B=makesiaf.create_siaf_oneband('4B')
    values4C=makesiaf.create_siaf_oneband('4C')
    
    # Recenter the pattern
    v2_fieldmean=(values4A['inscr_v2ref']+values4B['inscr_v2ref']+values4C['inscr_v2ref'])/3.
    v3_fieldmean=(values4A['inscr_v3ref']+values4B['inscr_v3ref']+values4C['inscr_v3ref'])/3.
    v2_mean=np.mean(pat_v2)
    v3_mean=np.mean(pat_v3)

    pat_v2 = pat_v2 - v2_mean + v2_fieldmean
    pat_v3 = pat_v3 - v3_mean + v3_fieldmean
    
    return pat_v2,pat_v3

# Routine to generate the extended-source patterns

# Plot the pt-source points for a given channel with field bound in ab space

# Plot points in v2/v3 space
