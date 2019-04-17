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

# Import offline MIRI MRS tools for CDP-6 specifically
#import miricoord.miricoord.mrs.toolversions.mrs_tools_cdp6 as tv

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

# A routine to get the average slice width

# A routine to get the average pixel size

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

# Generate the Ch2 point-source patterns

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

# Ch3

# Ch4

# Routine to generate the extended-source patterns

# Plot the pt-source points for a given channel with field bound in ab space

# Plot points in v2/v3 space
