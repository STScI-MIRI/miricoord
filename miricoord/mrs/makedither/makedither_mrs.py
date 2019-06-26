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
    if (channel is 1): wave=8.0
    if (channel is 2): wave=12.0
    if (channel is 3): wave=18.0
    if (channel is 4): wave=29.0

    return 0.31*wave/8.0
    
#############################

# A routine to recenter a given dither pattern within a particular channel FOV

def recenterFOV(pat_v2,pat_v3,channel):
    if (channel is 1): bands=np.array(['1A','1B','1C'])
    if (channel is 2): bands=np.array(['2A','2B','2C'])
    if (channel is 3): bands=np.array(['3A','3B','3C'])
    if (channel is 4): bands=np.array(['4A','4B','4C'])
            
    # Get the field boundaries for all three sub-bands
    valuesA=makesiaf.create_siaf_oneband(bands[0])
    valuesB=makesiaf.create_siaf_oneband(bands[1])
    valuesC=makesiaf.create_siaf_oneband(bands[2])
    
    # Average the three mean field positions
    v2_fieldmean=(valuesA['inscr_v2ref']+valuesB['inscr_v2ref']+valuesC['inscr_v2ref'])/3.
    v3_fieldmean=(valuesA['inscr_v3ref']+valuesB['inscr_v3ref']+valuesC['inscr_v3ref'])/3.
    v2_mean=np.mean(pat_v2)
    v3_mean=np.mean(pat_v3)

    newpat_v2 = pat_v2 - v2_mean + v2_fieldmean
    newpat_v3 = pat_v3 - v3_mean + v3_fieldmean

    return newpat_v2,newpat_v3

#############################

# A routine to recenter a given dither pattern with respect to a given channel reference point
# (which is not quite the same thing as centering wrt the FOV)

def recenterRP(pat_v2,pat_v3,channel):
    if (channel is 1): band='1A'
    if (channel is 2): band='2A'
    if (channel is 3): band='3A'
    if (channel is 4): band='4A'
    
    values=makesiaf.create_siaf_oneband(band)
    v2ref,v3ref=values['inscr_v2ref'],values['inscr_v3ref']
    
    v2_mean=np.mean(pat_v2)
    v3_mean=np.mean(pat_v3)

    newpat_v2 = pat_v2 - v2_mean + v2ref
    newpat_v3 = pat_v3 - v3_mean + v3ref

    return newpat_v2,newpat_v3

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

    # Recenter the pattern within the field
    pat_v2,pat_v3=recenterFOV(pat_v2,pat_v3,1)
    
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

    # Recenter the pattern within the field
    pat_v2,pat_v3=recenterFOV(pat_v2,pat_v3,2)

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

    # Recenter the pattern within the field
    pat_v2,pat_v3=recenterFOV(pat_v2,pat_v3,3)
    
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

    # Recenter the pattern within the field
    pat_v2,pat_v3=recenterFOV(pat_v2,pat_v3,4)
    
    return pat_v2,pat_v3

#############################

# Routine to generate the extended-source pattern optimized for ALL channels

def makepattern_ext_all():
    # See if the pixel and slice sizes have already been calculated
    try:
        slicewidth
        pixsize
    # If not, calculate them
    except:
        setsizes()

    # Ch3 and Ch4 are well-sampled in the pixel direction already, so optimize the along-slice
    # offset to be half-integer in Ch1 and Ch2
    da=pixsize[0]/2.# Ch1
    # Use the mathematically related slice widths in each channel to construct a half-integer
    # offset for all channels
    db=slicewidth[0]*5.5# Ch1

    pat_a=np.array([-da/2.,da/2.,da/2.,-da/2.])
    pat_b=np.array([db/2.,-db/2.,db/2.,-db/2.])

    # Transform assuming input in Ch1A alpha-beta
    pat_v2,pat_v3=mrst.abtov2v3(pat_a,pat_b,'1A')

    # Recenter the pattern within the field
    pat_v2,pat_v3=recenterFOV(pat_v2,pat_v3,1)
    
    return pat_v2,pat_v3

#############################

# Routine to generate the extended-source pattern optimized for a given channel

def makepattern_ext_ChX(ptpat_v2,ptpat_v3,channel):
    # First dither pair; pull out short-dithers with parity = 1
    temp1_v2=ptpat_v2[[0,2]]
    temp1_v3=ptpat_v3[[0,2]]
    # Recenter within field
    temp1_v2,temp1_v3=recenterRP(temp1_v2,temp1_v3,channel)

    # Second dither pair; pull out short-dithers with parity = -1
    temp2_v2=ptpat_v2[[4,6]]
    temp2_v3=ptpat_v3[[4,6]]
    # Recenter within field
    temp2_v2,temp2_v3=recenterRP(temp2_v2,temp2_v3,channel)

    # Combine the dither pairs
    pat_v2,pat_v3=np.append(temp1_v2,temp2_v2),np.append(temp1_v3,temp2_v3)
    # And recenter the combined dithers
    pat_v2,pat_v3=recenterRP(pat_v2,pat_v3,channel)
    
    return pat_v2,pat_v3

#############################

# Routine to convert fixed v2,v3 dither points into actual xideal,yideal offsets
# relative to the fiducial reference point for a given channel

def compute_dxdyideal(pat_v2,pat_v3,channel):
    # Reference point of this channel
    if (channel is 1): band='1A'
    if (channel is 2): band='2A'
    if (channel is 3): band='3A'
    if (channel is 4): band='4A'
    values=makesiaf.create_siaf_oneband(band)
    v2ref,v3ref=values['inscr_v2ref'],values['inscr_v3ref']

    # Ideal coordinate of the dither position
    x,y=mrst.v2v3_to_xyideal(pat_v2,pat_v3)
    # Ideal coordinate of the fiducial (undithered) point
    xref,yref=mrst.v2v3_to_xyideal(v2ref,v3ref)

    # Delta offsets
    dxidl=x-xref
    dyidl=y-yref

    return dxidl,dyidl
    
#############################

# Routine to write abbreviated results to a file (APT info only)

def writeresults_apt(index,dxidl,dyidl):
    # Set the output data directory
    data_dir=os.path.expandvars('$MIRICOORD_DATA_DIR')
    outdir=os.path.join(data_dir,'dithers/temp/')
    # Set the output filename
    outfile=os.path.join(outdir,'mrs_dithers_apt.txt')

    now=datetime.datetime.now()
    thisfile=__file__
    _,thisfile=os.path.split(thisfile)
    
    # Write header information to the output text file
    print('# Created ',now.isoformat(),file=open(outfile,"w"))
    print('# Using program',thisfile,file=open(outfile,"a"))
    
    # Column names
    print("{:<10} {:<15} {:<15}".format('PosnIndex','dXIdeal(arcsec)','dYIdeal(arcsec)'),file=open(outfile,"a"))
    
    for i in range(0,len(index)):
        # Write information to a text file
        print("{0:<10} {1:<15.5f} {2:<15.5f}".format(index[i],dxidl[i],dyidl[i]),file=open(outfile,"a"))
   
#############################

# Routine to write full results to a file

def writeresults_full(index,ch,v2,v3,dxidl,dyidl):
    # Set the output data directory
    data_dir=os.path.expandvars('$MIRICOORD_DATA_DIR')
    outdir=os.path.join(data_dir,'dithers/temp/')
    # Set the output filename
    outfile=os.path.join(outdir,'mrs_dithers.txt')

    now=datetime.datetime.now()
    thisfile=__file__
    _,thisfile=os.path.split(thisfile)
    
    # Write header information to the output text file
    print('# Created ',now.isoformat(),file=open(outfile,"w"))
    print('# Using program',thisfile,file=open(outfile,"a"))
    
    # Column names
    print("{:<10} {:<8} {:<10} {:<10} {:<15} {:<15}".format('PosnIndex','Channel','V2(arcsec)','V3(arcsec)','dXIdeal(arcsec)','dYIdeal(arcsec)'),file=open(outfile,"a"))
    
    for i in range(0,len(index)):
        # Write information to a text file
        print("{0:<10} {1:<8} {2:<10.5f} {3:<10.5f} {4:<15.5f} {5:<15.5f}".format(index[i],ch[i],v2[i],v3[i],dxidl[i],dyidl[i]),file=open(outfile,"a"))

#############################

# Plot showing the location of the point-source dithers

def qaplot_ptsourceloc(v2,v3):
    # Set the output data directory
    data_dir=os.path.expandvars('$MIRICOORD_DATA_DIR')
    outdir=os.path.join(data_dir,'dithers/temp/')
    # Set the output filename
    filename=os.path.join(outdir,'dithers_pt.pdf')

    # Field locations
    values1A=makesiaf.create_siaf_oneband('1A')
    values2A=makesiaf.create_siaf_oneband('2A')
    values3A=makesiaf.create_siaf_oneband('3A')
    values4A=makesiaf.create_siaf_oneband('4A')
    values1B=makesiaf.create_siaf_oneband('1B')
    values2B=makesiaf.create_siaf_oneband('2B')
    values3B=makesiaf.create_siaf_oneband('3B')
    values4B=makesiaf.create_siaf_oneband('4B')
    values1C=makesiaf.create_siaf_oneband('1C')
    values2C=makesiaf.create_siaf_oneband('2C')
    values3C=makesiaf.create_siaf_oneband('3C')
    values4C=makesiaf.create_siaf_oneband('4C')
    
    # Plot thickness
    mpl.rcParams['axes.linewidth'] = 1.5

    plt.figure(figsize=(5,5),dpi=150)
    plt.tick_params(axis='both',direction='in',which='both',top=True,right=True,width=2.)
    plt.minorticks_on()
    plt.xlim(-497.4,-509.4)
    plt.ylim(-325.8,-313.8)

    plt.plot(values1A['inscr_v2_corners'],values1A['inscr_v3_corners'],color='b',linewidth=1.5)
    plt.plot(values1B['inscr_v2_corners'],values1B['inscr_v3_corners'],color='b',linewidth=1.5)
    plt.plot(values1C['inscr_v2_corners'],values1C['inscr_v3_corners'],color='b',linewidth=1.5)
    plt.plot(values2A['inscr_v2_corners'],values2A['inscr_v3_corners'],color='g',linewidth=1.5)
    plt.plot(values2B['inscr_v2_corners'],values2B['inscr_v3_corners'],color='g',linewidth=1.5)
    plt.plot(values2C['inscr_v2_corners'],values2C['inscr_v3_corners'],color='g',linewidth=1.5)
    plt.plot(values3A['inscr_v2_corners'],values3A['inscr_v3_corners'],color='gold',linewidth=1.5)
    plt.plot(values3B['inscr_v2_corners'],values3B['inscr_v3_corners'],color='gold',linewidth=1.5)
    plt.plot(values3C['inscr_v2_corners'],values3C['inscr_v3_corners'],color='gold',linewidth=1.5)
    plt.plot(values4A['inscr_v2_corners'],values4A['inscr_v3_corners'],color='r',linewidth=1.5)
    plt.plot(values4B['inscr_v2_corners'],values4B['inscr_v3_corners'],color='r',linewidth=1.5)
    plt.plot(values4C['inscr_v2_corners'],values4C['inscr_v3_corners'],color='r',linewidth=1.5)

    plt.plot(v2[0:8],v3[0:8],'+',color='b',linewidth=1.5)
    plt.plot(v2[8:16],v3[8:16],'+',color='g')
    plt.plot(v2[16:24],v3[16:24],'+',color='gold')
    plt.plot(v2[24:32],v3[24:32],'+',color='r')
            
    plt.xlabel('V2 (arcsec)')
    plt.ylabel('V3 (arcsec)')
    plt.title('MRS Dithers: Pre-flight (May 2019)')

    plt.savefig(filename)
    plt.show()
    plt.close()

#############################

# Plot showing the location of the extended-source dithers

def qaplot_extsourceloc(v2,v3):
    # Set the output data directory
    data_dir=os.path.expandvars('$MIRICOORD_DATA_DIR')
    outdir=os.path.join(data_dir,'dithers/temp/')
    # Set the output filename
    filename=os.path.join(outdir,'dithers_ext.pdf')

    # Field locations
    values1A=makesiaf.create_siaf_oneband('1A')
    values2A=makesiaf.create_siaf_oneband('2A')
    values3A=makesiaf.create_siaf_oneband('3A')
    values4A=makesiaf.create_siaf_oneband('4A')
    values1B=makesiaf.create_siaf_oneband('1B')
    values2B=makesiaf.create_siaf_oneband('2B')
    values3B=makesiaf.create_siaf_oneband('3B')
    values4B=makesiaf.create_siaf_oneband('4B')
    values1C=makesiaf.create_siaf_oneband('1C')
    values2C=makesiaf.create_siaf_oneband('2C')
    values3C=makesiaf.create_siaf_oneband('3C')
    values4C=makesiaf.create_siaf_oneband('4C')
    
    # Plot thickness
    mpl.rcParams['axes.linewidth'] = 1.5

    plt.figure(figsize=(5,5),dpi=150)
    plt.tick_params(axis='both',direction='in',which='both',top=True,right=True,width=2.)
    plt.minorticks_on()
    plt.xlim(-497.4,-509.4)
    plt.ylim(-325.8,-313.8)

    plt.plot(values1A['inscr_v2_corners'],values1A['inscr_v3_corners'],color='b',linewidth=1.5)
    plt.plot(values1B['inscr_v2_corners'],values1B['inscr_v3_corners'],color='b',linewidth=1.5)
    plt.plot(values1C['inscr_v2_corners'],values1C['inscr_v3_corners'],color='b',linewidth=1.5)
    plt.plot(values2A['inscr_v2_corners'],values2A['inscr_v3_corners'],color='g',linewidth=1.5)
    plt.plot(values2B['inscr_v2_corners'],values2B['inscr_v3_corners'],color='g',linewidth=1.5)
    plt.plot(values2C['inscr_v2_corners'],values2C['inscr_v3_corners'],color='g',linewidth=1.5)
    plt.plot(values3A['inscr_v2_corners'],values3A['inscr_v3_corners'],color='gold',linewidth=1.5)
    plt.plot(values3B['inscr_v2_corners'],values3B['inscr_v3_corners'],color='gold',linewidth=1.5)
    plt.plot(values3C['inscr_v2_corners'],values3C['inscr_v3_corners'],color='gold',linewidth=1.5)
    plt.plot(values4A['inscr_v2_corners'],values4A['inscr_v3_corners'],color='r',linewidth=1.5)
    plt.plot(values4B['inscr_v2_corners'],values4B['inscr_v3_corners'],color='r',linewidth=1.5)
    plt.plot(values4C['inscr_v2_corners'],values4C['inscr_v3_corners'],color='r',linewidth=1.5)

    plt.plot(v2[32:36],v3[32:36],'+',color='black',linewidth=1.5)
    plt.plot(v2[36:40],v3[36:40],'+',color='b',linewidth=1.5)
    plt.plot(v2[40:44],v3[40:44],'+',color='g')
    plt.plot(v2[44:48],v3[44:48],'+',color='gold')
    plt.plot(v2[48:52],v3[48:52],'+',color='r')
            
    plt.xlabel('V2 (arcsec)')
    plt.ylabel('V3 (arcsec)')
    plt.title('MRS Dithers: Pre-flight (May 2019)')

    plt.savefig(filename)
    plt.show()
    plt.close()
    
#############################

# Plot showing field coverage of a 4-pt ALL point-source dither

def qaplot_ps4all(v2,v3,dx,dy):
    # Set the output data directory
    data_dir=os.path.expandvars('$MIRICOORD_DATA_DIR')
    outdir=os.path.join(data_dir,'dithers/temp/')
    # Set the output filename
    filename=os.path.join(outdir,'dithers_ps4ALL.pdf')

    # Field locations
    values1A=makesiaf.create_siaf_oneband('1A')
    values4A=makesiaf.create_siaf_oneband('4A')

    # Recenter everything to be based around zero
    v2ref,v3ref=values1A['inscr_v2ref'],values1A['inscr_v3ref']
    v2corn_1A=values1A['inscr_v2_corners']-v2ref
    v3corn_1A=values1A['inscr_v3_corners']-v3ref  
    v2corn_4A=values4A['inscr_v2_corners']-v2ref
    v3corn_4A=values4A['inscr_v3_corners']-v3ref

    
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
    plt.title('MRS Dithers: Pre-flight (May 2019)')
    plt.text(5,5,'ALL, 4-PT, point source')
    plt.legend()

    plt.savefig(filename)
    plt.show()
    plt.close()

#############################

# Plot showing field coverage of a 2-pt Ch4 point-source dither

def qaplot_ps2ch4(v2,v3,dx,dy):
    # Set the output data directory
    data_dir=os.path.expandvars('$MIRICOORD_DATA_DIR')
    outdir=os.path.join(data_dir,'dithers/temp/')
    # Set the output filename
    filename=os.path.join(outdir,'dithers_ps2ch4.pdf')

    # Field locations
    values1A=makesiaf.create_siaf_oneband('1A')
    values4A=makesiaf.create_siaf_oneband('4A')

    # Recenter everything to be based around zero
    v2ref,v3ref=values4A['inscr_v2ref'],values4A['inscr_v3ref']
    v2corn_1A=values1A['inscr_v2_corners']-v2ref
    v3corn_1A=values1A['inscr_v3_corners']-v3ref  
    v2corn_4A=values4A['inscr_v2_corners']-v2ref
    v3corn_4A=values4A['inscr_v3_corners']-v3ref

    
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
    plt.title('MRS Dithers: Pre-flight (May 2019)')
    plt.text(7,7,'Ch4, 2-PT, point source')
    plt.legend()

    plt.savefig(filename)
    plt.show()
    plt.close()

#############################

# Plot showing field coverage of a 2-pt ALL extended-source dither

def qaplot_ext2all(v2,v3,dx,dy):
    # Set the output data directory
    data_dir=os.path.expandvars('$MIRICOORD_DATA_DIR')
    outdir=os.path.join(data_dir,'dithers/temp/')
    # Set the output filename
    filename=os.path.join(outdir,'dithers_ext2all.pdf')

    # Field locations
    values1A=makesiaf.create_siaf_oneband('1A')
    values4A=makesiaf.create_siaf_oneband('4A')

    # Recenter everything to be based around zero
    v2ref,v3ref=values1A['inscr_v2ref'],values1A['inscr_v3ref']
    v2corn_1A=values1A['inscr_v2_corners']-v2ref
    v3corn_1A=values1A['inscr_v3_corners']-v3ref  
    v2corn_4A=values4A['inscr_v2_corners']-v2ref
    v3corn_4A=values4A['inscr_v3_corners']-v3ref

    
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
    plt.title('MRS Dithers: Pre-flight (May 2019)')
    plt.text(5,5,'ALL, 2-PT, extended source')
    plt.legend()

    plt.savefig(filename)
    plt.show()
    plt.close()

#############################

# Plot showing field coverage of a 4-pt ALL extended-source dither

def qaplot_ext4all(v2,v3,dx,dy):
    # Set the output data directory
    data_dir=os.path.expandvars('$MIRICOORD_DATA_DIR')
    outdir=os.path.join(data_dir,'dithers/temp/')
    # Set the output filename
    filename=os.path.join(outdir,'dithers_ext4all.pdf')

    # Field locations
    values1A=makesiaf.create_siaf_oneband('1A')
    values4A=makesiaf.create_siaf_oneband('4A')

    # Recenter everything to be based around zero
    v2ref,v3ref=values1A['inscr_v2ref'],values1A['inscr_v3ref']
    v2corn_1A=values1A['inscr_v2_corners']-v2ref
    v3corn_1A=values1A['inscr_v3_corners']-v3ref  
    v2corn_4A=values4A['inscr_v2_corners']-v2ref
    v3corn_4A=values4A['inscr_v3_corners']-v3ref

    
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
    plt.title('MRS Dithers: Pre-flight (May 2019)')
    plt.text(5,5,'ALL, 4-PT, extended source')
    plt.legend()

    plt.savefig(filename)
    plt.show()
    plt.close()
    
#############################

# Plot showing field coverage of a 2-pt Ch3 extended-source dither

def qaplot_ext2ch3(v2,v3,dx,dy):
    # Set the output data directory
    data_dir=os.path.expandvars('$MIRICOORD_DATA_DIR')
    outdir=os.path.join(data_dir,'dithers/temp/')
    # Set the output filename
    filename=os.path.join(outdir,'dithers_ext2all.pdf')

    # Field locations
    values1A=makesiaf.create_siaf_oneband('1A')
    values3A=makesiaf.create_siaf_oneband('3A')
    values4A=makesiaf.create_siaf_oneband('4A')

    # Recenter everything to be based around zero
    v2ref,v3ref=values3A['inscr_v2ref'],values3A['inscr_v3ref']
    v2corn_1A=values1A['inscr_v2_corners']-v2ref
    v3corn_1A=values1A['inscr_v3_corners']-v3ref
    v2corn_3A=values3A['inscr_v2_corners']-v2ref
    v3corn_3A=values3A['inscr_v3_corners']-v3ref 
    v2corn_4A=values4A['inscr_v2_corners']-v2ref
    v3corn_4A=values4A['inscr_v3_corners']-v3ref

    
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
    plt.title('MRS Dithers: Pre-flight (May 2019)')
    plt.text(5,5,'Ch3, 2-PT, extended source')
    plt.legend()

    plt.savefig(filename)
    plt.show()
    plt.close()
        
#############################

# Plot showing field coverage of a 4-pt Ch3 extended-source dither

def qaplot_ext4ch3(v2,v3,dx,dy):
    # Set the output data directory
    data_dir=os.path.expandvars('$MIRICOORD_DATA_DIR')
    outdir=os.path.join(data_dir,'dithers/temp/')
    # Set the output filename
    filename=os.path.join(outdir,'dithers_ext4all.pdf')

    # Field locations
    values1A=makesiaf.create_siaf_oneband('1A')
    values3A=makesiaf.create_siaf_oneband('3A')
    values4A=makesiaf.create_siaf_oneband('4A')

    # Recenter everything to be based around zero
    v2ref,v3ref=values3A['inscr_v2ref'],values3A['inscr_v3ref']
    v2corn_1A=values1A['inscr_v2_corners']-v2ref
    v3corn_1A=values1A['inscr_v3_corners']-v3ref
    v2corn_3A=values3A['inscr_v2_corners']-v2ref
    v3corn_3A=values3A['inscr_v3_corners']-v3ref 
    v2corn_4A=values4A['inscr_v2_corners']-v2ref
    v3corn_4A=values4A['inscr_v3_corners']-v3ref

    
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
    plt.title('MRS Dithers: Pre-flight (May 2019)')
    plt.text(5,5,'Ch3, 4-PT, extended source')
    plt.legend()

    plt.savefig(filename)
    plt.show()
    plt.close()

#############################




# Plot the pt-source points for a given channel with field bound in ab space

# Plot points in v2/v3 space
