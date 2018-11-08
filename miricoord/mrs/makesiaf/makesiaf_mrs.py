#
"""
Python tools for creating the MIRI MRS SIAF input files.
These define the slice locations and per-channel field boundaries
in the v2,v3 coordinate system.

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
30-Jul-2015  Written by David Law (dlaw@stsci.edu)
27-Oct-2015  Use library routines (D. Law)
24-Jan-2016  Update to CDP5 (D. Law)
13-Sep-2016  Fix bug in slice indexing (D. Law)
30-Sep-2016  Add Yanny par file option (D. Law)
17-Oct-2016  Input/output v2/v3 in arcsec (D. Law)
14-Dec-2017  Adapt for new github/central store interaction (D. Law)
15-Oct-2018  Adapt from IDL to python (D. Law)
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

import miricoord.miricoord.mrs.mrs_fields as mf

# Import offline MIRI MRS tools for CDP-6 specifically
import miricoord.miricoord.mrs.toolversions.mrs_tools_cdp6 as tv

#############################

# This routine will run all of the channels/bands automatically
# if you don't want to use the python notebook to examine the
# results.
def create_siaf_all():
    # Set the output data directory
    data_dir=os.path.expandvars('$MIRICOORD_DATA_DIR')
    outdir=os.path.join(data_dir,'siaf/mrs/temp/')
    # Set the output csv filename
    outfile=os.path.join(outdir,'miri_siaf_mrs.txt')

    now=datetime.datetime.now()
    thisfile=__file__
    _,thisfile=os.path.split(thisfile)

    # Write header information to the output text file
    print('#',now.isoformat(),file=open(outfile,"w"))
    print('# Created by user',getpass.getuser(),'at',socket.gethostname(),file=open(outfile,"a"))
    print('# Using program',thisfile,file=open(outfile,"a"))
    print('# Using input files:',file=open(outfile,"a"))

    # Initial loop over all channels to make note of the input files
    channel=['1A','1B','1C','2A','2B','2C','3A','3B','3C','4A','4B','4C']
    nchan=len(channel)
    for i in range(0,nchan):
        reffile=tv.get_fitsreffile(channel[i])
        reffile0,reffile1=os.path.split(reffile)
        print('# '+channel[i]+' : ' + reffile1,file=open(outfile,"a"))

    # Set up a loop over all channels to do the calculation
    # Column names
    print("{:>20}, {:>10}, {:>10}, {:>8}, {:>12}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('AperName','AperType','SliceName','SliceNum','V3IdlYAngle','VIdlParity','v2ref','v3ref','v2ll','v2ul','v2ur','v2lr','v3ll','v3ul','v3ur','v3lr'),file=open(outfile,"a"))
    for i in range(0,nchan):
        # Create the values dictionary
        values=create_siaf_oneband(channel[i])

        # Make QA plots
        plotfile=os.path.join(outdir,'siaf_'+channel[i]+'ab.ps')
        makeabplot(values,channel[i],filename=plotfile)
        plotfile=os.path.join(outdir,'siaf_'+channel[i]+'v2v3.ps')
        makev2v3plot(values,channel[i],filename=plotfile)
        plotfile=os.path.join(outdir,'siaf_'+channel[i]+'v2v3_common.ps')
        makev2v3plot(values,channel[i],filename=plotfile,xlim=[-8.29*60.,-8.49*60.],ylim=[-5.43*60.,-5.23*60.])

        # Write information to a text file
        print("{val1:>20}, {val2:>10}, {val3:>10}, {val4:>8}, {val5:>12}, {val6:>10}, {0:>10.4f}, {1:>10.4f}, {2:>10.4f}, {3:>10.4f}, {4:>10.4f}, {5:>10.4f}, {6:>10.4f}, {7:>10.4f}, {8:>10.4f}, {9:>10.4f}".format(values['inscr_v2ref'],values['inscr_v3ref'],values['inscr_v2_corners'][0],values['inscr_v2_corners'][1],values['inscr_v2_corners'][2],values['inscr_v2_corners'][3],values['inscr_v3_corners'][0],values['inscr_v3_corners'][1],values['inscr_v3_corners'][2],values['inscr_v3_corners'][3],val1=values['inscr_apername'],val2='COMPOUND',val3='-1',val4='-1',val5='0.',val6='-1'),file=open(outfile,"a"))
        nslice=len(values['slice_num'])
        for j in range(0,nslice):
            print("{val1:>20}, {val2:>10}, {val3:>10}, {val4:>8}, {val5:>12}, {val6:>10}, {0:>10.4f}, {1:>10.4f}, {2:>10.4f}, {3:>10.4f}, {4:>10.4f}, {5:>10.4f}, {6:>10.4f}, {7:>10.4f}, {8:>10.4f}, {9:>10.4f}".format(values['slice_v2ref'][j],values['slice_v3ref'][j],values['slice_v2_corners'][j][0],values['slice_v2_corners'][j][1],values['slice_v2_corners'][j][2],values['slice_v2_corners'][j][3],values['slice_v3_corners'][j][0],values['slice_v3_corners'][j][1],values['slice_v3_corners'][j][2],values['slice_v3_corners'][j][3],val1=values['slice_apername'][j],val2='SLIT',val3=values['slice_name'][j],val4=values['slice_num'][j],val5='0.',val6='-1'),file=open(outfile,"a"))

    # Print out what all of the field sizes in this file are
    mf.getfov(file=outfile)


#############################

# This routine will create the v2-v3 QA plots for one channel of information
def makev2v3plot(values,channel,**kwargs):
    # Plot thickness
    mpl.rcParams['axes.linewidth'] = 1.5

    plt.figure(figsize=(8,5),dpi=250)
    
    # Box limits
    if ('xlim') in kwargs:
        xlim=kwargs['xlim']
    else:
        xlim=[values['slice_v2_corners'].max(),values['slice_v2_corners'].min()]
    if ('ylim') in kwargs:
        ylim=kwargs['ylim']
    else:
        ylim=[values['slice_v3_corners'].min(),values['slice_v3_corners'].max()]

    plt.tick_params(axis='both',direction='in',which='both',top=True,right=True)

    plt.xlim(xlim[0],xlim[1])
    plt.ylim(ylim[0],ylim[1])
    nslice=len(values['slice_num'])
    for i in range(0,nslice):
        thiscol='C'+str(i%10)
        plt.plot(values['slice_v2_corners'][i],values['slice_v3_corners'][i],c=thiscol)
        plt.plot(values['slice_v2ref'][i],values['slice_v3ref'][i],marker='x',linestyle='',c=thiscol)
    plt.plot(values['inscr_v2_corners'],values['inscr_v3_corners'],color='#000000',linewidth=2)
    plt.plot(values['inscr_v2ref'],values['inscr_v3ref'],color='#000000',marker='x',markersize=10,mew=2)
    plt.xlabel('v2 (arcsec)')
    plt.ylabel('v3 (arcsec)')
    plt.title(channel)

    # Determine whether we're sending the plot to screen or to a file
    if ('filename' in kwargs):
        plt.savefig(kwargs['filename'])
        plt.close()
    else:
        plt.show()

#############################

# This routine will create the alpha-beta QA plots for one channel of information
def makeabplot(values,channel,**kwargs):
    # Plot thickness
    mpl.rcParams['axes.linewidth'] = 1.5

    plt.figure(figsize=(8,5),dpi=250)
    plt.tick_params(axis='both',direction='in',which='both',top=True,right=True)

    nslice=len(values['slice_num'])
    for i in range(0,nslice):
        thiscol='C'+str(i%10)
        plt.plot(values['slice_alpha_corners'][i],values['slice_beta_corners'][i],c=thiscol)
        plt.plot(values['slice_alpharef'][i],values['slice_betaref'][i],marker='x',linestyle='',c=thiscol)
    plt.plot(values['inscr_alpha_corners'],values['inscr_beta_corners'],color='#000000',linewidth=2)
    plt.plot(values['inscr_alpharef'],values['inscr_betaref'],color='#000000',marker='x',markersize=10,mew=2)
    plt.xlabel('Alpha (arcsec)')
    plt.ylabel('Beta (arcsec)')
    plt.title(channel)

    # Determine whether we're sending the plot to screen or to a file
    if ('filename' in kwargs):
        plt.savefig(kwargs['filename'])
        plt.close()
    else:
        plt.show()

#############################

# This routine will compute the field information for one
# channel/band (e.g., 1A) and return a dictionary of information
# that can either be plotted or simply added to the output
def create_siaf_oneband(channel):
    # Split input channel into components, e.g.
    # if channel='1A' then ch=1 and sband=A
    ch=channel[0]
    sband=channel[1]
    inscr_apername='MIRIFU_CHANNEL'+channel
    
    # Open relevant distortion file
    distfilename=tv.get_fitsreffile(channel)
    distfile=fits.open(distfilename)

    # Read global header
    hdr=distfile[0].header

    # Get beta zeropoint and spacing from header
    beta0=hdr['B_ZERO'+ch]
    dbeta=hdr['B_DEL'+ch]

    # Read FoV alpha boundaries
    alphalimits=distfile['FoV_CH'+ch].data

    # Determine number of slices
    nslices=alphalimits.size

    # Create a 1-indexed vector of slice numbers and slice names
    # (the names will be of the form 112A for ch 1, slice 12, band A)
    slicenum=np.arange(1,nslices+1)
    slicename=np.array(['JUNK' for i in range(0,nslices)])
    apername=np.array(['JUNKJUNKJUNKJUNK' for i in range(0,nslices)])
    for i in range(0,nslices):
        slicename[i]=str(int(ch)*100+slicenum[i])+sband
        apername[i]='MIRIFU_'+channel+'SLICE'+str(slicenum[i]).zfill(2)
 
    # Figure out beta boundaries of each slice
    beta1=beta0+(slicenum-1.5)*dbeta# Lower bound
    beta2=beta1+dbeta# Upper bound

    # Figure out central reference point locations of each slice
    slice_beta_ref=(beta1+beta2)/2.
    slice_alpha_ref=np.zeros(nslices)

    # Convert from our list of maximum and minimum alpha,beta
    # to actual corner coordinates for each slice
    alpha_corners=np.zeros((nslices,5))
    beta_corners=np.zeros((nslices,5))

    # Order is lower-left, upper-left, upper-right, lower-right, lower-left
    # We're duplicating the lower-left point to aid plotting routines, it will
    # be stripped out when we actually write the files
    # Populate values
    for i in range(0,nslices):
        alpha_corners[i,0]=alphalimits[i][0]
        alpha_corners[i,1]=alphalimits[i][0]
        alpha_corners[i,2]=alphalimits[i][1]
        alpha_corners[i,3]=alphalimits[i][1]
        alpha_corners[i,4]=alphalimits[i][0]
        beta_corners[i,0]=beta1[i]
        beta_corners[i,1]=beta2[i]
        beta_corners[i,2]=beta2[i]
        beta_corners[i,3]=beta1[i]
        beta_corners[i,4]=beta1[i]
    
    # Compute corner coordinates for an inscribed footprint
    inscr_alpha=np.zeros(5)
    inscr_beta=np.zeros(5)
    inscr_alpha[0]=alpha_corners[:,0].max()
    inscr_alpha[1]=alpha_corners[:,1].max()
    inscr_alpha[2]=alpha_corners[:,2].min()
    inscr_alpha[3]=alpha_corners[:,3].min()
    inscr_alpha[4]=alpha_corners[:,0].max()
    inscr_beta[0]=beta_corners.min()
    inscr_beta[1]=beta_corners.max()
    inscr_beta[2]=beta_corners.max()
    inscr_beta[3]=beta_corners.min()
    inscr_beta[4]=beta_corners.min()

    # Figure out central reference point locations of the inscribed footprint
    # Fix it to alpha=beta=0.
    beta_ref=0.
    alpha_ref=0.

    # Convert to v2,v3 reference points
    v2_ref,v3_ref=tv.abtov2v3(alpha_ref,beta_ref,channel)
    slice_v2_ref,slice_v3_ref=tv.abtov2v3(slice_alpha_ref,slice_beta_ref,channel)
    # Convert to v2,v3 corner coordinates
    v2_corners,v3_corners=tv.abtov2v3(alpha_corners,beta_corners,channel)
    # Convert to v2,v3 inscribed box
    inscr_v2,inscr_v3=tv.abtov2v3(inscr_alpha,inscr_beta,channel)

    # Create a dictionary to return
    values=dict();
    values['distfilename']=distfilename
    values['slice_name']=slicename
    values['slice_num']=slicenum
    values['slice_apername']=apername
    values['slice_alpharef']=slice_alpha_ref
    values['slice_betaref']=slice_beta_ref
    values['slice_v2ref']=slice_v2_ref
    values['slice_v3ref']=slice_v3_ref
    values['slice_alpha_corners']=alpha_corners
    values['slice_beta_corners']=beta_corners
    values['slice_v2_corners']=v2_corners
    values['slice_v3_corners']=v3_corners
    values['inscr_apername']=inscr_apername
    values['inscr_alpharef']=alpha_ref
    values['inscr_betaref']=beta_ref
    values['inscr_v2ref']=v2_ref
    values['inscr_v3ref']=v3_ref
    values['inscr_alpha_corners']=inscr_alpha
    values['inscr_beta_corners']=inscr_beta
    values['inscr_v2_corners']=inscr_v2
    values['inscr_v3_corners']=inscr_v3

    return values
