#
"""
Code to create CRDS reference files for the distortion of the
MIRI MRS using reference files delivered with FLT-8:

Note FLT-8 only changed 3C/4A/4B/4C wavelength solution from FLT-7.

MIRI_FM_MIRIFULONG_34LONG_DISTORTION_FLT8.fits
MIRI_FM_MIRIFUSHORT_12MEDIUM_DISTORTION_FLT8.fits
MIRI_FM_MIRIFULONG_34MEDIUM_DISTORTION_FLT8.fits
MIRI_FM_MIRIFUSHORT_12SHORT_DISTORTION_FLT8.fits
MIRI_FM_MIRIFULONG_34SHORT_DISTORTION_FLT8.fits
MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_FLT8.fits

MIRI MRS uses 4 reference files of type:

DISTORTION
REGIONS
SPECWCS
WAVELENGTHRANGE

In this version the CDP file goes from 0-indexed detector pixels
(0,0) in the middle of the lower left detector pixel to V2,V3

make_references() creates all reference files.

In addition to making the usual configurations where the DGAA
and DGAB wheels are in the same configuration (e.g., A-A) this
also created crossed-configuration reference files (e.g., A-B).
This mixes multiple input FITS files since the DGAA wheel sets
the grating for Channels 1/4 and the DGAB wheel for Channels 2/3.

Since the first flight version FLT-1, this code allows specifying
tearfix=True.  If set, this will ensure that all slices in a given
channel have the same wavelength range by zeroing out the slice
information for all slices that go beyond the most-restricted
slice wavelength range.  This fixes a number of bugs that can occur
when the MRS gets data only over a tiny piece of the FOV.

Author: David R. Law (dlaw@stsci.edu), Nadia Dencheva

REVISION HISTORY:
2015         Written by Nadia Dencheva
2016         Adapted for new formats by David Law (dlaw@stsci.edu)
11-Oct-2018  Adapted to new miricoord structure (D. Law)
26-Apr-2019  Updated for CDP-8b (D. Law)
15-Jul-2019  Bugfix to slice mask in CDP version 8B.05.01 (D. Law)
07-Dec-2020  Add cross-dichroic configurations (D. Law)
04-Apr-2022  Update for FLT-1, add fixes for wavelength tearing (D. Law)
26-May-2022  FLT-1 real data from MIR-013 (D. Law)
14-Jun-2022  FLT-2 update from APT-1050 Obs 9 (D. Law)
01-Aug-2022  FLT-3 updated from APT-1524 Obs 16 (D. Law)
27-Aug-2022  FLT-4 wave cal update from APT 1031 (D. Law)
13-Mar-2023  FLT-5 updates using distortion updates plus Jupiter/Saturn wavecal info (D. Law)
31-May-2023  FLT-6 update to 1A+1B wave solution ONLY (D. Law)
30-Aug-2023  FLT-7 update to 4B distortion solution ONLY (D. Law)
06-Dec-2023  FLT-8 update to 3C/4A/4B/4C wave solution (D. Law)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os as os
import time
import numpy as np
from asdf import AsdfFile
import pdb as pdb
from astropy.io import fits
from astropy.modeling import models
from astropy import units as u
from jwst import datamodels
from jwst.assign_wcs import miri
#from numpy.testing import utils

import miricoord.mrs.mrs_tools as mrst
import miricoord.mrs.mrs_pipetools as mrspt
import miri3d.cubepar.make_cubepar as mcp

import miricoord.general.drltimer as drltimer

from asdf.tags.core import Software, HistoryEntry
import datetime

from jwst.datamodels import *

#############################

# Function to loop over all 6 MIRI MRS distortion files
# making reference files for all of them

def create_flt8_all(outdir, **kwargs):
    # Regular grating wheel configurations
    detbands='12A','12B','12C','34A','34B','34C'
    nbands=len(detbands)
    for i in range(nbands):
        create_flt8_setfiles(detbands[i],outdir,**kwargs)

    print('Creating cross-dichroic reference files')
    # Cross-dichroic grating wheel configurations
    xbands='12AB','12AC','12BA','12BC','12CA','12CB','34AB','34AC','34BA','34BC','34CA','34CB'
    nxbands=len(xbands)
    for i in range(nxbands):
        create_flt8_setxfiles(xbands[i],outdir,**kwargs)
        
#############################

# Function to automatically figure out the input/output required to make
# a FLT-2 reference file for a particular detector band (e.g., 12A)

def create_flt8_setfiles(detband,outdir, **kwargs):
    rootdir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    rootdir=os.path.join(rootdir,'data/fits/flt8/')

    if (detband == '12A'):
        fname=os.path.join(rootdir,'MIRI_FM_MIRIFUSHORT_12SHORT_DISTORTION_flt8.fits')
    elif (detband == '34A'):
        fname=os.path.join(rootdir,'MIRI_FM_MIRIFULONG_34SHORT_DISTORTION_flt8.fits')
    elif (detband == '12B'):
        fname=os.path.join(rootdir,'MIRI_FM_MIRIFUSHORT_12MEDIUM_DISTORTION_flt8.fits')
    elif (detband == '34B'):
        fname=os.path.join(rootdir,'MIRI_FM_MIRIFULONG_34MEDIUM_DISTORTION_flt8.fits')
    elif (detband == '12C'):
        fname=os.path.join(rootdir,'MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_flt8.fits')
    elif (detband == '34C'):
        fname=os.path.join(rootdir,'MIRI_FM_MIRIFULONG_34LONG_DISTORTION_flt8.fits')

    distfile=outdir+'jwst_miri_mrs'+detband+'_distortion_flt8.asdf'
    regfile=outdir+'jwst_miri_mrs'+detband+'_regions_flt8.asdf'
    specfile=outdir+'jwst_miri_mrs'+detband+'_specwcs_flt8.asdf'
    wavefile=outdir+'jwst_miri_mrs_wavelengthrange_flt8.asdf'
    refs={'distortion': distfile, 'regions':regfile, 'specwcs':specfile, 'wavelengthrange':wavefile}
    print('Working on: '+detband)
    create_flt8_onereference(fname,refs,**kwargs)
    print('Testing: '+detband)
    test_flt8_onereference(detband,refs)
    print('Done testing: '+detband)

#############################

# Function to automatically figure out the input/output required to make
# a reference file for crossed DGA configurations (e.g., 12AB)

def create_flt8_setxfiles(detxband,outdir,**kwargs):
    rootdir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    rootdir=os.path.join(rootdir,'data/fits/flt8/')

    if (detxband == '12AB'):
        fname1=os.path.join(rootdir,'MIRI_FM_MIRIFUSHORT_12SHORT_DISTORTION_flt8.fits')
        fname2=os.path.join(rootdir,'MIRI_FM_MIRIFUSHORT_12MEDIUM_DISTORTION_flt8.fits')
    elif (detxband == '12AC'):
        fname1=os.path.join(rootdir,'MIRI_FM_MIRIFUSHORT_12SHORT_DISTORTION_flt8.fits')
        fname2=os.path.join(rootdir,'MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_flt8.fits')
    elif (detxband == '12BA'):
        fname1=os.path.join(rootdir,'MIRI_FM_MIRIFUSHORT_12MEDIUM_DISTORTION_flt8.fits')
        fname2=os.path.join(rootdir,'MIRI_FM_MIRIFUSHORT_12SHORT_DISTORTION_flt8.fits')
    elif (detxband == '12BC'):
        fname1=os.path.join(rootdir,'MIRI_FM_MIRIFUSHORT_12MEDIUM_DISTORTION_flt8.fits')
        fname2=os.path.join(rootdir,'MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_flt8.fits')
    elif (detxband == '12CA'):
        fname1=os.path.join(rootdir,'MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_flt8.fits')
        fname2=os.path.join(rootdir,'MIRI_FM_MIRIFUSHORT_12SHORT_DISTORTION_flt8.fits')
    elif (detxband == '12CB'):
        fname1=os.path.join(rootdir,'MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_flt8.fits')
        fname2=os.path.join(rootdir,'MIRI_FM_MIRIFUSHORT_12MEDIUM_DISTORTION_flt8.fits')

    # Note that we are going to SWAP fname1 and fname2 for Ch34; this is because later code will
    # assume that we have things in the order of Ch3-Ch4 and fname1-fname2 whereas the name
    # '34AB' means 3B+4A
    elif (detxband == '34AB'):
        fname2=os.path.join(rootdir,'MIRI_FM_MIRIFULONG_34SHORT_DISTORTION_flt8.fits')
        fname1=os.path.join(rootdir,'MIRI_FM_MIRIFULONG_34MEDIUM_DISTORTION_flt8.fits')
    elif (detxband == '34AC'):
        fname2=os.path.join(rootdir,'MIRI_FM_MIRIFULONG_34SHORT_DISTORTION_flt8.fits')
        fname1=os.path.join(rootdir,'MIRI_FM_MIRIFULONG_34LONG_DISTORTION_flt8.fits')
    elif (detxband == '34BA'):
        fname2=os.path.join(rootdir,'MIRI_FM_MIRIFULONG_34MEDIUM_DISTORTION_flt8.fits')
        fname1=os.path.join(rootdir,'MIRI_FM_MIRIFULONG_34SHORT_DISTORTION_flt8.fits')
    elif (detxband == '34BC'):
        fname2=os.path.join(rootdir,'MIRI_FM_MIRIFULONG_34MEDIUM_DISTORTION_flt8.fits')
        fname1=os.path.join(rootdir,'MIRI_FM_MIRIFULONG_34LONG_DISTORTION_flt8.fits')
    elif (detxband == '34CA'):
        fname2=os.path.join(rootdir,'MIRI_FM_MIRIFULONG_34LONG_DISTORTION_flt8.fits')
        fname1=os.path.join(rootdir,'MIRI_FM_MIRIFULONG_34SHORT_DISTORTION_flt8.fits')
    elif (detxband == '34CB'):
        fname2=os.path.join(rootdir,'MIRI_FM_MIRIFULONG_34LONG_DISTORTION_flt8.fits')
        fname1=os.path.join(rootdir,'MIRI_FM_MIRIFULONG_34MEDIUM_DISTORTION_flt8.fits')

    distfile=outdir+'jwst_miri_mrs'+detxband+'_distortion_flt8.asdf'
    regfile=outdir+'jwst_miri_mrs'+detxband+'_regions_flt8.asdf'
    specfile=outdir+'jwst_miri_mrs'+detxband+'_specwcs_flt8.asdf'
    wavefile=outdir+'jwst_miri_mrs_wavelengthrange_flt8.asdf'
    refs={'distortion': distfile, 'regions':regfile, 'specwcs':specfile, 'wavelengthrange':wavefile}
    print('Working on: '+detxband)
    create_flt8_onexreference(fname1,fname2,refs,**kwargs)
    print('Created: '+detxband)

    
#############################
    
def create_flt8_onereference(fname, ref, **kwargs):
    """
    Create ASDF WCS reference files for MIRI MRS data from a single reference file.
    Parameters
    ----------
    fname : str
        name of reference file
    ref : dict
        A dictionary {reftype: refname}, e.g.
        {'distortion': 'jwst_miri_distortion_0001.asdf',
         'regions': 'jwst_miri_regions_0001.asdf',
         'specwcs': 'jwst_miri_specwcs_0001.asdf',
         'wavelengthrange': 'jwst_miri_wavelengthrange_0001.asdf'}
    Examples
    --------
    >>> create_flt8_references('MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_flt8.fits', ref)
    """
    with fits.open(fname) as f:
        channel = f[0].header['CHANNEL']
        band = f[0].header['BAND']
        detector = f[0].header['DETECTOR']
        ch1 = 'CH{0}'.format(channel[0])
        ch2 = 'CH{0}'.format(channel[1])
        slices = f[1].data
        #fov1 = f[2].data
        #fov2 = f[3].data
        alpha1 = f[('Alpha_'+ch1, 1)].data
        lam1 = f[('Lambda_'+ch1, 1)].data
        alpha2 = f[('Alpha_'+ch2, 1)].data
        lam2 = f[('Lambda_'+ch2, 1)].data
        x1 = f[('X_'+ch1, 1)].data
        y1 = f[('Y_'+ch1, 1)].data
        x2 = f[('X_'+ch2, 1)].data
        y2 = f[('Y_'+ch2, 1)].data
        ab_v23 = f[('albe_to_V2V3', 1)].data.copy()
        v23_ab = f[('V2V3_to_albe', 1)].data.copy()
        b0_ch1 = f[0].header['B_ZERO'+ch1[2]]
        bdel_ch1 = f[0].header['B_DEL'+ch1[2]]
        b0_ch2 = f[0].header['B_ZERO'+ch2[2]]
        bdel_ch2 = f[0].header['B_DEL'+ch2[2]]
    # Get channel names, e.g. 1LONG, 2LONG
    channels = [c + band for c in channel]
    # Note that now 'channel' is (e.g.) 12, while 'channels' is (e.g.) '1SHORT','2SHORT'

    # If selected, crop slice mask to mitigate spectral tearing
    if ('tearfix' in kwargs):
        if kwargs['tearfix']:
            print('Adjusting slice mask to clean spectral tearing')
            slices = clean_tears(slices,channel,band)

    bzero = {}
    bdel = {}
    for c in channel:
        cb = c+band
        bzero[cb] = f[0].header['B_ZERO' + c]
        bdel[cb] = f[0].header['B_DEL' + c]

    # MRS reference files are long enough that keeping tables as inline
    # text is impractical
    outformat='inline'

    coeff_names = build_coeff_names(alpha1.names)
    amodel1 = create_poly_models(alpha1, int(channel[0]), coeff_names, name='det2local')
    lmodel1 = create_poly_models(lam1, int(channel[0]), coeff_names, name='det2local')
    amodel2 = create_poly_models(alpha2, int(channel[1]), coeff_names, name='det2local')
    lmodel2 = create_poly_models(lam2, int(channel[1]), coeff_names, name='det2local')
    # reverse models

    # 'x' in the report corresponds to y in python and 'y' to x,
    # The x/y models take (lam, alpha)
    xmodel1 = create_xy_models(x1, int(channel[0]), coeff_names, name='x')
    ymodel1 = create_xy_models(y1, int(channel[0]), coeff_names, name='y')
    xmodel2 = create_xy_models(x2, int(channel[1]), coeff_names, name='x')
    ymodel2 = create_xy_models(y2, int(channel[1]), coeff_names, name='y')
    amodel1.update(amodel2)
    xmodel1.update(xmodel2)
    ymodel1.update(ymodel2)
    lmodel1.update(lmodel2)

    bmodel1 = create_beta_models(b0_ch1, bdel_ch1, int(channel[0]), len(alpha1))
    bmodel2 = create_beta_models(b0_ch2, bdel_ch2, int(channel[1]), len(alpha2))

    bmodel1.update(bmodel2)
    useafter = "2022-05-01T00:00:00"
    author =  'David R. Law, Polychronis Patapis, Yannis Argyriou'  #Author of the data
    description = 'MIRI MRS FLT8 (flt8) distortion reference data.'

    create_distortion_file(reftype='distortion', detector=detector, band=band, channel=channel, channels=channels,
                           data=(amodel1, bmodel1, xmodel1, ymodel1, bzero, bdel, ab_v23, v23_ab), name=ref['distortion'],
                           author=author, useafter=useafter, description=description, outformat=outformat)

    create_specwcs_file('specwcs', detector, band, channel, lmodel1, ref['specwcs'], author,
                        useafter, description, outformat)

    create_regions_file(slices, detector, band, channel, ref['regions'], author, useafter,
               description, outformat)

    create_wavelengthrange_file(ref['wavelengthrange'], detector, author, useafter,
                                description, outformat)

#############################
    
def create_flt8_onexreference(fname1, fname2, ref, **kwargs):
    """
    Create ASDF WCS reference files for MIRI MRS data from a cross-dichroic.
    Parameters
    ----------
    fname1 : str
        name of reference file for Ch1/4
    fname2 : str
        name of reference file for Ch2/3
    ref : dict
        A dictionary {reftype: refname}, e.g.
        {'distortion': 'jwst_miri_distortion_0001.asdf',
         'regions': 'jwst_miri_regions_0001.asdf',
         'specwcs': 'jwst_miri_specwcs_0001.asdf',
         'wavelengthrange': 'jwst_miri_wavelengthrange_0001.asdf'}
    """
    with fits.open(fname1) as f:
        channel = f[0].header['CHANNEL']
        band1 = f[0].header['BAND']
        detector = f[0].header['DETECTOR']
        ch1 = 'CH{0}'.format(channel[0])
        slices1 = f[1].data
        alpha1 = f[('Alpha_'+ch1, 1)].data
        lam1 = f[('Lambda_'+ch1, 1)].data
        x1 = f[('X_'+ch1, 1)].data
        y1 = f[('Y_'+ch1, 1)].data
        ab_v23_1 = f[('albe_to_V2V3', 1)].data.copy()
        v23_ab_1 = f[('V2V3_to_albe', 1)].data.copy()
        b0_ch1 = f[0].header['B_ZERO'+ch1[2]]
        bdel_ch1 = f[0].header['B_DEL'+ch1[2]]

    with fits.open(fname2) as f:
        channel = f[0].header['CHANNEL']
        band2 = f[0].header['BAND']
        detector = f[0].header['DETECTOR']
        ch2 = 'CH{0}'.format(channel[1])
        slices2 = f[1].data
        alpha2 = f[('Alpha_'+ch2, 1)].data
        lam2 = f[('Lambda_'+ch2, 1)].data
        x2 = f[('X_'+ch2, 1)].data
        y2 = f[('Y_'+ch2, 1)].data
        ab_v23_2 = f[('albe_to_V2V3', 1)].data.copy()
        v23_ab_2 = f[('V2V3_to_albe', 1)].data.copy()
        b0_ch2 = f[0].header['B_ZERO'+ch2[2]]
        bdel_ch2 = f[0].header['B_DEL'+ch2[2]]

    # If selected, crop slice mask to mitigate spectral tearing
    if ('tearfix' in kwargs):
        if kwargs['tearfix']:
            print('Adjusting slice mask to clean spectral tearing')
            slices1 = clean_tears(slices1,channel,band1)

    # If selected, crop slice mask to mitigate spectral tearing
    if ('tearfix' in kwargs):
        if kwargs['tearfix']:
            print('Adjusting slice mask to clean spectral tearing')
            slices2 = clean_tears(slices2,channel,band2)            
        
    # Stich together the different reference files across middle of the detector
    slices = slices1.copy()

    if (channel == '12'):
        band=band1+'-'+band2
        slices[:,:,509:]=slices2[:,:,509:] # Paste in Ch2
    if (channel == '34'):
        band=band2+'-'+band1
        slices[:,:,0:509]=slices2[:,:,0:509] # Paste in Ch4

    ab_v23 = ab_v23_1.copy()
    ab_v23[2] = ab_v23_2[2]
    ab_v23[3] = ab_v23_2[3]
    v23_ab = v23_ab_1.copy()
    v23_ab[2] = v23_ab_2[2]
    v23_ab[3] = v23_ab_2[3]

    # Get channel names, e.g. 1LONG, 2LONG
    channels=channel[0]+band,channel[1]+band
    # Note that now 'channel' is (e.g.) 12, while 'channels' is (e.g.) '1SHORT-MEDIUM'
    # This naming is awkward, but necessary for pipeline to deal with the names passed
    # from individual data file headers

    bzero = {}
    bdel = {}
    for c in channel:
        cb = c+band
        bzero[cb] = f[0].header['B_ZERO' + c]
        bdel[cb] = f[0].header['B_DEL' + c]

    # MRS reference files are long enough that keeping tables as inline
    # text is impractical
    outformat='inline'

    coeff_names = build_coeff_names(alpha1.names)
    amodel1 = create_poly_models(alpha1, int(channel[0]), coeff_names, name='det2local')
    lmodel1 = create_poly_models(lam1, int(channel[0]), coeff_names, name='det2local')
    amodel2 = create_poly_models(alpha2, int(channel[1]), coeff_names, name='det2local')
    lmodel2 = create_poly_models(lam2, int(channel[1]), coeff_names, name='det2local')
    # reverse models

    # 'x' in the report corresponds to y in python and 'y' to x,
    # The x/y models take (lam, alpha)
    xmodel1 = create_xy_models(x1, int(channel[0]), coeff_names, name='x')
    ymodel1 = create_xy_models(y1, int(channel[0]), coeff_names, name='y')
    xmodel2 = create_xy_models(x2, int(channel[1]), coeff_names, name='x')
    ymodel2 = create_xy_models(y2, int(channel[1]), coeff_names, name='y')
    amodel1.update(amodel2)
    xmodel1.update(xmodel2)
    ymodel1.update(ymodel2)
    lmodel1.update(lmodel2)

    bmodel1 = create_beta_models(b0_ch1, bdel_ch1, int(channel[0]), len(alpha1))
    bmodel2 = create_beta_models(b0_ch2, bdel_ch2, int(channel[1]), len(alpha2))

    bmodel1.update(bmodel2)
    useafter = "2022-05-01T00:00:00"
    author =  'David R. Law, Polychronis Patapis, Yannis Argyriou'  #Author of the data
    description = 'MIRI MRS FLT8 (flt8) distortion reference data.'

    create_distortion_file(reftype='distortion', detector=detector, band=band, channel=channel, channels=channels,
                           data=(amodel1, bmodel1, xmodel1, ymodel1, bzero, bdel, ab_v23, v23_ab), name=ref['distortion'],
                           author=author, useafter=useafter, description=description, outformat=outformat)

    create_specwcs_file('specwcs', detector, band, channel, lmodel1, ref['specwcs'], author,
                        useafter, description, outformat)

    create_regions_file(slices, detector, band, channel, ref['regions'], author, useafter,
               description, outformat)

    # We don't need to make the wavelength range file again as that already has x-dichroic info
    
#############################

def create_regions_file(slices, detector, band, channel, name, author, useafter, description, outformat):
    model = RegionsModel()
    model = create_reffile_header(model, detector, band, channel, author, useafter,
                                  description)
    model.meta.filename = os.path.split(name)[-1]
    model.regions = slices
    model.save(name)

#############################

def create_reffile_header(model, detector, band, channel, author, useafter,
                          description=""):
    model.meta.description = description
    model.meta.author = author
    model.meta.useafter = useafter
    model.meta.pedigree = 'INFLIGHT 2022-06-08 2022-06-08'
    model.meta.instrument.name = "MIRI"
    model.meta.instrument.detector = detector
    model.meta.instrument.channel = channel
    model.meta.instrument.band = band
    model.meta.exposure.type = "MIR_MRS"

    entry = HistoryEntry({'description': "MRS wavelength and spatial distortion.  FLT5 updates spatial distortion (many programs) and wavelength calibration (based on programs 1246/1247 observations of Jupiter/Saturn)", 'time': datetime.datetime.utcnow()})
    entry = HistoryEntry({'description': "MRS wavelength and spatial distortion.  FLT6 updates wavelength calibration of 1A/1B slightly relative to FLT5)", 'time': datetime.datetime.utcnow()})
    entry = HistoryEntry({'description': "MRS wavelength and spatial distortion.  FLT7 updates 4B distortion solution.)", 'time': datetime.datetime.utcnow()})
    entry = HistoryEntry({'description': "MRS wavelength and spatial distortion.  FLT8 updates 3C/4A/4B/4C wavelength solution.)", 'time': datetime.datetime.utcnow()})
    software = Software({'name': 'miricoord', 'author': 'D.Law', 
                         'homepage': 'https://github.com/STScI-MIRI/miricoord', 'version': "master"})
    entry['software'] = software
    model.history = [entry]

    return model

#############################

def create_distortion_file(reftype, detector,  band, channel, channels, data, name, author,
                           useafter, description, outformat):
    dist = DistortionMRSModel()
    description = 'MIRI MRS Distortion Maps'
    dist = create_reffile_header(dist, detector, band, channel, author, useafter,
                                 description)

    dist.meta.filename = os.path.split(name)[-1]
    # Split the provided data vector into its pieces
    adata, bdata, xdata, ydata, bzero, bdel, ab_v23, v23_ab = data

    slices = list(xdata.keys())
    dist.slices = slices
    xd = [xdata[sl] for sl in slices]
    dist.x_model = xd
    yd = [ydata[sl] for sl in slices]
    dist.y_model = yd
    ad = [adata[sl] for sl in slices]
    dist.alpha_model = ad
    bd = [bdata[sl] for sl in slices]
    dist.beta_model = bd
    dist.bzero = {'channel_band': list(bzero.keys()), 'beta_zero': list(bzero.values())}
    dist.bdel = {'channel_band': list(bdel.keys()), 'delta_beta': list(bdel.values())}

    """
    Create the transform from MIRI Local to telescope V2/V3 system for all channels.
    """
    channel = "".join([ch[0] for ch in channels])

    # Read in coefficients from the tables.  Note that we'll flip the coefficient
    # ordering since they were set up for column-major indexing (IDL) but we're working in
    # python (row-major)
    # ab -> v2 transform for first channel
    c0_0, c1_0, c2_0, c0_1, c1_1, c2_1, c0_2, c1_2, c2_2 = ab_v23[0][1:]
    ch1_v2 = models.Polynomial2D(4, c0_0=c0_0, c1_0=c1_0, c2_0=c2_0, c0_1=c0_1, 
                    c1_1=c1_1, c2_1=c2_1, c0_2=c0_2, c1_2=c1_2, c2_2=c2_2, name="ab_v23")
    # v2,v3 -> a transform for first channel
    c0_0, c1_0, c2_0, c0_1, c1_1, c2_1, c0_2, c1_2, c2_2 = v23_ab[0][1:]
    ch1_a = models.Polynomial2D(4, c0_0=c0_0, c1_0=c1_0, c2_0=c2_0, c0_1=c0_1, 
                    c1_1=c1_1, c2_1=c2_1, c0_2=c0_2, c1_2=c1_2, c2_2=c2_2, name="v23_ab")
    # ab -> v3 transform for first channel
    c0_0, c1_0, c2_0, c0_1, c1_1, c2_1, c0_2, c1_2, c2_2 = ab_v23[1][1:]
    ch1_v3 = models.Polynomial2D(4, c0_0=c0_0, c1_0=c1_0, c2_0=c2_0, c0_1=c0_1, 
                    c1_1=c1_1, c2_1=c2_1, c0_2=c0_2, c1_2=c1_2, c2_2=c2_2, name="ab_v23")
    # v2,v3 -> b transform for first channel
    c0_0, c1_0, c2_0, c0_1, c1_1, c2_1, c0_2, c1_2, c2_2 = v23_ab[1][1:]
    ch1_b = models.Polynomial2D(4, c0_0=c0_0, c1_0=c1_0, c2_0=c2_0, c0_1=c0_1, 
                    c1_1=c1_1, c2_1=c2_1, c0_2=c0_2, c1_2=c1_2, c2_2=c2_2, name="v23_ab")
    # ab -> v2 transform for second channel
    c0_0, c1_0, c2_0, c0_1, c1_1, c2_1, c0_2, c1_2, c2_2 = ab_v23[2][1:]
    ch2_v2 = models.Polynomial2D(4, c0_0=c0_0, c1_0=c1_0, c2_0=c2_0, c0_1=c0_1, 
                    c1_1=c1_1, c2_1=c2_1, c0_2=c0_2, c1_2=c1_2, c2_2=c2_2, name="ab_v23")
    # v2,v3 -> a transform for second channel
    c0_0, c1_0, c2_0, c0_1, c1_1, c2_1, c0_2, c1_2, c2_2 = v23_ab[2][1:]
    ch2_a = models.Polynomial2D(4, c0_0=c0_0, c1_0=c1_0, c2_0=c2_0, c0_1=c0_1, 
                    c1_1=c1_1, c2_1=c2_1, c0_2=c0_2, c1_2=c1_2, c2_2=c2_2, name="v23_ab")
    # ab -> v3 transform for second channel
    c0_0, c1_0, c2_0, c0_1, c1_1, c2_1, c0_2, c1_2, c2_2 = ab_v23[3][1:]
    ch2_v3 = models.Polynomial2D(4, c0_0=c0_0, c1_0=c1_0, c2_0=c2_0, c0_1=c0_1, 
                    c1_1=c1_1, c2_1=c2_1, c0_2=c0_2, c1_2=c1_2, c2_2=c2_2, name="ab_v23")
    # v2,v3 -> b transform for second channel
    c0_0, c1_0, c2_0, c0_1, c1_1, c2_1, c0_2, c1_2, c2_2 = v23_ab[3][1:]
    ch2_b = models.Polynomial2D(4, c0_0=c0_0, c1_0=c1_0, c2_0=c2_0, c0_1=c0_1, 
                    c1_1=c1_1, c2_1=c2_1, c0_2=c0_2, c1_2=c1_2, c2_2=c2_2, name="v23_ab")

    # Since the matrix transforms need a 4-element input we need a mapping
    # to go from (0,1) to (0,1,0,1)
    mapping = models.Mapping((0, 1, 0, 1), n_inputs=2)

    # Put the mappings all together
    ch1 = mapping | ch1_v2 & ch1_v3
    ch2 = mapping | ch2_v2 & ch2_v3

    # And make the inverse mapping
    ch1.inverse =  mapping | ch1_a & ch1_b
    ch2.inverse =  mapping | ch2_a & ch2_b

    # save to file
    dist.abv2v3_model = {'channel_band': channels, 'model': [ch1, ch2]}
    dist.meta.input_units = u.pix
    dist.meta.output_units = u.arcsec
    dist.save(name)

#############################

def create_specwcs_file(reftype, detector, band, channel, lmodel, name, author, useafter, description, outformat):
    spec = SpecwcsModel()
    spec = create_reffile_header(spec, detector, band, channel, author, useafter,
                                 description)

    spec.meta.subarray.name = "N/A"
    spec.meta.filename = os.path.split(name)[-1]
    spec.meta.input_units = u.pix
    spec.meta.output_units = u.micron

    slices = list(lmodel.keys())
    spec.slices = slices
    lam_data = [lmodel[sl] for sl in slices]
    spec.model = lam_data

    spec.save(name)

#############################

# Create the x,y to a,b models
def create_poly_models(data, channel, coeff_names, name):
    """
    Create a 2D polynomial model for the transformation
    detector --> local MIRI frame
    Works for alpha and lambda coordinates.
    """
    nslices = len(data)
    sl = channel * 100 + np.arange(1, nslices+1)

    transforms = {}
    for i in range(nslices):
        sl = channel * 100 + i +1
        al = data[i]
        xs = al[0]
        coeffs = {}
        for c, val in zip(coeff_names, al[1:]):
            coeffs[c] = val

        # As of CDP-8b both the IDT transform as the pipeline use 0-indexed pixels, and
        # include the 4 reference pixels in their counting.  Therefore we do not need to
        # apply any index shift, just the transform.
        thisxform=models.Identity(1) & models.Shift(-xs) | models.Polynomial2D(8, name=name, **coeffs)
        # Put the models together
        transforms[sl] = thisxform

    return transforms

#############################

# Create the a,b, to x,y models
def create_xy_models(data, channel, coeff_names, name):
    """
    Create a 2D polynomial model for the transformation
    local_MIRI --> detector frame.
    """
    nslices = len(data)
    sl = channel * 100 + np.arange(1, nslices+1)
    shname = "shift_{0}".format(name)
    pname = "polynomial_{0}".format(name)
    transforms = {}
    for i in range(nslices):
        sl = channel * 100 + i +1
        al = data[i]
        xs = al[0]
        coeffs = {}
        for c, val in zip(coeff_names, al[1:]):
            coeffs[c] = val

        # As of CDP-8b both the IDT transform as the pipeline use 0-indexed pixels, and
        # include the 4 reference pixels in their counting.  Therefore we do not need to
        # apply any index shift, just the transform.
        thisxform=models.Shift(-xs, name=shname) & models.Identity(1) | models.Polynomial2D(8, name=pname, **coeffs)
        transforms[sl] =  thisxform

    return transforms

#############################

def build_coeff_names(names):
    names = names[1:]
    names = [name.replace('VAR2_', "c") for name in names]
    return names

#############################

def create_beta_models(b0, bdel, channel, nslices):
    beta = {}
    for s in range(nslices):
        sl = channel * 100 + s +1
        beta_s = b0 + s * bdel
        m = models.Const1D(beta_s, name='det2local') #xy2beta and xy2lam
        beta[sl] = m
    return beta

#############################

def create_wavelengthrange_file(name, detector, author, useafter, description, outformat):
    model = WavelengthrangeModel()

    # Based on common range across all slices, IDT as-designed values
    wavelengthrange = {'1SHORT': (4.68, 5.97),
                        '1MEDIUM': (5.24, 6.87),
                        '1LONG': (6.2, 7.90),
                        '2SHORT': (7.27, 9.03),
                        '2MEDIUM': (8.43, 10.39),
                        '2LONG': (9.76, 11.97),
                        '3SHORT': (11.29, 13.75),
                        '3MEDIUM': (13.08, 15.86),
                        '3LONG': (15.14, 18.29),
                        '4SHORT': (17.40, 21.20),
                        '4MEDIUM': (20.31, 24.68),
                        '4LONG': (23.72, 28.75)
                        }
    channels = ['1SHORT', '1MEDIUM', '1LONG', '2SHORT', '2MEDIUM', '2LONG',
                '3SHORT', '3MEDIUM', '3LONG', '4SHORT', '4MEDIUM', '4LONG']

    # We also need to add cross-dichroic information
    channels.append('1SHORT-MEDIUM'), channels.append('1SHORT-LONG')
    channels.append('1MEDIUM-SHORT'), channels.append('1MEDIUM-LONG')
    channels.append('1LONG-SHORT'), channels.append('1LONG-MEDIUM')
    wavelengthrange['1SHORT-MEDIUM'] = wavelengthrange['1SHORT']
    wavelengthrange['1SHORT-LONG'] = wavelengthrange['1SHORT']
    wavelengthrange['1MEDIUM-SHORT'] = wavelengthrange['1MEDIUM']
    wavelengthrange['1MEDIUM-LONG'] = wavelengthrange['1MEDIUM']
    wavelengthrange['1LONG-SHORT'] = wavelengthrange['1LONG']
    wavelengthrange['1LONG-MEDIUM'] = wavelengthrange['1LONG']    

    channels.append('2SHORT-MEDIUM'), channels.append('2SHORT-LONG')
    channels.append('2MEDIUM-SHORT'), channels.append('2MEDIUM-LONG')
    channels.append('2LONG-SHORT'), channels.append('2LONG-MEDIUM')
    wavelengthrange['2SHORT-MEDIUM'] = wavelengthrange['2MEDIUM']
    wavelengthrange['2SHORT-LONG'] = wavelengthrange['2LONG']
    wavelengthrange['2MEDIUM-SHORT'] = wavelengthrange['2SHORT']
    wavelengthrange['2MEDIUM-LONG'] = wavelengthrange['2LONG']
    wavelengthrange['2LONG-SHORT'] = wavelengthrange['2SHORT']
    wavelengthrange['2LONG-MEDIUM'] = wavelengthrange['2MEDIUM']   

    channels.append('3SHORT-MEDIUM'), channels.append('3SHORT-LONG')
    channels.append('3MEDIUM-SHORT'), channels.append('3MEDIUM-LONG')
    channels.append('3LONG-SHORT'), channels.append('3LONG-MEDIUM')
    wavelengthrange['3SHORT-MEDIUM'] = wavelengthrange['3MEDIUM']
    wavelengthrange['3SHORT-LONG'] = wavelengthrange['3LONG']
    wavelengthrange['3MEDIUM-SHORT'] = wavelengthrange['3SHORT']
    wavelengthrange['3MEDIUM-LONG'] = wavelengthrange['3LONG']
    wavelengthrange['3LONG-SHORT'] = wavelengthrange['3SHORT']
    wavelengthrange['3LONG-MEDIUM'] = wavelengthrange['3MEDIUM']  

    channels.append('4SHORT-MEDIUM'), channels.append('4SHORT-LONG')
    channels.append('4MEDIUM-SHORT'), channels.append('4MEDIUM-LONG')
    channels.append('4LONG-SHORT'), channels.append('4LONG-MEDIUM')
    wavelengthrange['4SHORT-MEDIUM'] = wavelengthrange['4SHORT']
    wavelengthrange['4SHORT-LONG'] = wavelengthrange['4SHORT']
    wavelengthrange['4MEDIUM-SHORT'] = wavelengthrange['4MEDIUM']
    wavelengthrange['4MEDIUM-LONG'] = wavelengthrange['4MEDIUM']
    wavelengthrange['4LONG-SHORT'] = wavelengthrange['4LONG']
    wavelengthrange['4LONG-MEDIUM'] = wavelengthrange['4LONG']   
    
    model = create_reffile_header(model, detector, band="N/A", channel="N/A", author=author,
                                 useafter=useafter, description=description)
    model.meta.filename = os.path.split(name)[-1]
    model.meta.instrument.detector = "N/A"
    model.waverange_selector = channels
    wr = [wavelengthrange[ch] for ch in channels]
    model.wavelengthrange = wr
    model.meta.wavelength_units = u.micron
    model.save(name)

#############################

# Function to test the implemented transforms and ASDF files
# Detband is (e.g.) '12A'
def test_flt8_onereference(detband,refs):
    ch1,ch2=mrspt.channel(detband)# Convert to (e.g.) '1A' and '2A'

    mrspt.testtransform(testchannel=[ch1,ch2],refs=refs)
 
#############################

# Function to zero-out slice map in regions where slices doesn't
# have full spatial coverage (i.e., to avoid 'tearing' in spec2
# cubes at wavelengths where not all slices map to the sky).

def clean_tears(slices,channel,band):
    if ((channel == '12') & (band == 'SHORT')):
        detband1, detband2 = '1A', '2A'
    if ((channel == '12') & (band == 'MEDIUM')):
        detband1, detband2 = '1B', '2B'
    if ((channel == '12') & (band == 'LONG')):
        detband1, detband2 = '1C', '2C'
    if ((channel == '34') & (band == 'SHORT')):
        detband1, detband2 = '3A', '4A'
    if ((channel == '34') & (band == 'MEDIUM')):
        detband1, detband2 = '3B', '4B'
    if ((channel == '34') & (band == 'LONG')):
        detband1, detband2 = '3C', '4C'

    newslices=np.zeros_like(slices)
    # How many planes does the slice mask have?
    nz = newslices.shape[0]

    mrst.set_toolversion('flt8')
    print(mrst.version())
    
    for ii in range(0,nz):
        wimg=mrst.waveimage(detband1,mapplane=ii)
        wmin, wmax = mcp.waveminmax(detband1,mapplane=ii,waveimage=wimg)
        indx=np.where((wimg >= wmin) & (wimg <= wmax))
        temp1 = slices[ii,:,:]
        temp2 = newslices[ii,:,:]
        temp2[indx] = temp1[indx]

    for ii in range(0,nz):
        wimg=mrst.waveimage(detband2,mapplane=ii)
        wmin, wmax = mcp.waveminmax(detband2,mapplane=ii,waveimage=wimg)
        indx=np.where((wimg >= wmin) & (wimg <= wmax))
        temp1 = slices[ii,:,:]
        temp2 = newslices[ii,:,:]
        temp2[indx] = temp1[indx]
    
    return newslices
