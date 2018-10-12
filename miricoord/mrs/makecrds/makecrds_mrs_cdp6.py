#
"""
Code to create CRDS reference files for the distortion of the
MIRI MRS using IDT reference files delivered with CDP-6:

MIRI_FM_MIRIFULONG_34LONG_DISTORTION_06.04.00.fits
MIRI_FM_MIRIFUSHORT_12MEDIUM_DISTORTION_06.04.00.fits
MIRI_FM_MIRIFULONG_34MEDIUM_DISTORTION_06.04.00.fits
MIRI_FM_MIRIFUSHORT_12SHORT_DISTORTION_06.04.00.fits
MIRI_FM_MIRIFULONG_34SHORT_DISTORTION_06.04.00.fits
MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_06.04.00.fits

MIRI MRS uses 4 reference files of type:

DISTORTION
REGIONS
SPECWCS
WAVELENGTHRANGE

In this version the CDP file goes from 1-indexed detector pixels
(0,0) in the middle of the lower left detector pixel to XAN,YAN

We will need to add additional transforms so that the mapping goes
from 0-indexed detector pixels (0,0) in the middle of the lower left
reference pixel to V2,V3

make_references() creates all reference files.

Author: David R. Law (dlaw@stsci.edu), Nadia Dencheva

REVISION HISTORY:
2015         Written by Nadia Dencheva
2016         Adapted for new formats by David Law (dlaw@stsci.edu)
11-Oct-2018  Adapted to new miricoord structure (D. Law)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os as os
import numpy as np
from asdf import AsdfFile
import pdb as pdb
from astropy.io import fits
from astropy.modeling import models
from astropy import units as u
from jwst import datamodels
from jwst.assign_wcs import miri
from numpy.testing import utils

import miricoord.miricoord.mrs.mrs_tools as mrst
import miricoord.miricoord.mrs.mrs_pipetools as mrspt

import miricoord.miricoord.general.drltimer as drltimer

from asdf.tags.core import Software, HistoryEntry
import datetime

from jwst.datamodels import *

#############################

# Function to loop over all 6 MIRI MRS distortion files
# making reference files for all of them
# create_cdp6_all('./')
#@drltimer.fn_timer
def create_cdp6_all(outdir):
    detbands='12A','12B','12C','34A','34B','34C'
    nbands=len(detbands)
    for i in range(nbands):
        create_cdp6_setfiles(detbands[i],outdir)

# Function to automatically figure out the input/output required to make
# a CDP-6 reference file for a particular detector band (e.g., 12A)
# create_cdp6_setfiles('12A','./')
#@drltimer.fn_timer
def create_cdp6_setfiles(detband,outdir):
    rootdir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    rootdir=os.path.join(rootdir,'data/fits/cdp6/')

    if (detband == '12A'):
        fname=os.path.join(rootdir,'MIRI_FM_MIRIFUSHORT_12SHORT_DISTORTION_06.04.00.fits')
    elif (detband == '34A'):
        fname=os.path.join(rootdir,'MIRI_FM_MIRIFULONG_34SHORT_DISTORTION_06.04.00.fits')
    elif (detband == '12B'):
        fname=os.path.join(rootdir,'MIRI_FM_MIRIFUSHORT_12MEDIUM_DISTORTION_06.04.00.fits')
    elif (detband == '34B'):
        fname=os.path.join(rootdir,'MIRI_FM_MIRIFULONG_34MEDIUM_DISTORTION_06.04.00.fits')
    elif (detband == '12C'):
        fname=os.path.join(rootdir,'MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_06.04.00.fits')
    elif (detband == '34C'):
        fname=os.path.join(rootdir,'MIRI_FM_MIRIFULONG_34LONG_DISTORTION_06.04.00.fits')

    distfile=outdir+'jwst_miri_mrs'+detband+'_distortion_cdp6.asdf'
    regfile=outdir+'jwst_miri_mrs'+detband+'_regions_cdp6.asdf'
    specfile=outdir+'jwst_miri_mrs'+detband+'_specwcs_cdp6.asdf'
    wavefile=outdir+'jwst_miri_mrs_wavelengthrange_cdp6.asdf'
    refs={'distortion': distfile, 'regions':regfile, 'specwcs':specfile, 'wavelengthrange':wavefile}
    print('Working on: '+detband)
    create_cdp6_onereference(fname,refs)
    print('Testing: '+detband)
    test_cdp6_onereference(detband,refs)
    print('Done testing: '+detband)

def create_cdp6_onereference(fname, ref):
    """
    Create ASDF WCS reference files for MIRI MRS data from a single CDP6 reference file.
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
    >>> create_cdp6_references('MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_06.04.00.fits', ref)
    """
    with fits.open(fname) as f:
        channel = f[0].header['CHANNEL']
        band = f[0].header['BAND']
        detector = f[0].header['DETECTOR']
        ch1 = 'CH{0}'.format(channel[0])
        ch2 = 'CH{0}'.format(channel[1])
        slices = f[1].data
        fov1 = f[2].data
        fov2 = f[3].data
        alpha1 = f[('Alpha_'+ch1, 1)].data
        lam1 = f[('Lambda_'+ch1, 1)].data
        alpha2 = f[('Alpha_'+ch2, 1)].data
        lam2 = f[('Lambda_'+ch2, 1)].data
        x1 = f[('X_'+ch1, 1)].data
        y1 = f[('Y_'+ch1, 1)].data
        x2 = f[('X_'+ch2, 1)].data
        y2 = f[('Y_'+ch2, 1)].data
        ab_v23 = f[('albe_to_XANYAN', 1)].data.copy()
        v23_ab = f[('XANYAN_to_albe', 1)].data.copy()
        b0_ch1 = f[0].header['B_ZERO'+ch1[2]]
        bdel_ch1 = f[0].header['B_DEL'+ch1[2]]
        b0_ch2 = f[0].header['B_ZERO'+ch2[2]]
        bdel_ch2 = f[0].header['B_DEL'+ch2[2]]
    # Get channel names, e.g. 1LONG, 2LONG
    channels = [c + band for c in channel]
    # Note that now 'channel' is (e.g.) 12, while 'channels' is (e.g.) '1SHORT','2SHORT'

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
    useafter = "2000-01-01T00:00:00"
    author =  'Adrian M. Glauser, David R. Law'  #Author of the data
    description = 'MIRI MRS CDP6 distortion reference data.'

    create_distortion_file(reftype='distortion', detector=detector, band=band, channel=channel, channels=channels,
                           data=(amodel1, bmodel1, xmodel1, ymodel1, bzero, bdel, ab_v23, v23_ab), name=ref['distortion'],
                           author=author, useafter=useafter, description=description, outformat=outformat)

    create_specwcs_file('specwcs', detector, band, channel, lmodel1, ref['specwcs'], author,
                        useafter, description, outformat)

    create_regions_file(slices, detector, band, channel, ref['regions'], author, useafter,
               description, outformat)

    create_wavelengthrange_file(ref['wavelengthrange'], detector, author, useafter,
                                description, outformat)


def create_regions_file(slices, detector, band, channel, name, author, useafter, description, outformat):
    model = RegionsModel()
    model = create_reffile_header(model, detector, band, channel, author, useafter,
                                  description)
    model.meta.filename = os.path.split(name)[-1]
    model.regions = slices
    model.save(name)


def create_reffile_header(model, detector, band, channel, author, useafter,
                          description=""):
    model.meta.description = description
    model.meta.author = author
    model.meta.useafter = useafter
    model.meta.pedigree = 'GROUND'
    model.meta.instrument.name = "MIRI"
    model.meta.instrument.detector = detector
    model.meta.instrument.channel = channel
    model.meta.instrument.band = band
    model.meta.exposure.type = "MIR_MRS"

    entry = HistoryEntry({'description': "DOCUMENT: MIRI-TN-00001-ETH_Iss2-1_Calibrationproduct_MRS_d2c.  New files created from CDP-6 with updated file structure and V2/V3 instead of XAN/YAN", 'time': datetime.datetime.utcnow()})
    software = Software({'name': 'coordinates', 'author': 'D.Law', 
                         'homepage': 'https://github.com/STScI-MIRI/coordinates', 'version': "20171219"})
    entry['software'] = software
    model.history = [entry]

    return model


def create_distortion_file(reftype, detector,  band, channel, channels, data, name, author,
                           useafter, description, outformat):
    dist = DistortionMRSModel()
    description = 'MIRI MRS Distortion Maps - build 7.1'
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
    c0_0, c1_0, c0_1, c1_1 = ab_v23[0][1:]
    ch1_v2 = models.Polynomial2D(2, c0_0=c0_0, c1_0=c1_0, c0_1=c0_1, c1_1=c1_1,
                                 name="ab_v23")
    # v2,v3 -> a transform for first channel
    c0_0, c1_0, c0_1, c1_1 = v23_ab[0][1:]
    ch1_a = models.Polynomial2D(2, c0_0=c0_0, c1_0=c1_0, c0_1=c0_1, c1_1=c1_1,
                                name="v23_ab")
    # ab -> v3 transform for first channel
    c0_0, c1_0, c0_1, c1_1 = ab_v23[1][1:]
    ch1_v3 = models.Polynomial2D(2, c0_0=c0_0, c1_0=c1_0, c0_1=c0_1, c1_1=c1_1,
                                 name="ab_v23")
    # v2,v3 -> b transform for first channel
    c0_0, c1_0, c0_1, c1_1 = v23_ab[1][1:]
    ch1_b = models.Polynomial2D(2, c0_0=c0_0, c1_0=c1_0, c0_1=c0_1, c1_1=c1_1,
                                name="v23_ab")
    # ab -> v2 transform for second channel
    c0_0, c1_0, c0_1, c1_1 = ab_v23[2][1:]
    ch2_v2 = models.Polynomial2D(2, c0_0=c0_0, c1_0=c1_0, c0_1=c0_1, c1_1=c1_1,
                                 name="ab_v23")
    # v2,v3 -> a transform for second channel
    c0_0, c1_0, c0_1, c1_1 = v23_ab[2][1:]
    ch2_a = models.Polynomial2D(2, c0_0=c0_0, c1_0=c1_0, c0_1=c0_1, c1_1=c1_1,
                                name="v23_ab")
    # ab -> v3 transform for second channel
    c0_0, c1_0, c0_1, c1_1 = ab_v23[3][1:]
    ch2_v3 = models.Polynomial2D(2, c0_0=c0_0, c1_0=c1_0, c0_1=c0_1, c1_1=c1_1,
                                 name="ab_v23")
    # v2,v3 -> b transform for second channel
    c0_0, c1_0, c0_1, c1_1 = v23_ab[3][1:]
    ch2_b = models.Polynomial2D(2, c0_0=c0_0, c1_0=c1_0, c0_1=c0_1, c1_1=c1_1,
                                name="v23_ab")

    # Transforms from the CDP only went to XAN,YAN, now need a transform to V2,V3
    # Mapping to transform from XAN,YAN in arcmin to V2,V3 in arcsec
    xanyan_to_v2v3 = models.Identity(1) & (models.Scale(-1) | models.Shift(-7.8)) | models.Scale(60.) & models.Scale(60.)

    # Since the matrix transforms need a 4-element input we need a mapping
    # to go from (0,1) to (0,1,0,1)
    mapping = models.Mapping((0, 1, 0, 1), n_inputs=2)

    # Put the mappings all together
    ch1 = mapping | ch1_v2 & ch1_v3 | xanyan_to_v2v3
    ch2 = mapping | ch2_v2 & ch2_v3 | xanyan_to_v2v3

    # And make the inverse mapping
    ch1.inverse =  xanyan_to_v2v3.inverse | mapping | ch1_a & ch1_b
    ch2.inverse =  xanyan_to_v2v3.inverse | mapping | ch2_a & ch2_b

    #pdb.set_trace()
    # save to file
    dist.abv2v3_model = {'channel_band': channels, 'model': [ch1, ch2]}
    dist.meta.input_units = u.pix
    dist.meta.output_units = u.arcsec
    dist.save(name)


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

        # First we need to shift from the 0-indexed pixels input by the pipeline to the
        # 1-indexed pixels used by Adrians transforms
        # (both Adrian and the pipeline include reference pixels)
        # Remember that the coordinates here are in order y,x
        thisshift=models.Shift(1) & models.Shift(1)
        # Now do Adrians transforms
        thisxform=models.Identity(1) & models.Shift(-xs) | models.Polynomial2D(8, name=name, **coeffs)
        # Put the models together
        transforms[sl] = thisshift | thisxform

    return transforms

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

        # First we do Adrian's transforms
        thisxform=models.Shift(-xs, name=shname) & models.Identity(1) | models.Polynomial2D(8, name=pname, **coeffs)
        # Then we need to shift from the 1-indexed pixels used by transforms to the
        # 0-indexed pixels used by the pipeline
        # (both Adrian and the pipeline include reference pixels)
        # Only a single output so only a single shift
        thisshift=models.Shift(-1)
        # Put the models together
        transforms[sl] =  thisxform | thisshift

    return transforms


def build_coeff_names(names):
    names = names[1:]
    names = [name.replace('VAR2_', "c") for name in names]
    return names

def create_beta_models(b0, bdel, channel, nslices):
    beta = {}
    for s in range(nslices):
        sl = channel * 100 + s +1
        beta_s = b0 + s * bdel
        m = models.Const1D(beta_s, name='det2local') #xy2beta and xy2lam
        beta[sl] = m
    return beta


def create_wavelengthrange_file(name, detector, author, useafter, description, outformat):
    model = WavelengthrangeModel()

    # Relaxing the range to match the distortion. The original table
    # comes from the MIRI IDT report and is "as designed".
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

    model = create_reffile_header(model, detector, band="N/A", channel="N/A", author=author,
                                 useafter=useafter, description=description)
    model.meta.filename = os.path.split(name)[-1]
    model.meta.instrument.detector = "N/A"
    model.waverange_selector = channels
    wr = [wavelengthrange[ch] for ch in channels]
    model.wavelengthrange = wr
    model.meta.wavelength_units = u.micron
    model.save(name)

# Function to test the implemented transforms and ASDF files
# Detband is (e.g.) '12A'
def test_cdp6_onereference(detband,refs):
    ch1,ch2=mrspt.channel(detband)# Convert to (e.g.) '1A' and '2A'

    mrspt.testtransform(testchannel=[ch1,ch2],refs=refs)
 
