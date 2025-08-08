#
"""
Code to create CRDS reference files for the distortion of the
MIRI Imager using IDT reference files for flight solution FLT-9.
FLT-9 updates the boresight of the entire instrument based on new FGS measurements.

MIRI_FM_MIRIMAGE_DISTORTION_SS.09.00.fits

MIRI Imager uses 2 reference files of type:

DISTORTION
FILTEROFFSET

In this version the CDP file goes from 0-indexed science pixels
(0,0) in the middle of the lower left science pixel to XAN,YAN

We will need to add additional transforms so that the mapping goes
from 0-indexed detector pixels (0,0) in the middle of the lower left
reference pixel to V2,V3

make_references() creates all reference files.

Author: David R. Law (dlaw@stsci.edu), Nadia Dencheva

REVISION HISTORY:
2015         Written by Nadia Dencheva
2016         Adapted for new formats by David Law (dlaw@stsci.edu)
11-Oct-2018  Adapted to new miricoord structure (D. Law)
02-Dec-2018  Adapt to CDP-7
18-May-2020  Change filter offset reference file format, added MIR_TACONFIRM (D. Law)
26-May-2022  Add FLT-1 version (D. Law)
17-Oct-2022  Add FLT-2 version (D. Law)
23-Jul-2025  Add FLT-5 version (D. Law)
"""

from __future__ import absolute_import, division, unicode_literals, print_function

import numpy as np
import pdb as pdb
from numpy.testing import assert_allclose
from astropy.io import fits
from astropy.modeling import models
from astropy import units as u
from asdf import AsdfFile
from jwst import datamodels
from jwst.assign_wcs import miri

import miricoord.imager.mirim_tools as mt

from jwst.datamodels import DistortionModel, FilteroffsetModel
from asdf.tags.core import Software, HistoryEntry
import datetime

# Print full arrays for debugging
np.set_printoptions(threshold=np.inf)

# Swap i,j order of the coefficient in this version
def polynomial_from_coeffs_matrix_swap(coefficients, name=None):
    n_dim = coefficients.ndim

    if n_dim == 1:
        model = models.Polynomial1D(coefficients.size - 1, name=name)
        model.parameters = coefficients
    elif n_dim == 2:
        shape = coefficients.shape
        degree = shape[0] - 1
        if shape[0] != shape[1]:
            raise TypeError("Coefficients must be an (n+1, n+1) matrix")

        coeffs = {}
        for i in range(shape[0]):
            for j in range(shape[0]):
                if i + j < degree + 1:
                    cname = 'c' + str(i) + '_' +str(j)
                    coeffs[cname] = coefficients[j, i]#DRL: I had to swap i,j order here
        model = models.Polynomial2D(degree, name=name, **coeffs)

    return model

# Keep i,j order of the coefficient in this version
def polynomial_from_coeffs_matrix(coefficients, name=None):
    n_dim = coefficients.ndim

    if n_dim == 1:
        model = models.Polynomial1D(coefficients.size - 1, name=name)
        model.parameters = coefficients
    elif n_dim == 2:
        shape = coefficients.shape
        degree = shape[0] - 1
        if shape[0] != shape[1]:
            raise TypeError("Coefficients must be an (n+1, n+1) matrix")

        coeffs = {}
        for i in range(shape[0]):
            for j in range(shape[0]):
                if i + j < degree + 1:
                    cname = 'c' + str(i) + '_' +str(j)
                    coeffs[cname] = coefficients[i, j]
        model = models.Polynomial2D(degree, name=name, **coeffs)

    return model

def make_filter_offset(distfile, outname, subarray):
    """
    Create an asdf reference file with the filter offsets for the MIRI imager and coronagraphs.

    Note: The IDT supplied distortion file lists sky to pixel as the
    forward transform. Since "forward" in the JWST pipeline is from
    pixel to sky, the offsets are taken with the opposite sign.

    Parameters
    ----------
    distfile : str
        MIRI imager DISTORTION file provided by the IDT team.
    outname : str
        Name of reference file to be wriiten to disk.
    subarray: str
        Which subarray to generate the file for (coron files are different than imager)
        Use pipeline-style names (e.g., FULL) rather than SIAF-style names (MIRIM_FULL)

    Returns
    -------
    fasdf : AsdfFile
        AsdfFile object

    Examples
    -------
    >>> make_filter_offset('MIRI_FM_MIRIMAGE_DISTORTION_SS.09.00.fits',
                                        'jwst_miri_filter_offset_xxxx.asdf')
    """

    # Coronagraphic subarrays require their own boresight offsets
    if (subarray == 'MASK1065'):
        extname = 'BoresightCORON1065'
        subvalues = "MASK1065"
    elif (subarray == 'MASK1140'):
        extname = 'BoresightCORON1140'
        subvalues = "MASK1140"
    elif (subarray == 'MASK1550'):
        extname = 'BoresightCORON1550'
        subvalues = "MASK1550"
    elif (subarray == 'MASKLYOT'):
        extname = 'BoresightCORONLYOT'
        subvalues = "MASKLYOT"
    else:
        extname = 'Boresight offsets'
        subvalues = "FULL"
        psubvalues = "FULL|BRIGHTSKY|SUB256|SUB128|SUB64|SLITLESSPRISM|"
        
    with fits.open(distfile) as f:
        data = f[extname].data

    d = []
    for i in data:
        d.append({'filter':i[0],'pupil': 'N/A','column_offset': -i[1], 'row_offset': -i[2]} )

    model = FilteroffsetModel()
    # Add general metadata
    model = create_reffile_header(model)
    # Add file-specific metadata
    model.meta.title = "MIRI imager filter offset - FLT9"
    model.meta.description = "FLT9 delivery"
    model.meta.input_units = "pixels"
    model.meta.subarray.name = subvalues
    if (subvalues == "FULL"):
        model.meta.subarray.p_subarray = psubvalues

    for item in data:
        model.filters = d
    model.save(outname)


def make_distortion(distfile, outname):
    """
    Create an asdf reference file with all distortion components for the MIRI imager.
    The filter offsets are stored in a separate file.

    Note: The IDT supplied distortion file lists sky to pixel as the
    forward transform. Since "forward" in the JWST pipeline is from
    pixel to sky, the meaning of forward and inverse matrices and the order
    in which they are applied is switched.

    The order of operation from pixel to sky is:
    - Apply MI matrix
    - Apply Ai and BI matrices
    - Apply the TI matrix (this gives V2/V3 coordinates)

    Parameters
    ----------
    distfile : str
        MIRI imager DISTORTION file provided by the IDT team.
    outname : str
        Name of reference file to be wriiten to disk.

    Returns
    -------
    fasdf : AsdfFile
        AsdfFile object

    Examples
    --------
    >>> make_distortion("MIRI_FM_MIRIMAGE_DISTORTION_SS.09.00.fits", 'test.asdf')
    """
    # Transform from 0-indexed Detector frame (used by pipeline) to 0-indexed Science frame (used by CDP)
    det_to_sci = models.Shift(-4) & models.Identity(1)

    fdist = fits.open(distfile)
    mi_matrix = fdist['MI matrix'].data
    mi_col = models.Polynomial1D(1, c0=mi_matrix[0, 2], c1=mi_matrix[0,0], name="M_column_correction")
    mi_row = models.Polynomial1D(1, c0=mi_matrix[1, 2], c1=mi_matrix[1,1], name="M_row_correction")
    m_matrix = fdist['M matrix'].data
    m_col = models.Polynomial1D(1, c0=m_matrix[0, 2], c1=m_matrix[0,0])
    m_row = models.Polynomial1D(1, c0=m_matrix[1, 2], c1=m_matrix[1,1])
    mi_col.inverse = m_col.copy()
    mi_row.inverse = m_row.copy()
    m_transform = mi_col & mi_row
    m_transform.inverse = m_col & m_row

    # This turns the output of the MI transform into the shape needed for the AI/BI transforms
    mapping = models.Mapping([0, 1, 0, 1])
    mapping.inverse = models.Identity(2)

    ai_matrix = fdist['AI matrix'].data
    a_matrix = fdist['A matrix'].data
    col_poly = polynomial_from_coeffs_matrix_swap(ai_matrix, name="A_correction")
    col_poly.inverse = polynomial_from_coeffs_matrix(a_matrix)
    bi_matrix = fdist['BI matrix'].data
    b_matrix = fdist['B matrix'].data
    row_poly = polynomial_from_coeffs_matrix_swap(bi_matrix, name="B_correction")
    row_poly.inverse = polynomial_from_coeffs_matrix(b_matrix)
    poly = row_poly & col_poly # DRL: I had to switch the order here
    poly.inverse = col_poly.inverse & row_poly.inverse # but not switch here

    ti_matrix = fdist['TI matrix'].data
    t_matrix = fdist['T matrix'].data
    ti_col = models.Polynomial2D(1, name='TI_column_correction')
    ti_col.parameters = ti_matrix[0][::-1]
    ti_row = models.Polynomial2D(1, name='TI_row_correction')
    ti_row.parameters = ti_matrix[1][::-1]

    t_col = models.Polynomial2D(1, name='T_column_correction')
    t_col.parameters = t_matrix[0][::-1]
    t_row = models.Polynomial2D(1, name='T_row_correction')
    t_row.parameters = t_matrix[1][::-1]
    t_transform = ti_row & ti_col
    t_transform.inverse = t_row & t_col


    # ident is created here so that mapping can be assigned as inverse
    ident = models.Identity(2)
    ident.inverse = models.Mapping([0,1,0,1])

    # This turns the output of the AI/BI transforms into the shape needed for the TI transform
    poly2t_mapping = models.Mapping([0, 1, 0, 1])
    poly2t_mapping.inverse = models.Mapping([0, 1, 0, 1])

    map_t2_xanyan = models.Mapping((1, 0))
    map_t2_xanyan.inverse = models.Mapping((0, 1, 0, 1))

    distortion_transform = det_to_sci | m_transform | mapping | poly | poly2t_mapping | t_transform | ident | models.Mapping([1,0])

    # Inverse transform created automatically, but if we needed to do it by hand
    # it would look like this
    #distortion_transform.inverse=models.Mapping([1,0]).inverse | ident.inverse | t_transform.inverse | poly2t_mapping.inverse | poly.inverse | mapping.inverse | m_transform.inverse | det_to_sci.inverse

    fdist.close()

    dist = DistortionModel()
    # Add general metadata
    dist = create_reffile_header(dist)
    # Add file-specific metadata
    dist.model = distortion_transform
    dist.meta.input_units = u.pix
    dist.meta.output_units = u.arcsec
    dist.meta.title = "MIRI imager distortion - FLT9"
    dist.meta.description = "FLT9 delivery"

    dist.save(outname)

def create_reffile_header(model):

    model.meta.instrument.name = "MIRI"
    model.meta.instrument.detector = "MIRIMAGE"
    model.meta.instrument.band = "N/A"
    model.meta.instrument.channel = "N/A"
    model.meta.exposure.type = "MIR_IMAGE"
    model.meta.author = "Alistair Glasse, Mattia Libralato, David R. Law"
    model.meta.pedigree = "INFLIGHT 2022-05-01 2022-05-01"
    model.meta.useafter = "2022-04-01T00:00:00"
    model.meta.reftype = "filteroffset"

    if (model.meta.model_type == 'DistortionModel'):
        model.meta.exposure.p_exptype = "MIR_IMAGE|MIR_TACQ|MIR_LYOT|MIR_4QPM|MIR_CORONCAL|MIR_LRS-FIXEDSLIT|MIR_LRS-SLITLESS|MIR_TACONFIRM|"
        
    if (model.meta.model_type == 'FilteroffsetModel'):
        model.meta.exposure.p_exptype = "MIR_IMAGE|MIR_TACQ|MIR_LYOT|MIR_4QPM|MIR_CORONCAL|MIR_TACONFIRM|"
    
    entry = HistoryEntry({'description': "First in-flight distortion created from MIR-007 (APT 1024) and adjusted to boresight of MIR-013 (APT 1029, Obs 1-3)", 'time': datetime.datetime.utcnow()})
    entry = HistoryEntry({'description': "FLT-2 adds coronagraph-specific boresight reference files.", 'time': datetime.datetime.utcnow()})
    entry = HistoryEntry({'description': "FLT-9 updates overall MIRI boresight based on the latest FGS solutions.", 'time': datetime.datetime.utcnow()})
    
    software = Software({'name': 'miricoord', 'author': 'D.Law', 
                         'homepage': 'https://github.com/STScI-MIRI/miricoord', 'version': "master"})
    entry['software'] = software
    model.history = [entry]

    return model

def make_references(filename, ref):
    """
    Create the two primary reference files. Writes the files in the current directory.

    Parameters
    ----------
    filename : str
        The name of the IDT file with the distortion.
        In FLT2 the file is called "MIRI_FM_MIRIMAGE_DISTORTION_SS.09.00.fits"
    ref : dict
        A dictionary {reftype: refname}, e.g.
        {'DISTORTION': 'jwst_miri_distortion_0001.asdf',
         'FILTEROFFSET': 'jwst_miri_filteroffset_0001.asdf'
         }

    Examples
    --------
    >>> make_references('MIRI_FM_MIRIMAGE_DISTORTION_SS.09.00.fits',
        {'DISTORTION': 'jwst_miri_distortion_0001.asdf',
        'FILTEROFFSET': 'jwst_miri_filter_offset_0001.asdf'})

    """
    try:
        make_distortion(filename, ref['DISTORTION'])
    except:
        print("Distortion file was not created.")
        raise
    
    try:
        make_filter_offset(filename, ref['FILTEROFFSET'], subarray='FULL')
    except:
        print("Filter offset file was not created.")
        raise

def test_transform(refs):
    """
    Parameters
    ----------
    refs: refs = {"distortion": distfile, "filteroffset": offfile}
        distfile="jwst_miri_imager_distortion_flt2.asdf"
        offfile="jwst_miri_filteroffset_flt2.asdf"

    xy, v2, v3 values are from hand computation from FLT-2 delivery
    First entry is the Imager reference point
    """
    # F770W tests, xy are 0-indexed detector pixels
    x_770=np.array([692.5 , 511.5 , 948.18, 676.75, 404.81, 132.65, 923.52, 653.11,
       382.37, 111.34, 899.64, 629.88, 360.  ,  89.77])
    y_770=np.array([511.5 , 511.5 , 724.94, 745.67, 767.77, 791.34, 455.4 , 476.54,
       498.57, 521.66, 184.81, 206.95, 229.12, 251.55])
    v2_770=np.array([-453.34829012, -433.41691881, -479.35702157, -449.34943073,
        -419.33092578, -389.32606367, -479.3617599 , -449.34596406,
        -419.32677586, -389.34044485, -479.37280538, -449.35190468,
        -419.34256534, -389.38958253])
    v3_770=np.array([-373.6679493 , -375.27358152, -347.87742698, -347.86857266,
        -347.86671455, -347.85671539, -377.89957418, -377.89105908,
        -377.88356193, -377.85860274, -407.91748198, -407.90462395,
        -407.88815064, -407.84818552])
    filt_770=['F770W' for i in range(0,len(v2_770))]

    # F1800W tests
    x_1800=np.array([692.5 , 511.5 , 948.18, 676.75, 404.81, 132.65, 923.52, 653.11,
       382.37, 111.34, 899.64, 629.88, 360.  ,  89.77])
    y_1800=np.array([511.5 , 511.5 , 724.94, 745.67, 767.77, 791.34, 455.4 , 476.54,
       498.57, 521.66, 184.81, 206.95, 229.12, 251.55])
    v2_1800=np.array([-453.39832496, -433.46669506, -479.4072696 , -449.3992657 ,
        -419.38029999, -389.37506905, -479.41204734, -449.39596267,
        -419.37637275, -389.38966299, -479.42281681, -449.40188008,
        -419.39232507, -389.43908253])
    v3_1800=np.array([-373.74551164, -375.35105926, -347.95484426, -347.94583891,
        -347.94369818, -347.93337295, -377.97704079, -377.96859868,
        -377.96093636, -377.93565983, -407.99380009, -407.98134963,
        -407.96493408, -407.92476492])
    filt_1800=['F1800W' for i in range(0,len(v2_1800))]
    
    input_model=datamodels.ImageModel()
    input_model.meta.instrument.filter='F770W'
    transform770w = miri.imaging_distortion(input_model, refs)
    transform770w = transform770w
    input_model.meta.instrument.filter='F1800W'
    transform1800w = miri.imaging_distortion(input_model, refs)
    transform1800w = transform1800w

    # Test the inverse transform for F770w
    x, y = transform770w.inverse(v2_770, v3_770)
    assert_allclose(x, x_770, atol=.05)
    assert_allclose(y, y_770, atol=.05)
    # Test the forward transform for F770w
    s1, s2 = transform770w (x_770, y_770)
    assert_allclose(s1, v2_770, atol=0.05)
    assert_allclose(s2, v3_770, atol=.05)

    # Test the inverse transform for F1800w
    x, y = transform1800w.inverse(v2_1800, v3_1800)
    assert_allclose(x, x_1800, atol=.05)
    assert_allclose(y, y_1800, atol=.05)
    # Test the forward transform for F1800w
    s1, s2 = transform1800w (x_1800, y_1800)
    assert_allclose(s1, v2_1800, atol=0.05)
    assert_allclose(s2, v3_1800, atol=.05)
