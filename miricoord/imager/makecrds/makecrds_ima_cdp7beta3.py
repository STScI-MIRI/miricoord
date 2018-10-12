#
"""
Code to create CRDS reference files for the distortion of the
MIRI Imager using IDT reference files delivered with CDP-7beta:

MIRI_FM_MIRIMAGE_DISTORTION_7B.03.00.fits

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

import miricoord.miricoord.imager.mirim_tools as mt
import miricoord.miricoord.imager.mirim_pipetools as mpt

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

def make_filter_offset(distfile, outname):
    """
    Create an asdf reference file with the filter offsets for the MIRI imager.

    Note: The IDT supplied distortion file lists sky to pixel as the
    forward transform. Since "forward" in the JWST pipeline is from
    pixel to sky, the offsets are taken with the opposite sign.

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
    -------
    >>> make_filter_offset('MIRI_FM_MIRIMAGE_DISTORTION_7B.03.00.fits',
                                        'jwst_miri_filter_offset_0001.asdf')
    """

    with fits.open(distfile) as f:
        data = f[9].data

    d = []
    for i in data:
        d.append({'name':i[0],'column_offset': -i[1], 'row_offset': -i[2]} )

    model = FilteroffsetModel()
    # Add general metadata
    model = create_reffile_header(model)
    # Add file-specific metadata
    model.meta.title = "MIRI imager filter offset - CDP7B"
    model.meta.description = "CDP7B delivery - new file structure"

    for item in data:
        model.filters = d
    model.save(outname)


def make_distortion(distfile, outname):
    """
    Create an asdf reference file with all distortion components for the MIRI imager.
    The filter offsets are stored in a sepaate file.

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
    >>> make_distortion("MIRI_FM_MIRIMAGE_DISTORTION_7B.03.00.fits", 'test.asdf')
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

    # Transform from XAN,YAN in arcmin to V2,V3 in arcsec
    xanyan_to_v2v3 = models.Identity(1) & (models.Scale(-1) | models.Shift(-7.8)) | models.Scale(60.) & models.Scale(60.)

    distortion_transform = det_to_sci | m_transform | mapping | poly | poly2t_mapping | t_transform | ident | models.Mapping([1,0]) | xanyan_to_v2v3

    # Inverse transform created automatically, but if we needed to do it by hand
    # it would look like this
    #distortion_transform.inverse=xanyan_to_v2v3.inverse | models.Mapping([1,0]).inverse | ident.inverse | t_transform.inverse | poly2t_mapping.inverse | poly.inverse | mapping.inverse | m_transform.inverse | det_to_sci.inverse

    # Define imager bounding boxes
    shape=1032,1024 # columns,rows
    # The python bounding box must have form ((ylow,yhigh),(xlow,xhigh))
    # NB- at the moment this doesn't do anything here and must be implemented in
    # pipeline code miri.py instead
    distortion_transform.bounding_box = ((-0.5, shape[1] - 0.5), (3.5, shape[0] - 4.5))

    fdist.close()

    dist = DistortionModel()
    # Add general metadata
    dist = create_reffile_header(dist)
    # Add file-specific metadata
    dist.model = distortion_transform
    dist.meta.input_units = u.pix
    dist.meta.output_units = u.arcsec
    dist.meta.title = "MIRI imager distortion - CDP7B"
    dist.meta.description = "CDP7B delivery - new reference file structure and new bounding box"

    dist.save(outname)

def create_reffile_header(model):

    model.meta.instrument.name = "MIRI"
    model.meta.instrument.detector = "MIRIMAGE"
    model.meta.instrument.band = "N/A"
    model.meta.instrument.channel = "N/A"
    model.meta.exposure.type = "MIR_IMAGE"
    model.meta.exposure.p_exptype = "MIR_IMAGE|MIR_LRS-FIXEDSLIT|MIR_LRS-SLITLESS|"
    model.meta.author = "Alistair Glasse, David R. Law"
    model.meta.pedigree = "GROUND"
    model.meta.useafter = "2000-01-01T00:00:00"

    entry = HistoryEntry({'description': "New version created from CDP-7b with updated file structure", 'time': datetime.datetime.utcnow()})
    software = Software({'name': 'coordinates', 'author': 'D.Law', 
                         'homepage': 'https://github.com/STScI-MIRI/miricoord', 'version': "master"})
    entry['software'] = software
    model.history = [entry]

    return model

def make_references(filename, ref):
    """
    Create the two reference files. Writes the files in the current directory.

    Parameters
    ----------
    filename : str
        The name of the IDT file with the distortion.
        In CDP6 the file is called "MIRI_FM_MIRIMAGE_DISTORTION_7B.03.00.fits"
    ref : dict
        A dictionary {reftype: refname}, e.g.
        {'DISTORTION': 'jwst_miri_distortion_0001.asdf',
         'FILTEROFFSET': 'jwst_miri_filteroffset_0001.asdf'
         }

    Examples
    --------
    >>> make_references('MIRI_FM_MIRIMAGE_DISTORTION_7B.03.00.fits',
        {'DISTORTION': 'jwst_miri_distortion_0001.asdf',
        'FILTEROFFSET': 'jwst_miri_filter_offset_0001.asdf'})

    """
    try:
        make_distortion(filename, ref['DISTORTION'])
    except:
        print("Distortion file was not created.")
        raise
    try:
        make_filter_offset(filename, ref['FILTEROFFSET'])
    except:
        print("Filter offset file was not created.")
        raise

def test_transform(refs):
    """
    Parameters
    ----------
    refs: refs = {"distortion": distfile, "filteroffset": offfile}
        distfile="jwst_miri_imager_distortion_cdp7b.asdf"
        offfile="jwst_miri_filteroffset_cdp7b.asdf"

    xy, v2, v3 values are from technical report with CDP-7B delivery
    First entry is the Imager reference point
    """
    xan = np.array([-7.5560561,-8, -7.5, -7, -6.5, -8, -7.5, -7, -6.5, -8, -7.5, -7, -6.5], dtype=np.float)
    yan = np.array([-1.5655228,-2, -2, -2, -2, -1.5, -1.5, -1.5, -1.5, -1, -1, -1, -1], dtype=np.float)
    v2,v3=mt.xanyan_to_v2v3(xan,yan)
    # xy reference values in 0-indexed science frame for F770W
    xy770w = np.array([[688.5,511.5],[946.30, 728.95], [677.07, 749.13], [408.79, 769.19], [138.52, 789.59],
                   [924.80, 457.09],[655.68, 478.39], [387.55, 499.49], [117.42, 520.46],
                   [904.81, 185.52], [635.59, 207.87], [367.03, 229.95], [96.08, 251.45]],
                  dtype=np.float)
    # Convert to 0-indexed detector frame for F770W
    xy770w = xy770w+[4,0]

    # xy reference values in 0-indexed science frame for F1800W
    xy1800w = np.array([[688.10606,512.22995],[945.91, 729.68], [676.68, 749.86], [408.40, 769.92], [138.13, 790.32],
                        [924.41, 457.82],[655.29, 479.12], [387.16, 500.22], [117.03, 521.19],
                        [904.42, 186.25], [635.20, 208.60], [366.64, 230.68], [95.69, 252.18]],
                  dtype=np.float)
    # Convert to 0-indexed detector frame for F1800W
    xy1800w = xy1800w+[4,0]

    input_model=datamodels.ImageModel()
    input_model.meta.instrument.filter='F770W'
    transform770w = miri.imaging_distortion(input_model, refs)
    transform770w = transform770w
    input_model.meta.instrument.filter='F1800W'
    transform1800w = miri.imaging_distortion(input_model, refs)
    transform1800w = transform1800w

    # Test the inverse transform for F770w
    x, y = transform770w.inverse(v2, v3)
    assert_allclose(x, xy770w[:,0], atol=.05)
    assert_allclose(y, xy770w[:,1], atol=.05)
    # Test the forward transform for F770w
    s1, s2 = transform770w (xy770w[:,0], xy770w[:,1])
    assert_allclose(s1, v2, atol=0.05)
    assert_allclose(s2, v3, atol=.05)

    # Test the inverse transform for F1800w
    x, y = transform1800w.inverse(v2, v3)
    assert_allclose(x, xy1800w[:,0], atol=.05)
    assert_allclose(y, xy1800w[:,1], atol=.05)
    # Test the forward transform for F1800w
    s1, s2 = transform1800w (xy1800w[:,0], xy1800w[:,1])
    assert_allclose(s1, v2, atol=0.05)
    assert_allclose(s2, v3, atol=.05)
