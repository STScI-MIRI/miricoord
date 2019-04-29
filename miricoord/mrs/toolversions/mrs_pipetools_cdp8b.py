#
"""
Useful python tools for working with the MIRI MRS.
This contains cdp8b specific code.

This version of the tools uses the JWST pipeline implementation
of the distortion solution to do the transformations,
and hooks into offline versions of the CRDS reference
files contained within this github repository.

Convert JWST v2,v3 locations (in arcsec) to MIRI MRS SCA x,y pixel locations.
Note that the pipeline uses a 0-indexed detector pixel (1032x1024) convention while
SIAF uses a 1-indexed detector pixel convention.  The CDP files define
the origin such that (0,0) is the middle of the lower-left pixel
(1032x1024)- note that this is a CHANGE of convention from earlier CDP!

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
10-Oct-2018  Written by David Law (dlaw@stsci.edu)
"""

import os as os
import numpy as np
import pdb as pdb
from astropy.modeling import models
from asdf import AsdfFile
from jwst import datamodels
from jwst.assign_wcs import miri

#############################

# Return the tools version
def version():
    return 'cdp8b'

#############################

# Set the relevant CRDS distortion file based on channel (e.g., '1A')
def get_fitsreffile(channel):
    rootdir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    rootdir=os.path.join(rootdir,'data/crds/')

    wavefile=rootdir+'jwst_miri_mrs_wavelengthrange_cdp8b.asdf'

    # Channel should be of the form (e.g.) '1A', '3C', etc
    # See https://jwst-crds.stsci.edu//display_result/52cef902-ad77-4792-9964-d26a0a8a96a8
    if ((channel is '1A')or(channel is '2A')):
       distfile=rootdir+'jwst_miri_mrs12A_distortion_cdp8b.asdf'
       regfile=rootdir+'jwst_miri_mrs12A_regions_cdp8b.asdf'
       specfile=rootdir+'jwst_miri_mrs12A_specwcs_cdp8b.asdf'
    elif ((channel is '3A')or(channel is '4A')):
       distfile=rootdir+'jwst_miri_mrs34A_distortion_cdp8b.asdf'
       regfile=rootdir+'jwst_miri_mrs34A_regions_cdp8b.asdf'
       specfile=rootdir+'jwst_miri_mrs34A_specwcs_cdp8b.asdf'
    elif ((channel is '1B')or(channel is '2B')):
       distfile=rootdir+'jwst_miri_mrs12B_distortion_cdp8b.asdf'
       regfile=rootdir+'jwst_miri_mrs12B_regions_cdp8b.asdf'
       specfile=rootdir+'jwst_miri_mrs12B_specwcs_cdp8b.asdf'
    elif ((channel is '3B')or(channel is '4B')):
       distfile=rootdir+'jwst_miri_mrs34B_distortion_cdp8b.asdf'
       regfile=rootdir+'jwst_miri_mrs34B_regions_cdp8b.asdf'
       specfile=rootdir+'jwst_miri_mrs34B_specwcs_cdp8b.asdf'
    elif ((channel is '1C')or(channel is '2C')):
       distfile=rootdir+'jwst_miri_mrs12C_distortion_cdp8b.asdf'
       regfile=rootdir+'jwst_miri_mrs12C_regions_cdp8b.asdf'
       specfile=rootdir+'jwst_miri_mrs12C_specwcs_cdp8b.asdf'
    elif ((channel is '3C')or(channel is '4C')):
       distfile=rootdir+'jwst_miri_mrs34C_distortion_cdp8b.asdf'
       regfile=rootdir+'jwst_miri_mrs34C_regions_cdp8b.asdf'
       specfile=rootdir+'jwst_miri_mrs34C_specwcs_cdp8b.asdf'
    else:
       print('Failure!')

    refs={'distortion': distfile, 'regions':regfile, 'specwcs':specfile, 'wavelengthrange':wavefile}
    return refs

#############################

# Convenience function to turn '1A' type name into '12' and 'SHORT' type names
def bandchan(channel):
    # Channel should be of the form (e.g.) '1A', '3C', etc
    if ((channel is '1A')or(channel is '2A')):
       newband='SHORT'
       newchannel='12'
    elif ((channel is '3A')or(channel is '4A')):
       newband='SHORT'
       newchannel='34'
    elif ((channel is '1B')or(channel is '2B')):
       newband='MEDIUM'
       newchannel='12'
    elif ((channel is '3B')or(channel is '4B')):
       newband='MEDIUM'
       newchannel='34'
    elif ((channel is '1C')or(channel is '2C')):
       newband='LONG'
       newchannel='12'
    elif ((channel is '3C')or(channel is '4C')):
       newband='LONG'
       newchannel='34'
    else:
       newband='FAIL'
       newchannel='FAIL'

    return newband,newchannel

#############################

# Convenience function to turn '12A' type name into '1A' and '2A' type names
def channel(detband):
    if (detband == '12A'):
       ch1='1A'
       ch2='2A'
    elif (detband == '12B'):
       ch1='1B'
       ch2='2B'
    elif (detband == '12C'):
       ch1='1C'
       ch2='2C'
    elif (detband == '34A'):
       ch1='3A'
       ch2='4A'
    elif (detband == '34B'):
       ch1='3B'
       ch2='4B'
    elif (detband == '34C'):
       ch1='3C'
       ch2='4C'
    else:
       ch1='FAIL'
       ch2='FAIL'

    return ch1,ch2

#############################

# Convenience function to return the rough middle wavelength of a given channel
# Note that this ISNT exact, just some valid value
def midwave(channel):
    if (channel is '1A'):
       thewave=5.32
    elif (channel is '1B'):
       thewave=6.145
    elif (channel is '1C'):
       thewave=7.09
    elif (channel is '2A'):
       thewave=8.135
    elif (channel is '2B'):
       thewave=9.395
    elif (channel is '2C'):
       thewave=10.85
    elif (channel is '3A'):
       thewave=12.505
    elif (channel is '3B'):
       thewave=14.5
    elif (channel is '3C'):
       thewave=16.745
    elif (channel is '4A'):
       thewave=19.29
    elif (channel is '4B'):
       thewave=22.47
    elif (channel is '4C'):
       thewave=26.2

    return thewave

#############################

# Convenience function to return model distortion object
# for the x,y to alpha,beta,lam transform
def xytoablmodel(channel,**kwargs):
    # Construct the reference data model in general JWST imager type
    input_model = datamodels.ImageModel()
    # Convert input of type '1A' into the band and channel that pipeline needs
    theband,thechan=bandchan(channel)
    # Set the filter in the data model meta header
    input_model.meta.instrument.band = theband
    input_model.meta.instrument.channel = thechan
 
    # If passed input refs keyword, unpack and use it
    if ('refs' in kwargs):
      therefs=kwargs['refs']
    # Otherwise use default reference files
    else:
      therefs=get_fitsreffile(channel)

    distortion = miri.detector_to_abl(input_model, therefs)
    # Return the distortion object that can then be queried
    return distortion

#############################

# Convenience function to return model distortion object
# for the alpha,beta to v2,v3 transform

def abtov2v3model(channel,**kwargs):
    # Construct the reference data model in general JWST imager type
    input_model = datamodels.ImageModel()
    # Convert input of type '1A' into the band and channel that pipeline needs
    theband,thechan=bandchan(channel)
    # Set the filter in the data model meta header
    input_model.meta.instrument.band = theband
    input_model.meta.instrument.channel = thechan
 
    # If passed input refs keyword, unpack and use it
    if ('refs' in kwargs):
      therefs=kwargs['refs']
    # Otherwise use default reference files
    else:
      therefs=get_fitsreffile(channel)

    # The pipeline transform actually uses the triple
    # (alpha,beta,lambda) -> (v2,v3,lambda)
    basedistortion = miri.abl_to_v2v3l(input_model, therefs)
    distortion = basedistortion

    # Therefore we need to hack a reasonable wavelength onto our input, run transform,
    # then hack it back off again

    thewave=midwave(channel)
    # Duplicate the beta value at first, then replace with wavelength value
    map=models.Mapping((0,1,1)) | models.Identity(1) & models.Identity(1) & models.Const1D(thewave)
    map.inverse=models.Mapping((0,1),n_inputs=3)

    allmap= map | distortion | map.inverse
    allmap.inverse= map | distortion.inverse | map.inverse

    # Return the distortion object that can then be queried
    return allmap

#############################

# MRS test reference data
# These values are all CDP-8b placeholders mocked up by DRL!
mrs_ref_data = {
    '1A': {'x': np.array([123, 468]),
           'y': np.array([245, 1000]),
           's': np.array([9, 12]),
           'alpha': np.array([0.10731135, 1.12529977]),
           'beta': np.array([-0.35442029,  0.17721014]),
           'lam': np.array([5.11499695, 5.74016832]),
           'v2': np.array([-503.49916454, -502.56838429]),
           'v3': np.array([-318.40628359, -319.08393052]),
           },
    '1B': {'x': np.array([51, 244]),
           'y': np.array([1016,  476]),
           's': np.array([21, 17]),
           'alpha': np.array([0.38199789, 0.63723143]),
           'beta': np.array([1.77204177, 1.06322506]),
           'lam': np.array([6.62421656, 6.12990972]),
           'v2': np.array([-503.5315322 , -503.17667842]),
           'v3': np.array([-320.71381511, -320.05014591]),
           },
    '1C': {'x': np.array([127, 394]),
           'y': np.array([747, 111]),
           's': np.array([9, 3]),
           'alpha': np.array([0.37487296, -0.87620923]),
           'beta': np.array([-0.35440438, -1.4176175]),
           'lam': np.array([7.26440645, 6.52961571]),
           'v2': np.array([-503.19062471, -504.27276937]),
           'v3': np.array([-318.31059564, -317.08599443]),
           },
    '2A': {'x': np.array([574, 913]),
           'y': np.array([578, 163]),
           's': np.array([10, 16]),
           'alpha': np.array([0.02652122, -1.44523112]),
           'beta': np.array([0.27971819, 1.9580273]),
           'lam': np.array([8.22398597, 7.66495464]),
           'v2': np.array([-503.65420691, -505.38172957]),
           'v3': np.array([-319.37148692, -320.82933868]),
           },
    '2B': {'x': np.array([634, 955]),
           'y': np.array([749,  12]),
           's': np.array([11, 17]),
           'alpha': np.array([-1.31986085, -1.66029886]),
           'beta': np.array([0.5594521 , 2.23780842]),
           'lam': np.array([9.85535403, 8.65341739]),
           'v2': np.array([-505.18703764, -505.80250684]),
           'v3': np.array([-319.7057936 , -321.32425399]),
           },
    '2C': {'x': np.array([530, 884]),
           'y': np.array([965, 346]),
           's': np.array([1, 7]),
           'alpha': np.array([1.17219936, -0.13199122]),
           'beta': np.array([-2.23777695, -0.55944424]),
           'lam': np.array([11.68798183, 10.65732315]),
           'v2': np.array([-502.0634552 , -503.62291245]),
           'v3': np.array([-317.2417194 , -318.70820411]),
           },
    '3A': {'x': np.array([573, 913]),
           'y': np.array([851, 323]),
           's': np.array([8, 10]),
           'alpha': np.array([-1.0181757 ,  0.65295329]),
           'beta': np.array([-0.19490689,  0.58472067]),
           'lam': np.array([11.84245153, 12.96396074]),
           'v2': np.array([-505.35888594, -503.7824966]),
           'v3': np.array([-318.46913272, -319.4685406]),
           },
    '3B': {'x': np.array([606, 861]),
           'y': np.array([926, 366]),
           's': np.array([15, 11]),
           'alpha': np.array([-1.5124193 , -0.79361415]),
           'beta': np.array([2.53378956, 0.97453445]),
           'lam': np.array([13.60306079, 14.94878428]),
           'v2': np.array([-505.82191056, -504.9372123]),
           'v3': np.array([-321.34413558, -319.90108102]),
           },
    '3C': {'x': np.array([663, 852]),
           'y': np.array([822,  86]),
           's': np.array([14, 11]),
           'alpha': np.array([0.83845626, -1.00005387]),
           'beta': np.array([2.14397578, 0.97453445]),
           'lam': np.array([16.01468948, 17.97678143]),
           'v2': np.array([-503.52817761, -505.23700039]),
           'v3': np.array([-321.27004219, -319.84577337]),
           },
    '4A': {'x': np.array([448, 409]),
           'y': np.array([873,  49]),
           's': np.array([1, 7]),
           'alpha': np.array([-0.45466621, -1.07614592]),
           'beta': np.array([-3.60820915,  0.32801901]),
           'lam': np.array([18.05366191, 20.88016154]),
           'v2': np.array([-502.89806847, -504.25439193]),
           'v3': np.array([-315.86847223, -319.65622713]),
           },
    '4B': {'x': np.array([380, 260]),
           'y': np.array([926, 325]),
           's': np.array([2, 9]),
           'alpha': np.array([1.64217386, -1.70062938]),
           'beta': np.array([-2.95217555,  1.64009753]),
           'lam': np.array([20.69573674, 23.17990504]),
           'v2': np.array([-501.01720495, -505.23791555]),
           'v3': np.array([-316.76598039, -320.79546159]),
           },
    '4C': {'x': np.array([309, 114]),
           'y': np.array([941, 196]),
           's': np.array([3, 11]),
           'alpha': np.array([1.65440228, -0.87408042]),
           'beta': np.array([-2.29611932,  2.95215341]),
           'lam': np.array([24.17180582, 27.63402178]),
           'v2': np.array([-501.1647203 , -504.64107203]),
           'v3': np.array([-317.34628   , -322.10088837]),
           }
    
}
