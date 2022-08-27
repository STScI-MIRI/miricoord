#
"""
Useful python tools for working with the MIRI MRS.
This contains cdp6 specific code.

This version of the tools uses the JWST pipeline implementation
of the distortion solution to do the transformations,
and hooks into offline versions of the CRDS reference
files contained within this github repository.

Convert JWST v2,v3 locations (in arcsec) to MIRI MRS SCA x,y pixel locations.
Note that the pipeline uses a 0-indexed detector pixel (1032x1024) convention while
SIAF uses a 1-indexed detector pixel convention.  The CDP files define
the origin such that (1,1) is the middle of the lower-left detector pixel
(1032x1024),therefore also need to transform between this science frame and detector frame.

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
10-Oct-2018  Written by David Law (dlaw@stsci.edu)
"""

import os as os
import numpy as np
import sys
import pdb as pdb
from astropy.modeling import models
from asdf import AsdfFile
from jwst import datamodels
from jwst.assign_wcs import miri

#############################

# Return the tools version
def version():
    return 'cdp6'

#############################

# Set the relevant CRDS distortion file based on channel (e.g., '1A')
def get_fitsreffile(channel):
    wavefile='jwst_miri_wavelengthrange_0003.asdf'

    # Channel should be of the form (e.g.) '1A', '3C', etc
    # See https://jwst-crds.stsci.edu//display_result/52cef902-ad77-4792-9964-d26a0a8a96a8
    if ((channel is '1A')or(channel is '2A')):
       distfile='jwst_miri_distortion_0021.asdf'
       regfile='jwst_miri_regions_0016.asdf'
       specfile='jwst_miri_specwcs_0015.asdf'
    elif ((channel is '3A')or(channel is '4A')):
       distfile='jwst_miri_distortion_0024.asdf'
       regfile='jwst_miri_regions_0013.asdf'
       specfile='jwst_miri_specwcs_0017.asdf'
    elif ((channel is '1B')or(channel is '2B')):
       distfile='jwst_miri_distortion_0022.asdf'
       regfile='jwst_miri_regions_0015.asdf'
       specfile='jwst_miri_specwcs_0013.asdf'
    elif ((channel is '3B')or(channel is '4B')):
       distfile='jwst_miri_distortion_0026.asdf'
       regfile='jwst_miri_regions_0014.asdf'
       specfile='jwst_miri_specwcs_0018.asdf'
    elif ((channel is '1C')or(channel is '2C')):
       distfile='jwst_miri_distortion_0025.asdf'
       regfile='jwst_miri_regions_0018.asdf'
       specfile='jwst_miri_specwcs_0014.asdf'
    elif ((channel is '3C')or(channel is '4C')):
       distfile='jwst_miri_distortion_0027.asdf'
       regfile='jwst_miri_regions_0017.asdf'
       specfile='jwst_miri_specwcs_0016.asdf'
    else:
       print('Failure!')

    # Try looking for the files in the expected location
    rootdir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    rootdir=os.path.join(rootdir,'data/crds/cdp6/')
    refs={'distortion': os.path.join(rootdir,distfile), 'regions':os.path.join(rootdir,regfile),
          'specwcs':os.path.join(rootdir,specfile), 'wavelengthrange':os.path.join(rootdir,wavefile)}
    if os.path.exists(os.path.join(rootdir,distfile)):
        return refs

    # If that didn't work, look in the system path
    rootdir=sys.prefix
    rootdir=os.path.join(rootdir,'data/crds/cdp6/')
    refs={'distortion': os.path.join(rootdir,distfile), 'regions':os.path.join(rootdir,regfile),
          'specwcs':os.path.join(rootdir,specfile), 'wavelengthrange':os.path.join(rootdir,wavefile)}
    if os.path.exists(os.path.join(rootdir,distfile)):
        return refs    

    # If that didn't work either, just return what we've got
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
# Convert all x,y values to 0-indexed
mrs_ref_data = {
    '1A': {'x': np.array([28.310396, 475.02154, 493.9777, 41.282537, 58.998266])-1,
           'y': np.array([512., 10, 100, 900, 1014])-1,
           's': np.array([11, 1, 1, 21, 21]),
           'alpha': np.array([0, -1.66946, 1.65180, -1.70573, 1.70244]),
           'beta': np.array([0, -1.77210, -1.77210, 1.77210, 1.77210]),
           'lam': np.array([5.34437, 4.86642, 4.95325, 5.65296, 5.74349]),
           'xan': np.array([-8.39424, -8.41746, -8.36306, -8.42653, -8.37026]),
           'yan': np.array([-2.48763, -2.52081, -2.51311, -2.46269, -2.45395]),
           },
    '1B': {'x': np.array([28.648221, 475.07259, 493.98157, 41.559386, 59.738296])-1,
           'y': np.array([512., 10, 100, 900, 1014])-1,
           's': np.array([11, 1, 1, 21, 21]),
           'alpha': np.array([0., -1.70796, 1.60161, -1.70854, 1.78261]),
           'beta': np.array([0., -1.77204, -1.77204, 1.77204, 1.77204]),
           'lam': np.array([6.17572, 5.62345, 5.72380, 6.53231, 6.63698]),
           'xan': np.array([-8.39426, -8.41808, -8.36368, -8.42682, -8.36899]),
           'yan': np.array([-2.48492, -2.51808, -2.51040, -2.46001, -2.45126])
           },
    '1C': {'x': np.array([30.461871, 477.23742, 495.96228, 43.905314, 60.995683])-1,
           'y': np.array([512., 10, 100, 900, 1014])-1,
           's': np.array([11, 1, 1, 21, 21]),
           'alpha': np.array([0., -1.60587, 1.67276, -1.60766, 1.68720]),
           'beta': np.array([0., -1.77202, -1.77202, 1.77202, 1.77202]),
           'lam': np.array([7.04951, 6.42424, 6.53753, 7.45360, 7.57167]),
           'xan': np.array([-8.39357, -8.41570, -8.36165, -8.42457, -8.36996]),
           'yan': np.array([-2.48987, -2.52271, -2.51525, -2.46467, -2.45649])
           },
    '2A': {'x': np.array([992.158, 545.38386, 525.76143, 969.29711, 944.19303])-1,
           'y': np.array([512., 10, 100, 900, 1014])-1,
           's': np.array([9, 1, 1, 17, 17]),
           'alpha': np.array([0., -2.11250, 2.10676, -2.17239, 2.10447]),
           'beta': np.array([0., -2.23775, -2.23775, 2.23775, 2.23775]),
           'lam': np.array([8.20797, 7.52144, 7.64907, 8.68677, 8.83051]),
           'xan': np.array([-8.39393, -8.42259, -8.35355, -8.43583, -8.36499]),
           'yan': np.array([-2.48181, -2.52375, -2.51357, -2.44987, -2.44022])
           },
    '2B': {'x': np.array([988.39977, 541.23447, 521.60207, 964.91753, 940.10325])-1,
           'y': np.array([512., 10, 100, 900, 1014])-1,
           's': np.array([9, 1, 1, 17, 17]),
           'alpha': np.array([0., -2.10593, 2.10015, -2.08817, 2.10422]),
           'beta': np.array([0., -2.23781, -2.23781, 2.23781, 2.23781]),
           'lam': np.array([9.44205, 8.65341, 8.79991, 9.99257, 10.15795]),
           'xan': np.array([-8.39645, -8.42502, -8.35603, -8.43716, -8.36742]),
           'yan': np.array([-2.47773, -2.51972, -2.50938, -2.44554, -2.43626])
           },
    '2C': {'x': np.array([990.89693, 543.82344, 524.34514, 967.98318, 942.77564])-1,
           'y': np.array([512., 10, 100, 900, 1014])-1,
           's': np.array([9, 1, 1, 17, 17]),
           'alpha': np.array([0., -2.07490, 2.11234, -2.14704, 2.14196]),
           'beta': np.array([0., -2.23778, -2.23778, 2.23778, 2.23778]),
           'lam': np.array([10.90225, 9.99162, 10.16079, 11.53780, 11.72887]),
           'xan': np.array([-8.39303, -8.42129, -8.35221, -8.43454, -8.36352]),
           'yan': np.array([-2.47869, -2.52052, -2.51036, -2.44668, -2.43712])
           },
    '3A': {'x': np.array([574.80828, 1001.0602, 984.6387, 547.27479, 518.89992])-1,
           'y': np.array([512., 10, 100, 900, 1014])-1,
           's': np.array([8, 1, 1, 16, 16]),
           'alpha': np.array([0., -2.86745, 3.20982, -3.01230, 2.96643]),
           'beta': np.array([-0.19491, -2.92360, -2.92360, 2.92360, 2.92360]),
           'lam': np.array([12.5335, 13.49968, 13.33846, 11.77148, 11.52350]),
           'xan': np.array([-8.40590, -8.44849, -8.34906, -8.46070, -8.36174]),
           'yan': np.array([-2.48992, -2.54104, -2.52854, -2.44547, -2.43112])
           },
    '3B': {'x': np.array([574.26012, 1001.7349, 985.30166, 548.016, 519.98])-1,
           'y': np.array([512., 10., 100, 900, 1014])-1,
           's': np.array([8, 1, 1, 16, 16]),
           'alpha': np.array([0, -3.17728, 2.92434, -3.29402, 2.60797]),
           'beta': np.array([-0.19491, -2.92360, -2.92360, 2.92360, 2.92360]),
           'lam': np.array([14.53997, 15.66039, 15.47355, 13.65622, 13.36833]),
           'xan': np.array([-8.40044, -8.44785, -8.34786, -8.46088, -8.36211]),
           'yan': np.array([-2.48588, -2.53771, -2.52512, -2.44219, -2.42776])
           },
    '3C': {'x': np.array([573.25446, 1000.21721, 983.92918, 546.00285, 518.2782])-1,
           'y': np.array([512., 10, 100, 900, 1014])-1,
           's': np.array([8, 1, 1, 16, 16]),
           'alpha': np.array([0., -2.94573, 3.09057, -3.07810, 2.73161]),
           'beta': np.array([-0.19491, -2.92360, -2.92360, 2.92360, 2.92360]),
           'lam': np.array([16.79017, 18.08441, 17.86845, 15.76948, 15.43724]),
           'xan': np.array([-8.40205, -8.44574, -8.34664, -8.45859, -8.36196]),
           'yan': np.array([-2.48627, -2.53761, -2.52502, -2.44221, -2.42787]),
           },
    '4A': {'x': np.array([80.987181, 434.34987, 461.90855, 26.322503, 53.674656])-1,
           'y': np.array([512., 10, 100, 900, 1014])-1,
           's': np.array([6, 1, 1, 12, 12]),
           'alpha': np.array([0., -3.74625, 3.72621, -3.94261, 3.62762]),
           'beta': np.array([-0.32802, -3.60821, -3.60821, 3.60821, 3.60821]),
           'lam': np.array([19.34914, 20.93078, 20.6464, 18.07975, 17.67221]),
           'xan': np.array([-8.38446, -8.43506, -8.31378, -8.46256, -8.33609]),
           'yan': np.array([-2.48058, -2.5444, -2.52426, -2.42449, -2.40839])
           },
    '4B': {'x': np.array([77.625553, 431.57061, 458.86869, 23.559111, 50.632416])-1,
           'y': np.array([512., 10, 100, 900, 1014])-1,
           's': np.array([6, 1, 1, 12, 12]),
           'alpha': np.array([0., -3.64817, 3.73313, -3.73558, 3.74096]),
           'beta': np.array([-0.32802, -3.60821, -3.60821, 3.60821, 3.60821]),
           'lam': np.array([22.38267, 24.21212, 23.88327, 20.91426, 20.44279]),
           'xan': np.array([-8.38581, -8.43443, -8.3141, -8.46152, -8.33604]),
           'yan': np.array([-2.48185, -2.54526, -2.52568, -2.42513, -2.40959])
           },
    '4C': {'x': np.array([79.662059, 433.73384, 460.75026, 25.820431, 52.412219])-1,
           'y': np.array([512., 10, 100, 900, 1014])-1,
           's': np.array([6, 1, 1, 12, 12]),
           'alpha': np.array([0., -3.61682, 3.69713, -3.66259, 3.69888]),
           'beta': np.array([-0.32802, -3.60819, -3.60819, 3.60819, 3.60819]),
           'lam': np.array([26.18343, 28.32354, 27.93894, 24.46574, 23.91417]),
           'xan': np.array([-8.38603, -8.43509, -8.31524, -8.45888, -8.33707]),
           'yan': np.array([-2.48315, -2.54647, -2.52661, -2.42721, -2.41060])
           }
}
