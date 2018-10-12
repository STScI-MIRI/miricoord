# Useful python tools for working with the MIRI MRS

import os as os
import numpy as np
import pdb as pdb
from astropy.modeling import models
from asdf import AsdfFile
from jwst import datamodels
from jwst.assign_wcs import miri

########################################################################################
# General convenience tools
########################################################################################

# Convenience function to set default most-recent CRDS reference files from
# jwst-drl/trunk/CRDS/
# Last updated for CDP-6
def setreffiles_default(channel):
    def_rootdir=os.path.expandvars('$MIRICOORD_DIR')

    wavefile=def_rootdir+'/CRDS/jwst_miri_wavelengthrange_0003.asdf'

    # Channel should be of the form (e.g.) '1A', '3C', etc
    # See https://jwst-crds.stsci.edu//display_result/52cef902-ad77-4792-9964-d26a0a8a96a8
    if ((channel is '1A')or(channel is '2A')):
       distfile=def_rootdir+'/CRDS/jwst_miri_distortion_0021.asdf'
       regfile=def_rootdir+'/CRDS/jwst_miri_regions_0016.asdf'
       specfile=def_rootdir+'/CRDS/jwst_miri_specwcs_0015.asdf'
    elif ((channel is '3A')or(channel is '4A')):
       distfile=def_rootdir+'/CRDS/jwst_miri_distortion_0024.asdf'
       regfile=def_rootdir+'/CRDS/jwst_miri_regions_0013.asdf'
       specfile=def_rootdir+'/CRDS/jwst_miri_specwcs_0017.asdf'
    elif ((channel is '1B')or(channel is '2B')):
       distfile=def_rootdir+'/CRDS/jwst_miri_distortion_0022.asdf'
       regfile=def_rootdir+'/CRDS/jwst_miri_regions_0015.asdf'
       specfile=def_rootdir+'/CRDS/jwst_miri_specwcs_0013.asdf'
    elif ((channel is '3B')or(channel is '4B')):
       distfile=def_rootdir+'/CRDS/jwst_miri_distortion_0026.asdf'
       regfile=def_rootdir+'/CRDS/jwst_miri_regions_0014.asdf'
       specfile=def_rootdir+'/CRDS/jwst_miri_specwcs_0018.asdf'
    elif ((channel is '1C')or(channel is '2C')):
       distfile=def_rootdir+'/CRDS/jwst_miri_distortion_0025.asdf'
       regfile=def_rootdir+'/CRDS/jwst_miri_regions_0018.asdf'
       specfile=def_rootdir+'/CRDS/jwst_miri_specwcs_0014.asdf'
    elif ((channel is '3C')or(channel is '4C')):
       distfile=def_rootdir+'/CRDS/jwst_miri_distortion_0027.asdf'
       regfile=def_rootdir+'/CRDS/jwst_miri_regions_0017.asdf'
       specfile=def_rootdir+'/CRDS/jwst_miri_specwcs_0016.asdf'
    else:
       print('Failure!')

    refs={'distortion': distfile, 'regions':regfile, 'specwcs':specfile, 'wavelengthrange':wavefile}
    return refs

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

# Convenience function to convert input X,Y pixel coordinates
# to alpha,beta.  By default it will read in the most recent
# CDP reference files for the specified channel, but the input
# reference files can also be provided as a dictionary
# xytoabl('1A')
# xytoabl('1A',refs=refs)
def xytoabl(channel,**kwargs):
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
      therefs=setreffiles_default(channel)

    distortion = miri.detector_to_abl(input_model, therefs)
    # Return the distortion object that can then be queried
    return distortion

# Convenience function to convert alpha,beta coordinates
# to v2,v3.  By default it will read in the most recent
# CDP reference files for the specified channel, but the input
# reference files can also be provided as a dictionary
# abtov2v3('1A')
# abtov2v3('1A',refs=refs)
def abtov2v3(channel,**kwargs):
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
      therefs=setreffiles_default(channel)

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

# Convert v2,v3 in arcsec to xan,yan in arcmin
def v2v3_to_xanyan(v2,v3):
    xan=v2/60.
    yan=-(v3+7.8*60.)/60.
    return xan,yan

# Convert xan,yan in arcmin to v2,v3 in arcsec
def xanyan_to_v2v3(xan,yan):
    v2=xan*60.
    v3=(-yan-7.8)*60.
    return v2,v3


