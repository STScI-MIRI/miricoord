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
# Provided by Polychronis 5/9/19
mrs_ref_data = {
    '1A': {'x': np.array([76.0,354.0]),
           'y': np.array([512.0,700.0]),
           's': np.array([10,4]),
           'alpha': np.array([0.05765538365149925,-0.017032619150995743]),
           'beta': np.array([-0.17721014379699995,-1.240471006579]),
           'lam': np.array([5.348546577257886,5.5136420569934925]),
           'v2': np.array([-503.57285226785064,-503.4979806620663]),
           'v3': np.array([-318.5749892859028,-317.5090073056335]),
           },
    '1B': {'x': np.array([76.0,355.0]),
           'y': np.array([512.0,700.0]),
           's': np.array([10,4]),
           'alpha': np.array([-0.012990737471741731,0.10766447914943456]),
           'beta': np.array([-0.17720417669099997,-1.240429236837]),
           'lam': np.array([6.168310398808807,6.358007642348213]),
           'v2': np.array([-503.643100332753,-503.37069816112813]),
           'v3': np.array([-318.72773306477103,-317.6938248759762]),
           },
    '1C': {'x': np.array([78.0,356.0]),
           'y': np.array([512.0,700.0]),
           's': np.array([10,4]),
           'alpha': np.array([0.02871804339196271,-0.028315822861031847]),
           'beta': np.array([-0.17720218765499984,-1.240415313585]),
           'lam': np.array([7.006608159574103,7.218455147089075]),
           'v2': np.array([-503.5598371896608,-503.45975848303885]),
           'v3': np.array([-318.4367657801553,-317.3779485524358]),
           },
    '2A': {'x': np.array([574.0,719.0]),
           'y': np.array([512.0,700.0]),
           's': np.array([10,4]),
           'alpha': np.array([0.022862344416012093,0.024104763006107532]),
           'beta': np.array([0.27971818633699996,-1.3985909316610001]),
           'lam': np.array([8.139463800053713, 8.423879719165456]),
           'v2': np.array([-503.65782416704644, -503.3907046961389]),
           'v3': np.array([-319.3709764579651, -317.71318662530217]),
           },
    '2B': {'x': np.array([570.0,715.0]),
           'y': np.array([512.0,700.0]),
           's': np.array([10,4]),
           'alpha': np.array([-0.04101483043351095,-0.021964438108625473]),
           'beta': np.array([0.27972605223,-1.39863026115]),
           'lam': np.array([9.49091778668766, 9.826112199836349]),
           'v2': np.array([-503.872441161987, -503.58468453126545]),
           'v3': np.array([-319.6066193816802, -317.9526192173689]),
           },
    '2C': {'x': np.array([573.0,718.0]),
           'y': np.array([512.0,700.0]),
           's': np.array([10,4]),
           'alpha': np.array([-0.08065540123411097,-0.07196315905207484]),
           'beta': np.array([0.2797221192789996, -1.3986105964070001]),
           'lam': np.array([10.909558387414732,11.292658213110698]),
           'v2': np.array([-503.7062367371822, -503.4292038385116]),
           'v3': np.array([-319.54349206004053, -317.8886490566051]),
           },
    '3A': {'x': np.array([918.0,827.0]),
           'y': np.array([512.0,700.0]),
           's': np.array([10,4]),
           'alpha': np.array([-0.14902640584477922, -0.1394111481404252]),
           'beta': np.array([0.5847206674920002, -1.7541620024759998]),
           'lam': np.array([12.586085291551054, 12.171803779467552]),
           'v2': np.array([-504.57532179184557, -504.3428404141017]),
           'v3': np.array([-319.3596209726561, -317.0363338552647]),
           },
    '3B': {'x': np.array([919.0, 827.0]),
           'y': np.array([512.0, 700.0]),
           's': np.array([10,4]),
           'alpha': np.array([-0.18610616903060873, 0.05223448620927229]),
           'beta': np.array([0.5847206674920002, -1.7541620024759998]),
           'lam': np.array([14.60074101845329, 14.120353260795175]),
           'v2': np.array([-504.29128783278026, -503.81513623681207]),
           'v3': np.array([-319.5977726217362, -317.30169796071453]),
           },
    '3C': {'x': np.array([917.0,826.0]),
           'y': np.array([512.0,700.0]),
           's': np.array([10,4]),
           'alpha': np.array([-0.08917305254544772, -0.09924683542340063]),
           'beta': np.array([0.5847206674920002, -1.7541620024759998]),
           'lam': np.array([16.860616228418674, 16.305648049347006]),
           'v2': np.array([-504.29179372150304, -504.06099473540036]),
           'v3': np.array([-319.5864222556306, -317.26146053061063]),
           },
    '4A': {'x': np.array([195.0, 232.0]),
           'y': np.array([512.0, 700.0]),
           's': np.array([10,4]),
           'alpha': np.array([-0.18281231856817595, -0.10820926727846612]),
           'beta': np.array([2.2961330928359995, -1.640095066308]),
           'lam': np.array([19.42967253041467, 18.733785802367724]),
           'v2': np.array([-503.73916258138155, -502.9287085654886]),
           'v3': np.array([-321.7198475574414, -317.8596067111157]),
           },
    '4B': {'x': np.array([192.0, 229.0]),
           'y': np.array([512.0, 700.0]),
           's': np.array([10,4]),
           'alpha': np.array([-0.03596952007447607, -0.10259402857181654]),
           'beta': np.array([2.2961365363689996, -1.640097525977]),
           'lam': np.array([22.47574268879503, 21.67074830984225]),
           'v2': np.array([-503.7051048327475, -502.9891450100565]),
           'v3': np.array([-321.6637327196876, -317.78403487305536]),
           },
    '4C': {'x': np.array([194.0, 231.0]),
           'y': np.array([512.0, 700.0]),
           's': np.array([10,4]),
           'alpha': np.array([-0.0661930805678849, -0.01176625661012924]),
           'beta': np.array([2.296119318687, -1.640085227631]),
           'lam': np.array([26.292379242285914, 25.350694577065074]),
           'v2': np.array([-503.7171854824459, -502.9282547181127]),
           'v3': np.array([-321.57006077329663, -317.7252303132135]),
           }
    
}
