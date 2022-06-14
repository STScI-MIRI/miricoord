#
"""
Useful python tools for working with the MIRI MRS.
This contains flt2 specific code.

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
27-May-2022  Adapted to FLT-1 (D. Law)
13-Jun-2022  Flight updates to distortion and wavecal FLT-2 (D. Law)
"""

import os as os
import sys
import numpy as np
import pdb as pdb
from astropy.modeling import models
from asdf import AsdfFile
from jwst import datamodels
from jwst.assign_wcs import miri

#############################

# Return the tools version
def version():
    return 'flt2'

#############################

# Set the relevant CRDS distortion file based on channel (e.g., '1A')
def get_fitsreffile(channel):
    wavefile='jwst_miri_mrs_wavelengthrange_flt2.asdf'

    # Channel should be of the form (e.g.) '1A', '3C', etc
    if ((channel == '1A')or(channel == '2A')):
       distfile='jwst_miri_mrs12A_distortion_flt2.asdf'
       regfile='jwst_miri_mrs12A_regions_flt2.asdf'
       specfile='jwst_miri_mrs12A_specwcs_flt2.asdf'
    elif ((channel == '3A')or(channel == '4A')):
       distfile='jwst_miri_mrs34A_distortion_flt2.asdf'
       regfile='jwst_miri_mrs34A_regions_flt2.asdf'
       specfile='jwst_miri_mrs34A_specwcs_flt2.asdf'
    elif ((channel == '1B')or(channel == '2B')):
       distfile='jwst_miri_mrs12B_distortion_flt2.asdf'
       regfile='jwst_miri_mrs12B_regions_flt2.asdf'
       specfile='jwst_miri_mrs12B_specwcs_flt2.asdf'
    elif ((channel == '3B')or(channel == '4B')):
       distfile='jwst_miri_mrs34B_distortion_flt2.asdf'
       regfile='jwst_miri_mrs34B_regions_flt2.asdf'
       specfile='jwst_miri_mrs34B_specwcs_flt2.asdf'
    elif ((channel == '1C')or(channel == '2C')):
       distfile='jwst_miri_mrs12C_distortion_flt2.asdf'
       regfile='jwst_miri_mrs12C_regions_flt2.asdf'
       specfile='jwst_miri_mrs12C_specwcs_flt2.asdf'
    elif ((channel == '3C')or(channel == '4C')):
       distfile='jwst_miri_mrs34C_distortion_flt2.asdf'
       regfile='jwst_miri_mrs34C_regions_flt2.asdf'
       specfile='jwst_miri_mrs34C_specwcs_flt2.asdf'
    else:
       print('Failure!')

    # Try looking for the files in the expected location
    rootdir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    rootdir=os.path.join(rootdir,'data/crds/')
    refs={'distortion': os.path.join(rootdir,distfile), 'regions':os.path.join(rootdir,regfile),
          'specwcs':os.path.join(rootdir,specfile), 'wavelengthrange':os.path.join(rootdir,wavefile)}
    if os.path.exists(os.path.join(rootdir,distfile)):
        return refs

    # If that didn't work, look in the system path
    rootdir=sys.prefix
    rootdir=os.path.join(rootdir,'data/crds/')
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
    if ((channel == '1A')or(channel == '2A')):
       newband='SHORT'
       newchannel='12'
    elif ((channel == '3A')or(channel == '4A')):
       newband='SHORT'
       newchannel='34'
    elif ((channel == '1B')or(channel == '2B')):
       newband='MEDIUM'
       newchannel='12'
    elif ((channel == '3B')or(channel == '4B')):
       newband='MEDIUM'
       newchannel='34'
    elif ((channel == '1C')or(channel == '2C')):
       newband='LONG'
       newchannel='12'
    elif ((channel == '3C')or(channel == '4C')):
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
    if (channel == '1A'):
       thewave=5.32
    elif (channel == '1B'):
       thewave=6.145
    elif (channel == '1C'):
       thewave=7.09
    elif (channel == '2A'):
       thewave=8.135
    elif (channel == '2B'):
       thewave=9.395
    elif (channel == '2C'):
       thewave=10.85
    elif (channel == '3A'):
       thewave=12.505
    elif (channel == '3B'):
       thewave=14.5
    elif (channel == '3C'):
       thewave=16.745
    elif (channel == '4A'):
       thewave=19.29
    elif (channel == '4B'):
       thewave=22.47
    elif (channel == '4C'):
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

# Convert from JWST v2,v3 coordinates to alpha,beta
# all coordinates are in arcsec
def v2v3toab(v2in,v3in,channel):
    # Split input channel into components, e.g.
    # if channel='1A' then ch=1 and sband=A
    ch=channel[0]
    sband=channel[1]

    v2=np.array(v2in)*1.
    v3=np.array(v3in)*1.

    # Open relevant distortion file
    distfile=fits.open(get_fitsreffile(channel))

    # Read the distortion table
    convtable=distfile['V2V3_to_albe'].data
    # Determine which rows we need
    alindex=(np.where(convtable['Label'] == 'U_CH'+channel+'_al'))[0][0]
    beindex=(np.where(convtable['Label'] == 'U_CH'+channel+'_be'))[0][0]

    if (np.logical_or(alindex < 0, beindex < 0)):
        print('Bad channel specification!')

    conv_al=convtable[alindex]
    conv_be=convtable[beindex]

    # Apply transform to alpha,beta
    alpha=conv_al[1]+conv_al[2]*v2+conv_al[3]*v2*v2+ \
       conv_al[4]*v3+conv_al[5]*v3*v2+conv_al[6]*v3*v2*v2+ \
       conv_al[7]*v3*v3+conv_al[8]*v3*v3*v2+conv_al[9]*v3*v3*v2*v2

    beta=conv_be[1]+conv_be[2]*v2+conv_be[3]*v2*v2+ \
       conv_be[4]*v3+conv_be[5]*v3*v2+conv_be[6]*v3*v2*v2+ \
       conv_be[7]*v3*v3+conv_be[8]*v3*v3*v2+conv_be[9]*v3*v3*v2*v2

    distfile.close()

    return alpha,beta

#############################

# MRS test reference data
# Created by DRL 6/14/22
mrs_ref_data = {
    '1A': {'x': np.array([ 76., 354.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([ 0.05652259, -0.02172324]),
           'beta': np.array([-0.17721014, -1.24047101]),
           'lam': np.array([5.35006707, 5.51533192]),
           'v2': np.array([-503.2457946 , -503.22525263]),
           'v3': np.array([-318.85913779, -317.78642102]),
           },
    '1B': {'x': np.array([ 76., 355.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([-0.01337304,  0.09966433]),
           'beta': np.array([-0.17720418, -1.24042924]),
           'lam': np.array([6.1781104 , 6.36780764]),
           'v2': np.array([-503.24796851, -503.01329913]),
           'v3': np.array([-318.96006569, -317.92254897]),
           },
    '1C': {'x': np.array([ 78., 356.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([ 0.02716106, -0.03223556]),
           'beta': np.array([-0.17720219, -1.24041531]),
           'lam': np.array([7.13833239, 7.3541621]),
           'v2': np.array([-503.19148941, -503.13095893]),
           'v3': np.array([-318.69482794, -317.62408021]),
           },
    '2A': {'x': np.array([574., 719.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([0.02861321, 0.02479319]),
           'beta': np.array([ 0.27971819, -1.39859093]),
           'lam': np.array([8.15655668, 8.44089171]),
           'v2': np.array([-503.47994056, -503.2617271]),
           'v3': np.array([-319.57873922, -317.9176351]),
           },
    '2B': {'x': np.array([570., 715.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([-0.03627037, -0.02182506]),
           'beta': np.array([ 0.27972605, -1.39863026]),
           'lam': np.array([9.43432838, 9.76942261 ]),
           'v2': np.array([-503.63246937, -503.38051277]),
           'v3': np.array([-319.84220657, -318.19366974]),
           },
    '2C': {'x': np.array([573., 718.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([-0.07776634, -0.07206809]),
           'beta': np.array([ 0.27972212, -1.3986106 ]),
           'lam': np.array([10.89124816, 11.27396532]),
           'v2': np.array([-503.47227198, -503.24953575]),
           'v3': np.array([-319.77419183, -318.12789646]),
           },    
    '3A': {'x': np.array([918., 827.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([-0.14851662, -0.12738753]),
           'beta': np.array([ 0.58472067, -1.754162  ]),
           'lam': np.array([12.55889518, 12.22094465]),
           'v2': np.array([-504.13278185, -504.00754217]),
           'v3': np.array([-319.56520658, -317.26165859]),
           },
    '3B': {'x': np.array([919., 827.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([-0.18606572,  0.06403284]),
           'beta': np.array([ 0.58472067, -1.754162  ]),
           'lam': np.array([14.50829058, 14.11452434]),
           'v2': np.array([-504.24721745, -503.82448341]),
           'v3': np.array([-319.72339638, -317.50002028]),
           },
    '3C': {'x': np.array([917., 826.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([-0.08360864, -0.09362804]),
           'beta': np.array([ 0.58472067, -1.754162  ]),
           'lam': np.array([16.77816236, 16.31206364]),
           'v2': np.array([-504.0360265 , -503.87985304]),
           'v3': np.array([-319.71643786, -317.47041881]),
           },
    '4A': {'x': np.array([195., 232.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([-0.18098345, -0.08642445]),
           'beta': np.array([ 2.29613309, -1.64009507]),
           'lam': np.array([19.4712413 , 18.77320581]),
           'v2': np.array([-503.31779031, -502.56161701]),
           'v3': np.array([-321.83128315, -318.15829335]),
           },    
    '4B': {'x': np.array([192., 229.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([-0.03437878, -0.07182671]),
           'beta': np.array([ 2.29613654, -1.64009753]),
           'lam': np.array([22.73579252, 21.92986248]),
           'v2': np.array([-503.28885231, -502.63240122]),
           'v3': np.array([-321.66352013, -318.09815946]),
           },    
    '4C': {'x': np.array([194., 231.]),
           'y': np.array([512., 700.]),
           's': np.array([10,4]),
           'alpha': np.array([-0.08228878, -0.0200367]),
           'beta': np.array([ 2.29611932, -1.64008523]),
           'lam': np.array([26.43633709, 25.49355614]),
           'v2': np.array([-503.39221343, -502.66650835]),
           'v3': np.array([-321.76840195, -317.87523125]),
           }
}
