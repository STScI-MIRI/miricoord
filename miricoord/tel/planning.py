#
"""
Useful python tools for planning JWST observations.  Given an observing date, pick
something X degrees away from the Sun at a given ecliptic latitude.

Note that these are all very rough by design, NOT intended to give
precise answers!!!

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
18-Mar-2021  Written by David Law (dlaw@stsci.edu)
"""

import os as os
import math
import numpy as np
from numpy.testing import assert_allclose
import pdb
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body_barycentric, get_body, get_moon

#############################

# Given a date, ecliptic latitude, and separation angle compute
# a position for JWST to observe

# E.g., radec_atlocation('2014-09-22 23:22',15.,110.)
# would give points at +15 degrees ecliptic latitute, 110 degrees separation
# (roughly the middle of 85-135 degree field of regard)

def radec_atlocation(time,elat,targetsep):
    t=Time(time)
    loc = EarthLocation.of_site('greenwich')
    sun=get_body('sun', t, loc)
    sunra,sundec=sun.ra.value,sun.dec.value
    sunhel=sun.geocentricmeanecliptic#heliocentrictrueecliptic
    sunlon,sunlat=sunhel.lon.value,sunhel.lat.value
    c0=SkyCoord(sunlon,sunlat,frame='geocentricmeanecliptic',unit='deg')

    # There are smarter ways to do this, but close enough to calculate
    # angle at a range of longitudes, and interpolate to get the most
    # desired offset angle
    lat1, lat2 = elat, elat
    temp1=np.arange((sunlon+targetsep)-50,(sunlon+targetsep)+50,1.)
    temp2=np.arange((sunlon-targetsep)-50,(sunlon-targetsep)+50,1.)
    dist1=temp1.copy()
    dist2=temp2.copy()
    for ii in range(0,len(temp1)):
        c1=SkyCoord(temp1[ii],lat1,frame='geocentricmeanecliptic',unit='deg')
        c2=SkyCoord(temp2[ii],lat2,frame='geocentricmeanecliptic',unit='deg')
        dist1[ii]=c0.separation(c1).value
        dist2[ii]=c0.separation(c2).value
    lon1=np.interp(targetsep,dist1,temp1,period=360.)
    lon2=np.interp(targetsep,dist2,temp2,period=360.)

    # Values ahead and behind spacecraft
    c1=SkyCoord(lon1,lat1,frame='geocentricmeanecliptic',unit='deg')
    c2=SkyCoord(lon2,lat2,frame='geocentricmeanecliptic',unit='deg')

    ra1=c1.transform_to('fk5').ra.value
    dec1=c1.transform_to('fk5').dec.value
    ra2=c2.transform_to('fk5').ra.value
    dec2=c2.transform_to('fk5').dec.value

    return ra1,dec1,ra2,dec2
