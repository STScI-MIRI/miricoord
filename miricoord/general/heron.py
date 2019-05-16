#
"""
Implement Heron's formula for area of a triangle.
NOT thouroughly tested or applicable yet.

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
16-May-2019  Written by David Law (dlaw@stsci.edu)
"""

import math
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

import pdb


#############################

# Area of a triangle whose points are determined in v2,v3 (arcsec)
def tri_area(v2,v3):
    # Not robust, assumes proper length!
    pt1=SkyCoord(v2[0],v3[0],unit='arcsec')
    pt2=SkyCoord(v2[1],v3[1],unit='arcsec')
    pt3=SkyCoord(v2[2],v3[2],unit='arcsec')

    len12=pt1.separation(pt2).arcsec
    len13=pt1.separation(pt3).arcsec
    len23=pt2.separation(pt3).arcsec

    S=(len12+len13+len23)/2.
    area=np.sqrt(S*(S-len12)*(S-len13)*(S-len23))

    return area

# Area of a quadrilateral whose points are determined in v2,v3 (arcsec)
def quad_area(v2,v3):
    # Not robust, assumes proper length!
    # Break into two triangles
    area1=tri_area(v2[0:3],v3[0:3])
    area2=tri_area(v2[1:4],v3[1:4])

    return area1+area2

