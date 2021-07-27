#
"""
Useful python tools for working with Modified Julian Dates.
Mostly just wrappers to the rather buried functionality
elsewhere.

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
27-Jul-2021  Written by David Law (dlaw@stsci.edu)
"""

import datetime
from astropy.time import Time
import pdb

#############################

# Print the current MJD
def current_mjd():
    utcnow=datetime.datetime.utcnow()
    tt=Time(utcnow)
    mjd=tt.mjd
    
    return mjd
    
