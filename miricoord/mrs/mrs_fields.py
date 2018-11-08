#
"""
Python routines to compute the effective field sizes for the MRS.
Based on the old IDL routine mmrs_getfov.pro

By default it will pull the aperture information directly from
the SIAF using pysiaf, although it can be overridden to instead
pull information from a miri_siaf_mrs.txt file.  These are the inputs
to SIAF deliveries, so this would be used when for example
computing the field boundaries corresponding to a new SIAF delivery
that has not yet been integrated into pysiaf.

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
08-Nov-2018  Written by David Law (dlaw@stsci.edu)
"""

# We will need both the mrs_tools code
import miricoord.miricoord.mrs.mrs_tools as mrst
# the imager tools code
import miricoord.miricoord.imager.mirim_tools as mt
# and the pysiaf code
import pysiaf

import pdb
import numpy as np
from astropy.io import ascii

#############################

# Compute the field size given a set of corner coordinates
# ordered lower-left, upper-left, upper-right, lower-right
def boxsize(xvert,yvert):
    # Neglect the cosine term in v2,v3 space since we're working very near the origin
    dist=np.array([0.,0.,0.,0.])
    dist[0]=np.sqrt(np.power(xvert[0]-xvert[1],2.)+np.power(yvert[0]-yvert[1],2.))
    dist[1]=np.sqrt(np.power(xvert[1]-xvert[2],2.)+np.power(yvert[1]-yvert[2],2.))
    dist[2]=np.sqrt(np.power(xvert[2]-xvert[3],2.)+np.power(yvert[2]-yvert[3],2.))
    dist[3]=np.sqrt(np.power(xvert[0]-xvert[3],2.)+np.power(yvert[0]-yvert[3],2.))

    # Average each pair of sides
    ysize=(dist[0]+dist[2])/2.
    xsize=(dist[1]+dist[3])/2.

    return xsize,ysize

#############################

# Return the aperture corner coordinates in v2,v3 using pysiaf.
# Note that we'll have to convert from the Ideal coordinates
# recorded in pysiaf.

def fromsiaf(apername):
    siaf=pysiaf.Siaf('MIRI')
    thissiaf=siaf[apername]

    v2ref,v3ref=thissiaf.V2Ref,thissiaf.V3Ref
    xvert=np.array([thissiaf.XIdlVert1,thissiaf.XIdlVert2,thissiaf.XIdlVert3,thissiaf.XIdlVert4])
    yvert=np.array([thissiaf.YIdlVert1,thissiaf.YIdlVert2,thissiaf.YIdlVert3,thissiaf.YIdlVert4])
    # Convert the Ideal frame corner coordinates to v2,v3 corner coordinates
    v2vert,v3vert=mt.Idealtov2v3(xvert,yvert,apername)

    # Convert from SIAF order (lower-left, lower-right, upper-right, upper-left) to 
    # lower-left, upper-left, upper-right, lower-right
    v2vert=v2vert[[0,3,2,1]]
    v3vert=v3vert[[0,3,2,1]]
    
    return v2vert,v3vert

#############################

# Return the aperture corner coordinates in v2,v3 using an input
# miri_siaf_mrs.txt file.  Note that this requires that the input file
# match the expected format.

def fromfile(apername,file):
    data=ascii.read(file)

    # Determine which rows we need
    indx=(np.where(data['AperName'] == apername))
    # Pull out the vertices from that row
    v2vert=np.array([data[indx]['v2ll'],data[indx]['v2ul'],data[indx]['v2ur'],data[indx]['v2lr']])
    v3vert=np.array([data[indx]['v3ll'],data[indx]['v3ul'],data[indx]['v3ur'],data[indx]['v3lr']])

    return v2vert,v3vert

#############################

# Determine whether to get the corner information from pysiaf or a file,
# and then get the corner information

def getcorners(apername,**kwargs):
    # If 'file' was set in kwargs, use that for the input
    if ('file') in kwargs:
        file=kwargs['file']
        v2,v3=fromfile(apername,file)
    else:
        v2,v3=fromsiaf(apername)

    return v2,v3
        
#############################

# This routine will compute the field sizes for one
# of the MRS channels
def getfov(**kwargs):
    # Define all valid aperture names
    apernames=['MIRIFU_CHANNEL1A','MIRIFU_CHANNEL1B','MIRIFU_CHANNEL1C','MIRIFU_CHANNEL2A','MIRIFU_CHANNEL2B','MIRIFU_CHANNEL2C','MIRIFU_CHANNEL3A','MIRIFU_CHANNEL3B','MIRIFU_CHANNEL3C','MIRIFU_CHANNEL4A','MIRIFU_CHANNEL4B','MIRIFU_CHANNEL4C']
    nbands=len(apernames)

    for i in range(0,nbands):
        v2corners,v3corners=getcorners(apernames[i],**kwargs)
        xsize,ysize=boxsize(v2corners,v3corners)
        print(apernames[i],round(xsize,ndigits=1),'x',round(ysize,ndigits=1))
        
    # Average over the subbands
    fovch=np.zeros((2, 4))
    for i in range(0,4):
        for j in range(0,3):
            v2corners,v3corners=getcorners(apernames[i*3+j],**kwargs)
            xsize,ysize=boxsize(v2corners,v3corners)
            fovch[0,i]+=xsize/3.
            fovch[1,i]+=ysize/3.
        print('Ch'+str(i+1),round(fovch[0,i],ndigits=1),'x',round(fovch[1,i],ndigits=1))
        
