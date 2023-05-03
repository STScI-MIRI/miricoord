#
"""
Useful python tools for working with JWST.  Specifically, for dealing
with the v2,v3 to RA,DEC transforms.  Although this could be done by
calling the pipeline code, it is often inconvenient to do so.

Convert v2,v3 coordinates in a JWST frame to RA,DEC coordinates
given a JWST attitude matrix (or relevant attitude keywords)
describing the relative orientation of the V3,V3 and RA,DEC
reference frames.  These can be derived from JWST file FITS headers.

This constructs the attitude matrix using the keywords V2_REF,
V3_REF, RA_REF, DEC_REF, and ROLL_REF where the first four
associate a fixed reference location in V2,V3 with a location in RA,DEC
and the ROLL_REF specifies the local roll (defined as the position
angle of the V3 axis measured from N towards E) of the V2,V3 coordinate
system at the reference location.

Note that all v2,v3 locations are given in arcseconds while all
RA,DEC information is given in degrees

In normal operation this function computes and uses the full JWST
attitude matrix; it can also be run in a /local approximation
that neglects the full matrix formalism for a local approximation
with simpler math.

The full attitude matrix calculations are based on section 6 of
technical report JWST-STScI-001550 'Description and Use of
the JWST Science Instrument Aperture File', author C. Cox.

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
12-Apr-2016  Written by David Law (dlaw@stsci.edu)
17-Oct-2016  Deal with zero inputs, v2/v3 in arcsec (D. Law)
17-Oct-2018  Converted from IDL to python
"""

import os as os
import re
import math
import numpy as np
from numpy.testing import assert_allclose
import pdb
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.coordinates import get_body_barycentric
from astropy.constants import au
import miricoord.imager.mirim_tools as mt

#############################

# Construct the JWST M1 attitude matrix (V2 and V3 rotations)
# V2REF and V3REF should be in radians
def jwst_att1(V2REF,V3REF):
    # M1=  a00  a01  a02
    #      a10  a11  a12
    #      a20  a21  a22
  
    thematrix=np.zeros((3,3))
    thematrix[0,0]=np.cos(V2REF)*np.cos(V3REF)
    thematrix[1,0]=np.sin(V2REF)*np.cos(V3REF)
    thematrix[2,0]=np.sin(V3REF)
    thematrix[0,1]=-np.sin(V2REF)
    thematrix[1,1]=np.cos(V2REF)
    thematrix[2,1]=0.
    thematrix[0,2]=-np.cos(V2REF)*np.sin(V3REF)
    thematrix[1,2]=-np.sin(V2REF)*np.sin(V3REF)
    thematrix[2,2]=np.cos(V3REF)

    return thematrix

#############################

# Construct the JWST M2 attitude matrix (RA,DEC,ROLL rotations)
# RAREF, DECREF, ROLLREF should be in radians
def jwst_att2(RAREF,DECREF,ROLLREF):
    # M2=  a00  a01  a02
    #      a10  a11  a12
    #      a20  a21  a22
  
    thematrix=np.zeros((3,3))
    thematrix[0,0]=np.cos(RAREF)*np.cos(DECREF)
    thematrix[1,0]=-np.sin(RAREF)*np.cos(ROLLREF)+np.cos(RAREF)*np.sin(DECREF)*np.sin(ROLLREF)
    thematrix[2,0]=-np.sin(RAREF)*np.sin(ROLLREF)-np.cos(RAREF)*np.sin(DECREF)*np.cos(ROLLREF)
    thematrix[0,1]=np.sin(RAREF)*np.cos(DECREF)
    thematrix[1,1]=np.cos(RAREF)*np.cos(ROLLREF)+np.sin(RAREF)*np.sin(DECREF)*np.sin(ROLLREF)
    thematrix[2,1]=np.cos(RAREF)*np.sin(ROLLREF)-np.sin(RAREF)*np.sin(DECREF)*np.cos(ROLLREF)
    thematrix[0,2]=np.sin(DECREF)
    thematrix[1,2]=-np.cos(DECREF)*np.sin(ROLLREF)
    thematrix[2,2]=np.cos(DECREF)*np.cos(ROLLREF)

    return thematrix

#############################

# JWST M = (M2 # M1) attitude matrix
# V2REF, V3REF, RAREF, DECREF, ROLLREF should be in radians
def jwst_attmatrix(V2REF,V3REF,RAREF,DECREF,ROLLREF):

    thematrix = np.matmul(jwst_att1(V2REF,V3REF),jwst_att2(RAREF,DECREF,ROLLREF))

    return thematrix

#############################

# Compute the local roll (the position angle measured from N
# towards E of the V3 axis) at any V2,V3 given an attitude matrix
# V2, V3 must be in radians, result is in radians
def jwst_localroll(V2,V3,ATTMATRIX):

    X=-(ATTMATRIX[0,2]*np.cos(V2)+ATTMATRIX[1,2]*np.sin(V2))*np.sin(V3)+ATTMATRIX[2,2]*np.cos(V3)
    Y=(ATTMATRIX[0,0]*ATTMATRIX[2,1]-ATTMATRIX[0,1]*ATTMATRIX[2,0])*np.cos(V2)+(ATTMATRIX[1,0]*ATTMATRIX[2,1]-ATTMATRIX[1,1]*ATTMATRIX[2,0])*np.sin(V2)

    return np.arctan2(Y,X)

#############################

# Compute RA,DEC in degrees given a v2,v3 in arcsec and attitude keywords.
# Attitude keywords v2ref,v3ref are in arcsec, raref,decref,rollref in degrees
# Either provide v2ref=,v3ref=,raref=,decref=,rollref=
# or provide a FITS header with these keywords in it.
# If provided neither, crash.  If provided both, use directly provided values instead of header values.
# E.g., ra,dec=jwst_v2v3toradec(-450.,-380.,hdr=hdr)

def jwst_v2v3toradec(v2in,v3in,**kwargs):
    if ('hdr' in kwargs):
        hdr=kwargs['hdr']
        v2ref=hdr['V2_REF']
        v3ref=hdr['V3_REF']
        raref=hdr['RA_REF']
        decref=hdr['DEC_REF']
        rollref=hdr['ROLL_REF']
    elif ('v2ref' in kwargs):
        v2ref=kwargs['v2ref']
        v3ref=kwargs['v3ref']
        raref=kwargs['raref']
        decref=kwargs['decref']
        rollref=kwargs['rollref']
    else:
        print('Error: no reference values provided!')

    # Convert reference values to units of radians
    v2ref=v2ref/3600.*np.pi/180.
    v3ref=v3ref/3600.*np.pi/180.
    raref=raref*np.pi/180.
    decref=decref*np.pi/180.
    rollref=rollref*np.pi/180.

    # Compute the JWST attitude matrix from the 5 attitude keywords
    attmat=jwst_attmatrix(v2ref,v3ref,raref,decref,rollref)

    # Number of input points
    v2=np.array(v2in)
    v3=np.array(v3in)
    npoints=len(v2)
    
    # Make empty vectors to hold the output ra,dec,NEWROLL
    ra=np.zeros(npoints)
    dec=np.zeros(npoints)
    newroll=np.zeros(npoints)

    # Loop over input points in the simplest way
    for i in range(0,npoints):
        # Compute the vector describing the input location
        invector=[np.cos(v2[i]/3600.*np.pi/180.)*np.cos(v3[i]/3600.*np.pi/180.),np.sin(v2[i]/3600.*np.pi/180.)*np.cos(v3[i]/3600.*np.pi/180.),np.sin(v3[i]/3600.*np.pi/180.)]

        # Compute the output vector (cos(RA)cos(dec),sin(RA)cos(dec),sin(dec))
        # by applying the attitude matrix
        outvector=np.matmul(invector,attmat)

        # Split the output vector into RA and DEC components and convert
        # back to degrees
        ra[i]=np.arctan2(outvector[1],outvector[0])*180./np.pi

        # Ensure 0-360 degrees
        if (ra[i] < 0.):
            ra[i]=ra[i]+360.

        dec[i]=np.arcsin(outvector[2])*180./np.pi

        # Compute the local roll at this location and convert
        # back to degrees
        newroll[i]=jwst_localroll(v2[i]/3600.*np.pi/180.,v3[i]/3600.*np.pi/180.,attmat)*180./np.pi
        # Ensure 0-360 degrees
        if (newroll[i] < 0.):
            newroll[i]=newroll[i]+360.

    return ra,dec,newroll

#############################

# Compute v2,v3 in arcsec given an RA,DEC in degrees and attitude keywords.
# Attitude keywords v2ref,v3ref are in arcsec, raref,decref,rollref in degrees
# Either provide v2ref=,v3ref=,raref=,decref=,rollref=
# or provide a FITS header with these keywords in it.
# If provided neither, crash.  If provided both, use directly provided values instead of header values.
# E.g., v2,v3=jwst_radectov2v3(225.34234,+43.234323,hdr=hdr)

def jwst_radectov2v3(rain,decin,**kwargs):
    if ('hdr' in kwargs):
        hdr=kwargs['hdr']
        v2ref=hdr['V2_REF']
        v3ref=hdr['V3_REF']
        raref=hdr['RA_REF']
        decref=hdr['DEC_REF']
        rollref=hdr['ROLL_REF']
    elif ('v2ref' in kwargs):
        v2ref=kwargs['v2ref']
        v3ref=kwargs['v3ref']
        raref=kwargs['raref']
        decref=kwargs['decref']
        rollref=kwargs['rollref']
    else:
        print('Error: no reference values provided!')

    # Convert reference values to units of radians
    v2ref=v2ref/3600.*np.pi/180.
    v3ref=v3ref/3600.*np.pi/180.
    raref=raref*np.pi/180.
    decref=decref*np.pi/180.
    rollref=rollref*np.pi/180.

    # Compute the JWST attitude matrix from the 5 attitude keywords
    attmat=jwst_attmatrix(v2ref,v3ref,raref,decref,rollref)

    # Number of input points
    ra=np.array(rain)
    dec=np.array(decin)
    npoints=len(ra)
    
    # Make empty vectors to hold the output v2,v3,NEWROLL
    v2=np.zeros(npoints)
    v3=np.zeros(npoints)
    newroll=np.zeros(npoints)

    # Loop over input points in the simplest way
    for i in range(0,npoints):
        # Compute the vector describing the input location
        invector=[np.cos(ra[i]*np.pi/180.)*np.cos(dec[i]*np.pi/180.),np.sin(ra[i]*np.pi/180.)*np.cos(dec[i]*np.pi/180.),np.sin(dec[i]*np.pi/180.)]

        # Compute the output vector (cos(v2)cos(v3),sin(v2)cos(v3),sin(v3))
        # by applying the attitude matrix
        outvector=np.matmul(attmat,invector)

        # Split the output vector into v2 and v3 components and convert
        # back to degrees
        v2[i]=np.arctan2(outvector[1],outvector[0])*180./np.pi
        # Convert to arcsec
        v2[i]=v2[i]*3600.
        v3[i]=np.arcsin(outvector[2])*180./np.pi*3600.# v3 in arcsec

        # Compute the local roll at this location and convert
        # back to degrees
        newroll[i]=jwst_localroll(v2[i]/3600.*np.pi/180.,v3[i]/3600.*np.pi/180.,attmat)*180./np.pi
        # Ensure 0-360 degrees
        if (newroll[i] < 0.):
            newroll[i]=newroll[i]+360.

    return v2,v3,newroll

#############################

# Project RA, DEC onto a tangent plane grid at a particular location.
# Output xi,eta are in arcseconds.
def radectoxieta(crval1,crval2,ra,dec):
    # arcsec per radian
    rad2arcsec = (180.0*3600.0)/np.pi
    # radians per degree
    deg2rad = np.pi/180.0

    ra0 = crval1*deg2rad
    dec0 = crval2*deg2rad
    radiff = ra*deg2rad - ra0;
    decr = dec*deg2rad

    h=np.sin(decr)*np.sin(dec0)+np.cos(decr)*np.cos(dec0)*np.cos(radiff)

    xi = np.cos(decr)*np.sin(radiff)/h
    eta = ( np.sin(decr)*np.cos(dec0) - np.cos(decr)*np.sin(dec0)*np.cos(radiff) )/h;

    xi = xi * rad2arcsec
    eta = eta * rad2arcsec

    return xi,eta

#############################

# Compute angle between spacecraft v1 optical axis and the sun vector
# based on information in the 'SCI' extension header.
# Returns separation angle in degrees.
def jwst_sunangle(hdr, verbose=True):
    # Note that the spacecraft orientation is given by Pitch, Roll, Yaw
    # Pitch is the most relevant to hot vs cold attitude and is related
    # to how far away from the Sun the V1 axis is pointing.
    # Roll is the roll about the V1 optical axis.
    # Yaw is the orientation about the sunline, so the longitude of pointing
    # around the observable torus.
    
    # ICRS X/Y/Z of JWST relative to solar system barycenter in km
    jwstx=hdr['JWST_X']
    jwsty=hdr['JWST_Y']
    jwstz=hdr['JWST_Z']

    # ICRS X/Y/Z of Sun relative to solar system barycenter
    ti=Time(hdr['EPH_TIME'],format='mjd')
    sun=get_body_barycentric('sun',ti) # This is in AU
    # Convert from AU to km
    sunx=sun.x.value*au.value/1000
    suny=sun.y.value*au.value/1000
    sunz=sun.z.value*au.value/1000

    # Total vector from JWST to Sun
    x=sunx-jwstx
    y=suny-jwsty
    z=sunz-jwstz
    r=np.sqrt(x*x + y*y + z*z)

    # ICRS RA/DEC of the vector from JWST to the Sun
    sun_dec=np.arcsin(z/r)*180./np.pi
    sun_ra=np.arctan2(y,x)*180./np.pi
    if (sun_ra < 0):
        sun_ra = sun_ra+360.

    # Apparent position of Sun from JWST tested against JPL Horizons
    # computation to within 0.1 arcsec or so
    if (verbose):
        print('Apparent Sun position from JWST: ',sun_ra, sun_dec)

    # Where is JWST pointing?
    ra=hdr['RA_V1']
    dec=hdr['DEC_V1']

    c1 = SkyCoord(sun_ra, sun_dec, frame='icrs', unit='deg')
    c2 = SkyCoord(ra, dec, frame='icrs', unit='deg')
    sep = c1.separation(c2)
    
    return sep.value

#############################

# Test a bunch of cases against the JWST pipeline transforms to make sure they agree

def testtransform():
    import jwst
    from jwst.jwst.transforms.models import V23ToSky
    from astropy.modeling import models as astmodels

    # 3 imager and an MRS transform
    v2ref=[-453.363,-453.363,-453.363,-502.65447]
    v3ref=[-374.069,-374.069,-374.069,-318.74246]
    raref=[56.,265.,355.,17.]
    decref=[43.,-89.,75.,-10.]
    rollref=[0.,0.,170.,355.]

    v2=[-450.,-440.,-400.,-504.]
    v3=[-420.,-340.,-420.,-319.]

    ntest=len(v2ref)
    ra1=np.zeros(ntest)
    ra2=np.zeros(ntest)
    dec1=np.zeros(ntest)
    dec2=np.zeros(ntest)
    v2back1=np.zeros(ntest)
    v3back1=np.zeros(ntest)
    v2back2=np.zeros(ntest)
    v3back2=np.zeros(ntest)    

    for i in range(0,ntest):
        ra1[i],dec1[i],roll1=jwst_v2v3toradec([v2[i]],[v3[i]],v2ref=v2ref[i],v3ref=v3ref[i],raref=raref[i],decref=decref[i],rollref=rollref[i])
        angles=[-v2ref[i]/3600., v3ref[i]/3600., -rollref[i], -decref[i], raref[i]]
        axes = "zyxyz"
        sky_rotation = V23ToSky(angles, axes_order=axes, name="v23tosky")
        model=astmodels.Scale(1/3600) & astmodels.Scale(1/3600) | sky_rotation
        ra2[i],dec2[i]=model(v2[i],v3[i])
        if (ra2[i] < 0):
            ra2[i]=ra2[i]+360.
        
        v2back2[i],v3back2[i]=model.inverse(ra2[i],dec2[i])
        v2back1[i],v3back1[i],junk=jwst_radectov2v3([ra1[i]],[dec1[i]],v2ref=v2ref[i],v3ref=v3ref[i],raref=raref[i],decref=decref[i],rollref=rollref[i])

    radiff=ra1-ra2
    maxra=(np.abs(radiff)).max()*3600.*np.cos(dec1[i]*np.pi/180.)
    decdiff=dec1-dec2
    maxdec=(np.abs(decdiff)).max()*3600.
    v2diff=v2back1-v2back2
    maxv2=(np.abs(v2diff)).max()
    v3diff=v3back1-v3back2
    maxv3=(np.abs(v3diff)).max()

    allmax=(np.array([maxra,maxdec,maxv2,maxv3])).max()
    print('Maximum difference from pipeline:',allmax,'arcsec')
 
#############################

# Test for consistency in a FITS header
# Additional optional keywords include:
# override_ra
# override_dec
# ta_x, ta_y: x/y pixel location (SIAF convention) of TA source
# ta2_x, ta2_y: x/y pixel location (SIAF convention) of secondary coron TA source
# ta_filt: TA filter

def testhdr(file,verbose=False,**kwargs):
    import pysiaf
    
    hdu=fits.open(file)
    hdr0=hdu[0].header
    hdr1=hdu['SCI'].header

    # Target location
    targra=hdr0['TARG_RA']
    targdec=hdr0['TARG_DEC']

    if ('override_ra' in kwargs):
        print('Overriding Target RA in header!')
        targra=kwargs['override_ra']
    if ('override_dec' in kwargs):
        print('Overriding Target DEC in header!')
        targdec=kwargs['override_dec']
    
    # Apply any proper motions
    mu_ra=hdr0['MU_RA'] # Given by FITS headers in arcsec/yr
    mu_dec=hdr0['MU_DEC'] # Given by FITS headers in arcsec/yr
    if ('MU_EPOCH' in hdr0):
        mu_epochfull=hdr0['MU_EPOCH']
        tt1=Time(mu_epochfull,format='iso', scale='utc')
        mu_epoch=tt1.to_value('decimalyear')
        mjdavg=hdr1['MJD-AVG']
        tt2=Time(mjdavg,format='mjd')
        obs_epoch=tt2.to_value('decimalyear')
        dtime=(obs_epoch-mu_epoch)
        targra += dtime*mu_ra/3600./np.cos(targdec*np.pi/180.)
        targdec += dtime*mu_dec/3600.
        
    # Boresight info
    rav1=hdr1['RA_V1']
    decv1=hdr1['DEC_V1']
    pav3=hdr1['PA_V3']

    # Pointing info
    v2ref=hdr1['V2_REF']
    v3ref=hdr1['V3_REF']
    raref=hdr1['RA_REF']
    decref=hdr1['DEC_REF']
    rollref=hdr1['ROLL_REF']

    # Make any necessary dither corrections to figure out where the target SHOULD
    # be in v2/v3
    dx=hdr0['XOFFSET']
    dy=hdr0['YOFFSET']
    # These are offsets in Ideal coordinates wrt the aperture in use
    apername=hdr0['PPS_APER']
    instr=hdr0['INSTRUME']
    
    v2nom,v3nom=mt.Idealtov2v3(dx,dy,apername,instr=instr)

    # Read in all MIRI apertures and load them into vectors of v2/v3/name
    # that we can refer to later
    siaf=pysiaf.Siaf('MIRI')
    siaf_v2=np.zeros(len(siaf))
    siaf_v3=np.zeros(len(siaf))
    siaf_name = [''  for siaf_name in np.arange(len(siaf))]
    ii=0
    for aperture_name,aperture in siaf.apertures.items():
        siaf_v2[ii]=aperture.V2Ref
        siaf_v3[ii]=aperture.V3Ref
        siaf_name[ii]=aperture.AperName
        ii+=1
    
    # If target acquisition keywords were specified, correct for
    # where the TA source was found and any filter-dep boresight offsets
    if (('ta_x' in kwargs)&('ta_y' in kwargs)&('ta_filt' in kwargs)):
        tax,tay,tafilt = kwargs['ta_x'],kwargs['ta_y'],kwargs['ta_filt']
        ta_v2act,ta_v3act=mt.xytov2v3(tax-1,tay-1,'F770W')# Because OSS only knows about F770W
        
        # Define current filter (F770W if either LRS or MRS)
        exptype=hdr0['EXP_TYPE']
        if ((exptype == 'MIR_MRS')or(exptype == 'MIR_LRS-SLITLESS')or(exptype == 'MIR_LRS')):
            curfilt='F770W'
        elif ('FILTER' in hdr0):
            curfilt=hdr0['FILTER']
        else:
            curfilt='F770W'
            
        # Find intended TA location (assuming closest MIRI aperture)
        temp_v2,temp_v3 = ta_v2act-siaf_v2, ta_v3act-siaf_v3
        dist=np.sqrt(temp_v2*temp_v2 + temp_v3*temp_v3)
        indx=np.argmin(dist)
        # And compute offset in v2,v3
        ta_v2nom,ta_v3nom=siaf_v2[indx],siaf_v3[indx]
        ta_dv2,ta_dv3=ta_v2nom-ta_v2act,ta_v3nom-ta_v3act
        # Compute filter-dep boresight offset
        bore_dv2=mt.xytov2v3(tax-1,tay-1,curfilt)[0]-mt.xytov2v3(tax-1,tay-1,tafilt)[0]
        bore_dv3=mt.xytov2v3(tax-1,tay-1,curfilt)[1]-mt.xytov2v3(tax-1,tay-1,tafilt)[1]

        # Apply to positions
        v2nom,v3nom = v2nom+ta_dv2+bore_dv2, v3nom+ta_dv3+bore_dv3
        
        if (verbose is True):
            print('Applying TA source correction.')
            print('TA source found at x,y = ',tax,tay)
            print('TA location best matches MIRI aperture ',siaf_name[indx])
            print('TA source shift dv2,dv3 = ', ta_dv2, ta_dv3)
            print('Boresight source shift dv2,dv3 = ',bore_dv2, bore_dv3)

    # If secondary TA keywords were specified (i.e., coron) then
    # correct for where source was found in secondary TA
    if (('ta2_x' in kwargs)&('ta2_y' in kwargs)):
        ta2x,ta2y = kwargs['ta2_x'],kwargs['ta2_y']
        ta_v2act,ta_v3act=mt.xytov2v3(ta2x-1,ta2y-1,'F770W')# Because OSS only knows about F770W
            
        # Find intended TA location (assuming closest MIRI aperture)
        temp_v2,temp_v3 = ta_v2act-siaf_v2, ta_v3act-siaf_v3
        dist=np.sqrt(temp_v2*temp_v2 + temp_v3*temp_v3)
        indx=np.argmin(dist)
        # And compute offset in v2,v3
        ta_v2nom,ta_v3nom=siaf_v2[indx],siaf_v3[indx]
        ta_dv2,ta_dv3=ta_v2nom-ta_v2act,ta_v3nom-ta_v3act

        # Apply to positions
        v2nom,v3nom = v2nom+ta_dv2, v3nom+ta_dv3
        
        if (verbose is True):
            print('Applying secondary TA source correction.')
            print('TA source found at x,y = ',ta2x,ta2y)
            print('TA location best matches MIRI aperture ',siaf_name[indx])
            print('TA source shift dv2,dv3 = ', ta_dv2, ta_dv3)
    
    if (verbose is True):
        print('Nominal source RA/DEC = ',targra,targdec)
        print('Nominal source V2/V3 = ',v2nom,v3nom)
        print('')
    
    # Figure out where source is in V2/V3 according to pointing info
    v2,v3,_=jwst_radectov2v3([targra],[targdec],v2ref=v2ref,v3ref=v3ref,raref=raref,decref=decref,rollref=rollref)
    dv2,dv3=v2nom-v2,v3nom-v3
    # Overall offset according to Ref data
    dv_point = np.sqrt(dv2*dv2 + dv3*dv3)
    if (verbose is True):
        print('Pointing keywords think target is at V2/V3 = ',v2,v3)
        print('Which is ',dv2,dv3,' arcsec away from nominal location')
        print('')

    # Figure out where source is in V2/V3 according to boresight info
    v2,v3,_=jwst_radectov2v3([targra],[targdec],v2ref=0.,v3ref=0.,raref=rav1,decref=decv1,rollref=pav3)
    dv2,dv3=v2nom-v2,v3nom-v3
    # Overall offset according to boresight
    dv_bore = np.sqrt(dv2*dv2 + dv3*dv3)
    if (verbose is True):
        print('Boresight keywords think target is at V2/V3 = ',v2,v3)
        print('Which is ',dv2,dv3,' arcsec away from nominal location')
        print('')

    # Figure out implied location of the guide star on FGS1
    gsra=hdr0['GS_RA']
    gsdec=hdr0['GS_DEC']
    gsv2,gsv3,_=jwst_radectov2v3([gsra],[gsdec],v2ref=v2ref,v3ref=v3ref,raref=raref,decref=decref,rollref=rollref)
    gsxsci,gsysci=mt.v2v3toIdeal(gsv2,gsv3,'FGS1_FULL_OSS',instr='FGS')
    if (verbose is True):
        print('Pointing keywords imply guide star is at XIdeal/YIdeal = ',gsxsci,gsysci)

    return dv_point,dv_bore

#############################

# Return the dither commands expected in a Visit file.
# Arguments are the v2,v3 locations from the corresponding
# APT pointing file- these will be converted to the format
# displayed in a Visit file for visual comparison.
#
# Visit files can specify dithers in two different ways.
#
# The 'dither' list at the top of a visit file is a set
# of position differences in the MIRIM_FULL_OSS Ideal
# frame executed by MIRMAIN.
#
# Mosaics (and any dithers therein) are instead handled
# by SCSAMMAIN commands that take position differences
# in the FGS1_FULL_OSS Ideal frame.

def check_visit_dithers(v2,v3):
    import pysiaf

    # Compute dither offsets in the MIRIM_FULL_OSS frame
    nloc=len(v2)
    xmirim=np.zeros(nloc)
    ymirim=np.zeros(nloc)
    for ii in range(0,nloc):
        tempx,tempy=mt.v2v3toIdeal(v2[ii],v3[ii],'MIRIM_FULL_OSS')
        xmirim[ii]=tempx
        ymirim[ii]=tempy
    dxmirim = np.diff(xmirim)
    dymirim = np.diff(ymirim)

    # If necessary keywords are set from Visit file, compute offsets
    # in the FGS1_FULL_OSS frame.  We could do this in a much more complex
    # manner, transforming all positions to guider x/y first and then
    # measuring offsets, but the differences are irrelevant (~1e-5 pixels)
    # and require way more pointing information.
    xfgs=np.zeros(nloc)
    yfgs=np.zeros(nloc)
    for ii in range(0,nloc):
        tempx,tempy=mt.v2v3toIdeal(v2[ii],v3[ii],'FGS1_FULL_OSS',instr='FGS')
        xfgs[ii]=tempx
        yfgs[ii]=tempy
    dxfgs = np.diff(xfgs)
    dyfgs = np.diff(yfgs)    

    return dxmirim,dymirim,dxfgs,dyfgs

#############################

# Check the initial pointing of a visit file by parsing it and looking
# for guide star keywords.
# Note that this can only work with PRD versions in pysiaf, NOT with
# draft PRD deliveries, against which the results must be compared by hand.

def check_visit_pointing(visitfile,scira=0.,scidec=0.):
    import pysiaf

    if ((scira == 0)or(scidec == 0)):
        print('WARNING: No science target SCIRA/SCIDEC location specified!  Using default.')

    file = open(visitfile, 'r')
    lines=file.readlines()
    nline=len(lines)

    # Find where SCSLEWMAIN is commanded
    flags=np.zeros(nline)
    for ii in range(0,nline):
        if (re.search('SCSLEWMAIN',lines[ii])):
            flags[ii]=1

    # If more than one slew option, it picks the first one
    slewline=(np.where(flags == 1))[0][0]

    # Search slew lines for guide star keywords
    entries=str.split(lines[slewline],',') + str.split(lines[slewline+1],',') + str.split(lines[slewline+2],',') + str.split(lines[slewline+3],',')
    for entry in entries:
        if re.search('GSRA',entry):
            values=str.split(entry,'=')
            gsra=float(values[1])
        if re.search('GSDEC',entry):
            values=str.split(entry,'=')
            gsdec=float(values[1])
        if re.search('GSXSCI',entry):
            values=str.split(entry,'=')
            gsxsci=float(values[1])
        if re.search('GSYSCI',entry):
            values=str.split(entry,'=')
            gsysci=float(values[1])
        if re.search('GSPA',entry):
            values=str.split(entry,'=')
            gspa=float(values[1])
        if re.search('DETECTOR',entry):
            values=str.split(entry,'=')
            fgsdet=str.strip(values[1])

    # Check that at least one of these has been set, otherwise fail out
    if (not gsra):
        print('Could not find guide star info in Visit file!')
        exit

    siaf = pysiaf.Siaf('FGS')
    if (fgsdet == 'GUIDER1'):
        fgsaper = 'FGS1_FULL_OSS'
    if (fgsdet == 'GUIDER2'):
        fgsaper = 'FGS2_FULL_OSS'
        print('Warning: FGS-2 in use, using V3IdlYAngle for FGS-1 (see JSOCINT-662)')

    # Note that regardless of whether FGS-1 or FGS-2 is being used, the gspa
    # commanded in a visit file must always be the PA appropriate for the FGS-1
    # detector since ACS only knows about FGS-1 relation to J frame
    fgsang = siaf['FGS1_FULL_OSS'].V3IdlYAngle
    GSPAV3=gspa-fgsang

    # Find where MIRTAMAIN or MIRMAIN are commanded
    # If found, this will be used to get the first relevant filter setting
    flags[:]=0
    for ii in range(0,nline):
        if (re.search('MIRTAMAIN',lines[ii])):
            flags[ii]=1
        if (re.search('MIRMAIN',lines[ii])):
            flags[ii]=1
    mirline=np.where(flags == 1)[0]
    nmir=len(mirline)
    if (nmir > 0):
        # Pick the first one as the first-commanded filter
        mirline=mirline[0]
        # Get the various entries
        entries=str.split(lines[mirline],',') + str.split(lines[mirline+1],',') + str.split(lines[mirline+2],',') + str.split(lines[mirline+3],',')
        for entry in entries:
            if re.search('FILTER',entry):
                values=str.split(entry,'=')
                filter=str.strip(values[1])
                filter=(str.split(filter,';'))[0]# Remove any semicolon

    gsv2,gsv3=mt.Idealtov2v3(gsxsci,gsysci,fgsaper,instr='FGS')
    v2,v3,_=jwst_radectov2v3([scira],[scidec],v2ref=gsv2,v3ref=gsv3,raref=gsra,decref=gsdec,rollref=GSPAV3)

    # If a MIRI filter was found, apply boresight correction to what the commanded
    # v2,v3 position would have been if F770W were in use
    pdb.set_trace()
    if filter:
        x,y=mt.v2v3toxy(v2,v3,filter)
        v2_770,v3_770=mt.xytov2v3(x,y,'F770W')
        v2,v3 = v2_770, v3_770

    print('Science target should be at (v2,v3) = ',v2,v3)

    # Read in all MIRI apertures to find the closest one
    siaf=pysiaf.Siaf('MIRI')
    allv2=np.zeros(len(siaf))
    allv3=np.zeros(len(siaf))
    allname = [''  for allname in np.arange(len(siaf))]
    ii=0
    for aperture_name,aperture in siaf.apertures.items():
        allv2[ii]=aperture.V2Ref
        allv3[ii]=aperture.V3Ref
        allname[ii]=aperture.AperName
        ii+=1

    # Find the closest aperture
    dv2,dv3 = v2-allv2, v3-allv3
    dist=np.sqrt(dv2*dv2 + dv3*dv3)
    indx=np.argmin(dist)

    print('Closest MIRI aperture is ',allname[indx], ' at (v2,v3) = ',allv2[indx],allv3[indx])
    print('Offset is ', dist[indx], ' arcsec')

    return v2,v3
