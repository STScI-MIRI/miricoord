; Project RA, DEC onto a tangent plane grid at a particular
; location.
; Largely copied from Jane's code
; Tested that this gives identical results to my own code
; (to within needed accuracy)
pro radec2xieta,crval1,crval2,ra,dec,xi,eta

    rad2arcsec = (180.0*3600.0)/!PI

    deg2rad = !PI/180.0

    ra0 = crval1*deg2rad
    dec0 = crval2*deg2rad
    radiff = ra*deg2rad - ra0;
    decr = dec*deg2rad

    h=sin(decr)*sin(dec0)+cos(decr)*cos(dec0)*cos(radiff)

    xi = cos(decr)*sin(radiff)/h
    eta = ( sin(decr)*cos(dec0) - cos(decr)*sin(dec0)*cos(radiff) )/h;

    xi = xi * rad2arcsec
    eta = eta * rad2arcsec

return
end
