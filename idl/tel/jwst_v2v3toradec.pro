;+
; NAME:
;   jwst_v2v3toradec
;
; PURPOSE:
;   Convert v2,v3 coordinates in a JWST frame to RA,DEC coordinates
;   given a JWST attitude matrix (or relevant attitude keywords)
;   describing the relative orientation of the V3,V3 and RA,DEC
;   reference frames.  These can be derived from JWST file FITS headers.
;
;   This constructs the attitude matrix using the keywords V2_REF,
;   V3_REF, RA_REF, DEC_REF, and ROLL_REF where the first four
;   associate a fixed reference location in V2,V3 with a location in RA,DEC
;   and the ROLL_REF specifies the local roll (defined as the position
;   angle of the V3 axis measured from N towards E) of the V2,V3 coordinate
;   system at the reference location.
;
;   Note that all v2,v3 locations are given in arcseconds while all
;   RA,DEC information is given in degrees
;
;   In normal operation this function computes and uses the full JWST
;   attitude matrix; it can also be run in a /local approximation
;   that neglects the full matrix formalism for a local approximation
;   with simpler math.
;
;   The full attitude matrix calculations are based on section 6 of
;   technical report JWST-STScI-001550 'Description and Use of
;   the JWST Science Instrument Aperture File', author C. Cox.
;
; CALLING SEQUENCE:
;   jwst_v2v3toradec,v2,v3,ra,dec,hdr=hdr,V2REF=V2REF,V3REF=V3REF,$
;      RAREF=RAREF, DECREF=DECREF, ROLLREF=ROLLREF,NEWROLL=NEWROLL,/local
;
; INPUTS:
;   v2       - v2 location in arcseconds
;   v3       - v3 location in arcseconds
;
; OPTIONAL INPUTS:
;   (note that either hdr or the 5 other keywords MUST be provided)
;   hdr    - JWST type FITS header containing the V2_REF, V3_REF, 
;            RA_REF, DEC_REF, and ROLL_REF keywords
;   V2_REF - V2_REF location in arcseconds if hdr not given
;   V3_REF - V3_REF location in arcseconds if hdr not given
;   RA_REF - RA_REF location in degrees (0 -> 360) if hdr not given
;   DEC_REF - DEC_REF location in degrees (-90 -> +90) if hdr not given
;   ROLL_REF - ROLL_REF location in degrees (0 -> 360) if hdr not given
;
;   /local  - Use the local approximation algorithms
;
; OUTPUT:
;   ra      - Right ascension corresponding to v2 in degrees (0 -> 360)
;   dec     - Declination corresponding to v3 in degrees (-90 -> +90)
;
; OPTIONAL OUTPUT:
;   NEWROLL - Local roll in degrees (0 -> 360) at the location v2,v3
;
; COMMENTS:
;   The input v2,v3 locations may be provided as vectors of values, in
;   which case the output ra,dec,NEWROLL will also be vectors.  The
;   current loop to do this in the full matrix transforms is a little slow.
;
; EXAMPLES:
;
; BUGS:
;   This has not been exhaustively coded to catch all edge cases
;   (e.g., exactly at the coordinate system poles, exactly at
;   RA 360 degrees, out of bound inputs, etc.).
;
;   Note that not specifying input values makes them default to zero
;
; PROCEDURES CALLED:
;
; INTERNAL SUPPORT ROUTINES:
;   jwst_att1()
;   jwst_att2()
;   jwst_attmatrix()
;   jwst_localroll()
;
; REVISION HISTORY:
;   12-Apr-2016  Written by David Law (dlaw@stsci.edu)
;   17-Oct-2016  Deal with zero inputs, v2/v3 in arcsec (D. Law)
;-
;------------------------------------------------------------------------------

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; JWST M1 attitude matrix (V2 and V3 rotations)
; V2REF and V3REF should be in radians
function jwst_att1,V2REF,V3REF
  ; M1=  a00  a01  a02
  ;      a10  a11  a12
  ;      a20  a21  a22
  thematrix=dblarr(3,3)
  thematrix[0,0]=cos(V2REF)*cos(V3REF)
  thematrix[0,1]=sin(V2REF)*cos(V3REF)
  thematrix[0,2]=sin(V3REF)
  thematrix[1,0]=-sin(V2REF)
  thematrix[1,1]=cos(V2REF)
  thematrix[1,2]=0.d
  thematrix[2,0]=-cos(V2REF)*sin(V3REF)
  thematrix[2,1]=-sin(V2REF)*sin(V3REF)
  thematrix[2,2]=cos(V3REF)
return,thematrix
end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; JWST M2 attitude matrix (RA,DEC,ROLL rotations)
; RAREF, DECREF, ROLLREF should be in radians
function jwst_att2,RAREF,DECREF,ROLLREF
  ; M2=  a00  a01  a02
  ;      a10  a11  a12
  ;      a20  a21  a22
  thematrix=dblarr(3,3)
  thematrix[0,0]=cos(RAREF)*cos(DECREF)
  thematrix[0,1]=-sin(RAREF)*cos(ROLLREF)+cos(RAREF)*sin(DECREF)*sin(ROLLREF)
  thematrix[0,2]=-sin(RAREF)*sin(ROLLREF)-cos(RAREF)*sin(DECREF)*cos(ROLLREF)
  thematrix[1,0]=sin(RAREF)*cos(DECREF)
  thematrix[1,1]=cos(RAREF)*cos(ROLLREF)+sin(RAREF)*sin(DECREF)*sin(ROLLREF)
  thematrix[1,2]=cos(RAREF)*sin(ROLLREF)-sin(RAREF)*sin(DECREF)*cos(ROLLREF)
  thematrix[2,0]=sin(DECREF)
  thematrix[2,1]=-cos(DECREF)*sin(ROLLREF)
  thematrix[2,2]=cos(DECREF)*cos(ROLLREF)
return,thematrix
end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; JWST M = (M2 # M1) attitude matrix
; V2REF, V3REF, RAREF, DECREF, ROLLREF should be in radians
function jwst_attmatrix,V2REF,V3REF,RAREF,DECREF,ROLLREF

thematrix = jwst_att2(RAREF,DECREF,ROLLREF) # jwst_att1(V2REF,V3REF)

return,thematrix
end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; Compute the local roll (the position angle measured from N
; towards E of the V3 axis) at any V2,V3 given an attitude matrix
; V2, V3 must be in radians, result is in radians
function jwst_localroll,V2,V3,ATTMATRIX

X=-(ATTMATRIX[2,0]*cos(V2)+ATTMATRIX[2,1]*sin(V2))*sin(V3)+ATTMATRIX[2,2]*cos(V3)
Y=(ATTMATRIX[0,0]*ATTMATRIX[1,2]-ATTMATRIX[1,0]*ATTMATRIX[0,2])*cos(V2)+(ATTMATRIX[0,1]*ATTMATRIX[1,2]-ATTMATRIX[1,1]*ATTMATRIX[0,2])*sin(V2)

return,atan(Y,X)
end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; Everything is provided and returned in degrees
; Note that v2 has range 0 -> 360
; v3 has range -90 -> +90
pro jwst_v2v3toradec,v2,v3,ra,dec,hdr=hdr,V2REF=V2REF,V3REF=V3REF,RAREF=RAREF,DECREF=DECREF,ROLLREF=ROLLREF,NEWROLL=NEWROLL,local=local

; Read in attitude keywords and convert from arcseconds to radians
; If a header was provided start by using those attitude keywords
if (keyword_set(hdr)) then begin
  thisV2REF=fxpar(hdr,'V2_REF')/3600.d*!DPI/180.d
  thisV3REF=fxpar(hdr,'V3_REF')/3600.d*!DPI/180.d
  thisRAREF=fxpar(hdr,'RA_REF')*!DPI/180.d
  thisDECREF=fxpar(hdr,'DEC_REF')*!DPI/180.d
  thisROLLREF=fxpar(hdr,'ROLL_REF')*!DPI/180.d
; Case where no keywords set
endif else if ((~keyword_set(V2REF)) and (~keyword_set(V3REF)) and (~keyword_set(RAREF)) and (~keyword_set(DECREF)) and (~keyword_set(ROLLREF))) then begin
  ; Fail out
  print,'No attitude information provided!'
  return
endif else begin
  ; If individual keywords not set (or zero), default to zero
  if (~keyword_set(V2REF)) then V2REF=0.d
  if (~keyword_set(V3REF)) then V3REF=0.d
  if (~keyword_set(RAREF)) then RAREF=0.d
  if (~keyword_set(DECREF)) then DECREF=0.d
  if (~keyword_set(ROLLREF)) then ROLLREF=0.d  
  ; If attitude keywords were provided, use them
  thisV2REF=V2REF/3600.d*!DPI/180.d
  thisV3REF=V3REF/3600.d*!DPI/180.d
  thisRAREF=RAREF*!DPI/180.d
  thisDECREF=DECREF*!DPI/180.d
  thisROLLREF=ROLLREF*!DPI/180.d
endelse
 
; If running in /local mode, use the local approximate transform
if (keyword_set(local)) then begin
  dv2=(v2/3600.d*!PI/180.-thisV2REF)*cos(thisV3REF) ; Offset from V2REF in radians
  dv3=v3/3600.d*!PI/180.-thisV3REF ; Offset from V3REF in radians
  dra=dv2*cos(thisROLLREF)+dv3*sin(thisROLLREF) ; Offset from RAREF in radians
  ddec=-dv2*sin(thisROLLREF)+dv3*cos(thisROLLREF) ; Offset from DECREF in radians
  ra=(thisRAREF+dra/cos(thisDECREF))*180.d/!DPI ; New RA in degrees
  dec=(thisDECREF+ddec)*180.d/!DPI ; New DEC in degrees
  NEWROLL=thisROLLREF*180.d/!DPI ; New roll (identical to old) in degrees
; If running in normal mode, use the full attitude matrix transform
endif else begin
  ; Compute the JWST attitude matrix from the 5 attitude keywords
  attmat=jwst_attmatrix(thisV2REF,thisV3REF,thisRAREF,thisDECREF,thisROLLREF)

  ; Make empty vectors to hold the output ra,dec,NEWROLL
  ra=dblarr(n_elements(v2))
  dec=dblarr(n_elements(v2))
  NEWROLL=dblarr(n_elements(v2))

  ; If the input was a vector, loop over elements in the simplest way
  for i=0L,n_elements(v2)-1 do begin
    ; Compute the vector describing the input location
    invector=[cos(v2[i]/3600.d*!DPI/180.d)*cos(v3[i]/3600.d*!DPI/180.d),sin(v2[i]/3600.d*!DPI/180.d)*cos(v3[i]/3600.d*!DPI/180.d),sin(v3[i]/3600.d*!DPI/180.d)]

    ; Compute the output vector (cos(RA)cos(dec),sin(RA)cos(dec),sin(dec))
    ; by applying the attitude matrix
    outvector=attmat # invector

    ; Split the output vector into RA and DEC components and convert
    ; back to degrees
    ra[i]=atan(outvector[1],outvector[0])*180.d/!DPI
    ; Ensure 0-360 degrees
    if (ra[i] lt 0.d) then ra[i]=ra[i]+360.d
    dec[i]=asin(outvector[2])*180.d/!DPI

    ; Compute the local roll at this location and convert
    ; back to degrees
    NEWROLL[i]=jwst_localroll(v2[i]/3600.d*!DPI/180.d,v3[i]/3600.d*!DPI/180.d,attmat)*180.d/!DPI
    ; Ensure 0-360 degrees
    if (NEWROLL[i] lt 0.d) then NEWROLL[i]=NEWROLL[i]+360.d
  endfor
endelse

return
end
