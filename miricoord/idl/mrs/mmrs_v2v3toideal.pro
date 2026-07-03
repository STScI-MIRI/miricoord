;+
; NAME:
;   mmrs_v2v3toideal
;
; PURPOSE:
;   Convert V2, V3 coordinates to MRS Ideal coordinates
;
; CALLING SEQUENCE:
;   mmrs_
;
; INPUTS:
;   v2      - V2 coordinate in arcsec
;   v3      - V3 coordinate in arcsec
;
; OUTPUT:
;   XIdl    - XIdeal coordinate in arcsec
;   YIdl    - YIdeal coordinate in arcsec
;
; COMMENTS:
;   Does not do tangent-plane correction for Ideal frame; this is
;   minute effect
;
; EXAMPLES:
;
; BUGS:
;
; PROCEDURES CALLED:
;
; INTERNAL SUPPORT ROUTINES:
;
; REVISION HISTORY:
;   09-Aug-2016  Written by David Law (dlaw@stsci.edu)
;   17-Oct-2016  Input/output v2/v3 in arcsec (D. Law)
;-
;------------------------------------------------------------------------------

pro mmrs_v2v3toideal,v2,v3,xidl,yidl

; Define V2REF, V3REF at middle of 1A field (alpha=beta=0.0)
V2REF = -8.3942412d*60. ; In arcsec
V3REF = -5.3123744d*60. ; In arcsec

xidl=-(v2-V2REF)
yidl=(v3-V3REF)

return
end
