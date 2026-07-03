;+
; NAME:
;   mmrs_v2v3toab
;
; PURPOSE:
;   Convert JWST v2,v3 coordinates to MRS local alpha,beta coordinates
;
; CALLING SEQUENCE:
;   mmrs_v2v3toab,v2,v3,a,b,channel,[refdir=]
;
; INPUTS:
;   v2      - V2 coordinate in arcsec
;   v3      - V3 coordinate in arcsed
;   channel - channel name (e.g, '1A')
;
; OPTIONAL INPUTS:
;   refdir - Root directory for distortion files
;
; OUTPUT:
;   a       - Alpha coordinate in arcsec
;   b       - Beta coordinate in arcsec
;
; COMMENTS:
;   Works with CDP5 delivery files.  Inverse function is mmrs_abtov2v3.pro
;
; EXAMPLES:
;
; BUGS:
;   Current the values returned by this function are WRONG.
;
; PROCEDURES CALLED:
;
; INTERNAL SUPPORT ROUTINES:
;
; REVISION HISTORY:
;   30-Jul-2015  Written by David Law (dlaw@stsci.edu)
;   27-Oct-2015  Use real V2, V3 (D. Law)
;   24-Jan-2016  Update reference files to CDP5 (D. Law)
;   17-Oct-2016  Input/output v2/v3 in arcsec (D. Law)
;   13-Dec-2017  Update directory path for new STScI-MIRI workspace
;   10-Oct-2018  Update directory path for new miricoord structure
;-
;------------------------------------------------------------------------------

pro mmrs_v2v3toab,v2,v3,a,b,channel,refdir=refdir

if (~keyword_set(refdir)) then $
  refdir=concat_dir(ml_getenv('MIRICOORD_DIR'),'data/fits/cdp6/')

; Strip input channel into components, e.g.
; if channel='1A' then
; ch=1 and sband='A'
ch=fix(strmid(channel,0,1))
sband=strmid(channel,1,1)

; Ensure we're not using integer inputs
; and convert to XAN,YAN in units of arcmin
xan=double(v2/60.)
yan=-double(v3/60.)-7.8

; Determine input reference FITS file
case channel of
  '1A': reffile='MIRI_FM_MIRIFUSHORT_12SHORT_DISTORTION_06.04.00.fits'
  '1B': reffile='MIRI_FM_MIRIFUSHORT_12MEDIUM_DISTORTION_06.04.00.fits'
  '1C': reffile='MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_06.04.00.fits'
  '2A': reffile='MIRI_FM_MIRIFUSHORT_12SHORT_DISTORTION_06.04.00.fits'
  '2B': reffile='MIRI_FM_MIRIFUSHORT_12MEDIUM_DISTORTION_06.04.00.fits'
  '2C': reffile='MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_06.04.00.fits'
  '3A': reffile='MIRI_FM_MIRIFULONG_34SHORT_DISTORTION_06.04.00.fits'
  '3B': reffile='MIRI_FM_MIRIFULONG_34MEDIUM_DISTORTION_06.04.00.fits'
  '3C': reffile='MIRI_FM_MIRIFULONG_34LONG_DISTORTION_06.04.00.fits'
  '4A': reffile='MIRI_FM_MIRIFULONG_34SHORT_DISTORTION_06.04.00.fits'
  '4B': reffile='MIRI_FM_MIRIFULONG_34MEDIUM_DISTORTION_06.04.00.fits'
  '4C': reffile='MIRI_FM_MIRIFULONG_34LONG_DISTORTION_06.04.00.fits'
  else: begin
    print,'Invalid band'
    return
    end
endcase
reffile=concat_dir(refdir,reffile)

; Read xan,yan -> alpha,beta table
convtable=mrdfits(reffile,'XANYAN_to_albe')
; Determine which rows we need
alindex=where(strcompress(convtable.(0),/remove_all) eq 'U_CH'+channel+'_al')
beindex=where(strcompress(convtable.(0),/remove_all) eq 'U_CH'+channel+'_be')
if ((alindex lt 0)or(beindex lt 0)) then exit
; Trim to relevant al, be rows for this channel
conv_al=convtable[alindex]
conv_be=convtable[beindex]

a=conv_al.(1)+conv_al.(2)*xan + $
      conv_al.(3)*yan + conv_al.(4)*xan*yan
b=conv_be.(1)+conv_be.(2)*xan + $
      conv_be.(3)*yan + conv_be.(4)*xan*yan

return
end
