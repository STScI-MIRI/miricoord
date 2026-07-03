;+
; NAME:
;   mmrs_abtov2v3_cdp8b
;
; PURPOSE:
;   Convert MRS local alpha,beta coordinates to JWST v2,v3 coordinates
;
; CALLING SEQUENCE:
;   mmrs_abtov2v3_cdp8b,a,b,v2,v3,channel,[refdir=]
;
; INPUTS:
;   a       - Alpha coordinate in arcsec
;   b       - Beta coordinate in arcsec
;   channel - channel name (e.g, '1A')
;
; OPTIONAL INPUTS:
;   refdir - Root directory for distortion files
;
; OUTPUT:
;   v2      - V2 coordinate in arcsec
;   v3      - V3 coordinate in arcsec
;
; COMMENTS:
;   Works with CDP8b delivery files.  Inverse function is mmrs_v2v3toab_cdp8b.pro
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
;   30-Jul-2015  Written by David Law (dlaw@stsci.edu)
;   27-Oct-2015  Add conversion to REAL V2,V3 (D. Law)
;   16-Nov-2015  Add conversion to XAN,YAN (D. Law)
;   24-Jan-2016  Update reference files to CDP5 (D. Law)
;   17-Oct-2016  Input/output v2/v3 in arcsec (D. Law)
;   13-Dec-2017  Update directory path for new STScI-MIRI workspace
;   10-Oct-2018  Update directory path for new miricoord structure
;   26-Apr-2019  Update to CDP-8b
;-
;------------------------------------------------------------------------------

pro mmrs_abtov2v3_cdp8b,a,b,v2,v3,channel,refdir=refdir,xan=xan,yan=yan

if (~keyword_set(refdir)) then $
  refdir=concat_dir(ml_getenv('MIRICOORD_DIR'),'data/fits/cdp8b/')

; Strip input channel into components, e.g.
; if channel='1A' then
; ch=1 and sband='A'
ch=fix(strmid(channel,0,1))
sband=strmid(channel,1,1)

; Ensure we're not using integer inputs
adbl=double(a)
bdbl=double(b)

; Determine input reference FITS file
case channel of
  '1A': reffile='MIRI_FM_MIRIFUSHORT_12SHORT_DISTORTION_8B.05.01.fits'
  '1B': reffile='MIRI_FM_MIRIFUSHORT_12MEDIUM_DISTORTION_8B.05.01.fits'
  '1C': reffile='MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_8B.05.01.fits'
  '2A': reffile='MIRI_FM_MIRIFUSHORT_12SHORT_DISTORTION_8B.05.01.fits'
  '2B': reffile='MIRI_FM_MIRIFUSHORT_12MEDIUM_DISTORTION_8B.05.01.fits'
  '2C': reffile='MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_8B.05.01.fits'
  '3A': reffile='MIRI_FM_MIRIFULONG_34SHORT_DISTORTION_8B.05.01.fits'
  '3B': reffile='MIRI_FM_MIRIFULONG_34MEDIUM_DISTORTION_8B.05.01.fits'
  '3C': reffile='MIRI_FM_MIRIFULONG_34LONG_DISTORTION_8B.05.01.fits'
  '4A': reffile='MIRI_FM_MIRIFULONG_34SHORT_DISTORTION_8B.05.01.fits'
  '4B': reffile='MIRI_FM_MIRIFULONG_34MEDIUM_DISTORTION_8B.05.01.fits'
  '4C': reffile='MIRI_FM_MIRIFULONG_34LONG_DISTORTION_8B.05.01.fits'
  else: begin
    print,'Invalid band'
    return
    end
endcase
reffile=concat_dir(refdir,reffile)

; Read alpha,beta -> V2,V3 table
convtable=mrdfits(reffile,'albe_to_V2V3')
; Determine which rows we need
v2index=where(strcompress(convtable.(0),/remove_all) eq 'T_CH'+channel+'_V2')
v3index=where(strcompress(convtable.(0),/remove_all) eq 'T_CH'+channel+'_V3')
if ((v2index lt 0)or(v3index lt 0)) then exit
; Trim to relevant v2, v3 rows for this channel
conv_v2=convtable[v2index]
conv_v3=convtable[v3index]

v2=conv_v2.(1)+conv_v2.(2)*adbl +conv_v2.(3)*adbl*adbl + $
   conv_v2.(4)*bdbl + conv_v2.(5)*adbl*bdbl + conv_v2.(6)*adbl*adbl*bdbl + $
   conv_v2.(7)*bdbl*bdbl + conv_v2.(8)*adbl*bdbl*bdbl + conv_v2.(9)*adbl*adbl*bdbl*bdbl
v3=conv_v3.(1)+conv_v3.(2)*adbl + conv_v3.(3)*adbl*adbl + $
   conv_v3.(4)*bdbl + conv_v3.(5)*adbl*bdbl+ conv_v3.(6)*adbl*adbl*bdbl + $
   conv_v3.(7)*bdbl*bdbl + conv_v3.(8)*adbl*bdbl*bdbl+ conv_v3.(9)*adbl*adbl*bdbl*bdbl

return
end
