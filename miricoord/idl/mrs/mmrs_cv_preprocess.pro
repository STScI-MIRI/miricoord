;+
; NAME:
;   mmrs_cv_preprocess
;
; PURPOSE:
;   Add WCS keywords to CV ground test data.  Note that while early
;   versions of this code originally changed the FORMAT of the data too,
;   applying a very simple minded ramps-to-slopes module.  It no
;   longer does so.
;
;   Note also that the WCS keyword addition is unnecessary if
;   Jane's code to do the same thing has already been run.
;
; CALLING SEQUENCE:
;   mmrs_cv_preprocess
;
; INPUTS:
;   directory - cv input directory
;
; OPTIONAL INPUTS:
;
; OUTPUT:
;   Outputs are the input files cloned into the new directory with
;   modified headers.

; OPTIONAL OUTPUT:
;
; COMMENTS:
;   Works with CV2 and CV3 data.  Used to be CV3-specific.
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
;   Early 2016   Written by David Law (dlaw@stsci.edu)
;   07-Mar-2017  Overhaul to only modify header information
;-
;------------------------------------------------------------------------------

pro mmrs_cv_preprocess,outdir=outdir

; Select input files to process
files = dialog_pickfile( title='Read Files to Process', $
                                     filter='*.fits', $
                                     get_path=rootdir, $
                                     /MUST_EXIST        , $
                                     /MULTIPLE_FILES)
nfiles=n_elements(files)

; Default output directory
if (~keyword_set(outdir)) then outdir=concat_dir(rootdir,'preproc/')
; Ensure output directory exists
if file_test(outdir,/directory) eq 0 then spawn, '\mkdir -p '+outdir

; Pretend we're pointing at RA=45 degrees, dec=0 degrees
; with local roll 0
razp=45.D
deczp=0.D
ROLLREF=0.D

; Define V2REF, V3REF at middle of 1A field (alpha=beta=0.0)
V2REF = -8.3942412d*60. ; In arcsec
V3REF = -5.3123744d*60. ; In arcsec

; Copy files to output directory
outfiles=outdir+fileandpath(files)
for i=0,nfiles-1 do begin
  spawn, 'cp '+files[i]+' '+outfiles[i]
  ; Ensure new files are not gzipped
  spawn, 'gunzip '+outfiles[i]
endfor

; Loop over files
for i=0,nfiles-1 do begin
  hdr=headfits(files[i])

  ; Derive the offsets from the XACTPOS, YACTPOS in the headers
  osim_xact=fxpar(hdr,'XACTPOS')
  osim_yact=fxpar(hdr,'YACTPOS')
  ; Convert to V2,V3 positions of the OSIM point source
  v2=osim_xact*60.
  v3=-(osim_yact+7.8)*60.
  ; Calculate the deltav2 and deltav3 relative to the reference point
  dv2=(v2-V2REF)/3600.*cos(V3REF/60.*!PI/180.); Arc offset in degrees
  dv3=(v3-V3REF)/3600.; Arc offset in degrees
  dra=dv2*cos(ROLLREF*!PI/180.)+dv3*sin(ROLLREF*!PI/180.)
  ddec=-dv2*sin(ROLLREF*!PI/180.)+dv3*cos(ROLLREF*!PI/180.)
  ; Figure out corresponding RA,DEC location
  ; (subtract because we're pretending that the telescope moved
  ; instead of the point source?)
  RAREF=razp-dra/cos(deczp*!PI/180.)
  DECREF=deczp-ddec

  ; Add dither offset keywords
  fxaddpar,hdr,'V2_REF',V2REF
  fxaddpar,hdr,'V3_REF',V3REF
  fxaddpar,hdr,'ROLL_REF',ROLLREF
  fxaddpar,hdr,'RA_REF',RAREF
  fxaddpar,hdr,'DEC_REF',DECREF

  ; Modify 0th extension headers
  modfits,outfiles[i],0,hdr,exten_no=0
endfor


return
end
