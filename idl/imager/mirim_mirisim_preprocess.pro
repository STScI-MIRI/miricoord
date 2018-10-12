;+
; NAME:
;   mirim_mirisim_preprocess
;
; PURPOSE:
;   Convert mirisim output files to a format that can be further processed
;   by the JWST pipeline.
;
;   This is a simple port from the MRS version- all it does is modify
;   the relevant already-existing header keywords.  Note that this
;   function will be obsolete in a future update of mirisim that
;   correct addresses v2,v3ref coordinates
;
;   Added in an optional call to hack the absolute pointing of
;   exposures so that the first one points at a given ra,dec.
;
; CALLING SEQUENCE:
;   mmrs_mirisim_preprocess,directory
;
; INPUTS:
;   none-prompts for files
;
; OPTIONAL INPUTS:
;   ra0: Desired RA of the reference point in 1st exposure (decimal degrees)
;   dec0: Desired DEC of the reference point in 1st exposure (decimal degrees)
;
; OUTPUT:
;   Outputs are the input files cloned into the new directory with
;   modified headers.

; OPTIONAL OUTPUT:
;
; COMMENTS:
;
; EXAMPLES:
;
; BUGS:
;   This program will start failing very near the poles- it
;   isn't designed to handle these edge cases where different
;   exposures might be on different sides of the pole.  It also
;   doesn't handle badly-formatted input- it's a quick kludge for testing.
;
; PROCEDURES CALLED:
;
; INTERNAL SUPPORT ROUTINES:
;
; REVISION HISTORY:
;   11-Apr-2016  Ported from similar MRS code by David Law (dlaw@stsci.edu)
;-
;------------------------------------------------------------------------------

pro mirim_mirisim_preprocess,ra0=ra0,dec0=dec0

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

; Copy files to output directory
outfiles=outdir+fileandpath(files)
for i=0,nfiles-1 do begin
  spawn, 'cp '+files[i]+' '+outfiles[i]
  ; Ensure new files are not gzipped
  spawn, 'gunzip '+outfiles[i]
endfor

; Look in the first file to see if we have header keywords
hdr=headfits(files[0])
temp=sxpar(hdr,'V3_REF',count=count)
; If not, fail out, we're not dealing with that case of really old data
if (count eq 0) then begin
  print,'Error- correct keywords not found!  Quitting.'
  exit
endif

v2ref=dblarr(nfiles)
v3ref=dblarr(nfiles)
raref=dblarr(nfiles)
decref=dblarr(nfiles)
rollref=dblarr(nfiles)

; Loop over files the first time to determine new keyword values
for i=0,nfiles-1 do begin
  hdr=headfits(files[i])

  ; Get the header
  temp=sxpar(hdr,'V3_REF',count=count)

  ; If there are header keywords, process them
  ; Translate from Oct 2016 mirisim header keywords
  ; to my conventional keywords.  Mirisim currently uses
  ; an odd frame with reference point nearly but not quite XAN,YAN
  ; that is flipped in the DEC direction.
  ; Hard code this for now in a way that will probably fail for 
  ; ROLL ne 0
  temp1=fxpar(hdr,'V2_REF')*60.
  temp2=fxpar(hdr,'V3_REF')*60.-7.8*60.
  temp3=fxpar(hdr,'RA_REF')
  temp4=-fxpar(hdr,'DEC_REF')
  temp5=fxpar(hdr,'ROLL_REF')

  ; Actual MIRIM reference point- hard code it properly
  v2ref[i] = -453.363 ; In arcsec
  v3ref[i] = -374.069 ; In arcsec
  ; Use the coordinate transform infrastructure to map provided 'bad'
  ; location to real 'good' location
  jwst_v2v3toradec,v2ref[i],v3ref[i],ra,dec,V2REF=temp1,V3REF=temp2,RAREF=temp3,DECREF=temp4,ROLLREF=temp5

  raref[i]=ra[0]
  decref[i]=dec[0]
  rollref[i]=temp5
endfor

; Do we need to further hack the headers to shift the entire set of pointings?
; If so, do this in a super-simple way that REQUIRES zero roll
if ((keyword_set(ra0))and(keyword_set(dec0))) then begin
  basera=raref[0]
  basedec=decref[0]
  for i=0,nfiles-1 do begin
    dra=(raref[i]-basera)*cos(basedec*!PI/180.)
    ddec=decref[i]-basedec

    newra=ra0+dra/cos(dec0*!PI/180.)
    newdec=dec0+ddec

    if (newra gt 360.d) then newra=newra-360.d
    if (newra lt 0.d) then newra=newra+360.d

    raref[i]=newra
    decref[i]=newdec
  endfor
endif

for i=0,nfiles-1 do begin
  ; Overwrite these keywords into header
  hdr=headfits(files[i])
  fxaddpar,hdr,'V2_REF',v2ref[i]
  fxaddpar,hdr,'V3_REF',v3ref[i]
  fxaddpar,hdr,'ROLL_REF',rollref[i]
  fxaddpar,hdr,'RA_REF',raref[i]
  fxaddpar,hdr,'DEC_REF',decref[i]

  ; Modify 0th extension headers
  modfits,outfiles[i],0,hdr,exten_no=0
endfor


return
end
