;+
; NAME:
;   mmrs_mirisim_preprocess
;
; PURPOSE:
;   Convert mirisim output files to a format that can be further processed
;   by DRL cube building code or the JWST pipeline.
;
;   Originally, this meant digging through the logs and mirisim code
;   structure to put the WCS reference keywords in the headers.
;   Later this information was already in the headers, but was
;   incorrectly defined so had to be tweaked.
;
;   Puts the results in a preproc/ directory.
;
;   Note that this code originally changed the FORMAT of the data too,
;   applying a very simple minded ramps-to-slopes module.  It no
;   longer does so.
;
; CALLING SEQUENCE:
;   mmrs_mirisim_preprocess
;
; INPUTS:
;   none- prompts for files
;
; OPTIONAL INPUTS:
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
;   The dither-table dependent method hasn't been checked recently.
;
; PROCEDURES CALLED:
;
; INTERNAL SUPPORT ROUTINES:
;   mmrs_mirisim_dloc()
;
; REVISION HISTORY:
;   Early 2016  Written by David Law (dlaw@stsci.edu)
;   17-Oct-2016  Use coordinate keywords in header
;   10-Nov-2016  Remove format changes, use arcsec for v2,v3, general overhaul
;-
;------------------------------------------------------------------------------

; Convert a nominal dither offset in alpha,beta
; to offset in ra, dec
pro mmrs_mirisim_dloc,aoff,boff,ROLLREF,raoff,decoff

; Convert alpha,beta offsets to v2, v3
mmrs_abtov2v3,0.,0.,v2ref,v3ref,'1A'
mmrs_abtov2v3,aoff,boff,v2off,v3off,'1A'
v2off=-(v2off-v2ref)
v3off=-(v3off-v3ref)

raoff=v2off*cos(ROLLREF*!PI/180.)+v3off*sin(ROLLREF*!PI/180.)
decoff=-v2off*sin(ROLLREF*!PI/180.)+v3off*cos(ROLLREF*!PI/180.)

return
end

;------------------------------------------------------------------------------

pro mmrs_mirisim_preprocess,oldwcs=oldwcs

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
; If not, read in reference dither table
if (count eq 0) then begin
  ditherfile=concat_dir(getenv('MIRISIM_DIR'),'obssim/data/mrs_recommended_dither.dat')
  ; Table of possible dither positions (2 x N)
  dithertable=(read_ascii(ditherfile)).field1

  ; Read the log file to pull out key information
  logfile=concat_dir(directory,'mirisim.log')
  openr,lun,logfile,/get_lun
  line=''
  while not eof(lun) do begin
    ; Read one line
    readf,lun,line
    ; Look for a line of info about the dither pattern start index (1-indexed)
    a=stregex(line,'Starting index of executed dither pattern:')
    if (a ge 0) then begin
      dstart=fix((strsplit(line,'Starting index of executed dither pattern:',/extract,/regex))[1])-1 ;0-indexed
      break
    endif
  endwhile
  close,lun
  free_lun,lun
  ; Pretend we're pointing at RA=45 degrees, dec=0 degrees
  ; with local roll 0
  razp=45.D
  deczp=0.D
  ROLLREF=0.D
  ; Define V2REF, V3REF at middle of 1A field (alpha=beta=0.0)
  V2REF = -8.3942412d*60 ; In arcsec
  V3REF = -5.3123744d*60 ; In arcsec
endif


; Loop over files
for i=0,nfiles-1 do begin
  hdr=headfits(files[i])

  ; Do the headers contain the WCS keywords?
  temp=sxpar(hdr,'V3_REF',count=count)

  ; If no header keywords, dig them up the old way
  if (count eq 0) then begin
    ditherno=dstart+fxpar(hdr,'PNTG_SEQ')-1; Add starting point, convert to 0 indexed
    ; Add dither offset keywords.
    aoffset=dithertable[0,ditherno]; In alpha
    boffset=dithertable[1,ditherno]; In beta
    ; Convert alpha,beta offsets to ra,dec
    mmrs_mirisim_dloc,aoffset,boffset,ROLLREF,dra,ddec
    ; Figure out corresponding RA,DEC location
    ; (subtract because we're pretending that the telescope moved
    ; instead of the point source)
    RAREF=razp-dra/cos(deczp*!PI/180.)/3600.D
    DECREF=deczp-ddec/3600.D
  endif else begin   ; If there are header keywords, process them
    ; Translate from Oct 2016 mirisim header keywords
    ; to my conventional keywords.  Mirisim currently uses
    ; an odd frame with reference point nearly but not quite XAN,YAN
    ; that is flipped in the DEC direction.
    ; Hard code this for now in a way that will probably fail for 
    ; ROLL ne 0
    ;temp1=-1.138;fxpar(hdr,'V2_REF')*60.
    ;temp2=-468.363;fxpar(hdr,'V3_REF')*60.-7.8*60.
    temp1=fxpar(hdr,'V2_REF')*60.
    temp2=fxpar(hdr,'V3_REF')*60.-7.8*60.
    temp3=fxpar(hdr,'RA_REF')
    temp4=-fxpar(hdr,'DEC_REF')
    temp5=fxpar(hdr,'ROLL_REF')

    V2REF = -8.3942412d*60. ; In arcsec
    V3REF = -5.3123744d*60. ; In arcsec
    jwst_v2v3toradec,V2REF,V3REF,ra,dec,V2REF=temp1,V3REF=temp2,RAREF=temp3,DECREF=temp4,ROLLREF=temp5

    RAREF=ra[0]
    DECREF=dec[0]
    ROLLREF=temp5
  endelse

  ; Add (or overwrite) these keywords into header
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
