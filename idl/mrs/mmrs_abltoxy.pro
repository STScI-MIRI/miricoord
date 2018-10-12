;+
; NAME:
;   mmrs_abltoxy
;
; PURPOSE:
;   Convert MRS local coordinates to MRS detector coordinates.
;   Convention is that x,y pixel locations follow the JWST pipeline
;   convention where the detector has 1032x1024 pixels and (0,0) is
;   the middle of the lower left detector pixel.
;
; CALLING SEQUENCE:
;   mmrs_abltoxy,a,b,l,x,y,channel,[phase=,slicenum=,slicename=,refdir=,/trim]
;
; INPUTS:
;   a       - Alpha coordinate in arcsec
;   b       - Beta coordinate in arcsec
;   l       - Lambda coordinate in microns
;   channel - channel name (e.g, '1A')
;
; OPTIONAL INPUTS:
;   refdir - Root directory for distortion files
;   /trim  - Return only valid pixels on the detector

;
; OUTPUT:
;   x      - X coordinate in 0-indexed pixels
;   y      - Y coordinate in 0-indexed pixels
;
; OPTIONAL OUTPUT:
;   slicenum - Slice number (e.g., 11)
;   slicename - Slice name (e.g., 211A for ch2, slice 11, sub-band A)
;   phase - Give pixel/slice phase for a given input point
;
; COMMENTS:
;   Works with CDP6 delivery files.  Inverse function is mmrs_xytoabl.pro
;   Not all input a,b,l can map to x,y because some may fall outside
;   the FOV.  x,y for these are set to -999.
;
;   CDP transforms provided by A. Glauser assume a 1-indexed detector
;   frame convention, whereas this code uses the JWST pipeline
;   convention that (0,0) is the middle of the lower-left detector pixel.
;   Therefore this code also does the 1- to 0-indexed transform after
;   calling the Glauser distortions.
;
; EXAMPLES:
;
; BUGS: We should really do something smart enough to figure out
; channel based on the given lambda, but that's actually hard
; b/c of the wavelength overlaps, so we'll require it to be provided.
;
; PROCEDURES CALLED:
;
; INTERNAL SUPPORT ROUTINES:
;
; REVISION HISTORY:
;   09-Feb-2017  Written by David Law (dlaw@stsci.edu)
;   07-Jun-2017  Update to use x,y in 0-indexed detector frame to
;                match python code
;   13-Dec-2017  Update directory path for new STScI-MIRI workspace
;   10-Oct-2018  Update directory path for new miricoord structure
;-
;------------------------------------------------------------------------------

pro mmrs_abltoxy,a,b,l,x,y,channel,slicenum=slicenum,slicename=slicename,refdir=refdir,trim=trim,phase=phase

if (~keyword_set(refdir)) then $
  refdir=concat_dir(ml_getenv('MIRICOORD_DIR'),'data/fits/cdp6/')

; Strip input channel into components, e.g.
; if channel='1A' then
; ch=1 and sband='A'
ch=fix(strmid(channel,0,1))
sband=strmid(channel,1,1)

; Ensure we're not using integer inputs
adbl=double(a)
bdbl=double(b)
ldbl=double(l)

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

; Read global header
hdr=headfits(reffile)
; Get wavelength ranges from header
wmin=fxpar(hdr,'L_MIN'+strcompress(string(ch),/remove_all))
wmax=fxpar(hdr,'L_MAX'+strcompress(string(ch),/remove_all))
; If input l was a single negative, replace with midpoint wavelength
if ((n_elements(ldbl) eq 1)and(ldbl[0] < 0.)) then $
  ldbl=replicate((wmin+wmax)/2.,n_elements(adbl))

; Get beta zeropoint and spacing from header
beta0=fxpar(hdr,'B_ZERO'+strcompress(string(ch),/remove_all))
dbeta=fxpar(hdr,'B_DEL'+strcompress(string(ch),/remove_all))

c2d_x=mrdfits(reffile,'X_CH'+strcompress(string(ch),/remove_all))
c2d_y=mrdfits(reffile,'Y_CH'+strcompress(string(ch),/remove_all))

fovalpha=mrdfits(reffile,'FoV_CH'+strcompress(string(ch),/remove_all))

; Determine slices
slicefloat=(bdbl-beta0)/dbeta + 1
slicenum=round(slicefloat)
slicename=strcompress(string(slicenum+ch*100)+sband,/remove_all)
c2d_slice=mrdfits(reffile,'Slice_Number')
if (ch eq 1) then c2d_slice=c2d_slice[0:507,*]-100
if (ch eq 2) then c2d_slice=c2d_slice[508:*,*]-200
if (ch eq 4) then c2d_slice=c2d_slice[0:492,*]-400
if (ch eq 3) then c2d_slice=c2d_slice[492:*,*]-300
smax=max(c2d_slice[where(c2d_slice ge 0)])
smin=min(c2d_slice[where(c2d_slice ge 0)])
badslice=where((slicenum lt smin)or(slicenum gt smax),nbads)
if (nbads ne 0) then begin
 slicenum[badslice]=-999
 slicename[badslice]='-999'
 slicefloat[badslice]=-999.
endif

; Define an index0 where slice number is physical
index0=where((slicenum gt 0)and(slicenum le 99),nindex0)

; Initialize x,y to -999.
; (they will only be set to something else if the pixel
; lands on a valid slice)
x=replicate(-999.,n_elements(slicenum))
y=replicate(-999.,n_elements(slicenum))

; Loop over number
epsilon=0.01
for k=0L,nindex0-1 do begin
  thisxcoeff=c2d_x[slicenum[index0[k]]-1] ; X coefficients for this slice
  thisycoeff=c2d_y[slicenum[index0[k]]-1] ; Y coefficients for this slice

  thisx=0.d
  thisy=0.d
  ; Check that input alpha is within epsilon of field boundary for this slice
  if(((adbl[index0[k]]+epsilon) ge fovalpha[slicenum[index0[k]]-1].alpha_min)and((adbl[index0[k]]-epsilon) le fovalpha[slicenum[index0[k]]-1].alpha_max)) then begin
    for i=0,4 do begin
      for j=0,4 do begin
         coind=1+(i*5)+j
         term1=((ldbl[index0[k]]-thisxcoeff.(0))^i)
         term2=(adbl[index0[k]]^j)
         termall=thisxcoeff.(coind)*term1*term2
        thisx += thisxcoeff.(coind)*(((ldbl[index0[k]]-thisxcoeff.(0))^i)   * (adbl[index0[k]]^j))
        thisy += thisycoeff.(coind)*(((ldbl[index0[k]]-thisycoeff.(0))^i)   * (adbl[index0[k]]^j))
      endfor
    endfor
    x[index0[k]]=thisx
    y[index0[k]]=thisy
  endif
endfor

; Look for wherever x,y = -999 those are failure cases
index0=where((x ne -999.)and(y ne -999.),nindex0)

; Determine slice[0] and pixel[1] phase
; 0 is in the middle of a sample, -0.5 at the bottom edge, 0.5 at the
; top edge
phase=fltarr(n_elements(slicenum),2)
phase[*,*]=-999
phase[index0,0]=slicefloat[index0]-round(slicefloat[index0])
phase[index0,1]=x[index0]-round(x[index0])
stop
; Trim to only valid-slice pixels if so desired
if (keyword_set(trim)) then begin
  x=x[index0]
  y=y[index0]
  phase=phase[index0,*]
endif

; Transform from 1-indexed to 0-indexed values
x=x-1
y=y-1

return
end
