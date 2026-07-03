;+
; NAME:
;   mmrs_siaf
;
; PURPOSE:
;   Figure out the siaf locations for MRS slices based on reference
;   files.  Output as both text file and Yanny par file.
;
; CALLING SEQUENCE:
;   mrs_siaf,channel
;
; INPUTS:
;   channel - channel name (e.g, '1A')
;
; OPTIONAL INPUTS:
;   rootdir - Root directory for distortion files
;   outdir - Directory for output files to go into
;
; OUTPUT:
;   siaf_[channel].txt  Slicer corner coordinates
;   siaf_[channel]ab.ps  Plot of alpha,beta corner coordinates
;   siaf_[channel]v2v3.ps  Plot of v2,v3 corner coordinates
;
; COMMENTS:
;   Works with CDP6 delivery files.
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
;   27-Oct-2015  Use library routines (D. Law)
;   24-Jan-2016  Update to CDP5 (D. Law)
;   13-Sep-2016  Fix bug in slice indexing (D. Law)
;   30-Sep-2016  Add Yanny par file option (D. Law)
;   17-Oct-2016  Input/output v2/v3 in arcsec (D. Law)
;   14-Dec-2017  Adapt for new github/central store interaction (D. Law)
;-
;------------------------------------------------------------------------------

pro mmrs_siaf,channel,rootdir=rootdir,outdir=outdir

if (~keyword_set(rootdir)) then $
  rootdir=concat_dir(ml_getenv('MIRICOORD_DIR'),'data/fits/cdp6/')

; Write to a /temp subdirectory to be sure we don't accidentally
; overwrite something in a dated subdirectory!
if (~keyword_set(outdir)) then $
  outdir=concat_dir(ml_getenv('MIRICOORD_DATA_DIR'),'siaf/mrs/temp/')

; Strip input channel into components, e.g.
; if channel='1A' then
; ch=1 and sband='A'
channel=strupcase(channel)
ch=fix(strmid(channel,0,1))
sband=strmid(channel,1,1)

; Determine output files to put the results in
outfile=outdir+'siaf_'+channel+'.txt'
outpar=outdir+'siaf_'+channel+'.par'
openw,lun,outfile,/get_lun,width=450

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
reffile=concat_dir(rootdir,reffile)

; Read global header
hdr=headfits(reffile)
; Get beta zeropoint and spacing from header
beta0=fxpar(hdr,'B_ZERO'+strcompress(string(ch),/remove_all))
dbeta=fxpar(hdr,'B_DEL'+strcompress(string(ch),/remove_all))

; Read FoV alpha boundaries
extname='FoV_CH'+strcompress(string(ch),/remove_all)
alphalimits=mrdfits(reffile,extname)

; Determine number of slices
nslices=n_elements(alphalimits)
; Create a 1-indexed vector of slice numbers and slice names
; (the names will be of the form 112A for ch 1, slice 12,
; band A)
slicenum=indgen(nslices)+1
slicename=string(ch*100+slicenum)+sband

; Figure out beta boundaries of each slice
beta1=beta0+(slicenum-1.5)*dbeta; Lower bound
beta2=beta1+dbeta; Upper bound

; Figure out central reference point locations of each slice
slice_beta_ref=(beta1+beta2)/2.
slice_alpha_ref=replicate(0.,nslices)

; Convert from our list of maximum and minimum alpha,beta
; to actual corner coordinates for each slice
alpha_corners=fltarr(4,nslices)
beta_corners=fltarr(4,nslices)
; Order is lower-left, upper-left, upper-right, lower-right
for i=0,nslices-1 do begin
  alpha_corners[0,i]=alphalimits[i].(0)
  alpha_corners[1,i]=alphalimits[i].(0)
  alpha_corners[2,i]=alphalimits[i].(1)
  alpha_corners[3,i]=alphalimits[i].(1)
  beta_corners[0,i]=beta1[i]
  beta_corners[1,i]=beta2[i]
  beta_corners[2,i]=beta2[i]
  beta_corners[3,i]=beta1[i]
endfor

; Compute corner coordinates for an inscribed footprint
inscr_alpha=fltarr(4)
inscr_beta=fltarr(4)
inscr_alpha[0]=max(alpha_corners[0,*])
inscr_alpha[1]=max(alpha_corners[1,*])
inscr_alpha[2]=min(alpha_corners[2,*])
inscr_alpha[3]=min(alpha_corners[3,*])
inscr_beta[0]=min(beta_corners)
inscr_beta[1]=max(beta_corners)
inscr_beta[2]=max(beta_corners)
inscr_beta[3]=min(beta_corners)

; Figure out central reference point locations of the inscribed footprint
; Fix it to alpha=beta=0.
beta_ref=0.
alpha_ref=0.

; Convert to v2,v3 reference points
mmrs_abtov2v3,alpha_ref,beta_ref,v2_ref,v3_ref,channel,refdir=refdir
mmrs_abtov2v3,slice_alpha_ref,slice_beta_ref,slice_v2_ref,slice_v3_ref,channel,refdir=refdir
; Convert to v2,v3 corner coordinates
mmrs_abtov2v3,alpha_corners,beta_corners,v2_corners,v3_corners,channel,refdir=refdir
; Convert to v2,v3 inscribed box
mmrs_abtov2v3,inscr_alpha,inscr_beta,inscr_v2,inscr_v3,channel,refdir=refdir

  ; Print to a Yanny par file
  siaf_line=create_struct(name='SIAF', $
    'BAND', channel, $
    'SliceName', channel, $
    'SliceNum', 0, $
    'alpha_ref', alpha_ref, $
    'beta_ref', beta_ref, $
    'v2_ref', v2_ref, $
    'v3_ref', v3_ref, $
    'alpha_ll', inscr_alpha[0], $
    'beta_ll', inscr_beta[0], $
    'v2_ll', inscr_v2[0], $
    'v3_ll', inscr_v3[0], $
    'alpha_ul', inscr_alpha[1], $
    'beta_ul', inscr_beta[1], $
    'v2_ul', inscr_v2[1], $
    'v3_ul', inscr_v3[1], $
    'alpha_ur', inscr_alpha[2], $
    'beta_ur', inscr_beta[2], $
    'v2_ur', inscr_v2[2], $
    'v3_ur', inscr_v3[2], $
    'alpha_lr', inscr_alpha[3], $
    'beta_lr', inscr_beta[3], $
    'v2_lr', inscr_v2[3], $
    'v3_lr', inscr_v3[3])
  siaf=replicate(siaf_line,nslices+1)
  for i=0,nslices-1 do begin
    siaf[i+1].SliceName=slicename[i]
    siaf[i+1].SliceNum=slicenum[i]
    siaf[i+1].alpha_ref=slice_alpha_ref[i]
    siaf[i+1].beta_ref=slice_beta_ref[i]
    siaf[i+1].v2_ref=slice_v2_ref[i]
    siaf[i+1].v3_ref=slice_v3_ref[i]
    siaf[i+1].alpha_ll=alpha_corners[0,i]
    siaf[i+1].beta_ll=beta_corners[0,i]
    siaf[i+1].v2_ll=v2_corners[0,i]
    siaf[i+1].v3_ll=v3_corners[0,i]
    siaf[i+1].alpha_ul=alpha_corners[1,i]
    siaf[i+1].beta_ul=beta_corners[1,i]
    siaf[i+1].v2_ul=v2_corners[1,i]
    siaf[i+1].v3_ul=v3_corners[1,i]
    siaf[i+1].alpha_ur=alpha_corners[2,i]
    siaf[i+1].beta_ur=beta_corners[2,i]
    siaf[i+1].v2_ur=v2_corners[2,i]
    siaf[i+1].v3_ur=v3_corners[2,i]
    siaf[i+1].alpha_lr=alpha_corners[3,i]
    siaf[i+1].beta_lr=beta_corners[3,i]
    siaf[i+1].v2_lr=v2_corners[3,i]
    siaf[i+1].v3_lr=v3_corners[3,i]
  endfor
  yanny_write,outpar,ptr_new(siaf)

  ; Print all of the corner coordinates to a text file
  printf,lun,'# SliceName SliceNum a_ref b_ref v2_ref v3_ref a_ll b_ll v2_ll v3_ll a_ul b_ul v2_ul v3_ul a_ur b_ur v2_ur v3_ur a_lr b_lr v2_lr v3_lr'
  printf,lun,channel,'   -1',alpha_ref,beta_ref,v2_ref,v3_ref,inscr_alpha[0],inscr_beta[0],inscr_v2[0],inscr_v3[0],$
    inscr_alpha[1],inscr_beta[1],inscr_v2[1],inscr_v3[1],$
    inscr_alpha[2],inscr_beta[2],inscr_v2[2],inscr_v3[2],$
    inscr_alpha[3],inscr_beta[3],inscr_v2[3],inscr_v3[3]
  for i=0,nslices-1 do begin
    printf,lun,slicename[i],slicenum[i],$
    slice_alpha_ref[i],slice_beta_ref[i],slice_v2_ref[i],slice_v3_ref[i],$
    alpha_corners[0,i],beta_corners[0,i],v2_corners[0,i],v3_corners[0,i],$
    alpha_corners[1,i],beta_corners[1,i],v2_corners[1,i],v3_corners[1,i],$
    alpha_corners[2,i],beta_corners[2,i],v2_corners[2,i],v3_corners[2,i],$
    alpha_corners[3,i],beta_corners[3,i],v2_corners[3,i],v3_corners[3,i]
  endfor


; Plot the corners in alpha,beta for this subband
plotname=outdir+'siaf_'+channel+'ab.ps'
set_plot,'ps'
device,filename=plotname,/color
loadct,39
plot,alpha_corners[*,0],beta_corners[*,0],/nodata,xrange=[min(alpha_corners),max(alpha_corners)],yrange=[min(beta_corners),max(beta_corners)],xstyle=1,ystyle=1,xtitle='Alpha',ytitle='Beta',xcharsize=1.3,ycharsize=1.3,title=channel,thick=5,charthick=5,xthick=5,ythick=5
seed=56
colors=randomu(seed,nslices)*250
for i=0,nslices-1 do begin
  oplot,[alpha_corners[*,i],alpha_corners[0,i]],[beta_corners[*,i],beta_corners[0,i]],color=colors[i],thick=3
  oplot,[slice_alpha_ref[i]],[slice_beta_ref[i]],psym=1,color=colors[i],thick=3
endfor
oplot,[alpha_ref],[beta_ref],psym=1,thick=3,symsize=2
oplot,inscr_alpha,inscr_beta,thick=3
device,/close

; Plot the corners in v2,v3
plotname=outdir+'siaf_'+channel+'v2v3.ps'
device,filename=plotname,/color
loadct,39
plot,v2_corners[*,0],v3_corners[*,0],/nodata,xrange=[max(v2_corners),min(v2_corners)],yrange=[min(v3_corners),max(v3_corners)],xstyle=1,ystyle=1,xtitle='V2',ytitle='V3',xcharsize=1.3,ycharsize=1.3,xmargin=12,ymargin=5,title=channel,thick=5,charthick=5,xthick=5,ythick=5
for i=0,nslices-1 do begin
  oplot,[v2_corners[*,i],v2_corners[0,i]],[v3_corners[*,i],v3_corners[0,i]],color=colors[i],thick=3
  oplot,[slice_v2_ref[i]],[slice_v3_ref[i]],psym=1,color=colors[i],thick=3
endfor
oplot,[v2_ref],[v3_ref],psym=1,thick=3,symsize=2
oplot,[inscr_v2,inscr_v2[0]],[inscr_v3,inscr_v3[0]],thick=3
device,/close

; Plot the corners in v2,v3 in a constant box size
plotname=outdir+'siaf_'+channel+'v2v3_common.ps'
device,filename=plotname,/color,xsize=16,ysize=15
loadct,39
plot,v2_corners[*,0],v3_corners[*,0],/nodata,xrange=[-8.29*60.,-8.49*60.],yrange=[-5.43*60.,-5.23*60.],xstyle=1,ystyle=1,xtitle='V2',ytitle='V3',xcharsize=1.3,ycharsize=1.3,xmargin=12,ymargin=5,title=channel,thick=5,charthick=5,xthick=5,ythick=5
for i=0,nslices-1 do begin
  oplot,[v2_corners[*,i],v2_corners[0,i]],[v3_corners[*,i],v3_corners[0,i]],color=colors[i],thick=3
  oplot,[slice_v2_ref[i]],[slice_v3_ref[i]],psym=1,color=colors[i],thick=3
endfor
oplot,[v2_ref],[v3_ref],psym=1,thick=3,symsize=2
oplot,[inscr_v2,inscr_v2[0]],[inscr_v3,inscr_v3[0]],thick=3
oplot,[-8.3942412*60.], [-5.3123744*60.],psym=1,thick=3
device,/close
set_plot,'x'

close,lun
free_lun,lun

return
end
