; This program takes the MRS dither patterns and evaluates their
; pixel- and slice-phase coverage for a given subband.  By default it
; works at the middle wavelength of a given subband, but it can be
; overridden (wave=) to run at arbitrary wavelength.  It can also be
; overridden to run with specified alpha, beta moves (da=, db=)
; as offsets from the 0,0 midpoint in 1A.
;
; dithers are 1-indexed
;
; Example: mmrs_assessdither,'1A',[1,2,3,4]

pro mmrs_assessdither,channel,dith,wave=wave,da=da,db=db,outfile=outfile,rootdir=rootdir,siafdir=siafdir

; This is where the dither input file from the EC lives
if (~keyword_set(rootdir)) then $
  rootdir=concat_dir(ml_getenv('MIRICOORD_DATA_DIR'),'dithers/mirimrs/feb2017/')

; This is where the SIAF parameter files created from mmrs_siaf live
if (~keyword_set(siafdir)) then $
  siafdir=concat_dir(ml_getenv('MIRICOORD_DATA_DIR'),'siaf/mrs/feb2017/')

; This is where the results will go
if (~keyword_set(outdir)) then $
  outdir='./'

; By default test dither positions 1,2 for Ch1A
if (~keyword_set(channel)) then channel='1A'
if (~keyword_set(dith)) then dith=[1,2]

ch=fix(strmid(channel,0,1))
sband=strmid(channel,1,1)

; Read the dithers file
dithers=yanny_readone(concat_dir(rootdir,'dithers.par'))

; Read the SIAF file to define field boundaries
siaffile=strcompress('siaf_'+channel+'.par',/remove_all)
siaf=yanny_readone(concat_dir(siafdir,siaffile))
; Define rough field sizes encompassing a working area
; (so 2x bigger than real single field)
maxalpha=siaf[0].alpha_ur*2
minalpha=siaf[0].alpha_ll*2
maxbeta=siaf[0].beta_ur*2
minbeta=siaf[0].beta_ll*2

; Convert dithers from XYIdeal to RA/DEC offsets
dra=dithers[dith-1].dxidl/3600.d
ddec=dithers[dith-1].dyidl/3600.d

; Figure out dither in xyidl from override if specified
if ((keyword_set(da))and(keyword_set(db))) then begin
  mmrs_abtov2v3,da,db,tempv2,tempv3,'1A'
  ; Figure out the zeropoint location of this band
  ; (local alpha=beta=0 for 1A,2A,3A,or 4A) in v2,v3 coordinates
  mmrs_abtov2v3,0.,0.,zpv2,zpv3,'1A'
  ; Convert both the dither locations and the zeropoint locations
  ; to the XIdl, YIdl reference frame
  mmrs_v2v3toideal,zpv2,zpv3,zpx,zpy
  mmrs_v2v3toideal,tempv2,tempv3,tempx,tempy
  ; Determine dXIdl, dYIdl offsets.  Note that these dXIdl,dYIdl are OFFSETS
  ; from the base location rather than POSITIONS (as for v2,v3)
  dra=(tempx-zpx)/3600.d
  ddec=(tempy-zpy)/3600.d
endif

; Test with a 0.05'' grid
racen=45.d;degrees
decen=0.d;degrees
; WARNING- this code will fail if DEC != 0 !!!!
dtheta=0.02d ;arcsec
nra=long((maxalpha-minalpha)/dtheta)
ndec=long((maxbeta-minbeta)/dtheta)
ra=(findgen(nra)*dtheta+minalpha)/3600.+racen
dec=(findgen(ndec)*dtheta+minbeta)/3600.+decen
racen=(min(ra)+max(ra))/2.
decen=(min(dec)+max(dec))/2.

; Grid of RA,DEC on the sky to sample
skyloc=fltarr(nra,ndec,2)
for i=0L,nra-1 do skyloc[i,*,0]=ra[i]
for i=0L,ndec-1 do skyloc[*,i,1]=dec[i]

rall=reform(skyloc[*,*,0],nra*ndec)
deall=reform(skyloc[*,*,1],nra*ndec)

; If a wavelength was set, use it
; Otherwise use default of -1
if (keyword_set(wave)) then lam=replicate(wave,nra*ndec) $
else lam=-1

; Loop over exposures
ndith=n_elements(dith)
phase_slice=fltarr(nra,ndec,ndith)
phase_pix=fltarr(nra,ndec,ndith)
for i=0,ndith-1 do begin
  ; Transform to V2,V3 locations
  ; WARNING; code will fail if ROLL != 0
  ; Note that we MUST do -dra and +ddec cuz of how the Ideal frame is defined!!
  jwst_radectov2v3,rall,deall,v2all,v3all,V2REF=-8.3942412d*60,V3REF=-5.3123744d*60,ROLLREF=1d-5,RAREF=racen-dra[i],DECREF=decen+ddec[i]
  v2=reform(v2all,[nra,ndec])
  v3=reform(v3all,[nra,ndec])
  ; Tranform to a,b locations
  mmrs_v2v3toab,v2all,v3all,aall,ball,channel
  a=reform(aall,[nra,ndec])
  b=reform(ball,[nra,ndec])
  ; Transform to 1-indexed pixel locations
  mmrs_abltoxy,aall,ball,lam,xall,yall,channel,phase=phase
  x=reform(xall,[nra,ndec])
  y=reform(yall,[nra,ndec])
  ; Construct phase images; they go from -0.5 to 0.5 originally, convert to 0-1
  phase_slice[*,*,i]=reform(phase[*,0],[nra,ndec])+0.5
  phase_pix[*,*,i]=reform(phase[*,1],[nra,ndec])+0.5
endfor

; Collapse over 3rd dimension and find where there are always good values
temp=total(phase_slice,3)
goodval=where(temp gt -10,ngood,complement=badval)

; Create a coverage map that will go from 0 (no coverage) to 1 (ideal
; coverage).  Ideal coverage is defined as half integer
covmap=fltarr(nra,ndec,2)
; Loop over good locations
for k=0L,ngood-1 do begin
  ind=array_indices(covmap,goodval[k])
  slicevals=phase_slice[ind[0],ind[1],*]
  pixvals=phase_pix[ind[0],ind[1],*]

  hit_s=intarr(100)
  for m=0L,n_elements(slicevals)-1 do begin
    sval=round(slicevals[m]*100)
    if (sval eq 100) then sval=0; Wrap around
    if (sval lt 25) then begin
      hit_s[sval:sval+25]=1
      hit_s[0:sval]=1
      hit_s[100-(25-sval):*]=1
    endif
    if ((sval ge 25)and(sval lt 75)) then begin
      hit_s[sval-25:sval+25]=1
    endif
    if (sval ge 75) then begin
      hit_s[sval-25:sval]=1
      hit_s[sval:*]=1
      hit_s[0:sval-75]=1
    endif
  endfor

  hit_p=intarr(100)
  for m=0,n_elements(pixvals)-1 do begin
    pval=round(pixvals[m]*100)
    if (pval eq 100) then pval=0; Wrap around
    if (pval lt 25) then begin
      hit_p[pval:pval+25]=1
      hit_p[0:pval]=1
      hit_p[100-(25-pval):*]=1
    endif
    if ((pval ge 25)and(pval lt 75)) then begin
      hit_p[pval-25:pval+25]=1
    endif
    if (pval ge 75) then begin
      hit_p[pval-25:pval]=1
      hit_p[pval:*]=1
      hit_p[0:pval-75]=1
    endif
  endfor

  temp_s=where(hit_s eq 1,nhit_s)
  temp_p=where(hit_p eq 1,nhit_p)
  covmap[ind[0],ind[1],0]=nhit_s/100.
  covmap[ind[0],ind[1],1]=nhit_p/100.
endfor

writefits,'phase_pix.fits',phase_pix
writefits,'phase_slice.fits',phase_slice
if (~keyword_set(outfile)) then outfile='covmap.fits'

writefits,outfile,covmap


; Print out some metrics
; Total coverage area
temp=covmap[*,*,0]
indx=where(temp ne 0,nindx)
print,'Coverage area: ',nindx*dtheta*dtheta,' arcsec^2'
offsets=sqrt((dra-dra[0])^2+(ddec-ddec[0])^2)
maxoffset=max(offsets)*3600.
if (channel eq '1A') then psf=0.31
if (channel eq '1B') then psf=0.31
if (channel eq '1C') then psf=0.31
if (channel eq '2A') then psf=0.31*(8.90/8.0)
if (channel eq '2B') then psf=0.31*(10.28/8.0)
if (channel eq '2C') then psf=0.31*(11.87/8.0)
if (channel eq '3A') then psf=0.31*(13.67/8.0)
if (channel eq '3B') then psf=0.31*(15.80/8.0)
if (channel eq '3C') then psf=0.31*(18.24/8.0)
if (channel eq '4A') then psf=0.31*(21.10/8.0)
if (channel eq '4B') then psf=0.31*(24.72/8.0)
if (channel eq '4C') then psf=0.31*(28.82/8.0)
print,'Max offset (arcsec): ',maxoffset
print,'Max offset (FWHM): ',maxoffset/psf

return
end
