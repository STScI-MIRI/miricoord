; Function to compute size of box in angular space with arbitrary corners
function mmrs_boxsize,xcor,ycor

; Neglect cosine term since very close to origin
dist=fltarr(4)
dist[0]=sqrt((xcor[0]-xcor[1])^2+(ycor[0]-ycor[1])^2)
dist[1]=sqrt((xcor[1]-xcor[2])^2+(ycor[1]-ycor[2])^2)
dist[2]=sqrt((xcor[2]-xcor[3])^2+(ycor[2]-ycor[3])^2)
dist[3]=sqrt((xcor[0]-xcor[3])^2+(ycor[0]-ycor[3])^2)

return,dist
end

; Function to compute the MRS FOV
; based on the output of mmrs_siaf
pro mmrs_getfov,siafdir=siafdir

; This is where the SIAF parameter files created from mmrs_siaf live
if (~keyword_set(siafdir)) then $
  siafdir=concat_dir(ml_getenv('MIRICOORD_DATA_DIR'),'siaf/mrs/20171214/')

bands=['1A','1B','1C','2A','2B','2C','3A','3B','3C','4A','4B','4C']
nband=n_elements(bands)

fov_inscribed=fltarr(2,nband)
fov_inscribedv23=fltarr(2,nband)
fov_circum=fltarr(2,nband)

; Loop over channels/bands
for i=0,nband-1 do begin
  thisband=bands[i]
  thisfile=strcompress('siaf_'+thisband+'.par',/remove_all)
  siaf=yanny_readone(concat_dir(siafdir,thisfile))

  ; Nominal inscribed footprint based on input file
  alsznom=siaf[0].alpha_lr-siaf[0].alpha_ll
  besznom=siaf[0].beta_ul-siaf[0].beta_ll

  ; Working in alpha, beta space compute inscribed footprint
  maxa=min(siaf[1:*].alpha_lr)
  mina=max(siaf[1:*].alpha_ll)
  maxb=max(siaf[1:*].beta_ul)
  minb=min(siaf[1:*].beta_ll)
  alsz=abs(maxa-mina)
  besz=abs(maxb-minb)
  
  epsilon=0.01
  if ((abs(alsz-alsznom) ge epsilon)or(abs(besz-besznom) ge epsilon)) then begin
    print,'Warning: Calculated inscribed footprint differs from that in reference file!'
    stop
  endif

  ; Compute inscribed v2,v3 corner coordinates
  v2=fltarr(4)
  v3=fltarr(4)
  mmrs_abtov2v3,mina,minb,temp1,temp2,bands[i]
  v2[0]=temp1
  v3[0]=temp2
  mmrs_abtov2v3,maxa,minb,temp1,temp2,bands[i]
  v2[1]=temp1
  v3[1]=temp2
  mmrs_abtov2v3,maxa,maxb,temp1,temp2,bands[i]
  v2[2]=temp1
  v3[2]=temp2
  mmrs_abtov2v3,mina,maxb,temp1,temp2,bands[i]
  v2[3]=temp1
  v3[3]=temp2
  ; Compute distance between these corner locations
  dist=mmrs_boxsize(v2,v3)

  ; Working in alpha, beta space compute circumscribed footprint
  maxa=max(siaf[1:*].alpha_lr)
  mina=min(siaf[1:*].alpha_ll)
  maxb=max(siaf[1:*].beta_ul)
  minb=min(siaf[1:*].beta_ll)
  alsz2=abs(maxa-mina)
  besz2=abs(maxb-minb)

  fov_inscribed[0,i]=alsz
  fov_inscribed[1,i]=besz
  ; Average the side distance for the v2/v3 quadrilateral
  fov_inscribedv23[0,i]=(dist[0]+dist[2])/2.
  fov_inscribedv23[1,i]=(dist[1]+dist[3])/2.
  fov_circum[0,i]=alsz2
  fov_circum[1,i]=besz2
endfor

; Print inscribed sizes
sigdigits=2

;print,'Inscribed FoV in alpha,beta:'
;for i=0,nband-1 do begin
;  outstr=strcompress(bands[i]+' '+sigfig(fov_inscribed[0,i],sigdigits)+' x '+sigfig(fov_inscribed[1,i],sigdigits))
;  print,outstr
;endfor

; Print inscribed sizes
print,'Inscribed FoV in V2,V3:'
for i=0,nband-1 do begin
  outstr=strcompress(bands[i]+' '+sigfig(fov_inscribedv23[0,i],sigdigits)+' x '+sigfig(fov_inscribedv23[1,i],sigdigits))
  print,outstr
endfor

;print,'Circumscribed FoV in alpha,beta:'
;for i=0,nband-1 do begin
;  outstr=strcompress(bands[i]+' '+sigfig(fov_circum[0,i],sigdigits)+' x '+sigfig(fov_circum[1,i],sigdigits))
;  print,outstr
;endfor

; Average over subbands
ch=['Ch1','Ch2','Ch3','Ch4']
fovch=fltarr(2,4)
print,'Inscribed FoV in V2,V3 (Ch average):'
for i=0,3 do begin
  fovch[0,i]=(fov_inscribedv23[0,i*3]+fov_inscribedv23[0,i*3+1]+fov_inscribedv23[0,i*3+2])/3.
  fovch[1,i]=(fov_inscribedv23[1,i*3]+fov_inscribedv23[1,i*3+1]+fov_inscribedv23[1,i*3+2])/3.

  outstr=strcompress(ch[i]+' '+sigfig(fovch[0,i],sigdigits)+' x '+sigfig(fovch[1,i],sigdigits))
  print,outstr
endfor

return
end
