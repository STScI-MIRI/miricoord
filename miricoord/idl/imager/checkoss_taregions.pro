pro checkoss_taregions

; Mike's 1-2-3-4 ordering is UR, UL,LL,LR in form outer,inner

lyot_xsize=320
lyot_ysize=304
lyot_x1=1
lyot_y1=717
lyot_xref=142.5
lyot_yref=884.5

lyot_ta_x1=[180,163,39,110,34,108,180,164]
lyot_ta_y1=[921,900,921,900,782,853,779,853]
lyot_ta_xsize=[64,16,64,16,64,16,64,16]
lyot_ta_ysize=[64,16,64,16,64,16,64,16]
lyot_ta_xref=[211.5,170.5,70.5,117.5,65.5,115.5,211.5,171.5]
lyot_ta_yref=[952.5,907.5,952.5,907.5,813.5,860.5,810.5,860.5]

mike_x1=[162,163,40,110,40,108,162,164]
mike_y1=[192,184,192,184,117,137,117,137]

plot,[0,0],[0,0],/nodata,xrange=[0,300],yrange=[0,300]

for i=0,7 do begin
  x=lyot_ta_x1[i]-lyot_x1+1
  xs=lyot_ta_xsize[i]
  y=lyot_ta_y1[i]-lyot_y1+1
  ys=lyot_ta_ysize[i]

  mx=mike_x1[i]
  my=mike_y1[i]

  oplot,[x,x+xs,x+xs,x,x],[y,y,y+ys,y+ys,y]
  oplot,[mx,mx+xs,mx+xs,mx,mx],[my,my,my+ys,my+ys,my],color=250
endfor
xyouts,150,250,'Lyot',charsize=2.0
oplot,lyot_ta_xref-lyot_x1+1,lyot_ta_yref-lyot_y1+1,psym=1

; Coordinate xforms in DMS units
mirim_xytov2v3,lyot_ta_xref-1,lyot_ta_yref-1,v2,v3,'F770W'
mirim_xytov2v3,lyot_xref-1,lyot_yref-1,v2ref,v3ref,'F770W'

print,'LYOT'
print,v2ref,v3ref
print,v2,v3

names=['1P','1S','2P','2S','3P','3S','4P','4S']
for i=0,7 do begin
  print,names[i]
  print,'regionXcorner = ',lyot_ta_x1[i]-lyot_x1+1
  print,'regionYcorner = ',lyot_ta_y1[i]-lyot_y1+1
endfor

stop

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

c1550_xsize=288
c1550_ysize=224
c1550_x1=1
c1550_y1=467
c1550_xref=119.5
c1550_yref=575.5; RevB incorrectly says 575.0

c1550_ta_x1=[143,126,31,99,29,97,139,126]
c1550_ta_y1=[596,580,607,584,494,556,487,553]
c1550_ta_xsize=[64,16,64,16,64,16,64,16]
c1550_ta_ysize=[64,16,64,16,64,16,64,16]
c1550_ta_xref=[174.5,133.5,62.5,106.5,60.5,104.5,170.5,133.5]
c1550_ta_yref=[627.5,587.5,638.5,591.5,525.5,563.5,518.5,560.5]

mike1550_x1=[131,113,42,97,42,97,131,113]
mike1550_y1=[127,94,126,94,9,77,9,77]

plot,[0,0],[0,0],/nodata,xrange=[0,300],yrange=[0,300]

for i=0,7 do begin
  x=c1550_ta_x1[i]-c1550_x1+1
  xs=c1550_ta_xsize[i]
  y=c1550_ta_y1[i]-c1550_y1+1
  ys=c1550_ta_ysize[i]

  mx=mike1550_x1[i]
  my=mike1550_y1[i]

  oplot,[x,x+xs,x+xs,x,x],[y,y,y+ys,y+ys,y]
  oplot,[mx,mx+xs,mx+xs,mx,mx],[my,my,my+ys,my+ys,my],color=250
endfor
xyouts,150,250,'C1550',charsize=2.0
oplot,c1550_ta_xref-c1550_x1+1,c1550_ta_yref-c1550_y1+1,psym=1

; Coordinate xforms in DMS units
mirim_xytov2v3,c1550_ta_xref-1,c1550_ta_yref-1,v2,v3,'F770W'
mirim_xytov2v3,c1550_xref-1,c1550_yref-1,v2ref,v3ref,'F770W'

print,'C1550'
print,v2ref,v3ref
print,v2,v3

names=['1P','1S','2P','2S','3P','3S','4P','4S']
for i=0,7 do begin
  print,names[i]
  print,'regionXcorner = ',c1550_ta_x1[i]-c1550_x1+1
  print,'regionYcorner = ',c1550_ta_y1[i]-c1550_y1+1
endfor

stop

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; Mike's 1-2-3-4 ordering is UR, UL,LL,LR in form outer,inner

c1140_xsize=288
c1140_ysize=224
c1140_x1=1
c1140_y1=245
c1140_xref=119.5
c1140_yref=354.5

c1140_ta_x1=[143,130,31,97,26,94,139,126]
c1140_ta_y1=[380,362,383,363,269,333,265,331]
c1140_ta_xsize=[64,16,64,16,64,16,64,16]
c1140_ta_ysize=[64,16,64,16,64,16,64,16]
c1140_ta_xref=[174.5,137.5,62.5,104.5,57.5,101.5,170.5,133.5]
c1140_ta_yref=[411.5,369.5,414.5,370.5,300.5,340.5,296.5,338.5]

mike1140_x1=[131,113,42,97,42,97,131,113]
mike1140_y1=[126,94,126,94,9,78,9,78]

plot,[0,0],[0,0],/nodata,xrange=[0,300],yrange=[0,300]

for i=0,7 do begin
  x=c1140_ta_x1[i]-c1140_x1+1
  xs=c1140_ta_xsize[i]
  y=c1140_ta_y1[i]-c1140_y1+1
  ys=c1140_ta_ysize[i]

  mx=mike1140_x1[i]
  my=mike1140_y1[i]

  oplot,[x,x+xs,x+xs,x,x],[y,y,y+ys,y+ys,y]
  oplot,[mx,mx+xs,mx+xs,mx,mx],[my,my,my+ys,my+ys,my],color=250
endfor
xyouts,150,250,'C1140',charsize=2.0
oplot,c1140_ta_xref-c1140_x1+1,c1140_ta_yref-c1140_y1+1,psym=1

; Coordinate xforms in DMS units
mirim_xytov2v3,c1140_ta_xref-1,c1140_ta_yref-1,v2,v3,'F770W'
mirim_xytov2v3,c1140_xref-1,c1140_yref-1,v2ref,v3ref,'F770W'

print,'C1140'
print,v2ref,v3ref
print,v2,v3

names=['1P','1S','2P','2S','3P','3S','4P','4S']
for i=0,7 do begin
  print,names[i]
  print,'regionXcorner = ',c1140_ta_x1[i]-c1140_x1+1
  print,'regionYcorner = ',c1140_ta_y1[i]-c1140_y1+1
endfor

stop

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; Mike's 1-2-3-4 ordering is UR, UL,LL,LR in form outer,inner

c1065_xsize=288
c1065_ysize=224
c1065_x1=1
c1065_y1=19
c1065_xref=120.0
c1065_yref=132.0

c1065_ta_x1=[145,128,30,97,26,94,139,126]
c1065_ta_y1=[157,138,160,141,47,110,42,108]
c1065_ta_xsize=[64,16,64,16,64,16,64,16]
c1065_ta_ysize=[64,16,64,16,64,16,64,16]
c1065_ta_xref=[176.5,135.5,61.5,104.5,57.5,101.5,170.5,133.5]
c1065_ta_yref=[188.5,145.5,191.5,148.5,78.5,117.5,73.5,115.5]

mike1065_x1=[131,113,42,97,42,97,131,113]
mike1065_y1=[123,91,123,91,7,75,7,75]

plot,[0,0],[0,0],/nodata,xrange=[0,300],yrange=[0,300]

for i=0,7 do begin
  x=c1065_ta_x1[i]-c1065_x1+1
  xs=c1065_ta_xsize[i]
  y=c1065_ta_y1[i]-c1065_y1+1
  ys=c1065_ta_ysize[i]

  mx=mike1065_x1[i]
  my=mike1065_y1[i]

  oplot,[x,x+xs,x+xs,x,x],[y,y,y+ys,y+ys,y]
  oplot,[mx,mx+xs,mx+xs,mx,mx],[my,my,my+ys,my+ys,my],color=250
endfor
xyouts,150,250,'C1065',charsize=2.0
oplot,c1065_ta_xref-c1065_x1+1,c1065_ta_yref-c1065_y1+1,psym=1

; Coordinate xforms in DMS units
mirim_xytov2v3,c1065_ta_xref-1,c1065_ta_yref-1,v2,v3,'F770W'
mirim_xytov2v3,c1065_xref-1,c1065_yref-1,v2ref,v3ref,'F770W'

print,'C1065'
print,v2ref,v3ref
print,v2,v3

names=['1P','1S','2P','2S','3P','3S','4P','4S']
for i=0,7 do begin
  print,names[i]
  print,'regionXcorner = ',c1065_ta_x1[i]-c1065_x1+1
  print,'regionYcorner = ',c1065_ta_y1[i]-c1065_y1+1
endfor

return
end
