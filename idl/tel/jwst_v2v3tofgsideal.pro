; Function to convert V2, V3 coordinates to Ideal
; coordinates for the FGS
pro jwst_v2v3tofgsideal,v2,v3,xideal,yideal

; Use FGS1_FULL from the 2017-03-22 FGS SIAF
v2ref=207.19d
v3ref=-697.5d
a=-1.2508d *!PI/180. ; (v3sciyangle)
P=-1 ; parity

; Per Colin Cox:
;V2 = V2Ref + VIdlParity*XIdl*cos(a) + YIdl*sin(a)
;V3 = V3Ref - VIdlParity*XIdl*sin(a) + YIdl*cos(a)
xideal=P*((v2-v2ref)-(v3-v3ref)*tan(a))/(cos(a)+sin(a)*tan(a))
yideal=((v2-v2ref)-P*xideal*cos(a))/sin(a)

return
end
