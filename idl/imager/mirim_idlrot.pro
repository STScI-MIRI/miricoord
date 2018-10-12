pro mirim_idlrot,x,y

; Return the rotation of the Ideal coordinate system at a given
; detector x,y (pipeline 0-indexed convention)
mirim_xytov2v3,x,y,v2,v3,'F770W'
mirim_xytov2v3,x,y+1,v2a,v3a,'F770W'
a=atan((v3a-v3),(v2a-v2))*180./!PI
print,'V3IdlYAngle',90-a

mirim_xytov2v3,x,y,v2,v3,'F770W'
mirim_xytov2v3,x-1,y,v2a,v3a,'F770W'
a=atan((v3a-v3),(v2a-v2))*180./!PI
print,'V2XAngle',a

return
end
