;+
;	NAME:
;		ml_getenv
;
;	PURPOSE:
;		Same as the IDL function getenv, except now checks if last character of the directory path is a / or not
;		If not, then adds one ; otherwise it does nothing
;
;	INPUTS:
;		path  - a string containing the directory path ; usually an environment variable
;
;	OUTPUTS:
;		dir - the path as output by GETENV but with a slash added to the end
;		
;	KEYWORDS:
;		ADD - set this keyword to just add a / at the end of your input path ; does not use GETENV
;
;	HISTORY:
;		Oct 2012 - written by B. Cherinka
;-

function ml_getenv, path, add=add

on_error, 0
compile_opt idl2
compile_opt idl2, hidden

;catch error
if size(path,/type) ne 7 then begin
  print, 'ERROR: INPUT MUST BE OF TYPE STRING!'
  return,''
endif

dir = keyword_set(add) ? path : getenv(path)

;check if last part is slash
if strlen(dir) ne 0 then if strmid(dir,strlen(dir)-1) ne '/' then dir=dir+'/'

return, dir

end
