;+
; NAME:
;   hogg_strsplit
; PURPOSE:
;   split strings on whitespace, except inside double quotes, plus other
;     stuff specialized for yanny_read
; CALLING SEQUENCE:
;   hogg_strsplit, line, output[, count, recurse=recurse, verbose=verbose, $
;     keepquote=keepquote
; INPUTS:
;   line - A string to split.
; OPTIONAL INPUTS:
;   recurse   - For internal use by hogg_strsplit
;   verbose   - Print debug information.
;   keepquote - Do not strip double quotes from quoted strings.
; OUTPUTS:
;   output - An array of strings.
; OPTIONAL OUTPUTS:
;   count  - The number of strings returned.
; BUGS:
;   - Does not preserve the input string.
;   - Comment stripping may fail in certain pathological cases,
;     for example if a real trailing comment contains only a
;     single double-quote, or if a trailing comment contains more
;     than one # character.
; REVISION HISTORY:
;   2002-10-10  written - Hogg
;   2014-03-06  Allow # characters to exist inside quoted strings. - BAW
;-
PRO hogg_strsplit, line, output, count, recurse=recurse, verbose=verbose, $
    keepquote=keepquote
    ;
    IF KEYWORD_SET(verbose) THEN splog, 'splitting string >'+line+'<'
    ;
    ; Initialize unset variables, putting in first-element dummy value
    ; and splitting out comments
    ;
    IF ~KEYWORD_SET(recurse) THEN BEGIN
        output= 'NULL'
        count= 0
        ;
        ; For some reason this function only works on single spaces between
        ; words.  This takes care of that.
        ;
        line = ' '+STRCOMPRESS(line)
        ;line= (STRSPLIT(' '+STRCOMPRESS(line),'#',/EXTRACT))[0]
        ;
        ; Remove trailing comments, but not if # is enclosed in quotes.
        ;
        lastmark = STRPOS(line,'#',/REVERSE_SEARCH)
        IF lastmark GT -1 THEN BEGIN
            ;
            ; Count the number of double quotes between the lastmark
            ; position and the end of the line.
            ;
            nq = 0
            pos = STRPOS(STRMID(line,lastmark),'"')
            IF pos GT -1 THEN BEGIN
                nq = 1
                WHILE pos GT -1 DO BEGIN
                    pos = STRPOS(STRMID(line,lastmark),'"',pos+1)
                    IF pos GT -1 THEN nq = nq + 1
                ENDWHILE
            ENDIF
            IF (nq MOD 2) EQ 0 THEN $
                line = STRTRIM(STRMID(line,0,lastmark))
        ENDIF
    ENDIF
    ;
    ; Do the dumbest thing, if possible
    ;
    IF (STRCOMPRESS(line,/REMOVE_ALL) EQ '') THEN RETURN
    ;
    ; Do the second-dumbest thing, if possible
    ;
    IF STREGEX(line,'[\"]') LT 0 THEN BEGIN
        pos = STREGEX(line,'\{ *\{ *\} *\}',LENGTH=len)
        IF (pos GE 0) THEN BEGIN
            line= STRMID(line,0,pos)+' "" '+STRMID(line,pos+len)
            hogg_strsplit, line, output, count, /recurse, verbose=verbose
        ENDIF ELSE BEGIN
            ;
            ; just split on spaces
            ;
            word = STRSPLIT(line, '[ ,;]+', /REGEX,/EXTRACT)
            output= [output, word]
            count = count + N_ELEMENTS(word)
        ENDELSE
    ENDIF ELSE BEGIN
        ;
        ; split on quotation marks and operate recursively
        ; Find the position and length of the first double-quoted string.
        ;
        pos = STREGEX(line,'\"([^"]*)\"',LENGTH=len)
        IF (pos GE 0) THEN BEGIN
            ;
            ; Split everyting prior to that quote, appending to OUTPUT
            ;
            hogg_strsplit, STRMID(line,0,pos), output, count, /recurse, $
                verbose=verbose
            ;
            ; Now add to that the quoted string, but excluding the quotation
            ; marks themselves. (Unless we are requested to keep the quotes).
            ;
            IF ~KEYWORD_SET(keepquote) THEN word = STRMID(line,pos+1,len-2) $
            ELSE word = STRMID(line,pos,len-1)
            output= [output, word]
            count = count + 1
            ;
            ; Finally, split everything after the quoted part,
            ; which might contain more quoted strings.
            ;
            hogg_strsplit, STRMID(line,pos+len), output, count, /recurse, $
                verbose=verbose
        ENDIF
    ENDELSE
    ;
    ; Remove first-element dummy value
    ;
    IF (~KEYWORD_SET(recurse) && (count GT 0)) THEN BEGIN
        output= output[1:count]
        IF KEYWORD_SET(verbose) THEN FOR i=0,count-1 DO PRINT, i,'>'+output[i]+'<'
    ENDIF
    RETURN
END
