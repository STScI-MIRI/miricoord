;+
; Like regular meanclip, but it returns mean as a function would do
;
;+
; NAME:
;	ml_MEANCLIP
;
; PURPOSE:
;	Computes an iteratively sigma-clipped mean on a data set
; EXPLANATION:
;       Clipping is done about median, but mean is returned.
;       Called by SKYADJ_CUBE
;
; CATEGORY:
;	Statistics
;
; CALLING SEQUENCE:
;	MEANCLIP, Data, Mean, Sigma
;
; INPUT POSITIONAL PARAMETERS:
;	Data: 	  Input data, any numeric array
;	
; OUTPUT POSITIONAL PARAMETERS:
;       Mean:     N-sigma clipped mean.
;       Sigma:    Standard deviation of remaining pixels.
;
; INPUT KEYWORD PARAMETERS:
;       CLIPSIG:  Number of sigma at which to clip.  Default=3
;	MAXITER:  Ceiling on number of clipping iterations.  Default=5
;       CONVERGE_NUM:  If the proportion of rejected pixels is less
;           than this fraction, the iterations stop.  Default=0.02, i.e.,
;           iteration stops if fewer than 2% of pixels excluded.
;       IGNORE: If there is a 'bad value' flag in your data, specify
;           it here if you don't want it included in the mean
;       /VERBOSE:  Set this flag to get messages.
;
; OUTPUT KEYWORD PARAMETER:
;       SUBS:     Subscript array for pixels finally used.
;
;
; MODIFICATION HISTORY:
; 	Written by:	RSH, RITSS, 21 Oct 98
;       20 Jan 99 - Added SUBS, fixed misplaced paren on float call, 
;                   improved doc.  RSH
;-
; Modified 6/14/07: Tweaked to ignore 0-values when it
; does calculations.  Also ignores 1 high and 1 low value when
; calculating the initial sigma for rejection with n geq 4 data
; points, ignores 1 high on initial sigma pass when n = 3 data points
;
; Modified 6/22/07: Fixed a bug in the ct>=3 routines where it would
; bomb out if the lower two values were identical
;
; Modified 6/25/07: Tweaked the case for ct>=3 where the code properly
; rejected extreme positive outliers, but not extreme negative.
;
; Modified 7/8/10: Tweaked it to allow auto-return of sigma as well
;
; Modified 8/27/10: Tweaked to allow specification of a parameter
; value to ignore, not just to *assume* that zero is ignored
;-

FUNCTION ml_meanclip, Image, result_mean, result_sigma, CLIPSIG=clipsig, MAXITER=maxiter, $
    CONVERGE_NUM=converge_num, IGNORE=ignore, VERBOSE=verbose, SUBS=subs

IF n_params(0) LT 1 THEN BEGIN
    print, 'CALLING SEQUENCE:  MEANCLIP, Image, Mean, Sigma'
    print, 'KEYWORD PARAMETERS:  CLIPSIG, MAXITER, CONVERGE_NUM, ' $
        + 'VERBOSE, SUBS'
    RETALL
ENDIF

prf = 'MEANCLIP:  '

verbose = keyword_set(verbose)
IF n_elements(maxiter) LT 1 THEN maxiter = 5
IF n_elements(clipsig) LT 1 THEN clipsig = 3
IF n_elements(converge_num) LT 1 THEN converge_num = 0.02

; Reject any values where non-finite value, or (if option chosen) bad valued
IF n_elements(ignore) LT 1 THEN subs = where(finite(image),ct)
IF n_elements(ignore) GE 1 THEN subs = where((finite(image) AND (image ne ignore)),ct)

; Case where no remaining members, return 0
if (ct lt 1) then begin
  mean=0.
  sigma=0.
  iter=0
; Case where only one remaining member, return that member
endif else if (ct eq 1) then begin
  mean=image[subs]
  sigma=0.
  iter=0
; Case where two remaining members, return average of the two
endif else if (ct eq 2) then begin
  skpix=image[subs]
  mean=(skpix[0]+skpix[1])/2.
  sigma=0.
  iter=0

; Case where three remaining members, pick the 2 values that are
; closest to the median for the first sigma calculation.
; If all three are consistent, average all 3.  If not, just
; average the original 2.
endif else if (ct eq 3) then begin
  skpix=image[subs]
  skpix=skpix[sort(skpix)]
  medval=ml_median(skpix)
  devtn=abs(skpix-medval)
  ; By default subset_skpix is just skpix for safety
  subset_skpix=skpix
  ; If there is ONE obvious maximum devtn then we can
  ; safely consider the 2 that are not the max
  if ((size(where(devtn eq max(devtn))))[1] eq 1) then subset_skpix=skpix[where(devtn ne max(devtn))]
  medval=(subset_skpix[0]+subset_skpix[1])/2.
  sig=stdev(subset_skpix)
  ; 6/22/07: Fixed glitch here where (if using LT) you can
  ; crash out if sigma=0, as would happen if you had the two
  ; lower values exactly identical.  Works fine using LE.
  wsm=where(abs(skpix-medval) LE clipsig*sig,ct)
  mean=total(skpix[wsm])/((size(wsm))[1])
  sigma=0.
  iter=0
; Otherwise, there are at least 4 members. Excluding top and bottom
; values calculate mean and sigma for 0th iteration. Then loop through
; *all* values with rejection.
endif else begin
  skpix=image[subs]
  subset_skpix=skpix[sort(skpix)]
  subset_skpix=subset_skpix[1:(size(skpix))[1]-2]
  medval=ml_median(subset_skpix)
  sig=stdev(subset_skpix)
  wsm=where(abs(skpix-medval) LE clipsig*sig,ct)
  subs=subs[wsm]

  iter=0
  REPEAT BEGIN
    skpix = image[subs]
    iter = iter + 1
    lastct = ct
    medval = ml_median(skpix)
    sig = stdev(skpix)
    wsm = where(abs(skpix-medval) LT clipsig*sig,ct)
    IF ct GT 0 THEN subs = subs[wsm]
  ENDREP UNTIL (float(abs(ct-lastct))/lastct LT converge_num) $
          OR (iter GT maxiter) 
  mom = moment(image[subs])
  mean = mom[0]
  sigma = sqrt(mom[1])
endelse

IF verbose THEN BEGIN
    print, prf+strn(clipsig)+'-sigma clipped mean'
    print, prf+'Mean computed in ',iter,' iterations'
    print, prf+'Mean = ',mean,',  sigma = ',sigma
ENDIF

result_mean=mean
result_sigma=sigma

RETURN,mean
END
