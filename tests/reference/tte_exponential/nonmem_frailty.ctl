$PROBLEM TTE Exponential (FRAILTY) – Phase 1 ferx validation
; Mixed-effects (frailty) fit: estimates var(eta.lambda) via CONDITIONAL LAPLACIAN
; INTERACTION. Fills the "NONMEM LAPLACIAN" column of expected.md.
;
; DATA LAYOUT (the fix for error 350): NONMEM forbids CONDITIONAL/LAPLACIAN methods
; on "single-subject" data, and 1 record per subject is inferred to be exactly that
; ("350 $ESTIM: METHOD=CONDITIONAL INVALID WITH SINGLE-SUBJECT DATA"). So this reads
; tte_exp_nm.csv, which adds a TIME=0 "at-risk entry" record per subject (2 records/
; subject => population data). The entry record contributes S(0)=exp(-0)=1 -> 0 to the
; -2LL, so the estimates are IDENTICAL to the 1-record file the other tools fit.
;
; Other required fixes already present: HAZNOW (not the reserved $PRED name H); clean
; $TABLE (no IPRED/PRED, undefined for F_FLAG=1); IGNORE=@ (skip the alphabetic header).
$INPUT ID TIME DV EVID CMT MDV
$DATA tte_exp_nm.csv IGNORE=@

$PRED
  LAMBDA = EXP(THETA(1) + ETA(1))
  CHZ    = LAMBDA * TIME        ; H(t) = lambda * t  (= 0 at the TIME=0 entry record)
  HAZNOW = LAMBDA               ; h(t) = lambda  (do NOT name this H — reserved $PRED arg)
  F_FLAG = 1
  IF (DV.EQ.0) Y = EXP(-CHZ)            ; entry (TIME=0) -> S(0)=1; right-censored -> S(T)
  IF (DV.EQ.1) Y = HAZNOW * EXP(-CHZ)   ; exact event:    h(T)*S(T)

$THETA (-10, -2.3, 5)   ; log(lambda): init log(0.1) = -2.303
$OMEGA 0.25             ; var(eta.lambda): ESTIMATED frailty, init 0.25

$ESTIMATION METHOD=COND LAPLACIAN INTERACTION MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE PRINT=E
$TABLE ID TIME DV NOPRINT NOAPPEND FILE=tte_exp_frailty.sdtab
