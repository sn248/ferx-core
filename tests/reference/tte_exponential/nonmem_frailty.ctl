$PROBLEM TTE Exponential (FRAILTY) – Phase 1 ferx validation
; Corrected mixed-effects (frailty) control file: estimates var(eta.lambda) via
; CONDITIONAL LAPLACIAN INTERACTION. This is the model the hand-off intended (it
; fills the "NONMEM LAPLACIAN" column of the frailty table in expected.md).
;
; Differences vs the fixed-effects nonmem.ctl that was actually run:
;   * $OMEGA is ESTIMATED (0.25 init), NOT `0 FIX`.
;   * $ESTIMATION uses CONDITIONAL LAPLACIAN INTERACTION, NOT METHOD=0 (FO).
; The earlier frailty attempts only failed NM-TRAN translation (reserved name `H`,
; and `$TABLE IPRED PRED` which are undefined in an F_FLAG=1 likelihood model) —
; both are already avoided here (HAZNOW; minimal $TABLE). Sample size was never the
; issue: NONMEM never reached the estimator.
$INPUT ID TIME DV EVID CMT MDV
$DATA tte_exp.csv IGNORE=@

$PRED
  LAMBDA = EXP(THETA(1) + ETA(1))
  CHZ    = LAMBDA * TIME        ; H(t) = lambda * t  (Exponential)
  HAZNOW = LAMBDA               ; h(t) = lambda  (do NOT name this H — reserved $PRED arg)
  F_FLAG = 1
  IF (DV.EQ.0) Y = EXP(-CHZ)            ; right-censored: S(T)
  IF (DV.EQ.1) Y = HAZNOW * EXP(-CHZ)   ; exact event:    h(T)*S(T)

$THETA (-10, -2.3, 5)   ; log(lambda): init log(0.1) = -2.303
$OMEGA 0.25             ; var(eta.lambda): ESTIMATED frailty, init 0.25

$ESTIMATION METHOD=COND LAPLACIAN INTERACTION MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE PRINT=E
$TABLE ID TIME DV NOPRINT NOAPPEND FILE=tte_exp_frailty.sdtab
