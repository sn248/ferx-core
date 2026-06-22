$PROBLEM TTE Weibull (FRAILTY on shape) – Phase 1 ferx validation
; Corrected mixed-effects (frailty) control file: estimates var(eta.shape) via
; CONDITIONAL LAPLACIAN INTERACTION. This is the model the hand-off intended (it
; fills the "NONMEM LAPLACIAN" column of the frailty table in expected.md) and is
; the cross-tool check for the #440 nonlinear-frailty omega^2 finding (ferx 0.204
; vs nlmixr2 0.173).
;
; Differences vs the fixed-effects nonmem.ctl that was actually run:
;   * $OMEGA is ESTIMATED (0.20 init), NOT `0 FIX`.
;   * $ESTIMATION uses CONDITIONAL LAPLACIAN INTERACTION, NOT METHOD=0 (FO).
; The earlier frailty attempts only failed NM-TRAN translation (reserved name `H`,
; and `$TABLE IPRED PRED` undefined in an F_FLAG=1 likelihood model) — both avoided
; here. Sample size was never the issue: NONMEM never reached the estimator.
$INPUT ID TIME DV EVID CMT MDV
$DATA tte_weibull.csv IGNORE=@

$PRED
  SCALE  = EXP(THETA(1))           ; scale (time units): H=(t/scale)^shape
  SHAPE  = EXP(THETA(2) + ETA(1))  ; shape (dimensionless), frailty on shape
  CHZ    = (TIME/SCALE)**SHAPE      ; H(t)
  HAZNOW = (SHAPE/SCALE)*(TIME/SCALE)**(SHAPE-1)  ; h(t)  (NOT H — reserved $PRED arg)
  F_FLAG = 1
  IF (DV.EQ.0) Y = EXP(-CHZ)
  IF (DV.EQ.1) Y = HAZNOW * EXP(-CHZ)

$THETA (0.001, 3.0, 10)   ; log(scale): init log(20) = 2.996
$THETA (0.001, 0.693, 5)  ; log(shape): init log(2.0) = 0.693
$OMEGA 0.20               ; var(eta.shape): ESTIMATED frailty, init 0.20

$ESTIMATION METHOD=COND LAPLACIAN INTERACTION MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE PRINT=E
$TABLE ID TIME DV NOPRINT NOAPPEND FILE=tte_weibull_frailty.sdtab
