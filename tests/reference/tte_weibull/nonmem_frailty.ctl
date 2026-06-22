$PROBLEM TTE Weibull (FRAILTY on shape) – Phase 1 ferx validation
; Mixed-effects (frailty on shape) fit via CONDITIONAL LAPLACIAN INTERACTION. Fills
; the "NONMEM LAPLACIAN" column / the #440 nonlinear-frailty omega^2 cross-check
; (ferx 0.204 vs nlmixr2 0.173).
;
; DATA LAYOUT (the fix for error 350): NONMEM forbids CONDITIONAL/LAPLACIAN on
; "single-subject" data, which 1 record/subject is inferred to be. So this reads
; tte_weibull_nm.csv, which adds a TIME=0 "at-risk entry" record per subject (2
; records/subject => population data). The entry record contributes S(0)=1 -> 0 to
; the -2LL, so the estimates are IDENTICAL to the 1-record file the other tools fit.
;
; HAZNOW is computed ONLY for event rows (DV=1, TIME>0): at TIME=0 the term
; (TIME/SCALE)**(SHAPE-1) is 0**(negative) if a trial SHAPE<1 during the eta search
; -> Inf/NaN. CHZ=(TIME/SCALE)**SHAPE is safe at TIME=0 (SHAPE>0 -> 0**positive = 0).
; HAZNOW (not the reserved name H); IGNORE=@; clean $TABLE — all kept.
$INPUT ID TIME DV EVID CMT MDV
$DATA tte_weibull_nm.csv IGNORE=@

$PRED
  SCALE  = EXP(THETA(1))           ; scale (time units): H=(t/scale)^shape
  SHAPE  = EXP(THETA(2) + ETA(1))  ; shape (dimensionless), frailty on shape
  CHZ    = (TIME/SCALE)**SHAPE      ; H(t)  (= 0 at the TIME=0 entry record)
  F_FLAG = 1
  IF (DV.EQ.0) Y = EXP(-CHZ)                        ; entry/censored -> S(T)
  IF (DV.EQ.1) THEN
     HAZNOW = (SHAPE/SCALE)*(TIME/SCALE)**(SHAPE-1) ; h(t); event rows only (TIME>0)
     Y = HAZNOW * EXP(-CHZ)
  ENDIF

$THETA (0.001, 3.0, 10)   ; log(scale): init log(20) = 2.996
$THETA (0.001, 0.693, 5)  ; log(shape): init log(2.0) = 0.693
$OMEGA 0.20               ; var(eta.shape): ESTIMATED frailty, init 0.20

$ESTIMATION METHOD=COND LAPLACIAN INTERACTION MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE PRINT=E
$TABLE ID TIME DV NOPRINT NOAPPEND FILE=tte_weibull_frailty.sdtab
