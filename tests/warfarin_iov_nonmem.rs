//! NONMEM 7.5.1 FOCEI cross-check for the inter-occasion variability (IOV) path
//! — issue #101.
//!
//! Validates ferx's augmented IOV marginal (`foce_subject_nll_iov`) against a
//! NONMEM reference fit on the 10-subject warfarin IOV dataset
//! (`data/warfarin_iov.csv`, 2 occasions/subject): 1-cpt oral, proportional
//! error, IOV on CL. Model: `examples/warfarin_iov.ferx`.
//!
//! ## What this characterizes
//!
//! `iov_objective_characterizes_nonmem_gap` — ferx's FOCEI objective, evaluated
//! at NONMEM's final MLE (all parameters FIXed), compared to NONMEM's
//! OFV-without-constant. With the continuous per-occasion-aware prediction
//! (issue #104, `pk::predict_iov`) ferx lands at ≈291.9 vs NONMEM's 308.83 — a
//! ≈17-unit gap, down from ≈37 under the old Option-A superposition.
//!
//! The ≈20-unit improvement is the carryover fix: ferx now propagates each
//! dose's amount continuously across occasion boundaries with the occasion's
//! clearance (via the event-driven solver), matching NONMEM's integration model
//! rather than scoring each occasion against the whole dose history with a
//! single clearance.
//!
//! The **residual ≈17 units** is the simultaneous cross-occasion event ordering:
//! warfarin's occasion-1 obs and occasion-2 dose are both at t=120, and ferx's
//! event sort processes the dose before the obs (`Dose < Obs` tie-break), so the
//! [96,120] interval decays with occasion-2's clearance instead of occasion-1's.
//! NONMEM processes records in data order (obs row first → occasion-1). Closing
//! this fully requires retaining the original record order for simultaneous
//! events (not currently stored — doses and observations live in separate
//! arrays). Non-IOV NONMEM cross-checks (`multi_endpoint`, `ss_lagtime`) match
//! tightly, confirming the residual is IOV-boundary-specific.
//!
//! This test is `#[ignore]`d: it characterizes and bounds the residual gap. The
//! optimizer-from-cold-start issue is tracked separately in
//! `tests/iov_convergence.rs` (issue #101 rec #2).
//!
//! ## Reproducing the NONMEM reference
//!
//! NONMEM 7.5.1, FOCEI (`METHOD=1 INTER`), from `tests/nonmem/warfarin_iov.ctl`
//! over `data/warfarin_iov.csv`. IOV on CL is coded with one ETA per occasion
//! sharing a single variance via `$OMEGA BLOCK(1) ... SAME`:
//!
//! ```text
//! $SUBROUTINES ADVAN2 TRANS2
//! $PK
//!   OCC1 = 0
//!   OCC2 = 0
//!   IF(OCC.EQ.1) OCC1 = 1
//!   IF(OCC.EQ.2) OCC2 = 1
//!   IOVCL = OCC1*ETA(4) + OCC2*ETA(5)
//!   CL = THETA(1)*EXP(ETA(1) + IOVCL)
//!   V  = THETA(2)*EXP(ETA(2))
//!   KA = THETA(3)*EXP(ETA(3))
//!   S2 = V
//! $ERROR
//!   Y = F*(1 + EPS(1))
//! $OMEGA 0.09 ; 0.04 ; 0.30        (ETA_CL, ETA_V, ETA_KA)
//! $OMEGA BLOCK(1) 0.01   ; occasion 1
//! $OMEGA BLOCK(1) SAME   ; occasion 2  -> IOV
//! $SIGMA 0.04
//! $ESTIMATION METHOD=1 INTER MAXEVAL=9999 NSIG=3 SIGL=9 NOABORT
//! ```
//!
//! Final estimates (run.ext, iteration 151), OBJECTIVE FUNCTION VALUE WITHOUT
//! CONSTANT = 308.8305 (WITH CONSTANT 713.1634 = 308.8305 + 220·ln(2π)).
//! NONMEM minimization TERMINATED on rounding errors (ERROR=134) but the OFV
//! and estimates are stable across the last iterations.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions, GradientMethod};
use std::path::Path;

// NONMEM 7.5.1 FOCEI MLE (run.ext final iteration; OFV without constant).
const NM_TVCL: f64 = 0.172776;
const NM_TVV: f64 = 8.62821;
const NM_TVKA: f64 = 1.17856;
const NM_OMEGA_CL: f64 = 0.0399349;
const NM_OMEGA_V: f64 = 0.0107782;
const NM_OMEGA_KA: f64 = 0.0254197;
const NM_OMEGA_IOV: f64 = 0.0357084;
const NM_SIGMA_PROP_SD: f64 = 0.188116; // sqrt(0.0353877)
const NM_OFV_NO_CONST: f64 = 308.8305;

#[test]
#[ignore = "issue #101: ferx Option-A IOV carryover differs from NONMEM's continuous integration on this washout-free design (~37 OFV units); characterization only"]
fn iov_objective_characterizes_nonmem_gap() {
    // examples/warfarin_iov.ferx structural model, parameters FIXed at NONMEM's
    // MLE. omega/kappa are variances; sigma is the SD ferx reports.
    let fixed = format!(
        r"
[parameters]
  theta TVCL({NM_TVCL}, FIX)
  theta TVV({NM_TVV}, FIX)
  theta TVKA({NM_TVKA}, FIX)
  omega ETA_CL ~ {NM_OMEGA_CL} FIX
  omega ETA_V  ~ {NM_OMEGA_V} FIX
  omega ETA_KA ~ {NM_OMEGA_KA} FIX
  kappa KAPPA_CL ~ {NM_OMEGA_IOV} FIX
  sigma PROP_ERR ~ {prop} (sd) FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = foce
  iov_column = OCC
",
        prop = NM_SIGMA_PROP_SD,
    );

    let model = parse_model_string(&fixed).expect("fixed-param IOV model parses");
    let pop = read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, Some("OCC"))
        .expect("warfarin_iov data loads");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.interaction = true; // match NONMEM METHOD=1 INTER
    opts.gradient_method = GradientMethod::Fd;
    opts.run_covariance_step = false;
    opts.verbose = false;

    let result = fit(&model, &pop, &model.default_params, &opts)
        .expect("fixed-param IOV objective evaluation must run");

    // Characterize the residual gap (~17 units) after the continuous-prediction
    // fix (issue #104): ferx sits BELOW NONMEM (the simultaneous-event ordering
    // decays the [96,120] interval with occasion-2's clearance). If this band is
    // broken, something changed — either the simultaneous-event ordering was
    // fixed (gap → ~0, tighten/retire this test) or a regression crept in (gap
    // grew back toward the old ≈37, or blew up).
    let diff = NM_OFV_NO_CONST - result.ofv; // expected ≈ +17
    assert!(
        result.ofv.is_finite() && (8.0..28.0).contains(&diff),
        "ferx FOCEI at NONMEM's MLE = {:.4}; NONMEM = {:.4}; gap {:.4} outside the \
         documented residual band [8, 28]",
        result.ofv,
        NM_OFV_NO_CONST,
        diff
    );
}
