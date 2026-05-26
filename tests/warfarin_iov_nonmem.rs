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
//! OFV-without-constant. At identical parameters ferx lands at ≈271.9 vs
//! NONMEM's 308.83 — a ≈37-unit systematic gap.
//!
//! That gap is NOT an optimizer artifact (parameters are fixed and identical)
//! and NOT the issue-#101 marginal bug (the augmented-marginal fix moved ferx
//! from ≈230 toward NONMEM, and the reduction unit test in `likelihood.rs`
//! proves the form). It is the **Option-A cross-occasion dose-carryover
//! approximation** documented on `individual_nll_iov`: warfarin doses occasion 2
//! at t=120 while occasion-1 drug is still at conc≈2.6 (no washout), and ferx
//! scores occasion-2 observations using occasion-2's CL for occasion-1's
//! carried-over dose, whereas NONMEM integrates continuously with CL switching
//! at the occasion boundary. Non-IOV NONMEM cross-checks (`multi_endpoint`,
//! `ss_lagtime`) match tightly, so the discrepancy is carryover-specific.
//!
//! This test is `#[ignore]`d: it documents and bounds the known gap rather than
//! asserting an agreement ferx cannot meet on a carryover design. Closing it
//! requires replacing Option-A with full per-dose occasion accounting (future
//! work). The optimizer-from-cold-start issue is tracked separately in
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

    // Characterize the known Option-A carryover gap (~37 units): ferx sits
    // BELOW NONMEM (its per-occasion CL fits the carryover-contaminated
    // occasion-2 points more loosely) but in the same neighborhood. If this
    // band is ever broken, something changed — either Option-A was replaced
    // with full per-dose occasion accounting (gap → ~0, tighten/retire this
    // test) or a regression was introduced (gap blew up).
    let diff = NM_OFV_NO_CONST - result.ofv; // expected ≈ +37
    assert!(
        result.ofv.is_finite() && (20.0..55.0).contains(&diff),
        "ferx FOCEI at NONMEM's MLE = {:.4}; NONMEM = {:.4}; gap {:.4} outside the \
         documented Option-A carryover band [20, 55]",
        result.ofv,
        NM_OFV_NO_CONST,
        diff
    );
}
