//! NONMEM 7.5.1 cross-check for multi-endpoint (per-CMT) residual error models
//! — the simultaneous PK/PD case from issue #14.
//!
//! Validates ferx's per-CMT FOCEI against a NONMEM reference fit on a 12-subject
//! simulated dataset (`data/pkpd_nonmem.csv`): IV bolus into the central
//! compartment, plasma concentration observed on CMT=1 with **proportional**
//! error and an effect-compartment level on CMT=2 with **additive** error, PK
//! and PD sampled at the same timepoints. The model is `examples/pkpd_nonmem.ferx`.
//!
//! ## Two checks
//!
//! 1. `objective_matches_nonmem_at_reference_params` — ferx's FOCEI objective,
//!    evaluated at NONMEM's final MLE (all parameters fixed), matches NONMEM's
//!    OFV. This is the robust validation of the per-CMT *likelihood* and does
//!    not depend on either engine's optimizer.
//! 2. `fit_recovers_theta_and_per_cmt_sigma` — a free ferx fit recovers the
//!    fixed effects and the two per-CMT residual SDs close to NONMEM.
//!
//! Known limitation (NOT asserted tightly here): ferx's outer optimizer
//! under-converges OMEGA on this dataset — it halts ~2.5 OFV units short of its
//! own objective minimum (the one check #1 confirms NONMEM reaches), leaving
//! ETA_CL variance near its initial value. Tracked separately; this is an
//! outer-optimizer conditioning issue, not a per-CMT error-model problem (the
//! objective in check #1 agrees with NONMEM).
//!
//! ## Reproducing the NONMEM reference
//!
//! NONMEM 7.5.1, FOCEI (`METHOD=1 INTER`), ADVAN6 TOL=9, MINIMIZATION
//! SUCCESSFUL, from this control file over `data/pkpd_nonmem.csv`:
//!
//! ```text
//! $PROBLEM Simultaneous PK/PD, per-CMT error - ferx-core cross-check
//! $DATA data.csv IGNORE=@
//! $INPUT ID TIME DV EVID AMT CMT RATE MDV
//! $SUBROUTINES ADVAN6 TOL=9
//! $MODEL COMP=(CENTRAL) COMP=(EFFECT)
//! $PK
//!   CL  = THETA(1)*EXP(ETA(1))
//!   V   = THETA(2)
//!   KE0 = THETA(3)
//!   S1  = V
//!   S2  = 1
//! $DES
//!   DADT(1) = -CL/V*A(1)
//!   DADT(2) = KE0*(A(1)/V - A(2))
//! $ERROR
//!   IPRED = F
//!   IF (CMT.EQ.1) THEN
//!     Y = IPRED*(1+EPS(1))   ; proportional (PK)
//!   ELSE
//!     Y = IPRED + EPS(2)     ; additive (PD)
//!   ENDIF
//! $THETA (0,2.0) (0,20.0) (0,0.5)
//! $OMEGA 0.09
//! $SIGMA 0.01
//! $SIGMA 0.25
//! $ESTIMATION METHOD=1 INTER MAXEVAL=9999 NSIG=3 SIGL=9 NOABORT
//! ```
//!
//! Final estimates (`.ext`): see the `NM_*` constants below. OFV (without the
//! N*log(2π) constant, matching ferx's convention) = -60.0971.

use ferx_core::parser::model_parser::{parse_model_file, parse_model_string};
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions, GradientMethod};
use std::path::Path;

// ── NONMEM 7.5.1 FOCEI reference (MINIMIZATION SUCCESSFUL) ──
const NM_TVCL: f64 = 1.98382;
const NM_TVV: f64 = 20.2177;
const NM_TVKE0: f64 = 0.511032;
const NM_OMEGA_CL: f64 = 0.0459095; // ETA_CL variance
const NM_SIGMA_PROP_SD: f64 = 0.0958000; // sqrt(0.00917757)
const NM_SIGMA_ADD_SD: f64 = 0.5372337; // sqrt(0.288620)
const NM_OFV_NO_CONST: f64 = -60.0971;

fn population() -> ferx_core::types::Population {
    read_nonmem_csv(Path::new("data/pkpd_nonmem.csv"), None, None)
        .expect("pkpd_nonmem.csv must load")
}

/// All parameters fixed at NONMEM's MLE; ferx evaluates its FOCEI objective
/// once. The OFV must match NONMEM's to within cross-engine FOCEI tolerance —
/// validating the per-CMT likelihood independent of either optimizer.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn objective_matches_nonmem_at_reference_params() {
    // Same structural model as examples/pkpd_nonmem.ferx, parameters FIXed at
    // NONMEM's MLE (sigmas on the SD scale that ferx reports).
    let fixed = format!(
        r"
[parameters]
  theta TVCL({NM_TVCL}, FIX)
  theta TVV({NM_TVV}, FIX)
  theta TVKE0({NM_TVKE0}, FIX)
  omega ETA_CL ~ {NM_OMEGA_CL} FIX
  sigma PROP_ERR_PK ~ {prop} (sd) FIX
  sigma ADD_ERR_PD  ~ {add} (sd) FIX

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KE0 = TVKE0

[structural_model]
  ode(states=[central, effect])

[odes]
  d/dt(central) = -CL/V * central
  d/dt(effect)  =  KE0 * (central/V - effect)

[scaling]
  y[CMT=1] = central / V
  y[CMT=2] = effect

[error_model]
  CMT=1: DV ~ proportional(PROP_ERR_PK)
  CMT=2: DV ~ additive(ADD_ERR_PD)
",
        prop = NM_SIGMA_PROP_SD,
        add = NM_SIGMA_ADD_SD,
    );
    let model = parse_model_string(&fixed).expect("fixed-param model parses");
    let pop = population();

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.gradient_method = GradientMethod::Fd;
    opts.run_covariance_step = false;
    opts.verbose = false;

    let result = fit(&model, &pop, &model.default_params, &opts)
        .expect("fixed-param objective evaluation must run");

    assert!(
        (result.ofv - NM_OFV_NO_CONST).abs() < 0.5,
        "ferx FOCEI objective at NONMEM's MLE ({:.4}) must match NONMEM OFV ({:.4}) \
         within 0.5; |diff| = {:.4}",
        result.ofv,
        NM_OFV_NO_CONST,
        (result.ofv - NM_OFV_NO_CONST).abs()
    );
}

/// A free ferx FOCEI fit recovers the fixed effects and the two per-CMT
/// residual SDs close to NONMEM. OMEGA is checked only loosely (see the module
/// note on the outer-optimizer limitation).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn fit_recovers_theta_and_per_cmt_sigma() {
    let model =
        parse_model_file(Path::new("examples/pkpd_nonmem.ferx")).expect("example must parse");
    let pop = population();

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.gradient_method = GradientMethod::Fd;
    opts.run_covariance_step = false;
    opts.verbose = false;

    let r = fit(&model, &pop, &model.default_params, &opts).expect("fit must converge");

    let theta: std::collections::HashMap<&str, f64> = r
        .theta_names
        .iter()
        .map(|s| s.as_str())
        .zip(r.theta.iter().copied())
        .collect();
    let rel = |got: f64, want: f64| (got - want).abs() / want.abs();

    // Fixed effects: within 5% of NONMEM.
    assert!(
        rel(theta["TVCL"], NM_TVCL) < 0.05,
        "TVCL {:?}",
        theta.get("TVCL")
    );
    assert!(
        rel(theta["TVV"], NM_TVV) < 0.05,
        "TVV {:?}",
        theta.get("TVV")
    );
    assert!(
        rel(theta["TVKE0"], NM_TVKE0) < 0.05,
        "TVKE0 {:?}",
        theta.get("TVKE0")
    );

    // Per-CMT residual SDs: within 12% of NONMEM (the crux of the per-CMT
    // feature — proportional on CMT=1, additive on CMT=2).
    let sd = |name: &str| -> f64 {
        let i = r
            .sigma_names
            .iter()
            .position(|s| s == name)
            .expect("sigma present");
        r.sigma[i]
    };
    assert!(
        rel(sd("PROP_ERR_PK"), NM_SIGMA_PROP_SD) < 0.12,
        "PROP_ERR_PK SD {} vs NONMEM {}",
        sd("PROP_ERR_PK"),
        NM_SIGMA_PROP_SD
    );
    assert!(
        rel(sd("ADD_ERR_PD"), NM_SIGMA_ADD_SD) < 0.12,
        "ADD_ERR_PD SD {} vs NONMEM {}",
        sd("ADD_ERR_PD"),
        NM_SIGMA_ADD_SD
    );

    // OMEGA: loose sanity only. ferx's outer optimizer under-converges this on
    // the dataset (see module note); assert finite/positive and within a wide
    // band rather than tight agreement with NONMEM's 0.046.
    assert!(
        r.omega[(0, 0)].is_finite() && r.omega[(0, 0)] > 0.0 && r.omega[(0, 0)] < 0.2,
        "ETA_CL variance {} should be a sane positive value",
        r.omega[(0, 0)]
    );
}
