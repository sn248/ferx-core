//! Equivalence between the analytical `[initial_conditions]` block (issue #521)
//! and its ODE transcription `init(state) = <expr>` in `[odes]`.
//!
//! A non-zero initial compartment amount on an analytical (closed-form) model
//! is layered on as the closed-form impulse of an F-bypassed bolus. This test
//! builds a 1-cpt oral model with a parameter-dependent central baseline
//! (`init(central) = 30 * V`) plus an oral dose, then `predict()`s both the
//! analytical form and a hand-written ODE twin and asserts the population PRED
//! agrees pointwise. This is the same structural-equivalence strategy as
//! `analytical_ode_equivalence.rs`, extended to the initial-condition path.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::predict;
use ferx_core::types::{DoseEvent, Population};

mod common;

// RK45 defaults (abstol 1e-6, reltol 1e-4) cap how exactly the ODE twin can
// reproduce the closed form; use a combined absolute+relative bound so the
// near-zero early-absorption points don't blow up a pure relative check.
const ATOL: f64 = 1e-5;
const RTOL: f64 = 1e-4;

const THETAS_AND_INDIV: &str = r"
[parameters]
  theta TVCL(3.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  theta TVKA(1.0, 0.01, 50.0)

  omega ETA_CL ~ 0.09

  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
";

fn analytical_src() -> String {
    format!(
        "{THETAS_AND_INDIV}
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[initial_conditions]
  init(central) = 30 * V

[error_model]
  DV ~ proportional(PROP_ERR)
"
    )
}

fn ode_src() -> String {
    // ODE twin: central state holds an AMOUNT, seeded to 30*V; the observed
    // concentration is amount/V via `obs_scale = V` (NONMEM's S2). This mirrors
    // the analytical model, whose closed form reads concentration directly.
    format!(
        "{THETAS_AND_INDIV}
[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  init(central) = 30 * V
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - CL / V * central

[scaling]
  obs_scale = V

[error_model]
  DV ~ proportional(PROP_ERR)
"
    )
}

fn population() -> Population {
    // One oral dose (depot = cmt 1) plus a baseline already in central; observe
    // across the absorption rise and the baseline decay.
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let obs_times = vec![0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0];
    let n = obs_times.len();
    let subj = common::subject("1", doses, obs_times, vec![0.0; n], vec![2; n]);
    Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![subj],
    }
}

#[test]
fn analytical_central_init_matches_ode_init() {
    let an = parse_full_model(&analytical_src())
        .expect("analytical model parses")
        .model;
    let ode = parse_full_model(&ode_src())
        .expect("ODE model parses")
        .model;

    // The analytical model must actually carry the parsed init.
    assert_eq!(
        an.analytical_init.len(),
        1,
        "[initial_conditions] should populate analytical_init"
    );
    assert!(
        ode.analytical_init.is_empty(),
        "ODE model seeds state via init_fn, not analytical_init"
    );

    let pop = population();
    let pa = predict(&an, &pop, &an.default_params);
    let po = predict(&ode, &pop, &ode.default_params);
    assert_eq!(pa.len(), po.len());
    assert!(!pa.is_empty());

    // At t=0 the baseline dominates: concentration = 30 (amount 30*V over V).
    assert!(
        (pa[0].pred - 30.0).abs() < 1e-6,
        "analytical baseline at t=0 should be 30, got {}",
        pa[0].pred
    );

    for (x, y) in pa.iter().zip(po.iter()) {
        let tol = ATOL + RTOL * x.pred.abs();
        assert!(
            (x.pred - y.pred).abs() <= tol,
            "t={:.3}: analytical PRED {:.6} vs ODE PRED {:.6} (|diff| {:.2e} > tol {:.2e})",
            x.time,
            x.pred,
            y.pred,
            (x.pred - y.pred).abs(),
            tol
        );
    }
}

/// `predict_iov` is the prediction path the importance-sampling (IMP) estimator
/// and the IOV likelihood use — even for non-IOV models. It must carry the
/// initial-compartment amount too, otherwise IMP mispredicts baseline subjects
/// and their importance weights collapse. Here we check that, for the non-IOV
/// init model, `predict_iov` (empty kappas) agrees with the public `predict()`.
#[test]
fn predict_iov_carries_analytical_init() {
    use ferx_core::pk::predict_iov;

    let an = parse_full_model(&analytical_src())
        .expect("analytical model parses")
        .model;
    let pop = population();
    let subject = &pop.subjects[0];

    let public = predict(&an, &pop, &an.default_params);
    // Non-IOV: no kappa groups. `predict_iov` takes the theta slice directly.
    let iov = predict_iov(&an, subject, &an.default_params.theta, &[0.0; 1], &[]);

    assert_eq!(iov.len(), public.len());
    // Baseline at t=0 must survive the predict_iov path (≈30), not be dropped.
    assert!(
        (iov[0] - 30.0).abs() < 1e-6,
        "predict_iov baseline at t=0 should be 30, got {}",
        iov[0]
    );
    for (i, p) in public.iter().enumerate() {
        let tol = ATOL + RTOL * p.pred.abs();
        assert!(
            (iov[i] - p.pred).abs() <= tol,
            "obs {i}: predict_iov {:.6} vs predict() {:.6}",
            iov[i],
            p.pred
        );
    }
}
