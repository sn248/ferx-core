//! Integration tests for issue #205: compartment state vector in DerivedContext.
//!
//! Tier 2: fast-returning tests (max 2 outer iterations or immediate parse/check).
//! Tier 3 slow tests are gated with `#[cfg_attr(not(feature = "slow-tests"), ignore)]`.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::read_nonmem_csv;
use ferx_core::types::{DoseEvent, Population, Subject};
use ferx_core::{fit, FitOptions};
use std::collections::HashMap;
use std::path::Path;

fn simple_iv_population() -> Population {
    let obs_times = vec![1.0, 4.0, 12.0, 24.0];
    let n = obs_times.len();
    Population {
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times,
            obs_raw_times: vec![],
            observations: vec![5.0, 3.0, 1.5, 0.7],
            obs_cmts: vec![1; n],
            covariates: HashMap::new(),
            dose_covariates: vec![],
            obs_covariates: vec![],
            pk_only_times: vec![],
            pk_only_covariates: vec![],
            reset_times: vec![],
            cens: vec![0; n],
            occasions: vec![],
            dose_occasions: vec![],
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }],
    }
}

/// 1-cpt IV model: `compartments[0]` (central concentration) must equal IPRED
/// for every observation (no scaling applied).
#[test]
fn analytical_1cpt_iv_compartments_equals_ipred() {
    const MODEL: &str = "
[parameters]
  theta CL(2.0, 0.01, 50.0)
  theta V(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V  = V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)

[derived]
  C_central = compartments[0]

[fit_options]
  method  = focei
  maxiter = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = simple_iv_population();
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        let c_central = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "C_central")
            .expect("C_central column must exist");
        assert_eq!(
            c_central.1.len(),
            sr.ipred.len(),
            "C_central must have same length as IPRED"
        );
        for (j, (&cmpt, &ip)) in c_central.1.iter().zip(sr.ipred.iter()).enumerate() {
            assert!(
                (cmpt - ip).abs() < 1e-10,
                "subject {}: compartments[0] at obs {j}: {cmpt:.6} != ipred {ip:.6}",
                sr.id
            );
        }
    }
}

/// 1-cpt IV model: named compartment access `central` must equal IPRED.
#[test]
fn analytical_1cpt_iv_named_access_equals_ipred() {
    const MODEL: &str = "
[parameters]
  theta CL(2.0, 0.01, 50.0)
  theta V(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V  = V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)

[derived]
  C_named = central

[fit_options]
  method  = focei
  maxiter = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = simple_iv_population();
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        let col = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "C_named")
            .expect("C_named column must exist");
        for (j, (&v, &ip)) in col.1.iter().zip(sr.ipred.iter()).enumerate() {
            assert!(
                (v - ip).abs() < 1e-10,
                "subject {}: central at obs {j}: {v:.6} != ipred {ip:.6}",
                sr.id
            );
        }
    }
}

/// 2-cpt IV model: `compartments[1]` (peripheral concentration) must be
/// finite and positive at every observation time.
#[test]
fn analytical_2cpt_iv_peripheral_is_finite_and_positive() {
    const MODEL: &str = "
[parameters]
  theta CL(2.0, 0.01, 50.0)
  theta V1(10.0, 0.1, 500.0)
  theta Q(1.0, 0.01, 50.0)
  theta V2(20.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V1 = V1
  Q  = Q
  V2 = V2

[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)

[error_model]
  DV ~ proportional(PROP)

[derived]
  C_periph = compartments[1]

[fit_options]
  method  = focei
  maxiter = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = simple_iv_population();
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        let col = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "C_periph")
            .expect("C_periph column must exist");
        for (j, &v) in col.1.iter().enumerate() {
            assert!(
                v.is_finite() && v >= 0.0,
                "subject {}: C_periph at obs {j} should be finite/non-negative, got {v}",
                sr.id
            );
        }
    }
}

/// 1-cpt oral model: `compartments[0]` is depot amount (>= 0),
/// `compartments[1]` is central concentration (= IPRED for no-scaling model).
#[test]
fn analytical_1cpt_oral_depot_and_central() {
    let obs_times = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let n = obs_times.len();
    let pop = Population {
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times,
            obs_raw_times: vec![],
            observations: vec![1.0, 2.0, 3.5, 4.0, 3.0, 2.0, 0.5],
            obs_cmts: vec![1; n],
            covariates: HashMap::new(),
            dose_covariates: vec![],
            obs_covariates: vec![],
            pk_only_times: vec![],
            pk_only_covariates: vec![],
            reset_times: vec![],
            cens: vec![0; n],
            occasions: vec![],
            dose_occasions: vec![],
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }],
    };

    const MODEL: &str = "
[parameters]
  theta CL(2.0, 0.01, 50.0)
  theta V(10.0, 0.1, 500.0)
  theta KA(1.0, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V  = V
  KA = KA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)

[derived]
  A_depot   = compartments[0]
  C_central = compartments[1]

[fit_options]
  method  = focei
  maxiter = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        let depot_col = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "A_depot")
            .expect("A_depot must exist");
        let central_col = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "C_central")
            .expect("C_central must exist");
        for (j, &d) in depot_col.1.iter().enumerate() {
            assert!(
                d >= 0.0,
                "depot amount at obs {j} must be non-negative, got {d}"
            );
        }
        for (j, (&c, &ip)) in central_col.1.iter().zip(sr.ipred.iter()).enumerate() {
            assert!(
                (c - ip).abs() < 1e-10,
                "C_central at obs {j}: {c:.6} != ipred {ip:.6}"
            );
        }
    }
}

/// Parse-time check: `compartments[0]` in an integral sets `uses_compartments=true`.
/// This is a fast Tier 2 test — no fit required.
#[test]
fn parse_integral_compartment_subscript_sets_uses_compartments() {
    const MODEL: &str = "
[parameters]
  theta CL(1.0, 0, 100)
  theta V(10.0, 0, 1000)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V  = V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)

[derived]
  AUC_C0 = integral(compartments[0], from=0, to=24)
  AUC_IP = integral(IPRED, from=0, to=24)
";
    let result = parse_model_string(MODEL).expect("model must parse");
    let derived = &result.derived_exprs;

    let auc_c0 = derived
        .iter()
        .find(|s| s.name == "AUC_C0")
        .expect("AUC_C0 must be present");
    if let ferx_core::types::DerivedKind::Integral {
        uses_compartments, ..
    } = &auc_c0.kind
    {
        assert!(uses_compartments, "AUC_C0 must have uses_compartments=true");
    } else {
        panic!("AUC_C0 must be Integral kind");
    }

    let auc_ip = derived
        .iter()
        .find(|s| s.name == "AUC_IP")
        .expect("AUC_IP must be present");
    if let ferx_core::types::DerivedKind::Integral {
        uses_compartments, ..
    } = &auc_ip.kind
    {
        assert!(
            !uses_compartments,
            "AUC_IP must have uses_compartments=false"
        );
    } else {
        panic!("AUC_IP must be Integral kind");
    }
}

// ── ODE model tests ───────────────────────────────────────────────────────────
//
// These cover code paths that the analytical tests above cannot reach:
//   • `ode_predictions_with_states`   (src/ode/predictions.rs)
//   • `ode_dense_solve_states`        (src/ode/predictions.rs)
//   • `compute_predictions_with_states` ODE branch (src/pk/mod.rs)
//   • `compute_extra_output_columns`  with `uses_compartments=true` (src/api.rs)
//
// Model: 2-cpt IV + effect compartment (3 ODE states).
//   State 0: central    (mg, amount)
//   State 1: peripheral (mg, amount)
//   State 2: effect     (mg/L, concentration, as written in the ODE)
// Dose lands in CMT=1 → central. Observations are from central (scaled by V1).

const ODE_2CPT_EFFECT: &str = "
[parameters]
  theta CL(2.0, 0.01, 50.0)
  theta V1(10.0, 0.1, 500.0)
  theta Q(1.0, 0.01, 50.0)
  theta V2(20.0, 0.1, 1000.0)
  theta KE0(0.5, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL  = CL * exp(ETA_CL)
  V1  = V1
  Q   = Q
  V2  = V2
  KE0 = KE0

[structural_model]
  ode(states=[central, peripheral, effect])

[odes]
  d/dt(central)    = -(CL/V1 + Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) = (Q/V1) * central - (Q/V2) * peripheral
  d/dt(effect)     = KE0 * (central/V1 - effect)

[scaling]
  y = central / V1

[error_model]
  DV ~ proportional(PROP)
";

/// ODE 2-cpt + effect compartment: `compartments[2]` (subscript) and named
/// state `effect` must yield identical, finite, positive values at every
/// observation. Exercises `ode_predictions_with_states` end-to-end.
#[test]
fn ode_2cpt_effect_compartment_derived() {
    let model_str = format!(
        "{ODE_2CPT_EFFECT}
[derived]
  Ce       = compartments[2]
  Ce_named = effect

[fit_options]
  method   = focei
  maxiter  = 2
  gradient = fd
"
    );
    let model = parse_model_string(&model_str).expect("model must parse");
    let pop = simple_iv_population();
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        let ce_col = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "Ce")
            .expect("Ce column must exist");
        let ce_named_col = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "Ce_named")
            .expect("Ce_named column must exist");

        assert_eq!(
            ce_col.1.len(),
            sr.ipred.len(),
            "Ce must have same length as IPRED"
        );

        for (j, (&by_idx, &by_name)) in ce_col.1.iter().zip(ce_named_col.1.iter()).enumerate() {
            assert!(
                by_idx.is_finite() && by_idx > 0.0,
                "subject {}: Ce (compartments[2]) at obs {j} must be finite+positive, \
                 got {by_idx}",
                sr.id
            );
            assert!(
                (by_idx - by_name).abs() < 1e-10,
                "subject {}: subscript compartments[2]={by_idx:.8} != named \
                 effect={by_name:.8} at obs {j}",
                sr.id
            );
        }
    }
}

/// ODE effect compartment: `integral(compartments[2], ...)` and
/// `integral(effect, ...)` both trigger the `uses_compartments=true`
/// grid-integral re-solve path. Both AUCs must be finite, positive, and
/// identical. Exercises `ode_dense_solve_states` via the grid-integral path.
#[test]
fn ode_integral_over_compartment() {
    let model_str = format!(
        "{ODE_2CPT_EFFECT}
[derived]
  AUC_Ce    = integral(compartments[2], from=0, to=24, step=1.0)
  AUC_named = integral(effect, from=0, to=24, step=1.0)

[fit_options]
  method   = focei
  maxiter  = 2
  gradient = fd
"
    );
    let model = parse_model_string(&model_str).expect("model must parse");

    // Parse-time check: both integrands reference a compartment state, so
    // uses_compartments must be true for each.
    for spec in &model.derived_exprs {
        if let ferx_core::types::DerivedKind::Integral {
            uses_compartments, ..
        } = &spec.kind
        {
            assert!(
                uses_compartments,
                "{} integral must have uses_compartments=true (references ODE state)",
                spec.name
            );
        }
    }

    let pop = simple_iv_population();
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        let auc_col = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "AUC_Ce")
            .expect("AUC_Ce column must exist");
        let auc_named_col = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "AUC_named")
            .expect("AUC_named column must exist");

        for (j, &auc) in auc_col.1.iter().enumerate() {
            assert!(
                auc.is_finite() && auc > 0.0,
                "subject {}: AUC_Ce at obs {j} must be finite+positive, got {auc}",
                sr.id
            );
        }
        // Named-state integral and subscript integral must agree.
        for (j, (&by_idx, &by_name)) in auc_col.1.iter().zip(auc_named_col.1.iter()).enumerate() {
            assert!(
                (by_idx - by_name).abs() < 1e-8,
                "subject {}: AUC_Ce={by_idx:.6} != AUC_named={by_name:.6} at obs {j}",
                sr.id
            );
        }
    }
}

// ── Analytical peripheral-state tests ──────────────────────────────────────────
//
// These cover every multi-compartment analytical model variant:
//   two_cpt_oral  → [depot, central, peripheral]
//   three_cpt_iv  → [central, peripheral1, peripheral2]
//   three_cpt_oral → [depot, central, peripheral1, peripheral2]
//
// Shared oral population (100 mg oral dose, 7 observation times):

fn simple_oral_population() -> Population {
    let obs_times = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let n = obs_times.len();
    Population {
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times,
            obs_raw_times: vec![],
            observations: vec![1.0, 2.0, 3.5, 4.0, 3.0, 2.0, 0.5],
            obs_cmts: vec![1; n],
            covariates: HashMap::new(),
            dose_covariates: vec![],
            obs_covariates: vec![],
            pk_only_times: vec![],
            pk_only_covariates: vec![],
            reset_times: vec![],
            cens: vec![0; n],
            occasions: vec![],
            dose_occasions: vec![],
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }],
    }
}

/// 2-cpt oral: compartments[0] = depot (amount >= 0), compartments[1] = central
/// (= IPRED), compartments[2] = peripheral (>= 0).  Named access `depot`,
/// `central`, `peripheral` must each match the corresponding subscript form.
#[test]
fn analytical_2cpt_oral_peripheral_states() {
    const MODEL: &str = "
[parameters]
  theta CL(2.0, 0.01, 50.0)
  theta V1(10.0, 0.1, 500.0)
  theta Q(1.0, 0.01, 50.0)
  theta V2(20.0, 0.1, 1000.0)
  theta KA(1.0, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V1 = V1
  Q  = Q
  V2 = V2
  KA = KA

[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)

[error_model]
  DV ~ proportional(PROP)

[derived]
  A_depot      = compartments[0]
  C_central    = compartments[1]
  C_periph     = compartments[2]
  depot_named  = depot
  cent_named   = central
  periph_named = peripheral

[fit_options]
  method   = focei
  maxiter  = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = simple_oral_population();
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        macro_rules! col {
            ($name:expr) => {
                sr.extra_columns
                    .iter()
                    .find(|(n, _)| n == $name)
                    .unwrap_or_else(|| panic!("{} column must exist", $name))
                    .1
                    .as_slice()
            };
        }
        let depot = col!("A_depot");
        let central = col!("C_central");
        let periph = col!("C_periph");
        let depot_n = col!("depot_named");
        let cent_n = col!("cent_named");
        let periph_n = col!("periph_named");

        for (j, &d) in depot.iter().enumerate() {
            assert!(
                d >= 0.0,
                "subject {}: depot at obs {j} must be >= 0, got {d}",
                sr.id
            );
        }
        for (j, (&c, &ip)) in central.iter().zip(sr.ipred.iter()).enumerate() {
            assert!(
                (c - ip).abs() < 1e-10,
                "subject {}: compartments[1] = {c:.8} but ipred = {ip:.8} at obs {j}",
                sr.id
            );
        }
        for (j, &p) in periph.iter().enumerate() {
            assert!(
                p >= 0.0,
                "subject {}: peripheral at obs {j} must be >= 0, got {p}",
                sr.id
            );
        }
        // Named access must equal subscript access.
        for (j, (&d0, &dn)) in depot.iter().zip(depot_n.iter()).enumerate() {
            assert!(
                (d0 - dn).abs() < 1e-12,
                "subject {}: compartments[0]={d0:.8} != depot={dn:.8} at obs {j}",
                sr.id
            );
        }
        for (j, (&c0, &cn)) in central.iter().zip(cent_n.iter()).enumerate() {
            assert!(
                (c0 - cn).abs() < 1e-12,
                "subject {}: compartments[1]={c0:.8} != central={cn:.8} at obs {j}",
                sr.id
            );
        }
        for (j, (&p0, &pn)) in periph.iter().zip(periph_n.iter()).enumerate() {
            assert!(
                (p0 - pn).abs() < 1e-12,
                "subject {}: compartments[2]={p0:.8} != peripheral={pn:.8} at obs {j}",
                sr.id
            );
        }
    }
}

/// 3-cpt IV: compartments[0] = central (= IPRED), compartments[1] = peripheral1
/// (>= 0), compartments[2] = peripheral2 (>= 0).  Named access `central`,
/// `peripheral1`, `peripheral2` must each match the corresponding subscript form.
#[test]
fn analytical_3cpt_iv_peripheral_states() {
    const MODEL: &str = "
[parameters]
  theta CL(5.0, 0.1, 100.0)
  theta V1(10.0, 0.1, 500.0)
  theta Q2(2.0, 0.01, 50.0)
  theta V2(20.0, 0.1, 1000.0)
  theta Q3(1.5, 0.01, 50.0)
  theta V3(30.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.04 (sd)

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V1 = V1
  Q2 = Q2
  V2 = V2
  Q3 = Q3
  V3 = V3

[structural_model]
  pk three_cpt_iv(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3)

[error_model]
  DV ~ proportional(PROP)

[derived]
  C_central  = compartments[0]
  C_periph1  = compartments[1]
  C_periph2  = compartments[2]
  cent_named = central
  p1_named   = peripheral1
  p2_named   = peripheral2

[fit_options]
  method   = focei
  maxiter  = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = simple_iv_population();
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        macro_rules! col {
            ($name:expr) => {
                sr.extra_columns
                    .iter()
                    .find(|(n, _)| n == $name)
                    .unwrap_or_else(|| panic!("{} column must exist", $name))
                    .1
                    .as_slice()
            };
        }
        let central = col!("C_central");
        let periph1 = col!("C_periph1");
        let periph2 = col!("C_periph2");
        let cent_n = col!("cent_named");
        let p1_n = col!("p1_named");
        let p2_n = col!("p2_named");

        for (j, (&c, &ip)) in central.iter().zip(sr.ipred.iter()).enumerate() {
            assert!(
                (c - ip).abs() < 1e-10,
                "subject {}: compartments[0] = {c:.8} but ipred = {ip:.8} at obs {j}",
                sr.id
            );
        }
        for (j, &p) in periph1.iter().enumerate() {
            assert!(
                p >= 0.0,
                "subject {}: peripheral1 at obs {j} must be >= 0, got {p}",
                sr.id
            );
        }
        for (j, &p) in periph2.iter().enumerate() {
            assert!(
                p >= 0.0,
                "subject {}: peripheral2 at obs {j} must be >= 0, got {p}",
                sr.id
            );
        }
        // Named access must equal subscript access.
        for (j, (&c0, &cn)) in central.iter().zip(cent_n.iter()).enumerate() {
            assert!(
                (c0 - cn).abs() < 1e-12,
                "subject {}: compartments[0]={c0:.8} != central={cn:.8} at obs {j}",
                sr.id
            );
        }
        for (j, (&p0, &pn)) in periph1.iter().zip(p1_n.iter()).enumerate() {
            assert!(
                (p0 - pn).abs() < 1e-12,
                "subject {}: compartments[1]={p0:.8} != peripheral1={pn:.8} at obs {j}",
                sr.id
            );
        }
        for (j, (&p0, &pn)) in periph2.iter().zip(p2_n.iter()).enumerate() {
            assert!(
                (p0 - pn).abs() < 1e-12,
                "subject {}: compartments[2]={p0:.8} != peripheral2={pn:.8} at obs {j}",
                sr.id
            );
        }
    }
}

/// 3-cpt oral: compartments[0] = depot (amount >= 0), compartments[1] = central
/// (= IPRED), compartments[2] = peripheral1 (>= 0), compartments[3] = peripheral2
/// (>= 0).  Named access matches subscript for all four compartments.
#[test]
fn analytical_3cpt_oral_peripheral_states() {
    const MODEL: &str = "
[parameters]
  theta CL(5.0, 0.1, 100.0)
  theta V1(10.0, 0.1, 500.0)
  theta Q2(2.0, 0.01, 50.0)
  theta V2(20.0, 0.1, 1000.0)
  theta Q3(1.5, 0.01, 50.0)
  theta V3(30.0, 0.1, 1000.0)
  theta KA(1.0, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.04 (sd)

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V1 = V1
  Q2 = Q2
  V2 = V2
  Q3 = Q3
  V3 = V3
  KA = KA

[structural_model]
  pk three_cpt_oral(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3, ka=KA)

[error_model]
  DV ~ proportional(PROP)

[derived]
  A_depot    = compartments[0]
  C_central  = compartments[1]
  C_periph1  = compartments[2]
  C_periph2  = compartments[3]
  depot_n    = depot
  cent_n     = central
  p1_n       = peripheral1
  p2_n       = peripheral2

[fit_options]
  method   = focei
  maxiter  = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = simple_oral_population();
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        macro_rules! col {
            ($name:expr) => {
                sr.extra_columns
                    .iter()
                    .find(|(n, _)| n == $name)
                    .unwrap_or_else(|| panic!("{} column must exist", $name))
                    .1
                    .as_slice()
            };
        }
        let depot = col!("A_depot");
        let central = col!("C_central");
        let periph1 = col!("C_periph1");
        let periph2 = col!("C_periph2");
        let depot_n = col!("depot_n");
        let cent_n = col!("cent_n");
        let p1_n = col!("p1_n");
        let p2_n = col!("p2_n");

        for (j, &d) in depot.iter().enumerate() {
            assert!(
                d >= 0.0,
                "subject {}: depot at obs {j} must be >= 0, got {d}",
                sr.id
            );
        }
        for (j, (&c, &ip)) in central.iter().zip(sr.ipred.iter()).enumerate() {
            assert!(
                (c - ip).abs() < 1e-10,
                "subject {}: compartments[1] = {c:.8} but ipred = {ip:.8} at obs {j}",
                sr.id
            );
        }
        for (j, &p) in periph1.iter().enumerate() {
            assert!(
                p >= 0.0,
                "subject {}: peripheral1 at obs {j} must be >= 0, got {p}",
                sr.id
            );
        }
        for (j, &p) in periph2.iter().enumerate() {
            assert!(
                p >= 0.0,
                "subject {}: peripheral2 at obs {j} must be >= 0, got {p}",
                sr.id
            );
        }
        // Named access must equal subscript access.
        for (j, (&d0, &dn)) in depot.iter().zip(depot_n.iter()).enumerate() {
            assert!(
                (d0 - dn).abs() < 1e-12,
                "subject {}: compartments[0]={d0:.8} != depot={dn:.8} at obs {j}",
                sr.id
            );
        }
        for (j, (&c0, &cn)) in central.iter().zip(cent_n.iter()).enumerate() {
            assert!(
                (c0 - cn).abs() < 1e-12,
                "subject {}: compartments[1]={c0:.8} != central={cn:.8} at obs {j}",
                sr.id
            );
        }
        for (j, (&p0, &pn)) in periph1.iter().zip(p1_n.iter()).enumerate() {
            assert!(
                (p0 - pn).abs() < 1e-12,
                "subject {}: compartments[2]={p0:.8} != peripheral1={pn:.8} at obs {j}",
                sr.id
            );
        }
        for (j, (&p0, &pn)) in periph2.iter().zip(p2_n.iter()).enumerate() {
            assert!(
                (p0 - pn).abs() < 1e-12,
                "subject {}: compartments[3]={p0:.8} != peripheral2={pn:.8} at obs {j}",
                sr.id
            );
        }
    }
}

// ── Tier 2: AUC_Ce analytical reference check ─────────────────────────────────
//
// Verifies that `ode_dense_solve_states` (the grid-integral code path, activated
// when an integrand references a compartment state) produces AUC_Ce values that
// agree with the closed-form integral to within the expected trapezoidal error.
//
// Model: 1-cpt IV ODE + effect compartment (2 ODE states):
//   d/dt(central) = -(CL/V) * central
//   d/dt(effect)  = KE0 * (central/V − effect)
//
// Closed-form for a single IV dose D, ke = CL/V, c0 = D/V:
//   Ce(t) = c0 · KE0/(KE0−ke) · (exp(−ke·t) − exp(−KE0·t))
//
//   AUC_Ce(0→T) = c0·KE0/(KE0−ke) · [(1−e^{−ke·T})/ke − (1−e^{−KE0·T})/KE0]
//
// The test extracts the individual-level CL_i, V_i, and KE0 from the [derived]
// block (post-EBE values), computes the analytical AUC_Ce at those parameters,
// and compares with the ferx grid-integral result (step=0.5 h).  Agreement to
// within 3% is expected (trapezoidal error at step=0.5 plus small EBE movement).

fn analytical_auc_ce_1cpt_iv(dose: f64, cl: f64, v: f64, ke0: f64, t_end: f64) -> f64 {
    let c0 = dose / v;
    let ke = cl / v;
    if (ke - ke0).abs() < 1e-6 {
        // L'Hôpital limit: AUC_Ce = c0·KE0·[t·e^{−ke·t}/ke + (e^{−ke·t}−1)/ke²] ... +∞ integrands
        // For the finite-window case: AUC_Ce ≈ AUC_C (degenerate; return AUC_C as upper bound)
        return c0 / ke * (1.0 - (-ke * t_end).exp());
    }
    let factor = c0 * ke0 / (ke0 - ke);
    let integral_ke = (1.0 - (-ke * t_end).exp()) / ke;
    let integral_ke0 = (1.0 - (-ke0 * t_end).exp()) / ke0;
    factor * (integral_ke - integral_ke0)
}

/// Tier 2 analytical reference: 1-cpt IV ODE + effect compartment.
///
/// Extracts post-fit individual CL and V from `[derived]`, computes the
/// closed-form AUC_Ce, and verifies the ferx grid integral (step=0.5) agrees
/// to within 3% trapezoidal error.
#[test]
fn ode_auc_ce_matches_analytical_reference() {
    // Tight KE0 bounds keep it near 0.5 regardless of what theta CL/V do.
    const MODEL: &str = "
[parameters]
  theta CL(2.0, 0.01, 50.0)
  theta V(10.0, 0.1, 500.0)
  theta KE0(0.5, 0.45, 0.55)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL  = CL * exp(ETA_CL)
  V   = V
  KE0 = KE0

[structural_model]
  ode(states=[central, effect])

[odes]
  d/dt(central) = -(CL/V) * central
  d/dt(effect)  = KE0 * (central/V - effect)

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP)

[derived]
  AUC_Ce  = integral(effect, from=0, to=24, step=0.5)
  CL_indv = CL
  V_indv  = V
  KE0_val = KE0

[fit_options]
  method   = focei
  maxiter  = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = simple_iv_population(); // dose=100 at t=0, obs at 1/4/12/24
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit must not error");

    for sr in &result.subjects {
        macro_rules! col {
            ($name:expr) => {
                sr.extra_columns
                    .iter()
                    .find(|(n, _)| n == $name)
                    .unwrap_or_else(|| panic!("{} column must exist", $name))
                    .1[0]
            };
        }
        // Individual-level PK at last observation row (they're per-row constants, take any).
        let cl_i = col!("CL_indv");
        let v_i = col!("V_indv");
        let ke0 = col!("KE0_val");

        // Grid-integral AUC_Ce from ferx.
        let auc_ce_ferx = col!("AUC_Ce");

        // Closed-form reference at the same individual parameters.
        let auc_ce_ref = analytical_auc_ce_1cpt_iv(100.0, cl_i, v_i, ke0, 24.0);

        assert!(
            auc_ce_ferx.is_finite() && auc_ce_ferx > 0.0,
            "subject {}: AUC_Ce must be finite+positive, got {auc_ce_ferx}",
            sr.id
        );
        let rel_err = (auc_ce_ferx - auc_ce_ref).abs() / auc_ce_ref;
        assert!(
            rel_err < 0.03,
            "subject {}: ferx AUC_Ce={auc_ce_ferx:.4} vs analytical={auc_ce_ref:.4} \
             (relative error {:.2}% > 3%)",
            sr.id,
            rel_err * 100.0
        );
    }
}

// ── Tier 3 slow: full convergence + AUC_Ce analytical reference ───────────────
//
// Fits a 2-cpt IV ODE model extended with an effect compartment to the
// `data/two_cpt_iv.csv` dataset (18 subjects, single IV bolus, bi-exponential
// decline).  The effect compartment is not observed; it is estimated purely for
// its compartment-state AUC:
//
//   AUC_Ce = integral(effect, from=0, to=24, step=0.5)
//
// **Analytical reference** (population-average parameters near the generating
// values CL=4 L/h, V1=12 L, Q=2 L/h, V2=25 L, KE0=0.5 h⁻¹, dose=100 mg):
//
//   Eigenvalues:
//     α = (k₁₀+k₁₂+k₂₁ + √[(k₁₀+k₁₂+k₂₁)²−4·k₁₀·k₂₁]) / 2  ≈ 0.530 h⁻¹
//     β = (k₁₀+k₁₂+k₂₁ − √[(k₁₀+k₁₂+k₂₁)²−4·k₁₀·k₂₁]) / 2  ≈ 0.050 h⁻¹
//   (k₁₀ = CL/V₁ = 0.333, k₁₂ = Q/V₁ = 0.167, k₂₁ = Q/V₂ = 0.080)
//
//   AUC_Ce(0→24) ≈ 21.6 mg/L·h    (ferx grid step=0.5, trapezoidal error <1%)
//   AUC_central(0→24) ≈ 21.9 mg/L·h
//
// Note: NONMEM cannot directly evaluate an integral over an effect-compartment
// state without custom $DES/$PK code; the closed-form formula above is the
// appropriate analytical reference for this derived variable.
//
// The test asserts:
//   1. Fit converges (finite OFV).
//   2. Each subject's AUC_Ce and AUC_central are finite and positive.
//   3. AUC_Ce ≤ AUC_central × 1.05 — the effect compartment can briefly exceed
//      the central concentration, but the cumulative area under the curve should
//      remain close to the central AUC over [0, 24] for any KE0 > 0 (because at
//      t = 24 h all drug is nearly gone and ∫₀^∞ Ce = ∫₀^∞ C for linear PK).
//   4. AUC_Ce / AUC_central is in [0.70, 1.05] for each subject (the lower
//      bound corresponds to very slow KE0 ≈ 0.01 with the fast 2-cpt elimination;
//      the upper bound is 1.05 to accommodate brief Ce > C_central periods).

const TWO_CPT_IV_ODE_EFFECT_MODEL: &str = "
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0, 0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  theta TVKE0(0.5, 0.01, 10.0)

  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15

  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V1  = TVV1 * exp(ETA_V1)
  Q   = TVQ
  V2  = TVV2
  KE0 = TVKE0

[structural_model]
  ode(states=[central, peripheral, effect])

[odes]
  d/dt(central)    = -(CL/V1 + Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) = (Q/V1) * central - (Q/V2) * peripheral
  d/dt(effect)     = KE0 * (central/V1 - effect)

[scaling]
  y = central / V1

[error_model]
  DV ~ proportional(PROP_ERR)

[derived]
  AUC_Ce      = integral(effect, from=0, to=24, step=0.5)
  AUC_central = integral(IPRED, from=0, to=24)

[fit_options]
  method   = focei
  maxiter  = 300
  gradient = fd
";

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn auc_ce_2cpt_iv_effect_convergence_and_analytical_ref() {
    let model = parse_model_string(TWO_CPT_IV_ODE_EFFECT_MODEL).expect("model must parse");
    let population = read_nonmem_csv(Path::new("data/two_cpt_iv.csv"), None, None)
        .expect("two_cpt_iv data must load");

    let mut opts = FitOptions::default();
    opts.verbose = false;
    opts.run_covariance_step = false;
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("fit must not error");

    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );

    let mut auc_ce_vec: Vec<f64> = Vec::new();
    let mut auc_central_vec: Vec<f64> = Vec::new();

    for sr in &result.subjects {
        // Each row holds the same aggregate AUC value; take the first.
        let auc_ce = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "AUC_Ce")
            .expect("AUC_Ce column must exist")
            .1[0];
        let auc_central = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n == "AUC_central")
            .expect("AUC_central column must exist")
            .1[0];

        assert!(
            auc_ce.is_finite() && auc_ce > 0.0,
            "subject {}: AUC_Ce must be finite+positive, got {auc_ce}",
            sr.id
        );
        assert!(
            auc_central.is_finite() && auc_central > 0.0,
            "subject {}: AUC_central must be finite+positive, got {auc_central}",
            sr.id
        );

        // For linear PK: ∫₀^∞ Ce = ∫₀^∞ C_central.  By t=24 h, nearly all
        // drug is eliminated (exp(-β·24) ≈ 0.30 for β≈0.05), so AUC(0→24)
        // captures ≥ 88% of AUC(0→∞).  Consequently AUC_Ce and AUC_central
        // should be within 5% of each other for any KE0 ≥ 0.01 at T=24.
        // We allow slightly wider bounds to accommodate subjects with high ETA_V1
        // (slower β, more drug remaining at t=24) and any KE0 value.
        let ratio = auc_ce / auc_central;
        assert!(
            ratio >= 0.70 && ratio <= 1.05,
            "subject {}: AUC_Ce/AUC_central = {ratio:.4} (expected [0.70, 1.05]); \
             AUC_Ce={auc_ce:.3}, AUC_central={auc_central:.3}",
            sr.id
        );

        auc_ce_vec.push(auc_ce);
        auc_central_vec.push(auc_central);
    }

    // Median AUC_Ce across subjects should be near the analytical reference
    // of ≈21.6 mg/L·h for population-average parameters; allow a wide range
    // [12, 35] to accommodate IIV in CL/V1 and the unidentified KE0.
    auc_ce_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_auc_ce = if auc_ce_vec.len() % 2 == 0 {
        (auc_ce_vec[auc_ce_vec.len() / 2 - 1] + auc_ce_vec[auc_ce_vec.len() / 2]) / 2.0
    } else {
        auc_ce_vec[auc_ce_vec.len() / 2]
    };
    assert!(
        median_auc_ce >= 12.0 && median_auc_ce <= 35.0,
        "median AUC_Ce = {median_auc_ce:.3} mg/L·h is outside the expected range \
         [12, 35] for the two_cpt_iv dataset; analytical reference ≈ 21.6 mg/L·h \
         at population-average parameters (CL=4, V1=12, Q=2, V2=25, KE0=0.5)"
    );

    // Summarise for --nocapture inspection.
    auc_central_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_auc_c = auc_central_vec[auc_central_vec.len() / 2];
    println!(
        "[auc_ce_2cpt_iv_effect] OFV={:.2}  n_subj={}  \
         median AUC_Ce={median_auc_ce:.2}  median AUC_central={median_auc_c:.2}",
        result.ofv,
        result.subjects.len(),
    );
}
