//! Integration tests for issue #205: compartment state vector in DerivedContext.
//!
//! Tier 2: fast-returning tests (max 2 outer iterations or immediate parse/check).
//! Tier 3 slow tests are gated with `#[cfg_attr(not(feature = "slow-tests"), ignore)]`.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{DoseEvent, Population, Subject};
use ferx_core::{fit, FitOptions};
use std::collections::HashMap;

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
