//! Equivalence between the analytical PK closed forms and their amount-based
//! ODE transcription (ferx-r issue #127: "standard models - analytical + ode
//! form").
//!
//! For each of the six standard [`PkModel`] types we build an analytical model
//! (`pk <model>(...)`) and an ODE model that is its hand transcription, then
//! `predict()` both on identical populations spanning every dosing mode -
//! bolus, infusion, multiple doses, steady state, lag time and (for oral
//! models) bioavailability - and assert the population PRED agrees pointwise.
//!
//! `predict()` evaluates at eta = 0, so this is a pure check of the *structural*
//! transformation. The transcription rules verified here are:
//!   - ODE states carry **amounts**; the observed concentration is read out via
//!     `[scaling] obs_scale = V` (or `V1` for multi-compartment models) -
//!     equivalent to NONMEM's `S2 = V`.
//!   - inter-compartmental flux uses micro-constants `k10 = CL/V1`,
//!     `k12 = Q2/V1`, `k21 = Q2/V2`, `k13 = Q3/V1`, `k31 = Q3/V3`.
//!   - absorption adds a `depot` state (`-KA*depot` out, `+KA*depot` into
//!     central).
//!   - **bioavailability `F` and lag time are declared as individual
//!     parameters and applied by the engine at the dose** (reserved PK slots) -
//!     they are *never* baked into the `[odes]` RHS. Baking `F` into the flux
//!     would double-count it, since the engine already loads the dose with
//!     `F*AMT` (issue ferx-core #122).
//!
//! Runs on the default/CI feature set (no autodiff) and is fast (no `fit()`),
//! so it is not gated behind `slow-tests`.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::predict;
use ferx_core::types::{DoseEvent, Population, Subject};
use std::collections::HashMap;

/// The RK45 solver defaults to `abstol = 1e-6`, `reltol = 1e-4`
/// (`src/ode/solver.rs`), so the ODE path can only reproduce the exact closed
/// form to about that level. Use a combined absolute+relative tolerance so
/// near-zero predictions (oral early times, multi-dose troughs) don't blow up a
/// pure relative check.
const ATOL: f64 = 1e-5;
const RTOL: f64 = 1e-4;
/// Steady-state predictions accumulate the solver tolerance over the
/// dose-superposition / SS iteration, so they get a slightly looser bound.
const SS_RTOL: f64 = 1e-3;

/// One standard analytical model and the pieces needed to build its ODE twin.
struct Family {
    label: &'static str,
    /// `[parameters]` theta declarations (one per structural parameter).
    thetas: &'static str,
    /// `[individual_parameters]` body (shared verbatim by both forms).
    indiv: &'static str,
    /// Analytical `pk <model>(...` call with the closing paren omitted so
    /// `f=`/`lagtime=` mappings can be appended.
    pk_open: &'static str,
    /// ODE `[structural_model]` line.
    ode_struct: &'static str,
    /// ODE `[odes]` body.
    odes: &'static str,
    /// `[scaling] obs_scale = <expr>` right-hand side (`V` or `V1`).
    obs_scale: &'static str,
    /// Compartment the dose enters (1 = central for IV, 1 = depot for oral).
    dose_cmt: usize,
    /// Compartment observed (central): 1 for IV, 2 for oral.
    obs_cmt: usize,
    is_oral: bool,
}

fn families() -> Vec<Family> {
    vec![
        Family {
            label: "one_cpt_iv",
            thetas: "  theta TVCL(3.0, 0.01, 100.0)\n  theta TVV(20.0, 1.0, 500.0)\n",
            indiv: "  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n",
            pk_open: "pk one_cpt_iv(cl=CL, v=V",
            ode_struct: "ode(obs_cmt=central, states=[central])",
            odes: "  d/dt(central) = -(CL/V) * central\n",
            obs_scale: "V",
            dose_cmt: 1,
            obs_cmt: 1,
            is_oral: false,
        },
        Family {
            label: "one_cpt_oral",
            thetas: "  theta TVCL(0.13, 0.001, 10.0)\n  theta TVV(8.0, 0.1, 500.0)\n  \
                     theta TVKA(1.2, 0.01, 50.0)\n",
            indiv: "  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n  KA = TVKA\n",
            pk_open: "pk one_cpt_oral(cl=CL, v=V, ka=KA",
            ode_struct: "ode(obs_cmt=central, states=[depot, central])",
            odes: "  d/dt(depot)   = -KA * depot\n  \
                   d/dt(central) =  KA * depot - (CL/V) * central\n",
            obs_scale: "V",
            dose_cmt: 1,
            obs_cmt: 2,
            is_oral: true,
        },
        Family {
            label: "two_cpt_iv",
            thetas: "  theta TVCL(3.0, 0.01, 100.0)\n  theta TVV1(15.0, 1.0, 500.0)\n  \
                     theta TVQ(3.0, 0.01, 100.0)\n  theta TVV2(30.0, 1.0, 500.0)\n",
            indiv: "  CL = TVCL * exp(ETA_CL)\n  V1 = TVV1\n  Q  = TVQ\n  V2 = TVV2\n",
            pk_open: "pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2",
            ode_struct: "ode(obs_cmt=central, states=[central, periph])",
            odes: "  d/dt(central) = -(CL/V1 + Q/V1) * central + (Q/V2) * periph\n  \
                   d/dt(periph)  =  (Q/V1) * central - (Q/V2) * periph\n",
            obs_scale: "V1",
            dose_cmt: 1,
            obs_cmt: 1,
            is_oral: false,
        },
        Family {
            label: "two_cpt_oral",
            thetas: "  theta TVCL(3.0, 0.01, 100.0)\n  theta TVV1(15.0, 1.0, 500.0)\n  \
                     theta TVQ(3.0, 0.01, 100.0)\n  theta TVV2(30.0, 1.0, 500.0)\n  \
                     theta TVKA(1.1, 0.01, 50.0)\n",
            indiv: "  CL = TVCL * exp(ETA_CL)\n  V1 = TVV1\n  Q  = TVQ\n  V2 = TVV2\n  KA = TVKA\n",
            pk_open: "pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA",
            ode_struct: "ode(obs_cmt=central, states=[depot, central, periph])",
            odes: "  d/dt(depot)   = -KA * depot\n  \
                   d/dt(central) =  KA * depot - (CL/V1 + Q/V1) * central + (Q/V2) * periph\n  \
                   d/dt(periph)  =  (Q/V1) * central - (Q/V2) * periph\n",
            obs_scale: "V1",
            dose_cmt: 1,
            obs_cmt: 2,
            is_oral: true,
        },
        Family {
            label: "three_cpt_iv",
            thetas: "  theta TVCL(3.0, 0.01, 100.0)\n  theta TVV1(15.0, 1.0, 500.0)\n  \
                     theta TVQ2(3.0, 0.01, 100.0)\n  theta TVV2(30.0, 1.0, 500.0)\n  \
                     theta TVQ3(1.5, 0.01, 100.0)\n  theta TVV3(60.0, 1.0, 999.0)\n",
            indiv: "  CL = TVCL * exp(ETA_CL)\n  V1 = TVV1\n  Q2 = TVQ2\n  V2 = TVV2\n  \
                    Q3 = TVQ3\n  V3 = TVV3\n",
            pk_open: "pk three_cpt_iv(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3",
            ode_struct: "ode(obs_cmt=central, states=[central, periph1, periph2])",
            odes: "  d/dt(central)  = -(CL/V1 + Q2/V1 + Q3/V1) * central \
                    + (Q2/V2) * periph1 + (Q3/V3) * periph2\n  \
                    d/dt(periph1)  =  (Q2/V1) * central - (Q2/V2) * periph1\n  \
                    d/dt(periph2)  =  (Q3/V1) * central - (Q3/V3) * periph2\n",
            obs_scale: "V1",
            dose_cmt: 1,
            obs_cmt: 1,
            is_oral: false,
        },
        Family {
            label: "three_cpt_oral",
            thetas: "  theta TVCL(3.0, 0.01, 100.0)\n  theta TVV1(15.0, 1.0, 500.0)\n  \
                     theta TVQ2(3.0, 0.01, 100.0)\n  theta TVV2(30.0, 1.0, 500.0)\n  \
                     theta TVQ3(1.5, 0.01, 100.0)\n  theta TVV3(60.0, 1.0, 999.0)\n  \
                     theta TVKA(1.1, 0.01, 50.0)\n",
            indiv: "  CL = TVCL * exp(ETA_CL)\n  V1 = TVV1\n  Q2 = TVQ2\n  V2 = TVV2\n  \
                    Q3 = TVQ3\n  V3 = TVV3\n  KA = TVKA\n",
            pk_open: "pk three_cpt_oral(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3, ka=KA",
            ode_struct: "ode(obs_cmt=central, states=[depot, central, periph1, periph2])",
            odes: "  d/dt(depot)    = -KA * depot\n  \
                    d/dt(central)  =  KA * depot - (CL/V1 + Q2/V1 + Q3/V1) * central \
                    + (Q2/V2) * periph1 + (Q3/V3) * periph2\n  \
                    d/dt(periph1)  =  (Q2/V1) * central - (Q2/V2) * periph1\n  \
                    d/dt(periph2)  =  (Q3/V1) * central - (Q3/V3) * periph2\n",
            obs_scale: "V1",
            dose_cmt: 1,
            obs_cmt: 2,
            is_oral: true,
        },
    ]
}

/// Build the analytical and ODE source for a family, optionally adding lag time
/// and/or bioavailability. The two forms share the same `[parameters]` and
/// `[individual_parameters]`; only the structural section differs. Lag/F enter
/// the analytical form through the `pk(...)` mapping and the ODE form purely by
/// declaring a `LAGTIME`/`F` individual parameter (the engine applies both at
/// the dose).
fn build_pair(f: &Family, lag: bool, fbio: bool) -> (String, String) {
    let mut thetas = String::from(f.thetas);
    let mut indiv = String::from(f.indiv);
    let mut pk_extra = String::new();
    if lag {
        thetas.push_str("  theta TVLAG(0.5, 0.0, 5.0)\n");
        indiv.push_str("  LAGTIME = TVLAG\n");
        pk_extra.push_str(", lagtime=LAGTIME");
    }
    if fbio {
        thetas.push_str("  theta TVF(0.7, 0.01, 1.0)\n");
        indiv.push_str("  F = TVF\n");
        pk_extra.push_str(", f=F");
    }

    let header = format!(
        "[parameters]\n{thetas}  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.01 (sd)\n\n\
         [individual_parameters]\n{indiv}\n"
    );

    let analytical = format!(
        "{header}[structural_model]\n  {pk}{pk_extra})\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n",
        pk = f.pk_open
    );

    let ode = format!(
        "{header}[structural_model]\n  {st}\n\n[odes]\n{odes}\n\
         [scaling]\n  obs_scale = {scale}\n\n[error_model]\n  DV ~ proportional(PROP)\n",
        st = f.ode_struct,
        odes = f.odes,
        scale = f.obs_scale
    );

    (analytical, ode)
}

fn subject(doses: Vec<DoseEvent>, obs_times: Vec<f64>, obs_cmt: usize) -> Subject {
    let n = obs_times.len();
    Subject {
        id: "1".into(),
        doses,
        obs_times,
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n],
        obs_cmts: vec![obs_cmt; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; n],
        occasions: Vec::new(),
        dose_occasions: Vec::new(),
        #[cfg(feature = "survival")]
        obs_records: vec![],
    }
}

fn population(doses: Vec<DoseEvent>, obs_times: Vec<f64>, obs_cmt: usize) -> Population {
    Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![subject(doses, obs_times, obs_cmt)],
    }
}

/// Parse both forms, `predict()` them on `pop`, and assert pointwise agreement.
fn assert_equiv(label: &str, analytical_src: &str, ode_src: &str, pop: &Population, rtol: f64) {
    let an = parse_full_model(analytical_src)
        .unwrap_or_else(|e| panic!("[{label}] analytical model did not parse: {e}"))
        .model;
    let ode = parse_full_model(ode_src)
        .unwrap_or_else(|e| panic!("[{label}] ODE model did not parse: {e}"))
        .model;

    let pa = predict(&an, pop, &an.default_params);
    let po = predict(&ode, pop, &ode.default_params);
    assert_eq!(pa.len(), po.len(), "[{label}] prediction count mismatch");
    assert!(!pa.is_empty(), "[{label}] produced no predictions");

    for (x, y) in pa.iter().zip(po.iter()) {
        assert!(
            (x.time - y.time).abs() < 1e-9,
            "[{label}] time mismatch: {} vs {}",
            x.time,
            y.time
        );
        let tol = ATOL + rtol * x.pred.abs();
        assert!(
            (x.pred - y.pred).abs() <= tol,
            "[{label}] t={:.3}: analytical PRED {:.6} vs ODE PRED {:.6} (|diff| {:.2e} > tol {:.2e})",
            x.time,
            x.pred,
            y.pred,
            (x.pred - y.pred).abs(),
            tol
        );
    }
}

#[test]
fn analytical_ode_equivalence_across_models_and_dosing() {
    let obs = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];

    for f in families() {
        let dc = f.dose_cmt;
        let oc = f.obs_cmt;

        // -- base model (no lag, no F) across dosing modes -------------------
        let (an, ode) = build_pair(&f, false, false);

        // single bolus
        assert_equiv(
            &format!("{}/bolus", f.label),
            &an,
            &ode,
            &population(
                vec![DoseEvent::new(0.0, 100.0, dc, 0.0, false, 0.0)],
                obs.clone(),
                oc,
            ),
            RTOL,
        );

        // multiple doses (q24h x3)
        assert_equiv(
            &format!("{}/multidose", f.label),
            &an,
            &ode,
            &population(
                vec![
                    DoseEvent::new(0.0, 100.0, dc, 0.0, false, 0.0),
                    DoseEvent::new(24.0, 100.0, dc, 0.0, false, 0.0),
                    DoseEvent::new(48.0, 100.0, dc, 0.0, false, 0.0),
                ],
                vec![1.0, 6.0, 12.0, 25.0, 30.0, 49.0, 54.0, 72.0],
                oc,
            ),
            RTOL,
        );

        // steady state (II = 24)
        assert_equiv(
            &format!("{}/steady_state", f.label),
            &an,
            &ode,
            &population(
                vec![DoseEvent::new(0.0, 100.0, dc, 0.0, true, 24.0)],
                vec![1.0, 4.0, 8.0, 12.0, 23.0],
                oc,
            ),
            SS_RTOL,
        );

        // infusion (IV only - infusion into a depot is not a standard combo)
        if !f.is_oral {
            assert_equiv(
                &format!("{}/infusion", f.label),
                &an,
                &ode,
                &population(
                    vec![DoseEvent::new(0.0, 100.0, dc, 50.0, false, 0.0)],
                    obs.clone(),
                    oc,
                ),
                RTOL,
            );
        }

        // -- lag time -------------------------------------------------------
        let (an_l, ode_l) = build_pair(&f, true, false);
        assert_equiv(
            &format!("{}/lagtime", f.label),
            &an_l,
            &ode_l,
            // observe comfortably past the 0.5 h lag
            &population(
                vec![DoseEvent::new(0.0, 100.0, dc, 0.0, false, 0.0)],
                vec![1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
                oc,
            ),
            RTOL,
        );

        // -- bioavailability (oral only) -----------------------------------
        if f.is_oral {
            let (an_f, ode_f) = build_pair(&f, false, true);
            assert_equiv(
                &format!("{}/bioavailability", f.label),
                &an_f,
                &ode_f,
                &population(
                    vec![DoseEvent::new(0.0, 100.0, dc, 0.0, false, 0.0)],
                    obs.clone(),
                    oc,
                ),
                RTOL,
            );
        }
    }
}
