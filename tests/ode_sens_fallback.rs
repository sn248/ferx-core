//! Waterproofing matrix for the user-ODE analytic sensitivity fallback (#410).
//!
//! Arming the ODE analytic gradient is only safe if **every** out-of-scope model
//! or subject declines cleanly (`subject_sensitivities` → `None`, or
//! `sens_supported` → `false`) so the optimizer falls back to the FD/gradient-free
//! path. A case that returned a *wrong* `Some` would silently corrupt the gradient.
//! `population_gradient_sens` short-circuits to `None` if any subject declines, so
//! a single declining subject routes the whole population to FD — correct, never
//! partial. This test asserts the decline for each excluded feature, plus a
//! positive control that the in-scope model *is* armed (guarding against the gates
//! silently disabling everything).

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::sens::provider::{sens_supported, subject_sensitivities};
use ferx_core::types::{CompiledModel, DoseEvent, RateMode};

mod common;

/// In-scope 2-cpt IV user-ODE (Form-C readout), IIV on CL+V1 — the armed baseline.
const ARMED: &str = r"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1
[error_model]
  DV ~ proportional(PROP_ERR)
";

/// 1-cpt IV user-ODE with bioavailability `F` (estimated) — for the #419
/// rate-defined-infusion-under-F fallback.
const ODE_WITH_F: &str = r"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV(20.0,  1.0, 500.0)
  theta THETA_F(0.7, 0.01, 1.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  F  = THETA_F
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
";

fn parse(src: &str) -> CompiledModel {
    parse_model_string(src).expect("model parses")
}

/// Call the analytic provider at η = 0 / default θ; `Some` means "on the analytic
/// path", `None` means "declined → FD fallback".
fn declines(model: &CompiledModel, doses: Vec<DoseEvent>) -> bool {
    let times = vec![0.5, 1.0, 2.0, 4.0, 8.0];
    let obs_cmt = 1;
    let s = common::subject(
        "1",
        doses,
        times.clone(),
        vec![1.0; times.len()],
        vec![obs_cmt; times.len()],
    );
    let theta = &model.default_params.theta;
    let eta = vec![0.0; model.n_eta];
    subject_sensitivities(model, &s, theta, &eta).is_none()
}

#[test]
fn armed_baseline_is_on_the_analytic_path() {
    // Positive control: the in-scope ODE model must be armed and serve a bolus
    // subject — otherwise the fallback assertions below would be vacuous.
    let model = parse(ARMED);
    assert!(
        sens_supported(&model),
        "in-scope ODE model must be armed (#410)"
    );
    assert!(
        !declines(&model, vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)]),
        "in-scope bolus subject must get analytic sensitivities"
    );
    // A plain finite infusion is in scope too.
    assert!(
        !declines(
            &model,
            vec![DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0)]
        ),
        "in-scope finite-infusion subject must get analytic sensitivities"
    );
}

#[test]
fn modeled_duration_dose_declines() {
    // RATE=-2 (D{cmt}) arrives unresolved; the dual walk would read the raw
    // rate/duration. Must fall back (the production path resolves per-eval).
    let model = parse(ARMED);
    let dose = DoseEvent::modeled(0.0, 100.0, 1, false, 0.0, RateMode::ModeledDuration);
    assert!(
        declines(&model, vec![dose]),
        "modeled-duration dose must fall back to FD"
    );
}

#[test]
fn modeled_rate_dose_declines() {
    let model = parse(ARMED);
    let dose = DoseEvent::modeled(0.0, 100.0, 1, false, 0.0, RateMode::ModeledRate);
    assert!(
        declines(&model, vec![dose]),
        "modeled-rate dose must fall back to FD"
    );
}

#[test]
fn steady_state_dose_declines() {
    let model = parse(ARMED);
    let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0); // SS, II=24
    assert!(
        declines(&model, vec![dose]),
        "steady-state dosing must fall back to FD"
    );
}

#[test]
fn bioavailability_with_rate_defined_infusion_declines() {
    // #419: F reshapes a rate-defined infusion's window in production, but the dual
    // walk scales the magnitude — must fall back to stay consistent.
    let model = parse(ODE_WITH_F);
    let infusion = DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0); // RATE>0 = rate-defined
    assert!(
        declines(&model, vec![infusion]),
        "F + rate-defined infusion must fall back to FD (#419)"
    );
    // But the same F model with a *bolus* (no infusion window to reshape) stays armed.
    assert!(
        !declines(&model, vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)]),
        "F + bolus is in scope (no window reshape)"
    );
}

/// Amount-based ODE (`obs_cmt=central`) with `obs_scale = V1` (divisor form) — not
/// handled over Dual2.
const ODE_OBS_SCALE: &str = r"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(obs_cmt=central, states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  obs_scale = V1
[error_model]
  DV ~ proportional(PROP_ERR)
";

/// Form-C ODE under LTBS (`log(DV) ~ additive`) — the log output transform is not
/// handled over Dual2.
const ODE_LTBS: &str = r"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.04
  sigma ADD_LOG ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1
[error_model]
  log(DV) ~ additive(ADD_LOG)
";

#[test]
fn obs_scale_divisor_model_is_not_armed() {
    // `obs_scale = V1` (divisor form) is not handled over Dual2 — model-level decline.
    let model = parse(ODE_OBS_SCALE);
    assert!(
        !sens_supported(&model),
        "obs_scale divisor scaling must not be on the analytic path"
    );
}

#[test]
fn ltbs_model_is_not_armed() {
    let model = parse(ODE_LTBS);
    assert!(
        !sens_supported(&model),
        "LTBS log-transform must not be on the analytic path"
    );
}
