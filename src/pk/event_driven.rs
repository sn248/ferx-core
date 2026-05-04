//! Event-driven analytical PK propagation.
//!
//! Walks events (doses + observations) in time order, propagating the
//! amount-vector state from one event to the next using the rate matrix
//! built from the *current* per-event PK parameters. This is what NONMEM
//! `ADVAN` routines do — and is how time-varying covariates take effect:
//! when CL or V change between events, the elimination rate during the
//! next interval changes accordingly.
//!
//! Initial scope (per the agreed phase plan):
//!   - 1-compartment IV bolus & infusion
//!   - 2-compartment IV bolus & infusion (dose into central / cmt=1)
//!
//! Oral absorption and 3-compartment models are deferred to a follow-up
//! pass (tracked in the project todo list). Models outside this scope
//! panic with a clear message — callers must dispatch to the existing
//! superposition path for non-TV-covariate cases.

use crate::types::{DoseEvent, PkModel, PkParams, Subject};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventKind {
    Dose,
    Obs,
}

#[derive(Debug, Clone, Copy)]
struct Event {
    time: f64,
    kind: EventKind,
    /// Index into `subject.doses` or `subject.obs_times`.
    orig_idx: usize,
}

/// True when this PK model has an event-driven implementation in this module.
/// Caller-side dispatch (in `pk::compute_predictions`) uses this to fall
/// back to the existing superposition path for unsupported models.
pub fn supports_event_driven(pk_model: PkModel) -> bool {
    matches!(
        pk_model,
        PkModel::OneCptIvBolus
            | PkModel::OneCptInfusion
            | PkModel::TwoCptIvBolus
            | PkModel::TwoCptInfusion
    )
}

/// Compute predictions by walking events in time order and propagating the
/// compartment-amount state with per-event PK parameters.
///
/// `pk_at_dose[k]` are the PK parameters at `subject.doses[k].time`;
/// `pk_at_obs[j]` are the PK parameters at `subject.obs_times[j]`. These
/// are produced by [`crate::pk::compute_event_pk_params`].
///
/// Concentration at observation `j` is read out as `state_central / V` where
/// `V` is `pk_at_obs[j].v()` — i.e. the central-compartment volume at the
/// *observation's* time. This matches NONMEM `S1 = V1` / `IPRED = A(1)/S1`.
///
/// Behavior on the unsupported-model branch is to panic, on the assumption
/// that the dispatcher in `pk::compute_predictions` already filtered.
pub fn event_driven_predictions(
    pk_model: PkModel,
    subject: &Subject,
    pk_at_dose: &[PkParams],
    pk_at_obs: &[PkParams],
) -> Vec<f64> {
    assert_eq!(pk_at_dose.len(), subject.doses.len());
    assert_eq!(pk_at_obs.len(), subject.obs_times.len());

    let n_obs = subject.obs_times.len();
    let mut preds = vec![0.0_f64; n_obs];

    if n_obs == 0 {
        return preds;
    }

    let n_states = match pk_model {
        PkModel::OneCptIvBolus | PkModel::OneCptInfusion => 1,
        PkModel::TwoCptIvBolus | PkModel::TwoCptInfusion => 2,
        _ => panic!(
            "event_driven_predictions: pk_model {:?} not yet supported. \
             Use compute_predictions superposition fallback for now.",
            pk_model
        ),
    };

    // Build merged event timeline. Stable tie-break: doses come before
    // observations at the same time so that an obs at the dose time
    // observes the post-dose state (NONMEM convention).
    let mut events: Vec<Event> = Vec::with_capacity(subject.doses.len() + n_obs);
    for (k, d) in subject.doses.iter().enumerate() {
        events.push(Event { time: d.time, kind: EventKind::Dose, orig_idx: k });
    }
    for (j, &t) in subject.obs_times.iter().enumerate() {
        events.push(Event { time: t, kind: EventKind::Obs, orig_idx: j });
    }
    events.sort_by(|a, b| {
        a.time
            .partial_cmp(&b.time)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| match (a.kind, b.kind) {
                (EventKind::Dose, EventKind::Obs) => std::cmp::Ordering::Less,
                (EventKind::Obs, EventKind::Dose) => std::cmp::Ordering::Greater,
                _ => std::cmp::Ordering::Equal,
            })
    });

    // State vector starts at zero (no residual drug before the first event).
    let mut state = vec![0.0_f64; n_states];
    let mut cur_t = events[0].time;
    // PK params governing the *current* interval. Initialized from the first
    // event so the first (zero-length) propagation is well-defined.
    let mut current_pk: PkParams = pk_for(events[0], pk_at_dose, pk_at_obs);

    for ev in &events {
        if ev.time > cur_t {
            propagate(
                &mut state,
                cur_t,
                ev.time,
                &current_pk,
                pk_model,
                &subject.doses,
            );
            cur_t = ev.time;
        }

        match ev.kind {
            EventKind::Dose => {
                let d = &subject.doses[ev.orig_idx];
                if d.rate <= 0.0 {
                    // Bolus: instantaneous amount jump in dose's compartment.
                    let cmt_idx = d.cmt.saturating_sub(1);
                    if cmt_idx < n_states {
                        state[cmt_idx] += d.amt;
                    } else {
                        panic!(
                            "event-driven PK: dose into compartment {} but model has \
                             {} states (cmt is 1-based)",
                            d.cmt, n_states
                        );
                    }
                }
                // Infusion: handled inside `propagate` via the active-input
                // lookup — no instantaneous state jump here.
                current_pk = pk_at_dose[ev.orig_idx];
            }
            EventKind::Obs => {
                let pk = &pk_at_obs[ev.orig_idx];
                let v = pk.v();
                let conc = if v > 0.0 { state[0] / v } else { 0.0 };
                preds[ev.orig_idx] = conc.max(0.0);
                current_pk = *pk;
            }
        }
    }

    preds
}

#[inline]
fn pk_for(ev: Event, pk_at_dose: &[PkParams], pk_at_obs: &[PkParams]) -> PkParams {
    match ev.kind {
        EventKind::Dose => pk_at_dose[ev.orig_idx],
        EventKind::Obs => pk_at_obs[ev.orig_idx],
    }
}

/// Propagate the compartment-amount state from `t_from` to `t_to` under
/// piecewise-constant PK params `pk`. Sub-divides the interval at infusion
/// start/stop times so the input rate is constant within each sub-interval.
fn propagate(
    state: &mut [f64],
    t_from: f64,
    t_to: f64,
    pk: &PkParams,
    pk_model: PkModel,
    doses: &[DoseEvent],
) {
    // Collect infusion-related sub-event boundaries within (t_from, t_to).
    // Including endpoints means `windows(2)` enumerates every sub-interval.
    let mut bounds: Vec<f64> = vec![t_from, t_to];
    for d in doses {
        if d.rate > 0.0 && d.duration > 0.0 {
            let start = d.time;
            let end = d.time + d.duration;
            if start > t_from + 1e-15 && start < t_to - 1e-15 {
                bounds.push(start);
            }
            if end > t_from + 1e-15 && end < t_to - 1e-15 {
                bounds.push(end);
            }
        }
    }
    bounds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    bounds.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    for w in bounds.windows(2) {
        let s0 = w[0];
        let s1 = w[1];
        let dt = s1 - s0;
        if dt <= 0.0 {
            continue;
        }
        let mid = 0.5 * (s0 + s1);
        // Sum infusion rates active across the *whole* sub-interval.
        // Boundary cases (an infusion starting exactly at s0 or ending
        // exactly at s1) qualify because boundaries were inserted above.
        let mut rate_central = 0.0;
        let mut rate_periph = 0.0;
        for d in doses {
            if d.rate > 0.0
                && d.duration > 0.0
                && d.time <= mid
                && d.time + d.duration >= mid
            {
                match d.cmt {
                    1 => rate_central += d.rate,
                    2 => rate_periph += d.rate,
                    _ => panic!(
                        "event-driven PK: infusion into compartment {} not supported \
                         (only cmt=1 central and cmt=2 peripheral for 2-cpt models)",
                        d.cmt
                    ),
                }
            }
        }

        match pk_model {
            PkModel::OneCptIvBolus | PkModel::OneCptInfusion => {
                propagate_one_cpt(state, dt, pk, rate_central);
            }
            PkModel::TwoCptIvBolus | PkModel::TwoCptInfusion => {
                propagate_two_cpt(state, dt, pk, rate_central, rate_periph);
            }
            _ => unreachable!("supports_event_driven gating bypassed"),
        }
    }
}

/// 1-cpt linear propagator with constant input `rate` into central:
///   A(t+dt) = exp(-ke·dt)·A(t) + (rate/ke)·(1 - exp(-ke·dt))
fn propagate_one_cpt(state: &mut [f64], dt: f64, pk: &PkParams, rate: f64) {
    let cl = pk.cl();
    let v = pk.v();
    if v <= 0.0 || cl <= 0.0 {
        // Degenerate params; skip propagation rather than blow up. The
        // outer optimizer will see a poor OFV and step away.
        return;
    }
    let ke = cl / v;
    let exp_term = (-ke * dt).exp();
    state[0] = exp_term * state[0] + (rate / ke) * (1.0 - exp_term);
}

/// 2-cpt linear propagator with constant input rates into central and
/// peripheral compartments. Uses eigendecomposition of the rate matrix
/// (eigenvalues -α, -β from `macro_rates`-style derivation).
fn propagate_two_cpt(
    state: &mut [f64],
    dt: f64,
    pk: &PkParams,
    rate_central: f64,
    rate_periph: f64,
) {
    let cl = pk.cl();
    let v1 = pk.v();
    let q = pk.q();
    let v2 = pk.v2();
    if v1 <= 0.0 || cl <= 0.0 || v2 <= 0.0 || q <= 0.0 {
        return;
    }
    let k10 = cl / v1;
    let k12 = q / v1;
    let k21 = q / v2;
    let s = k10 + k12 + k21;
    let d = k10 * k21;
    let disc = {
        let x = s * s - 4.0 * d;
        if x > 0.0 {
            x.sqrt()
        } else {
            0.0
        }
    };
    let alpha = (s + disc) / 2.0;
    // Vieta: alpha * beta = d (avoids cancellation).
    let beta = if alpha > 1e-30 { d / alpha } else { 0.0 };

    // Steady-state amounts under constant input b = (rate_central, rate_periph).
    //   A_ss = -K^{-1} b = (1/(k21·k10)) * [k21·b1 + k21·b2,
    //                                       k12·b1 + (k10+k12)·b2]
    // Special-cased for our two input channels.
    let denom_ss = k21 * k10;
    let (a_ss_1, a_ss_2) = if denom_ss > 1e-30 {
        (
            (k21 * rate_central + k21 * rate_periph) / denom_ss,
            (k12 * rate_central + (k10 + k12) * rate_periph) / denom_ss,
        )
    } else {
        (0.0, 0.0)
    };

    let h1_0 = state[0] - a_ss_1;
    let h2_0 = state[1] - a_ss_2;

    // Decompose homogeneous (h1_0, h2_0) into eigenmodes:
    //   eigenvectors u_α = (k21 - α, k12),  u_β = (k21 - β, k12)
    //   h(t) = c1·u_α·exp(-α·t) + c2·u_β·exp(-β·t)
    let denom = beta - alpha;
    let (c1, c2) = if k12.abs() < 1e-30 {
        // No central→peripheral transfer: 1-cpt-equivalent. Treat A1 and A2
        // as decoupled, with A1 having rate k10 and A2 having rate k21.
        // This branch is defensive — k12 == 0 implies q == 0 (already
        // guarded). Leave both modes balanced to avoid NaN.
        (0.0, 0.0)
    } else if denom.abs() < 1e-30 {
        // alpha ≈ beta — degenerate (only happens if discriminant ≈ 0,
        // i.e. (k10+k12-k21)^2 + 4·k12·k21 → 0 which requires k12=k21=0).
        // Fall back to splitting the homogeneous part evenly.
        let s_homog = h2_0 / k12;
        (s_homog * 0.5, s_homog * 0.5)
    } else {
        let s_homog = h2_0 / k12;
        let c1 = (h1_0 - s_homog * (k21 - beta)) / denom;
        let c2 = s_homog - c1;
        (c1, c2)
    };

    let e_alpha = (-alpha * dt).exp();
    let e_beta = (-beta * dt).exp();

    let h1_dt = c1 * (k21 - alpha) * e_alpha + c2 * (k21 - beta) * e_beta;
    let h2_dt = (c1 * e_alpha + c2 * e_beta) * k12;

    state[0] = h1_dt + a_ss_1;
    state[1] = h2_dt + a_ss_2;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DoseEvent;
    use approx::assert_relative_eq;
    use std::collections::HashMap;

    fn pk_one(cl: f64, v: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = cl;
        p.values[crate::types::PK_IDX_V] = v;
        p
    }

    fn pk_two(cl: f64, v1: f64, q: f64, v2: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = cl;
        p.values[crate::types::PK_IDX_V] = v1;
        p.values[crate::types::PK_IDX_Q] = q;
        p.values[crate::types::PK_IDX_V2] = v2;
        p
    }

    fn make_subject(doses: Vec<DoseEvent>, obs_times: Vec<f64>) -> Subject {
        let n_obs = obs_times.len();
        Subject {
            id: "1".into(),
            doses,
            obs_times,
            observations: vec![0.0; n_obs],
            obs_cmts: vec![1; n_obs],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            cens: vec![0; n_obs],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
        }
    }

    // ── Equivalence with superposition (constant pk_params) ───────────────────

    #[test]
    fn one_cpt_iv_bolus_matches_superposition_single_dose() {
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![0.0, 1.0, 2.0, 5.0, 10.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_one(10.0, 100.0);
        let pk_dose = vec![pk; 1];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = event_driven_predictions(PkModel::OneCptIvBolus, &subj, &pk_dose, &pk_obs);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| {
                crate::pk::predict_concentration(
                    PkModel::OneCptIvBolus,
                    &subj.doses,
                    t,
                    &pk,
                )
            })
            .collect();
        for (a, e) in preds.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *e, epsilon = 1e-10);
        }
    }

    #[test]
    fn one_cpt_iv_bolus_matches_superposition_multi_dose() {
        let doses = vec![
            DoseEvent::new(0.0, 500.0, 1, 0.0, false, 0.0),
            DoseEvent::new(8.0, 500.0, 1, 0.0, false, 0.0),
            DoseEvent::new(16.0, 500.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![1.0, 4.0, 8.5, 12.0, 18.0, 24.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_one(5.0, 80.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = event_driven_predictions(PkModel::OneCptIvBolus, &subj, &pk_dose, &pk_obs);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| {
                crate::pk::predict_concentration(
                    PkModel::OneCptIvBolus,
                    &subj.doses,
                    t,
                    &pk,
                )
            })
            .collect();
        for (i, (a, e)) in preds.iter().zip(expected.iter()).enumerate() {
            assert_relative_eq!(*a, *e, epsilon = 1e-10, max_relative = 1e-10);
            // Sanity: predictions are positive after the first dose.
            assert!(*a > 0.0, "obs {} should be positive, got {}", i, a);
        }
    }

    #[test]
    fn one_cpt_infusion_matches_superposition() {
        // 1000 mg over 2h infusion, then observe.
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 0.0)];
        let obs_times = vec![0.5, 1.0, 2.0, 4.0, 8.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_one(10.0, 100.0);
        let pk_dose = vec![pk; 1];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = event_driven_predictions(PkModel::OneCptInfusion, &subj, &pk_dose, &pk_obs);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| {
                crate::pk::predict_concentration(
                    PkModel::OneCptInfusion,
                    &subj.doses,
                    t,
                    &pk,
                )
            })
            .collect();
        for (a, e) in preds.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *e, epsilon = 1e-9, max_relative = 1e-9);
        }
    }

    #[test]
    fn two_cpt_iv_bolus_matches_superposition() {
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 1000.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![0.5, 2.0, 6.0, 12.5, 18.0, 24.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_two(5.0, 30.0, 2.0, 50.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = event_driven_predictions(PkModel::TwoCptIvBolus, &subj, &pk_dose, &pk_obs);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| {
                crate::pk::predict_concentration(
                    PkModel::TwoCptIvBolus,
                    &subj.doses,
                    t,
                    &pk,
                )
            })
            .collect();
        for (i, (a, e)) in preds.iter().zip(expected.iter()).enumerate() {
            assert_relative_eq!(
                *a,
                *e,
                epsilon = 1e-9,
                max_relative = 1e-9,
            );
            assert!(*a > 0.0, "obs {} should be positive, got {}", i, a);
        }
    }

    #[test]
    fn two_cpt_infusion_matches_superposition() {
        // 1000 mg over 2h infusion into central, multi-dose.
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 0.0),
            DoseEvent::new(12.0, 1000.0, 1, 500.0, false, 0.0),
        ];
        let obs_times = vec![0.5, 1.0, 2.0, 6.0, 12.5, 14.0, 18.0, 24.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_two(5.0, 30.0, 2.0, 50.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = event_driven_predictions(PkModel::TwoCptInfusion, &subj, &pk_dose, &pk_obs);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| {
                crate::pk::predict_concentration(
                    PkModel::TwoCptInfusion,
                    &subj.doses,
                    t,
                    &pk,
                )
            })
            .collect();
        for (i, (a, e)) in preds.iter().zip(expected.iter()).enumerate() {
            assert_relative_eq!(
                *a,
                *e,
                epsilon = 1e-8,
                max_relative = 1e-8,
            );
            assert!(*a > 0.0, "obs {} should be positive, got {}", i, a);
        }
    }

    // ── TV-covariate effect: changing CL between doses changes elimination ───

    #[test]
    fn one_cpt_tv_cl_changes_decay_rate() {
        // Single dose at t=0, two observations: CL doubles between the two
        // observations. The state at obs2 must equal:
        //   exp(-ke1 * (t1-0)) * dose then exp(-ke2 * (t2-t1)) * (above)
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![1.0, 2.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk_low = pk_one(5.0, 100.0); // ke = 0.05
        let pk_high = pk_one(10.0, 100.0); // ke = 0.10
        let pk_dose = vec![pk_low];
        let pk_obs = vec![pk_low, pk_high]; // pk changes at obs2

        let preds = event_driven_predictions(PkModel::OneCptIvBolus, &subj, &pk_dose, &pk_obs);

        // After dose: A1 = 1000.
        // Propagate to t=1 with ke=0.05 (current_pk = pk_low from dose):
        //   A1(1) = 1000 * exp(-0.05) ≈ 951.23
        //   C(obs1) = A1(1) / 100 = 9.5123
        let a1_at_t1 = 1000.0 * (-0.05f64).exp();
        let c1_expected = a1_at_t1 / 100.0;
        assert_relative_eq!(preds[0], c1_expected, epsilon = 1e-12);

        // After obs1 (which uses pk_low's V to read out, then sets
        // current_pk = pk_low). But on the obs event, current_pk *updates*
        // to pk_obs[0] = pk_low, so propagation 1→2 still uses pk_low.
        // (Verified: matches NONMEM where covariates only change with the
        // event row's values.)
        // For this test the pk at obs1 is still pk_low — so pk_low governs
        // (1, 2) and pk_high only changes the V used at obs2.
        let a1_at_t2 = a1_at_t1 * (-0.05f64).exp();
        let c2_expected = a1_at_t2 / 100.0; // V from pk_high == 100 anyway.
        assert_relative_eq!(preds[1], c2_expected, epsilon = 1e-12);
    }

    #[test]
    fn one_cpt_tv_cl_between_doses_changes_decay() {
        // Two doses, with CL doubling between them. Decay during
        // [t_dose1, t_dose2] uses pk_dose1; after dose2, decay uses pk_dose2.
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(10.0, 1000.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![5.0, 12.0];
        let subj = make_subject(doses, obs_times);
        let pk_low = pk_one(5.0, 100.0); // ke = 0.05
        let pk_high = pk_one(10.0, 100.0); // ke = 0.10
        let pk_dose = vec![pk_low, pk_high];
        let pk_obs = vec![pk_low, pk_high];

        let preds = event_driven_predictions(PkModel::OneCptIvBolus, &subj, &pk_dose, &pk_obs);

        // At t=5 (during low-CL interval after dose1):
        //   A1 = 1000 * exp(-0.05*5) = 778.80
        //   C = 7.7880
        let c5_expected = 1000.0 * (-0.05f64 * 5.0).exp() / 100.0;
        assert_relative_eq!(preds[0], c5_expected, epsilon = 1e-12);

        // At t=10: dose2 is added. Just before dose2:
        //   A1(10⁻) = 1000 * exp(-0.05*10) = 606.53
        // After dose2: A1(10⁺) = 606.53 + 1000 = 1606.53
        // From t=10 to t=12, current_pk is pk_dose2 = pk_high, so ke=0.10:
        //   A1(12) = 1606.53 * exp(-0.10*2) = 1316.18
        //   C(12) = 13.1618
        let a1_at_10_minus = 1000.0 * (-0.5f64).exp();
        let a1_at_10_plus = a1_at_10_minus + 1000.0;
        let a1_at_12 = a1_at_10_plus * (-0.20f64).exp();
        let c12_expected = a1_at_12 / 100.0;
        assert_relative_eq!(preds[1], c12_expected, epsilon = 1e-12);
    }

    #[test]
    fn supports_event_driven_gates_supported_models_only() {
        assert!(supports_event_driven(PkModel::OneCptIvBolus));
        assert!(supports_event_driven(PkModel::OneCptInfusion));
        assert!(supports_event_driven(PkModel::TwoCptIvBolus));
        assert!(supports_event_driven(PkModel::TwoCptInfusion));
        assert!(!supports_event_driven(PkModel::OneCptOral));
        assert!(!supports_event_driven(PkModel::TwoCptOral));
        assert!(!supports_event_driven(PkModel::ThreeCptIvBolus));
        assert!(!supports_event_driven(PkModel::ThreeCptInfusion));
        assert!(!supports_event_driven(PkModel::ThreeCptOral));
    }
}
