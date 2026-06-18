//! Generic (`PkNum`) compartment-state propagators — the `Dual2`-differentiable
//! mirror of the `f64` event-driven propagators in [`crate::pk::event_driven`].
//!
//! The closed-form *superposition* provider ([`super::provider`]) sums
//! independent per-dose contributions using a single per-subject `pk`. That works
//! when the PK parameters are constant over the subject's timeline. **IOV** and
//! **time-varying covariates** break it: the parameters *switch mid-decay* (a dose
//! given in one occasion carries over into the next and continues decaying with
//! that occasion's clearance — NONMEM #104 semantics). Representing that exactly
//! needs an *event walk* that carries the compartment **amounts** across occasion
//! boundaries and swaps the parameters at each boundary — exactly what the
//! production `f64` walker does.
//!
//! These propagators evolve the **dual** state over one inter-event interval, so
//! the walk can be run over `Dual2` and yield exact `∂(amount)/∂(η,κ)` (and second
//! order) — closed-form per segment, no numerical ODE integration. This module is
//! the foundation; the dual event walk + readout + the per-occasion block-Ω outer
//! assembly build on top (issue #367).

use super::num::PkNum;
use crate::pk::event_driven::{Event, EventKind, EventSchedule};
use crate::types::{DoseEvent, Subject};

/// 1-cpt IV/central propagator: evolve `state[0]` (central amount) over `dt` with
/// a constant input `rate` into the central compartment. Mirror of
/// [`crate::pk::event_driven::propagate_one_cpt`].
pub fn propagate_one_cpt_g<T: PkNum>(state: &mut [T], dt: f64, cl: T, v: T, rate: T) {
    if v.val() <= 0.0 || cl.val() <= 0.0 {
        // Degenerate params: skip (the outer optimizer sees a poor OFV and steps
        // away), matching the production propagator.
        return;
    }
    let ke = cl / v;
    let exp_term = (-(ke * T::from_f64(dt))).exp();
    state[0] = exp_term * state[0] + (rate / ke) * (T::from_f64(1.0) - exp_term);
}

/// 1-cpt oral propagator. `state = [A_depot, A_central]`; the depot drains into
/// the central compartment at absorption rate `ka` (bolus dosing — doses are added
/// to the depot by the event handler). Mirror of
/// [`crate::pk::event_driven::propagate_one_cpt_oral`], including the `ka ≈ ke`
/// L'Hôpital limit.
pub fn propagate_one_cpt_oral_g<T: PkNum>(state: &mut [T], dt: f64, cl: T, v: T, ka: T) {
    if v.val() <= 0.0 || cl.val() <= 0.0 || ka.val() <= 0.0 {
        return;
    }
    let ke = cl / v;
    let dtt = T::from_f64(dt);
    let e_ka = (-(ka * dtt)).exp();
    let e_ke = (-(ke * dtt)).exp();
    let a_d_0 = state[0];
    let a_c_0 = state[1];

    // Depot decays exponentially (decoupled).
    state[0] = a_d_0 * e_ka;

    // Central: homogeneous decay of A_c(0) plus the depot-driven Bateman term,
    // with the `ka ≈ ke` L'Hôpital fallback (branch on `.val()`).
    if (ka.val() - ke.val()).abs() < 1e-9 {
        state[1] = a_c_0 * e_ke + ka * a_d_0 * dtt * e_ke;
    } else {
        state[1] = a_c_0 * e_ke + (ka * a_d_0 / (ke - ka)) * (e_ka - e_ke);
    }
}

// ─── 1-cpt event-driven sensitivity walk ─────────────────────────────
//
// The propagators above evolve the dual state across *one* inter-event interval.
// This section stacks them into a full event walk — the `Dual2`-differentiable
// mirror of `event_driven::event_driven_predictions_with_schedule_impl`, but for
// the 1-cpt models only and over `PkNum`. The walk carries the dual compartment
// **amounts** across every event boundary and uses the **per-event** PK params,
// so IOV (parameters that switch at occasion boundaries, NONMEM #104) and
// time-varying covariates are exact: occasion 1's dose decays with occasion-1
// params to the boundary, then the carried-over amount continues with occasion-2
// params. Steady-state doses (`ss` + `ii > 0`) equilibrate per-event with that
// event's params, so SS composes with IOV the same way production does.
//
// Per-event PK params are passed in already seeded as `T` (e.g. `Dual2<M>` with
// each occasion's `(η, κ)`-derived params on their axes); the walk is agnostic to
// how they were seeded, which keeps it testable in isolation against FD of the
// `f64` production walk.

/// Per-event 1-cpt PK params for the generic walk. `ka` is unused for IV models.
#[derive(Clone, Copy)]
pub struct OneCptPk<T: PkNum> {
    pub cl: T,
    pub v: T,
    pub ka: T,
    /// Bioavailability `F` (multiplies bolus amount and infusion rate).
    pub f: T,
}

/// Cycles to expand for SS equilibration — mirrors
/// `event_driven::EVENT_DRIVEN_SS_EQUILIBRATION_CYCLES` (kept private there).
const SS_EQUILIBRATION_CYCLES: usize = 50;

#[inline]
fn pk_for_g<T: PkNum>(
    ev: Event,
    pk_at_dose: &[OneCptPk<T>],
    pk_at_obs: &[OneCptPk<T>],
    pk_at_pk_only: &[OneCptPk<T>],
) -> OneCptPk<T> {
    match ev.kind {
        EventKind::Dose => pk_at_dose[ev.orig_idx],
        EventKind::Obs => pk_at_obs[ev.orig_idx],
        EventKind::PkOnly => pk_at_pk_only[ev.orig_idx],
        EventKind::Reset => unreachable!("Reset carries no PK params"),
    }
}

/// Propagate the dual 1-cpt state across pre-built sub-event bounds (the
/// `EventSchedule` sub-interval boundaries), applying any active central
/// infusion per sub-interval. Generic mirror of
/// `event_driven::propagate_with_bounds` restricted to the 1-cpt models.
fn propagate_one_cpt_bounds_g<T: PkNum>(
    state: &mut [T],
    bounds: &[f64],
    pk: &OneCptPk<T>,
    oral: bool,
    doses: &[DoseEvent],
    dose_lagtimes: &[f64],
    reset_floor: f64,
) {
    for w in bounds.windows(2) {
        let dt = w[1] - w[0];
        if dt <= 0.0 {
            continue;
        }
        let mid = 0.5 * (w[0] + w[1]);
        // Active central-infusion rate (F·rate). Only inputs into the central
        // compartment are summed — IV cmt 1, oral cmt 2; this matches the
        // production 1-cpt match arms. (The oral propagator ignores `rate`, as
        // production's `propagate_one_cpt_oral` does, so an oral central infusion
        // is a no-op here too — a deliberate parity choice, not a new gap.)
        let mut rate_central = T::from_f64(0.0);
        for (k, d) in doses.iter().enumerate() {
            let lag = dose_lagtimes.get(k).copied().unwrap_or(0.0);
            let t_start = d.time + lag;
            let t_end = t_start + d.duration;
            if t_start < reset_floor {
                continue;
            }
            if d.rate > 0.0 && d.duration > 0.0 && t_start <= mid && t_end >= mid {
                let into_central = (!oral && d.cmt == 1) || (oral && d.cmt == 2);
                if into_central {
                    rate_central = rate_central + pk.f * T::from_f64(d.rate);
                }
            }
        }
        if oral {
            propagate_one_cpt_oral_g(state, dt, pk.cl, pk.v, pk.ka);
        } else {
            propagate_one_cpt_g(state, dt, pk.cl, pk.v, rate_central);
        }
    }
}

/// Equilibrate the dual 1-cpt state to its SS value for an SS=1 dose, per-event
/// (uses `pk`, the dose-event's params). Generic mirror of
/// `event_driven::equilibrate_ss_state_event_driven` for the 1-cpt models;
/// overlapping SS infusions (`T_inf > II`) return the empty state, matching
/// production's reject.
fn equilibrate_ss_one_cpt_g<T: PkNum>(oral: bool, pk: &OneCptPk<T>, dose: &DoseEvent) -> Vec<T> {
    let n_states = if oral { 2 } else { 1 };
    let mut state = vec![T::from_f64(0.0); n_states];
    if dose.ii <= 0.0 || dose.cmt == 0 {
        return state;
    }
    let cmt_idx = dose.cmt.saturating_sub(1);
    if cmt_idx >= n_states {
        return state;
    }
    let is_inf = dose.rate > 0.0 && dose.duration > 0.0 && dose.duration.is_finite();
    if is_inf && dose.duration > dose.ii {
        return state;
    }
    let synthetic_dose = if is_inf {
        vec![DoseEvent::new(0.0, dose.amt, dose.cmt, dose.rate, false, 0.0)]
    } else {
        Vec::new()
    };
    let synthetic_lag: Vec<f64> = if is_inf { vec![0.0] } else { Vec::new() };
    let bounds: Vec<f64> = if is_inf {
        vec![0.0, dose.duration, dose.ii]
    } else {
        vec![0.0, dose.ii]
    };
    for _ in 0..SS_EQUILIBRATION_CYCLES {
        if !is_inf {
            state[cmt_idx] = state[cmt_idx] + pk.f * T::from_f64(dose.amt);
        }
        propagate_one_cpt_bounds_g(
            &mut state,
            &bounds,
            pk,
            oral,
            &synthetic_dose,
            &synthetic_lag,
            f64::NEG_INFINITY,
        );
    }
    state
}

/// Event-driven 1-cpt **sensitivity** walk: returns the dual concentration at
/// every observation, parallel to `subject.obs_times`. The `Dual2`-differentiable
/// mirror of `event_driven::event_driven_predictions_with_schedule_impl` for
/// `OneCptIv` (`oral = false`) and `OneCptOral` (`oral = true`).
///
/// `pk_at_dose` / `pk_at_obs` / `pk_at_pk_only` are the **per-event** PK params,
/// already seeded as `T` (parallel to `subject.doses` / `obs_times` /
/// `pk_only_times`). The walk carries dual amounts across boundaries and switches
/// to each event's params, so IOV / time-varying covariates are exact; SS doses
/// equilibrate per-event. Resets (EVID 3/4) zero the dual state; central infusion
/// is applied through the bounds.
///
/// The `f64` instantiation reproduces the production walk bit-for-bit (one source
/// of truth); the `Dual2` instantiation yields exact `∂(conc)/∂(seeded axes)` and
/// second order.
pub fn event_driven_sens_one_cpt_g<T: PkNum>(
    oral: bool,
    subject: &Subject,
    schedule: &EventSchedule,
    pk_at_dose: &[OneCptPk<T>],
    pk_at_obs: &[OneCptPk<T>],
    pk_at_pk_only: &[OneCptPk<T>],
) -> Vec<T> {
    let n_obs = subject.obs_times.len();
    let mut preds = vec![T::from_f64(0.0); n_obs];
    if n_obs == 0 || schedule.events.is_empty() {
        return preds;
    }
    let (n_states, central_slot) = if oral { (2, 1) } else { (1, 0) };

    let mut state = vec![T::from_f64(0.0); n_states];
    let mut cur_t = schedule.events[0].time;
    let mut reset_floor = f64::NEG_INFINITY;

    for (i, ev) in schedule.events.iter().enumerate() {
        if ev.kind == EventKind::Reset {
            state.iter_mut().for_each(|s| *s = T::from_f64(0.0));
            cur_t = ev.time;
            reset_floor = ev.time;
            continue;
        }
        let pk_now = pk_for_g(*ev, pk_at_dose, pk_at_obs, pk_at_pk_only);

        if ev.time > cur_t {
            let bounds = &schedule.bounds_per_interval[i - 1];
            propagate_one_cpt_bounds_g(
                &mut state,
                bounds,
                &pk_now,
                oral,
                &subject.doses,
                &schedule.dose_lagtimes,
                reset_floor,
            );
            cur_t = ev.time;
        }

        match ev.kind {
            EventKind::Dose => {
                let d = &subject.doses[ev.orig_idx];
                if d.ss && d.ii > 0.0 {
                    state = equilibrate_ss_one_cpt_g(oral, &pk_now, d);
                }
                if d.rate <= 0.0 {
                    let cmt_idx = d.cmt.saturating_sub(1);
                    if cmt_idx < n_states {
                        state[cmt_idx] = state[cmt_idx] + pk_now.f * T::from_f64(d.amt);
                    }
                }
            }
            EventKind::Obs => {
                let v = pk_now.v;
                let conc = if v.val() > 0.0 {
                    state[central_slot] / v
                } else {
                    T::from_f64(0.0)
                };
                // Mirror production's `conc.max(0.0)`: a negative value clamps to
                // 0, so its derivatives vanish there (consistency with the OFV).
                preds[ev.orig_idx] = if conc.val() < 0.0 {
                    T::from_f64(0.0)
                } else {
                    conc
                };
            }
            EventKind::PkOnly => {}
            EventKind::Reset => unreachable!("Reset handled before pk_for_g above"),
        }
    }

    preds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pk::event_driven::event_driven_predictions;
    use crate::sens::dual2::Dual2;
    use crate::types::{PkModel, PkParams, PK_IDX_CL, PK_IDX_KA, PK_IDX_V};

    fn pk_of(cl: f64, v: f64, ka: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[PK_IDX_CL] = cl;
        p.values[PK_IDX_V] = v;
        p.values[PK_IDX_KA] = ka;
        p
    }

    /// The `f64` instantiation of the generic propagator must reproduce the
    /// production `f64` propagator bit-for-bit (same closed form, one source of
    /// truth in disguise).
    #[test]
    fn one_cpt_g_matches_production_f64() {
        for &(cl, v, rate, dt, s0) in &[
            (3.0, 30.0, 0.0, 2.5, 10.0),
            (5.0, 40.0, 8.0, 1.0, 0.0),
            (1.2, 12.0, 2.0, 6.0, 3.5),
        ] {
            let mut s_g = [s0];
            propagate_one_cpt_g::<f64>(&mut s_g, dt, cl, v, rate);
            let mut s_p = [s0];
            crate::pk::event_driven::propagate_one_cpt(&mut s_p, dt, &pk_of(cl, v, 1.0), rate);
            approx::assert_relative_eq!(s_g[0], s_p[0], max_relative = 1e-12);
        }
    }

    #[test]
    fn one_cpt_oral_g_matches_production_f64() {
        for &(cl, v, ka, dt, ad, ac) in &[
            (3.0, 30.0, 1.5, 2.5, 50.0, 5.0),
            (1.2, 12.0, 0.8, 4.0, 20.0, 1.0),
            // ka ≈ ke (L'Hôpital): ke = cl/v = 0.1, ka = 0.1.
            (3.0, 30.0, 0.1, 3.0, 40.0, 2.0),
        ] {
            let mut s_g = [ad, ac];
            propagate_one_cpt_oral_g::<f64>(&mut s_g, dt, cl, v, ka);
            let mut s_p = [ad, ac];
            crate::pk::event_driven::propagate_one_cpt_oral(&mut s_p, dt, &pk_of(cl, v, ka));
            approx::assert_relative_eq!(s_g[0], s_p[0], max_relative = 1e-12);
            approx::assert_relative_eq!(s_g[1], s_p[1], max_relative = 1e-12);
        }
    }

    /// Central FD grad + 4-point Hessian of a 2-arg `f64` closure.
    fn fd2(p: [f64; 2], val: impl Fn([f64; 2]) -> f64) -> ([f64; 2], [[f64; 2]; 2]) {
        let h = [1e-6 * (1.0 + p[0].abs()), 1e-6 * (1.0 + p[1].abs())];
        let hh = [1e-4 * (1.0 + p[0].abs()), 1e-4 * (1.0 + p[1].abs())];
        let mut g = [0.0; 2];
        for i in 0..2 {
            let mut up = p;
            up[i] += h[i];
            let mut dn = p;
            dn[i] -= h[i];
            g[i] = (val(up) - val(dn)) / (2.0 * h[i]);
        }
        let mut he = [[0.0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                let mut pp = p;
                pp[i] += hh[i];
                pp[j] += hh[j];
                let mut pm = p;
                pm[i] += hh[i];
                pm[j] -= hh[j];
                let mut mp = p;
                mp[i] -= hh[i];
                mp[j] += hh[j];
                let mut mm = p;
                mm[i] -= hh[i];
                mm[j] -= hh[j];
                he[i][j] = (val(pp) - val(pm) - val(mp) + val(mm)) / (4.0 * hh[i] * hh[j]);
            }
        }
        (g, he)
    }

    /// The `Dual2` instantiation's `∂(amount)/∂(cl,v)` (and Hessian) must match
    /// finite differences of the `f64` propagator — the propagator differentiates
    /// the compartment amount w.r.t. the PK parameters exactly.
    #[test]
    fn one_cpt_g_dual_matches_fd() {
        let (cl, v, rate, dt, s0) = (3.0, 30.0, 4.0, 2.5, 10.0);
        let mut sd = [Dual2::<2>::constant(s0)];
        propagate_one_cpt_g::<Dual2<2>>(
            &mut sd,
            dt,
            Dual2::var(cl, 0),
            Dual2::var(v, 1),
            Dual2::constant(rate),
        );
        let (g, he) = fd2([cl, v], |p| {
            let mut s = [s0];
            propagate_one_cpt_g::<f64>(&mut s, dt, p[0], p[1], rate);
            s[0]
        });
        for i in 0..2 {
            approx::assert_relative_eq!(sd[0].grad[i], g[i], max_relative = 1e-4, epsilon = 1e-8);
            for j in 0..2 {
                approx::assert_relative_eq!(
                    sd[0].hess[i][j],
                    he[i][j],
                    max_relative = 3e-3,
                    epsilon = 1e-6
                );
            }
        }
    }

    // ── Full 1-cpt event-driven sensitivity walk ─────────────────────

    fn make_subject(doses: Vec<DoseEvent>, obs_times: Vec<f64>) -> Subject {
        use std::collections::HashMap;
        let n_obs = obs_times.len();
        Subject {
            id: "1".into(),
            doses,
            obs_times,
            obs_raw_times: Vec::new(),
            observations: vec![0.0; n_obs],
            obs_cmts: vec![1; n_obs],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n_obs],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }
    }

    fn pk_full(cl: f64, v: f64, ka: f64) -> PkParams {
        let mut p = pk_of(cl, v, ka);
        p.values[crate::types::PK_IDX_F] = 1.0;
        p
    }

    fn one_cpt_pk_f64(p: &PkParams) -> OneCptPk<f64> {
        OneCptPk {
            cl: p.cl(),
            v: p.v(),
            ka: p.ka(),
            f: p.f_bio(),
        }
    }

    /// The `f64` instantiation of the full event walk must reproduce the
    /// production event-driven predictions bit-for-bit across dose kinds — bolus,
    /// infusion, oral, and steady state — so the IOV/TV-cov sensitivity walk is the
    /// same closed form as the f64 predictor, only differentiable.
    #[test]
    fn event_walk_g_matches_production_f64() {
        struct Case {
            oral: bool,
            model: PkModel,
            dose: DoseEvent,
            cl: f64,
            v: f64,
            ka: f64,
        }
        let obs = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 18.0, 24.0];
        let cases = [
            // 1-cpt IV bolus.
            Case {
                oral: false,
                model: PkModel::OneCptIv,
                dose: DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                cl: 5.0,
                v: 50.0,
                ka: 1.0,
            },
            // 1-cpt IV infusion (rate=25 → 4 h).
            Case {
                oral: false,
                model: PkModel::OneCptIv,
                dose: DoseEvent::new(0.0, 100.0, 1, 25.0, false, 0.0),
                cl: 3.0,
                v: 40.0,
                ka: 1.0,
            },
            // 1-cpt oral bolus into depot (cmt 1).
            Case {
                oral: true,
                model: PkModel::OneCptOral,
                dose: DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                cl: 4.0,
                v: 30.0,
                ka: 1.2,
            },
            // 1-cpt IV bolus at steady state (II=12).
            Case {
                oral: false,
                model: PkModel::OneCptIv,
                dose: DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0),
                cl: 5.0,
                v: 50.0,
                ka: 1.0,
            },
            // 1-cpt oral bolus at steady state (II=24).
            Case {
                oral: true,
                model: PkModel::OneCptOral,
                dose: DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0),
                cl: 4.0,
                v: 30.0,
                ka: 1.2,
            },
        ];

        for (ci, c) in cases.iter().enumerate() {
            let subj = make_subject(vec![c.dose.clone()], obs.clone());
            let pk = pk_full(c.cl, c.v, c.ka);
            let prod = event_driven_predictions(
                c.model,
                &subj,
                &[pk],
                &vec![pk; obs.len()],
                &[],
            );
            let schedule = EventSchedule::for_subject(&subj, c.model, &[pk.lagtime()]);
            let pk_g = one_cpt_pk_f64(&pk);
            let walk = event_driven_sens_one_cpt_g::<f64>(
                c.oral,
                &subj,
                &schedule,
                &[pk_g],
                &vec![pk_g; obs.len()],
                &[],
            );
            for (j, (&p, &w)) in prod.iter().zip(walk.iter()).enumerate() {
                approx::assert_relative_eq!(w, p, max_relative = 1e-12, epsilon = 1e-12);
                assert!(p >= 0.0, "case {ci} obs {j}: production conc negative");
            }
        }
    }

    /// Two-occasion IOV shape: the dose in occasion 1 decays with occasion-1
    /// params; the carried-over amount continues decaying with occasion-2 params
    /// after the boundary. The `f64` walk with per-event params must match the
    /// production event-driven predictor fed the same per-event params (which is
    /// exactly how `predict_iov` runs), confirming the walk handles
    /// parameter-switching mid-decay.
    #[test]
    fn event_walk_g_iov_carryover_matches_production() {
        // One dose at t=0 (occasion 1), observations spanning the boundary at
        // t=12 into occasion 2 with a different clearance.
        let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
        let obs = vec![1.0, 6.0, 11.0, 13.0, 18.0, 24.0];
        let subj = make_subject(vec![dose], obs.clone());

        let pk_occ1 = pk_full(5.0, 50.0, 1.0);
        let pk_occ2 = pk_full(8.0, 50.0, 1.0); // faster clearance in occasion 2
        // Per-event params: dose is occasion 1; obs before t=12 are occasion 1,
        // after are occasion 2.
        let pk_at_obs: Vec<PkParams> = obs
            .iter()
            .map(|&t| if t < 12.0 { pk_occ1 } else { pk_occ2 })
            .collect();

        let prod = event_driven_predictions(
            PkModel::OneCptIv,
            &subj,
            &[pk_occ1],
            &pk_at_obs,
            &[],
        );
        let schedule = EventSchedule::for_subject(&subj, PkModel::OneCptIv, &[0.0]);
        let pk_at_obs_g: Vec<OneCptPk<f64>> = pk_at_obs.iter().map(one_cpt_pk_f64).collect();
        let walk = event_driven_sens_one_cpt_g::<f64>(
            false,
            &subj,
            &schedule,
            &[one_cpt_pk_f64(&pk_occ1)],
            &pk_at_obs_g,
            &[],
        );
        for (&p, &w) in prod.iter().zip(walk.iter()) {
            approx::assert_relative_eq!(w, p, max_relative = 1e-12, epsilon = 1e-12);
        }
    }

    /// The `Dual2` walk's `∂(conc)/∂(cl, v)` (and Hessian) at one observation must
    /// match finite differences of the `f64` walk — the walk differentiates the
    /// full multi-interval prediction exactly. Single-occasion seeding (the same
    /// `(cl, v)` on every event) isolates the propagation chain.
    #[test]
    fn event_walk_g_dual_matches_fd() {
        let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
        let obs = vec![6.0];
        let subj = make_subject(vec![dose], obs.clone());
        let (cl, v, ka) = (5.0, 50.0, 1.0);

        let seed = |cl: Dual2<2>, v: Dual2<2>| OneCptPk {
            cl,
            v,
            ka: Dual2::<2>::constant(ka),
            f: Dual2::<2>::constant(1.0),
        };
        let schedule = EventSchedule::for_subject(&subj, PkModel::OneCptIv, &[0.0]);
        let pk_d = seed(Dual2::var(cl, 0), Dual2::var(v, 1));
        let walk = event_driven_sens_one_cpt_g::<Dual2<2>>(
            false,
            &subj,
            &schedule,
            &[pk_d],
            &[pk_d],
            &[],
        );
        let out = walk[0];

        let (g, he) = fd2([cl, v], |p| {
            let pk_g = OneCptPk {
                cl: p[0],
                v: p[1],
                ka,
                f: 1.0,
            };
            let w = event_driven_sens_one_cpt_g::<f64>(
                false,
                &subj,
                &schedule,
                &[pk_g],
                &[pk_g],
                &[],
            );
            w[0]
        });
        for i in 0..2 {
            approx::assert_relative_eq!(out.grad[i], g[i], max_relative = 1e-5, epsilon = 1e-9);
            for j in 0..2 {
                approx::assert_relative_eq!(
                    out.hess[i][j],
                    he[i][j],
                    max_relative = 3e-3,
                    epsilon = 1e-7
                );
            }
        }
    }

    /// Same for 1-cpt oral, validating the central-compartment amount's `∂/∂(cl,v)`
    /// through the Bateman term.
    #[test]
    fn one_cpt_oral_g_dual_matches_fd() {
        let (cl, v, ka, dt, ad, ac) = (1.2, 12.0, 0.8, 4.0, 50.0, 3.0);
        let mut sd = [Dual2::<2>::constant(ad), Dual2::<2>::constant(ac)];
        propagate_one_cpt_oral_g::<Dual2<2>>(
            &mut sd,
            dt,
            Dual2::var(cl, 0),
            Dual2::var(v, 1),
            Dual2::constant(ka),
        );
        let (g, he) = fd2([cl, v], |p| {
            let mut s = [ad, ac];
            propagate_one_cpt_oral_g::<f64>(&mut s, dt, p[0], p[1], ka);
            s[1] // central amount
        });
        for i in 0..2 {
            approx::assert_relative_eq!(sd[1].grad[i], g[i], max_relative = 1e-4, epsilon = 1e-8);
            for j in 0..2 {
                approx::assert_relative_eq!(
                    sd[1].hess[i][j],
                    he[i][j],
                    max_relative = 3e-3,
                    epsilon = 1e-6
                );
            }
        }
    }
}
