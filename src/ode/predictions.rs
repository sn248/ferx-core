//! ODE-based predictions for subjects with dose events.
//!
//! Matches Julia's `_ode_predictions`: breaks the timeline at dose times,
//! applies bolus doses as state discontinuities, and integrates between.

use crate::ode::solver::{solve_ode, OdeSolverOptions};
use crate::types::{PkParams, Subject};
use std::collections::HashMap;

/// ODE specification for a model
pub struct OdeSpec {
    /// RHS function: (u, pk_params_flat, t, du) — writes derivatives into du
    pub rhs: Box<dyn Fn(&[f64], &[f64], f64, &mut [f64]) + Send + Sync>,
    /// Number of ODE states
    pub n_states: usize,
    /// Names of state variables (e.g., ["depot", "central"])
    pub state_names: Vec<String>,
    /// Index of the observable compartment (0-based) for DV
    pub obs_cmt_idx: usize,
}

/// Compute ODE-based predictions for a single subject.
///
/// `pk_params_flat` is a flat array of PK parameters passed to the RHS function.
/// Dose events are handled as state discontinuities between integration segments.
pub fn ode_predictions(ode: &OdeSpec, pk_params_flat: &[f64], subject: &Subject) -> Vec<f64> {
    let n = ode.n_states;
    let n_obs = subject.obs_times.len();
    let opts = OdeSolverOptions::default();

    let mut u = vec![0.0; n];
    let mut predictions = vec![f64::NAN; n_obs];

    // Build obs_time → index map
    let obs_map: HashMap<u64, usize> = subject
        .obs_times
        .iter()
        .enumerate()
        .map(|(i, &t)| (t.to_bits(), i))
        .collect();

    // Break timeline at dose times
    let t_last = subject.obs_times.iter().cloned().fold(0.0f64, f64::max);
    let mut break_times: Vec<f64> = vec![0.0];
    for dose in &subject.doses {
        break_times.push(dose.time);
    }
    break_times.push(t_last);
    break_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    break_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    for k in 0..(break_times.len() - 1) {
        let t_start = break_times[k];
        let t_end = break_times[k + 1];

        // Apply bolus doses at t_start
        for dose in &subject.doses {
            if (dose.time - t_start).abs() < 1e-12 {
                assert!(
                    dose.rate == 0.0,
                    "Infusion doses (rate > 0) not yet supported in ODE models"
                );
                // dose.cmt is 1-based; state indices are 0-based
                let cmt_idx = dose.cmt - 1;
                if cmt_idx < n {
                    u[cmt_idx] += dose.amt;
                }
            }
        }

        // Record observations exactly at t_start (after dose)
        if let Some(&obs_idx) = obs_map.get(&t_start.to_bits()) {
            predictions[obs_idx] = u[ode.obs_cmt_idx];
        }

        // Observation times in this segment (t_start < t <= t_end)
        let mut saveat: Vec<f64> = subject
            .obs_times
            .iter()
            .filter(|&&t| t > t_start + 1e-12 && t <= t_end + 1e-12)
            .cloned()
            .collect();
        // Always include t_end so u is updated for next segment
        if saveat.is_empty() || (saveat.last().unwrap() - t_end).abs() > 1e-12 {
            saveat.push(t_end);
        }
        saveat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        saveat.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

        if (t_end - t_start).abs() < 1e-15 {
            continue;
        }

        // Integrate
        let sol = solve_ode(
            &*ode.rhs,
            &u,
            (t_start, t_end),
            pk_params_flat,
            &saveat,
            &opts,
        );

        // Extract predictions and update state
        for pt in &sol {
            if let Some(&obs_idx) = obs_map.get(&pt.t.to_bits()) {
                predictions[obs_idx] = pt.u[ode.obs_cmt_idx];
            }
        }

        // State at end of segment
        if let Some(last) = sol.last() {
            u.copy_from_slice(&last.u);
        }
    }

    // Clamp negatives
    for p in &mut predictions {
        if *p < 0.0 || p.is_nan() {
            *p = 0.0;
        }
    }

    predictions
}

/// ODE-based predictions with per-event PK parameters (time-varying-covariate
/// aware). Walks the merged dose+obs+pk-only timeline, integrating each
/// segment with the PK params evaluated at the *segment-start* event — so
/// a covariate that changes at an event row (dose, obs, or EVID=2)
/// immediately changes the rate matrix used over the next interval.
///
/// The non-TV `ode_predictions` is preserved as a fast path; this function
/// is only invoked from the dispatcher when `subject.has_tv_covariates()`.
///
/// **Bolus doses only.** Infusions in ODE models still hit the existing
/// `Infusion doses (rate > 0) not yet supported in ODE models` assertion.
/// Lifting that is independent of the TV-cov work and tracked separately.
pub fn ode_predictions_event_driven(
    ode: &OdeSpec,
    subject: &Subject,
    pk_at_dose: &[PkParams],
    pk_at_obs: &[PkParams],
    pk_at_pk_only: &[PkParams],
) -> Vec<f64> {
    assert_eq!(pk_at_dose.len(), subject.doses.len());
    assert_eq!(pk_at_obs.len(), subject.obs_times.len());
    assert_eq!(pk_at_pk_only.len(), subject.pk_only_times.len());

    let n = ode.n_states;
    let n_obs = subject.obs_times.len();
    let opts = OdeSolverOptions::default();

    let mut u = vec![0.0_f64; n];
    let mut predictions = vec![f64::NAN; n_obs];

    if n_obs == 0 {
        return predictions;
    }

    // Build merged event timeline. Tie-break at the same time:
    //   dose < pk-only < obs
    // — matches the analytical event-driven path.
    #[derive(Clone, Copy)]
    enum Kind {
        Dose,
        PkOnly,
        Obs,
    }
    fn kind_order(k: Kind) -> u8 {
        match k {
            Kind::Dose => 0,
            Kind::PkOnly => 1,
            Kind::Obs => 2,
        }
    }
    let mut timeline: Vec<(f64, Kind, usize)> =
        Vec::with_capacity(subject.doses.len() + n_obs + subject.pk_only_times.len());
    for (k, d) in subject.doses.iter().enumerate() {
        timeline.push((d.time, Kind::Dose, k));
    }
    for (j, &t) in subject.obs_times.iter().enumerate() {
        timeline.push((t, Kind::Obs, j));
    }
    for (m, &t) in subject.pk_only_times.iter().enumerate() {
        timeline.push((t, Kind::PkOnly, m));
    }
    timeline.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| kind_order(a.1).cmp(&kind_order(b.1)))
    });

    let mut cur_t = timeline[0].0;
    // Initial pk params for the first (zero-length) segment.
    let mut current_pk: PkParams = match timeline[0].1 {
        Kind::Dose => pk_at_dose[timeline[0].2],
        Kind::Obs => pk_at_obs[timeline[0].2],
        Kind::PkOnly => pk_at_pk_only[timeline[0].2],
    };

    for &(t_event, kind, idx) in &timeline {
        if t_event > cur_t {
            // Integrate segment [cur_t, t_event] with current_pk.
            let saveat = vec![t_event];
            let sol = solve_ode(
                &*ode.rhs,
                &u,
                (cur_t, t_event),
                &current_pk.values,
                &saveat,
                &opts,
            );
            if let Some(last) = sol.last() {
                u.copy_from_slice(&last.u);
            }
            cur_t = t_event;
        }

        match kind {
            Kind::Dose => {
                let d = &subject.doses[idx];
                assert!(
                    d.rate == 0.0,
                    "Infusion doses (rate > 0) not yet supported in ODE models \
                     (independent of TV-cov work)"
                );
                let cmt_idx = d.cmt.saturating_sub(1);
                if cmt_idx < n {
                    u[cmt_idx] += d.amt;
                }
                current_pk = pk_at_dose[idx];
            }
            Kind::Obs => {
                let v = u[ode.obs_cmt_idx];
                predictions[idx] = if v.is_nan() || v < 0.0 { 0.0 } else { v };
                current_pk = pk_at_obs[idx];
            }
            Kind::PkOnly => {
                // EVID=2: refresh current_pk so the next segment uses
                // the values $PK would have computed at this row in
                // NONMEM. Compartment state is unchanged.
                current_pk = pk_at_pk_only[idx];
            }
        }
    }

    predictions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DoseEvent;
    use approx::assert_relative_eq;

    /// 1-cpt IV bolus ODE: dA/dt = -ke·A. RHS reads CL,V from pk_params_flat.
    fn one_cpt_ode_spec() -> OdeSpec {
        OdeSpec {
            rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
                let cl = p[crate::types::PK_IDX_CL];
                let v = p[crate::types::PK_IDX_V];
                let ke = if v > 0.0 { cl / v } else { 0.0 };
                dy[0] = -ke * y[0];
            }),
            n_states: 1,
            state_names: vec!["central".into()],
            obs_cmt_idx: 0,
        }
    }

    fn pk_one(cl: f64, v: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = cl;
        p.values[crate::types::PK_IDX_V] = v;
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
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            cens: vec![0; n_obs],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
        }
    }

    #[test]
    fn ode_event_driven_matches_constant_path_when_pk_constant() {
        // Equivalence: when the per-event PK params are all the same, the
        // event-driven ODE path must agree with the existing single-snapshot
        // path. This is the "no TV covariates" sanity check.
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(8.0, 1000.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![1.0, 4.0, 8.5, 12.0, 24.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_one(5.0, 80.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];
        let ode = one_cpt_ode_spec();

        let baseline = ode_predictions(&ode, &pk.values, &subj);
        let event_driven = ode_predictions_event_driven(&ode, &subj, &pk_dose, &pk_obs, &[]);
        assert_eq!(baseline.len(), event_driven.len());
        for (b, e) in baseline.iter().zip(event_driven.iter()) {
            // ODE solver tolerance is ~1e-4 relative — a tighter equality
            // would over-constrain RK45.
            assert_relative_eq!(*b, *e, epsilon = 1e-6, max_relative = 1e-4);
        }
    }

    #[test]
    fn ode_event_driven_picks_up_changing_cl() {
        // Same shape as the analytical TV test: CL doubles between two doses.
        // The decay over [t_dose1, t_dose2] uses pk_dose1; after dose2 uses
        // pk_dose2. Compares against analytical exp(-ke·dt) closed form.
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
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions_event_driven(&ode, &subj, &pk_dose, &pk_obs, &[]);

        // Expected at t=5 (low-CL interval, no V scaling needed at obs in
        // this ODE since it returns A directly): A = 1000 * exp(-0.05*5)
        let a5 = 1000.0 * (-0.05f64 * 5.0).exp();
        assert_relative_eq!(preds[0], a5, epsilon = 1e-3, max_relative = 1e-4);

        // At t=10⁻: A = 1000 * exp(-0.5) ≈ 606.53
        // After dose2: A = 1606.53. Then 2h with ke=0.10 (pk_dose2=high):
        //   A(12) = 1606.53 * exp(-0.20) ≈ 1316.18
        let a10_minus = 1000.0 * (-0.5f64).exp();
        let a10_plus = a10_minus + 1000.0;
        let a12 = a10_plus * (-0.20f64).exp();
        assert_relative_eq!(preds[1], a12, epsilon = 1e-2, max_relative = 1e-4);
    }
}
