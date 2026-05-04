//! Event-driven AD path for time-varying covariate subjects.
//!
//! Mirrors `ad_gradients.rs` but takes per-event `tv` arrays so the AD
//! path can honour per-event LOCF covariate snapshots — the analytical
//! superposition path in `ad_gradients.rs` only handles a single `tv`
//! per subject, which silently drops time-varying covariate updates.
//!
//! Initial scope (matches the non-AD event-driven path in
//! `pk::event_driven`):
//!   - 1-compartment IV bolus & infusion
//!   - 2-compartment IV bolus & infusion (dose into central / cmt=1)
//!
//! Oral and 3-cpt models still fall back to the single-snapshot AD path
//! in `ad_gradients.rs` (which is correct for them only when covariates
//! are time-constant — the dispatcher in `inner_optimizer.rs` handles
//! the fallback).
//!
//! ## AD safety
//!
//! Same constraints as `ad_gradients.rs` — see CLAUDE.md. In particular:
//!   - No `f64::max` / `f64::min` (lower to `llvm.maximumnum`/`minimumnum`
//!     intrinsics that Enzyme can't differentiate). Use the arithmetic
//!     forms `(a+b±|a-b|)/2` (`.abs()` lowers to `llvm.fabs`, which is
//!     handled).
//!   - Avoid branches on differentiated quantities; branch only on
//!     `Const` inputs (event metadata, dose properties, model IDs).
//!     Const branches get constant-folded by LLVM and don't poison the
//!     reverse-mode adjoint.

use crate::types::*;
use std::autodiff::autodiff_reverse;

// ─────────────────────────────────────────────────────────────────────
// Flat data layout for one subject's event timeline.
// ─────────────────────────────────────────────────────────────────────

/// Pre-built flat arrays describing one subject's merged event timeline.
/// Constructed once per inner-loop call and re-used across every gradient
/// evaluation (eta perturbations don't change the timeline shape).
pub struct FlatEventData {
    /// Sorted merged event times.
    pub event_times: Vec<f64>,
    /// 0.0 for dose, 1.0 for observation.
    pub event_kinds: Vec<f64>,
    /// Original index back into `subject.doses` (when kind=0) or
    /// `subject.obs_times` (when kind=1).
    pub event_orig_idx_f64: Vec<f64>,
    /// Dose-level arrays (parallel to `subject.doses`).
    pub dose_times: Vec<f64>,
    pub dose_amts: Vec<f64>,
    pub dose_rates: Vec<f64>,
    pub dose_durations: Vec<f64>,
}

impl FlatEventData {
    pub fn from_subject(subject: &Subject) -> Self {
        let n_obs = subject.obs_times.len();
        let n_dose = subject.doses.len();
        let n_events = n_obs + n_dose;

        let mut events: Vec<(f64, f64, f64)> = Vec::with_capacity(n_events);
        // (time, kind, orig_idx). Doses first, then obs — kind tie-breaks to
        // dose-before-obs at the same time so an obs at the dose time sees
        // the post-dose state.
        for (k, d) in subject.doses.iter().enumerate() {
            events.push((d.time, 0.0, k as f64));
        }
        for (j, &t) in subject.obs_times.iter().enumerate() {
            events.push((t, 1.0, j as f64));
        }
        events.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        });

        let mut event_times = Vec::with_capacity(n_events);
        let mut event_kinds = Vec::with_capacity(n_events);
        let mut event_orig_idx_f64 = Vec::with_capacity(n_events);
        for (t, k, idx) in events {
            event_times.push(t);
            event_kinds.push(k);
            event_orig_idx_f64.push(idx);
        }

        Self {
            event_times,
            event_kinds,
            event_orig_idx_f64,
            dose_times: subject.doses.iter().map(|d| d.time).collect(),
            dose_amts: subject.doses.iter().map(|d| d.amt).collect(),
            dose_rates: subject.doses.iter().map(|d| d.rate).collect(),
            dose_durations: subject.doses.iter().map(|d| d.duration).collect(),
        }
    }
}

/// Per-event tv arrays (length `n_events * n_tv`, row-major). Built by
/// evaluating `model.tv_fn` against each event's covariate snapshot.
/// Order must match `FlatEventData::event_*` arrays.
pub struct FlatEventTv {
    pub tv: Vec<f64>,
    pub n_tv: usize,
    pub n_events: usize,
}

impl FlatEventTv {
    pub fn from_subject(model: &CompiledModel, subject: &Subject, theta: &[f64]) -> Self {
        let tv_fn = model
            .tv_fn
            .as_ref()
            .expect("FlatEventTv::from_subject: model.tv_fn required for AD path");

        // Re-derive the same event order used by FlatEventData::from_subject.
        let mut events: Vec<(f64, f64, usize, bool)> =
            Vec::with_capacity(subject.doses.len() + subject.obs_times.len());
        for (k, d) in subject.doses.iter().enumerate() {
            events.push((d.time, 0.0, k, false /* is_obs */));
        }
        for (j, &t) in subject.obs_times.iter().enumerate() {
            events.push((t, 1.0, j, true));
        }
        events.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        });

        let n_events = events.len();
        let n_tv = model.pk_idx_f64.len();

        let mut tv = Vec::with_capacity(n_events * n_tv);
        for (_, _, orig, is_obs) in &events {
            let cov = if *is_obs {
                subject.obs_cov(*orig)
            } else {
                subject.dose_cov(*orig)
            };
            let row = tv_fn(theta, cov);
            assert_eq!(row.len(), n_tv, "tv_fn returned wrong length");
            tv.extend_from_slice(&row);
        }

        Self { tv, n_tv, n_events }
    }
}

// ─────────────────────────────────────────────────────────────────────
// AD-instrumented core: reverse-mode for the gradient w.r.t. eta.
// ─────────────────────────────────────────────────────────────────────

#[autodiff_reverse(
    individual_nll_event_driven_ad_grad,
    Duplicated, // eta
    Const,      // tv_per_event
    Const,      // omega_inv_flat
    Const,      // log_det_omega
    Const,      // sigma_values
    Const,      // event_times
    Const,      // event_kinds
    Const,      // event_orig_idx_f64
    Const,      // dose_times
    Const,      // dose_amts
    Const,      // dose_rates
    Const,      // dose_durations
    Const,      // observations
    Const,      // cens_f64
    Const,      // pk_idx_f64
    Const,      // sel_flat
    Const,      // pk_and_err_model
    Active      // return
)]
pub fn individual_nll_event_driven_ad(
    eta: &[f64],
    tv_per_event: &[f64],         // n_events * n_tv (row-major)
    omega_inv_flat: &[f64],
    log_det_omega: f64,
    sigma_values: &[f64],
    event_times: &[f64],
    event_kinds: &[f64],          // 0=dose, 1=obs
    event_orig_idx_f64: &[f64],
    dose_times: &[f64],
    dose_amts: &[f64],
    dose_rates: &[f64],
    dose_durations: &[f64],
    observations: &[f64],
    cens_f64: &[f64],
    pk_idx_f64: &[f64],
    sel_flat: &[f64],
    pk_and_err_model: f64,
) -> f64 {
    let n_eta = eta.len();
    let n_tv = pk_idx_f64.len();
    let n_events = event_times.len();
    let n_doses = dose_times.len();
    let pk_model_id = (pk_and_err_model as i32) / 10;
    let error_model_id = (pk_and_err_model as i32) % 10;

    // ── η prior: η' Ω⁻¹ η ───────────────────────────────────────────
    let mut eta_prior = 0.0;
    for i in 0..n_eta {
        for j in 0..n_eta {
            eta_prior += eta[i] * omega_inv_flat[i * n_eta + j] * eta[j];
        }
    }

    // ── State + walk events ─────────────────────────────────────────
    // State held as scalars rather than `[f64; 2]` — avoids the array
    // initialization (`llvm.memset` of 16 bytes) and the `&mut [f64;2]`
    // through the propagator boundary. Both forms triggered Enzyme
    // type-deduction failures in reverse mode on this function shape.
    let mut state0 = 0.0_f64;
    let mut state1 = 0.0_f64;

    // Current PK params governing the current interval. Zero until we
    // reach the first event; harmless because the first propagation has
    // dt = 0 (cur_t starts at events[0].time).
    let mut current_cl = 0.0_f64;
    let mut current_v = 0.0_f64;
    let mut current_q = 0.0_f64;
    let mut current_v2 = 0.0_f64;

    let mut cur_t = if n_events > 0 { event_times[0] } else { 0.0 };

    let mut data_ll = 0.0_f64;

    for ev_idx in 0..n_events {
        let t_ev = event_times[ev_idx];
        // Always call the propagator — when dt = 0 the math is a no-op
        // (exp(0) = 1, infusion contributions reduce to e^x - e^x = 0).
        // Branching on `dt > 0.0` here would create a phi-node on
        // `state` that Enzyme can't type-deduce in reverse mode.
        let (s0_new, s1_new) = propagate_state_ad(
            pk_model_id,
            state0,
            state1,
            cur_t,
            t_ev,
            current_cl,
            current_v,
            current_q,
            current_v2,
            dose_times,
            dose_rates,
            dose_durations,
            n_doses,
        );
        state0 = s0_new;
        state1 = s1_new;

        // Compute pk_params at this event from per-event tv row + eta.
        // Inlined as scalars (no `[f64; MAX_PK_PARAMS]` array) so we
        // avoid the array memset + per-iter aliasing that confuses
        // Enzyme. The scalar form gives identical numerics.
        let mut ev_cl = 0.0_f64;
        let mut ev_v = 0.0_f64;
        let mut ev_q = 0.0_f64;
        let mut ev_v2 = 0.0_f64;
        let row_off = ev_idx * n_tv;
        for i in 0..n_tv {
            let mut eta_contrib = 0.0;
            for j in 0..n_eta {
                eta_contrib += sel_flat[i * n_eta + j] * eta[j];
            }
            let val = tv_per_event[row_off + i] * eta_contrib.exp();
            let idx = pk_idx_f64[i] as usize;
            // Const-branch fan-out: dispatch the value to the right
            // scalar based on the (Const) pk index. Other slots
            // (KA, F, Q3, V3) aren't used by 1-/2-cpt IV/infusion.
            if idx == PK_IDX_CL {
                ev_cl = val;
            } else if idx == PK_IDX_V {
                ev_v = val;
            } else if idx == PK_IDX_Q {
                ev_q = val;
            } else if idx == PK_IDX_V2 {
                ev_v2 = val;
            }
        }

        let kind = event_kinds[ev_idx];
        let orig = event_orig_idx_f64[ev_idx] as usize;
        // `orig` is the original index in *either* the doses or the obs
        // arrays depending on `kind`. To keep both branches branch-free
        // (we evaluate both and mask with `is_dose` / `is_obs`), clamp
        // the index for the inactive branch so the load stays in-bounds
        // — its result is multiplied by 0 anyway.
        let dose_idx = if kind < 0.5 { orig } else { 0 };
        let obs_idx = if kind < 0.5 { 0 } else { orig };

        // ── Dose branch (kind < 0.5). All inputs here are Const so the
        // gating is Const-Const with no adjoint flow.
        let is_dose = if kind < 0.5 { 1.0 } else { 0.0 };
        let is_bolus = if dose_rates[dose_idx] == 0.0 { 1.0 } else { 0.0 };
        state0 += is_dose * is_bolus * dose_amts[dose_idx];

        // ── Observation branch. Mask via is_obs so non-obs events
        // contribute exactly zero to data_ll without an `if` phi.
        let is_obs = if kind < 0.5 { 0.0 } else { 1.0 };

        // Strictly positive divisor — handles transient ev_v ≤ 0 from
        // line-search trial steps.
        let v_safe = ev_v.abs() + 1e-30;
        let conc_raw = state0 / v_safe;
        let conc = (conc_raw + conc_raw.abs()) * 0.5;

        let v_resid = residual_variance_ad(error_model_id, conc, sigma_values);
        let cens_active = if cens_f64[obs_idx] > 0.5 { 1.0 } else { 0.0 };
        let resid = observations[obs_idx] - conc;
        let z = resid / v_resid.sqrt();
        let bloq_term = -2.0 * log_normal_cdf_ad(z);
        let gaussian_term = resid * resid / v_resid + v_resid.ln();
        let obs_term = cens_active * bloq_term + (1.0 - cens_active) * gaussian_term;
        data_ll += is_obs * obs_term;

        current_cl = ev_cl;
        current_v = ev_v;
        current_q = ev_q;
        current_v2 = ev_v2;
        cur_t = t_ev;
    }

    0.5 * (eta_prior + log_det_omega + data_ll)
}

// ─────────────────────────────────────────────────────────────────────
// Forward-mode AD path for the Jacobian d(predictions)/d(eta).
//
// **Disabled for now.** The function below is identical in shape to the
// reverse-mode NLL above but Enzyme's forward-mode pointer tracker hits
// `Assertion failed: forwardModeInvertedPointerFallback` on the
// per-event scalar-state propagation. The wrapper in
// `inner_optimizer.rs` falls back to FD for the H matrix on TV-cov
// subjects until this is resolved upstream — reverse-mode AD on the
// BFGS gradient (the hot path) still wins back the bulk of the cost.
// Tracked alongside the oral/3-cpt follow-up.
// ─────────────────────────────────────────────────────────────────────

#[allow(dead_code)]
fn predict_all_event_driven_ad(
    eta: &[f64],
    tv_per_event: &[f64],
    event_times: &[f64],
    event_kinds: &[f64],
    event_orig_idx_f64: &[f64],
    dose_times: &[f64],
    dose_amts: &[f64],
    dose_rates: &[f64],
    dose_durations: &[f64],
    pk_idx_f64: &[f64],
    sel_flat: &[f64],
    pk_model_id_f64: f64,
    out: &mut [f64],
) {
    let n_eta = eta.len();
    let n_tv = pk_idx_f64.len();
    let n_events = event_times.len();
    let n_doses = dose_times.len();
    let pk_model_id = pk_model_id_f64 as i32;

    // Scalar state — see rationale in the reverse-mode NLL.
    let mut state0 = 0.0_f64;
    let mut state1 = 0.0_f64;
    let mut current_cl = 0.0_f64;
    let mut current_v = 0.0_f64;
    let mut current_q = 0.0_f64;
    let mut current_v2 = 0.0_f64;
    let mut cur_t = if n_events > 0 { event_times[0] } else { 0.0 };

    for ev_idx in 0..n_events {
        let t_ev = event_times[ev_idx];
        let (s0_new, s1_new) = propagate_state_ad(
            pk_model_id,
            state0,
            state1,
            cur_t,
            t_ev,
            current_cl,
            current_v,
            current_q,
            current_v2,
            dose_times,
            dose_rates,
            dose_durations,
            n_doses,
        );
        state0 = s0_new;
        state1 = s1_new;

        let mut ev_cl = 0.0_f64;
        let mut ev_v = 0.0_f64;
        let mut ev_q = 0.0_f64;
        let mut ev_v2 = 0.0_f64;
        let row_off = ev_idx * n_tv;
        for i in 0..n_tv {
            let mut eta_contrib = 0.0;
            for j in 0..n_eta {
                eta_contrib += sel_flat[i * n_eta + j] * eta[j];
            }
            let val = tv_per_event[row_off + i] * eta_contrib.exp();
            let idx = pk_idx_f64[i] as usize;
            if idx == PK_IDX_CL {
                ev_cl = val;
            } else if idx == PK_IDX_V {
                ev_v = val;
            } else if idx == PK_IDX_Q {
                ev_q = val;
            } else if idx == PK_IDX_V2 {
                ev_v2 = val;
            }
        }

        let kind = event_kinds[ev_idx];
        let orig = event_orig_idx_f64[ev_idx] as usize;

        let is_dose = if kind < 0.5 { 1.0 } else { 0.0 };
        let is_bolus = if dose_rates[orig] == 0.0 { 1.0 } else { 0.0 };
        state0 += is_dose * is_bolus * dose_amts[orig];

        // Write the central-compartment concentration unconditionally
        // — every event slot in `out` gets one write per pass. The
        // wrapper picks out the obs slots by event_kinds. Without the
        // unconditional write Enzyme's forward-mode pointer tracking
        // hits `Assertion failed: (found == gutils->invertedPointers...)`.
        let v_safe = ev_v.abs() + 1e-30;
        let conc_raw = state0 / v_safe;
        let conc = (conc_raw + conc_raw.abs()) * 0.5;
        out[ev_idx] = conc;

        current_cl = ev_cl;
        current_v = ev_v;
        current_q = ev_q;
        current_v2 = ev_v2;
        cur_t = t_ev;
    }
}

// ─────────────────────────────────────────────────────────────────────
// Inlined event-driven propagators (AD-safe).
//
// Each propagator advances `state` from `t_from` to `t_to` under the
// linear ODE governed by (cl, v, q, v2) and adds the contributions of
// any infusions active during the interval. Infusion contribution per
// active window [p_start, p_end] uses the unified formula derived in
// the design notes:
//
//   contribution_to_central(t_to) = (r/ke) * [exp(-ke·τ_to) - exp(-ke·τ_total)]
//
// where τ_to = t_to - p_end, τ_total = t_to - p_start. Both are clamped
// to ensure τ_total ≥ τ_to ≥ 0 so that infusions outside the interval
// contribute exactly zero (formula reduces to e^x - e^x = 0).
// ─────────────────────────────────────────────────────────────────────

/// Returns `(state0_new, state1_new)`. State is passed by value (no `&mut`)
/// so the function returns scalars rather than mutating an array — Enzyme
/// reverse-mode handles scalar return values cleanly but trips on the
/// memset / mixed-active aliasing of `&mut [f64; 2]` here.
#[allow(clippy::too_many_arguments)]
fn propagate_state_ad(
    pk_model_id: i32,
    state0: f64,
    state1: f64,
    t_from: f64,
    t_to: f64,
    cl: f64,
    v: f64,
    q: f64,
    v2: f64,
    dose_times: &[f64],
    dose_rates: &[f64],
    dose_durations: &[f64],
    n_doses: usize,
) -> (f64, f64) {
    // Const-only branch on pk_model_id — constant-folds under LLVM,
    // same pattern as `single_dose_ad`. Only one of these branches
    // contains real adjoint flow per build.
    if pk_model_id == 0 || pk_model_id == 2 {
        let s0 = propagate_one_cpt_ad(
            state0,
            t_from,
            t_to,
            cl,
            v,
            dose_times,
            dose_rates,
            dose_durations,
            n_doses,
        );
        // 1-cpt has no peripheral compartment — `state1` carries through
        // unchanged.
        (s0, state1)
    } else if pk_model_id == 3 || pk_model_id == 5 {
        propagate_two_cpt_ad(
            state0,
            state1,
            t_from,
            t_to,
            cl,
            v,
            q,
            v2,
            dose_times,
            dose_rates,
            dose_durations,
            n_doses,
        )
    } else {
        // Other pk_model_ids return state unchanged — dispatcher in
        // inner_optimizer.rs forces FD for those.
        (state0, state1)
    }
}

/// 1-cpt linear propagator. Branch-free over both (cl, v) and the dose
/// loop: degenerate parameters are guarded by arithmetic clamps, and the
/// per-infusion contribution naturally evaluates to zero for bolus rows
/// (`r = 0`) and outside-the-interval infusions (`p_end ≤ p_start` makes
/// `tau_total = tau_to` so `e^{-ke·τ_to} - e^{-ke·τ_total} = 0`).
#[allow(clippy::too_many_arguments)]
fn propagate_one_cpt_ad(
    state0: f64,
    t_from: f64,
    t_to: f64,
    cl: f64,
    v: f64,
    dose_times: &[f64],
    dose_rates: &[f64],
    dose_durations: &[f64],
    n_doses: usize,
) -> f64 {
    let v_safe = v.abs() + 1e-30;
    let cl_safe = cl.abs() + 1e-30;
    let ke = cl_safe / v_safe;
    let dt = t_to - t_from;
    let exp_dt = (-ke * dt).exp();

    let mut s0 = state0 * exp_dt;

    for d in 0..n_doses {
        let s_i = dose_times[d];
        let e_i = s_i + dose_durations[d];
        let p_start = (s_i + t_from + (s_i - t_from).abs()) * 0.5;
        let p_end = (e_i + t_to - (e_i - t_to).abs()) * 0.5;
        let tau_to_raw = t_to - p_end;
        let tau_to = (tau_to_raw + tau_to_raw.abs()) * 0.5;
        let tau_total_raw = t_to - p_start;
        let diff = tau_total_raw - tau_to;
        let tau_total = tau_to + (diff + diff.abs()) * 0.5;
        let contribution =
            (dose_rates[d] / ke) * ((-ke * tau_to).exp() - (-ke * tau_total).exp());
        s0 += contribution;
    }
    s0
}

/// 2-cpt linear propagator. Returns `(state0_new, state1_new)`.
#[allow(clippy::too_many_arguments)]
fn propagate_two_cpt_ad(
    state0: f64,
    state1: f64,
    t_from: f64,
    t_to: f64,
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
    dose_times: &[f64],
    dose_rates: &[f64],
    dose_durations: &[f64],
    n_doses: usize,
) -> (f64, f64) {
    let v1_safe = v1.abs() + 1e-30;
    let cl_safe = cl.abs() + 1e-30;
    let q_safe = q.abs() + 1e-30;
    let v2_safe = v2.abs() + 1e-30;
    let k10 = cl_safe / v1_safe;
    let k12 = q_safe / v1_safe;
    let k21 = q_safe / v2_safe;
    let s = k10 + k12 + k21;
    let d_eig = k10 * k21;
    let arg = s * s - 4.0 * d_eig;
    let arg_clamped = (arg + arg.abs()) * 0.5;
    let disc = arg_clamped.sqrt();
    let alpha = (s + disc) * 0.5;
    let beta = d_eig / alpha;

    let dt = t_to - t_from;
    let e_alpha = (-alpha * dt).exp();
    let e_beta = (-beta * dt).exp();

    // Homogeneous decomposition into eigenmodes.
    let s_homog = state1 / k12;
    let denom = beta - alpha;
    let c1 = (state0 - s_homog * (k21 - beta)) / denom;
    let c2 = s_homog - c1;

    let mut s0 = c1 * (k21 - alpha) * e_alpha + c2 * (k21 - beta) * e_beta;
    let mut s1 = (c1 * e_alpha + c2 * e_beta) * k12;

    // Infusion contributions (cmt=1 / central only — follow-up TODO).
    for d in 0..n_doses {
        let s_i = dose_times[d];
        let e_i = s_i + dose_durations[d];
        let p_start = (s_i + t_from + (s_i - t_from).abs()) * 0.5;
        let p_end = (e_i + t_to - (e_i - t_to).abs()) * 0.5;
        let tau_to_raw = t_to - p_end;
        let tau_to = (tau_to_raw + tau_to_raw.abs()) * 0.5;
        let tau_total_raw = t_to - p_start;
        let diff = tau_total_raw - tau_to;
        let tau_total = tau_to + (diff + diff.abs()) * 0.5;

        let a_ss_1 = dose_rates[d] * v1_safe / cl_safe;
        let a_ss_2 = dose_rates[d] * v2_safe / cl_safe;

        let s_ss = a_ss_2 / k12;
        let c1_ss = (a_ss_1 - s_ss * (k21 - beta)) / denom;
        let c2_ss = s_ss - c1_ss;

        let e_a_to = (-alpha * tau_to).exp();
        let e_a_tot = (-alpha * tau_total).exp();
        let e_b_to = (-beta * tau_to).exp();
        let e_b_tot = (-beta * tau_total).exp();

        let a1_contrib = c1_ss * (k21 - alpha) * (e_a_to - e_a_tot)
            + c2_ss * (k21 - beta) * (e_b_to - e_b_tot);
        let a2_contrib = k12 * (c1_ss * (e_a_to - e_a_tot) + c2_ss * (e_b_to - e_b_tot));

        s0 += a1_contrib;
        s1 += a2_contrib;
    }
    (s0, s1)
}

// ─── AD-safe special functions (re-exports from ad_gradients) ───────

/// Local copy of `erf_ad` so this module compiles independently of the
/// instrumented function in `ad_gradients`. Same Abramowitz & Stegun
/// 7.1.26 polynomial; see comment there for AD-safety rationale.
fn erf_ad(x: f64) -> f64 {
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = if x < 0.0 { -x } else { x };
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

fn log_normal_cdf_ad(z: f64) -> f64 {
    const INV_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
    const LOG_SQRT_2PI: f64 = 0.918_938_533_204_672_7;
    const MIN_PROB: f64 = 1e-300;

    if z > -5.0 {
        let p = 0.5 * (1.0 + erf_ad(z * INV_SQRT_2));
        let p_floor = if p < MIN_PROB { MIN_PROB } else { p };
        p_floor.ln()
    } else {
        let log_phi = -0.5 * z * z - LOG_SQRT_2PI;
        let inv_z2 = 1.0 / (z * z);
        let series = 1.0 - inv_z2 + 3.0 * inv_z2 * inv_z2 - 15.0 * inv_z2 * inv_z2 * inv_z2;
        log_phi - (-z).ln() + series.ln()
    }
}

fn residual_variance_ad(error_model_id: i32, f_pred: f64, sigma: &[f64]) -> f64 {
    let v = match error_model_id {
        0 => sigma[0] * sigma[0],
        1 => {
            let fs = f_pred * sigma[0];
            fs * fs
        }
        2 => {
            let p = f_pred * sigma[0];
            p * p + sigma[1] * sigma[1]
        }
        _ => sigma[0] * sigma[0],
    };
    if v < 1e-12 {
        1e-12
    } else {
        v
    }
}

// ─── Public wrappers ────────────────────────────────────────────────

/// True when this PK model has an event-driven AD implementation in
/// this module. Caller-side dispatch in `inner_optimizer.rs` uses this
/// to decide whether to take the AD fast path or fall back to FD.
pub fn supports_event_driven_ad(pk_model: PkModel) -> bool {
    matches!(
        pk_model,
        PkModel::OneCptIvBolus
            | PkModel::OneCptInfusion
            | PkModel::TwoCptIvBolus
            | PkModel::TwoCptInfusion
    )
}

/// Compute (NLL, ∂NLL/∂η) using the event-driven AD path with per-event
/// covariate snapshots.
///
/// Pads `observations` / `cens_f64` / `dose_*` arrays with single-element
/// placeholders when the subject has zero in either dimension. The masks
/// inside the AD kernel zero out their contribution, but the indices
/// must still be in-bounds — without padding a no-obs subject would
/// panic on `cens_f64[0]`.
#[allow(clippy::too_many_arguments)]
pub fn compute_nll_gradient_event_driven_ad(
    eta: &[f64],
    tv_per_event: &FlatEventTv,
    omega_inv_flat: &[f64],
    log_det_omega: f64,
    sigma_values: &[f64],
    event_data: &FlatEventData,
    observations: &[f64],
    cens_f64: &[f64],
    pk_model: PkModel,
    error_model: ErrorModel,
    pk_idx_f64: &[f64],
    sel_flat: &[f64],
) -> (f64, Vec<f64>) {
    let n_eta = eta.len();
    let mut d_eta = vec![0.0_f64; n_eta];

    let pk_and_err = (crate::ad::ad_gradients::pk_model_to_id(pk_model) * 10
        + crate::ad::ad_gradients::error_model_to_id(error_model)) as f64;

    // Pad zero-length arrays so the AD kernel's masked-but-still-evaluated
    // index loads stay in-bounds. The mask multiplies the read by 0 so
    // the value doesn't matter, only the address being valid.
    let observations_padded: &[f64] = if observations.is_empty() { &[0.0] } else { observations };
    let cens_padded: &[f64] = if cens_f64.is_empty() { &[0.0] } else { cens_f64 };
    let dose_times_padded: &[f64] = if event_data.dose_times.is_empty() {
        &[0.0]
    } else {
        &event_data.dose_times
    };
    let dose_amts_padded: &[f64] = if event_data.dose_amts.is_empty() {
        &[0.0]
    } else {
        &event_data.dose_amts
    };
    let dose_rates_padded: &[f64] = if event_data.dose_rates.is_empty() {
        &[0.0]
    } else {
        &event_data.dose_rates
    };
    let dose_durations_padded: &[f64] = if event_data.dose_durations.is_empty() {
        &[0.0]
    } else {
        &event_data.dose_durations
    };

    let nll = individual_nll_event_driven_ad_grad(
        eta,
        &mut d_eta,
        &tv_per_event.tv,
        omega_inv_flat,
        log_det_omega,
        sigma_values,
        &event_data.event_times,
        &event_data.event_kinds,
        &event_data.event_orig_idx_f64,
        dose_times_padded,
        dose_amts_padded,
        dose_rates_padded,
        dose_durations_padded,
        observations_padded,
        cens_padded,
        pk_idx_f64,
        sel_flat,
        pk_and_err,
        1.0,
    );

    (nll, d_eta)
}

// (Jacobian wrapper deleted alongside the disabled forward-mode AD
// function; the inner-loop dispatch routes the AdEventDriven branch
// through `compute_jacobian_fd` instead. Reinstate when Enzyme's
// `forwardModeInvertedPointerFallback` issue is resolved upstream.)
