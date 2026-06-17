//! Per-observation analytic sensitivities for **user-specified `[odes]` models**
//! (issue #367, Option A). The closed-form provider ([`super::provider`]) covers
//! the analytical 1-/2-/3-cpt PK models; this is its ODE counterpart.
//!
//! The state is integrated as [`Dual2<N>`](super::dual2::Dual2) seeded on the
//! `N` individual parameters: the compiled RHS program
//! ([`OdeRhsProgram`](crate::parser::model_parser::OdeRhsProgram)) is evaluated
//! over the dual numbers by the generic bytecode VM, and the generic RK45
//! ([`solve_ode_g`](crate::ode::solver::solve_ode_g)) propagates `∂u/∂p` and
//! `∂²u/∂p²` through the integration with **value-based step control**. The
//! readout then yields `∂f/∂p, ∂²f/∂p²` per observation, which feed the η/θ chain
//! via the **general** individual-parameter derivatives `∂p/∂η, ∂p/∂θ` (FD of
//! `pk_param_fn` — see [`param_derivatives`]; no log-normal assumption).
//!
//! **Supported:** single-endpoint `ObsCmt` or simple Form C (`y = central/V1`)
//! readout; **bolus and infusion** doses; **bioavailability F** (incl. estimated,
//! any parameterization — log-normal, logit-normal, additive); **EVID 3/4 resets
//! / multi-occasion**; static covariates; up to [`MAX_ODE_SENS_DIM`] individual
//! parameters.
//!
//! **Not yet supported** (falls back to the gradient-free path): steady-state
//! dosing, lagtime, non-zero `init(...)`, built-in input-rate absorption, IOV,
//! SDE/diffusion, `obs_scale`/LTBS output transforms, time-varying covariates,
//! per-CMT Form C.
#![allow(clippy::needless_range_loop)]

use super::dual2::Dual2;
use super::provider::{ObsSens, SubjectSens};
use crate::ode::predictions::OdeReadout;
use crate::ode::solver::solve_ode_g;
use crate::types::{CompiledModel, ScalingSpec, Subject, PK_IDX_F, PK_IDX_LAGTIME};
use std::cell::RefCell;

/// Largest individual-parameter count for which the `Dual2<N>` path is
/// monomorphised; models wider than this fall back to the gradient-free path.
const MAX_ODE_SENS_DIM: usize = 12;

/// True when [`ode_subject_sensitivities`] can serve this model: an ODE model
/// with a compiled RHS program, single `ObsCmt` readout, no built-in absorption,
/// no `init(...)`, no IOV/SDE, no output transform, and an individual-parameter
/// count within [`MAX_ODE_SENS_DIM`]. Per-subject gates (bolus-only doses, no TV
/// covariates/resets) are checked in [`ode_subject_sensitivities`].
pub fn ode_analytical_supported(model: &CompiledModel) -> bool {
    let Some(ode) = model.ode_spec.as_ref() else {
        return false;
    };
    if ode.rhs_program.is_none() {
        return false;
    }
    // Readout: either the state directly (`ObsCmt`) or a simple Form C output
    // program (`y = <expr>` over states/indiv params, e.g. `central / V1`).
    let readout_ok = match &ode.readout {
        OdeReadout::ObsCmt(_) => true,
        OdeReadout::Single(_) => ode.readout_program.as_ref().is_some_and(|p| p.is_simple()),
        OdeReadout::PerCmt(_) => false,
    };
    if !readout_ok {
        return false;
    }
    if !ode.input_rate.is_empty() || ode.init_fn.is_some() || !ode.diffusion_var.is_empty() {
        return false;
    }
    // The divisor (`obs_scale`) scaling form is not yet handled over Dual2; Form C
    // (`y = central/V1`) is handled via the readout program instead.
    if model.n_kappa != 0 || !matches!(model.scaling, ScalingSpec::None) || model.log_transform {
        return false;
    }
    // (ODE models have no `tv_fn` — typical values come from `pk_param_fn` at
    // η = 0 instead; see `run_subject`.)
    // Lagtime shifts the dosing timeline; supporting an estimated lagtime needs
    // ∂(timeline)/∂θ, which is not yet wired — exclude models that estimate it.
    // Bioavailability F *is* supported (it scales the dose amount/rate as a dual).
    if model.pk_indices.iter().any(|&s| s == PK_IDX_LAGTIME) {
        return false;
    }
    let n = model.pk_indices.len();
    (1..=MAX_ODE_SENS_DIM).contains(&n)
}

/// Compute per-observation analytic sensitivities for an ODE model, or `None` if
/// it is outside the supported scope (caller falls back to the gradient-free
/// path).
pub fn ode_subject_sensitivities(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<SubjectSens> {
    if !ode_analytical_supported(model) || subject.has_tv_covariates() {
        return None;
    }
    // Steady-state dosing is not yet supported over the dual loop (needs dual
    // SS-equilibration); bolus and (finite-duration) infusion doses are handled.
    if subject.doses.iter().any(|d| d.ss && d.ii > 0.0) {
        return None;
    }

    // Dispatch on the individual-parameter count so the dual width is right-sized.
    macro_rules! dispatch {
        ($($n:literal),+) => {
            match model.pk_indices.len() {
                $($n => run_subject::<$n>(model, subject, theta, eta),)+
                _ => None,
            }
        };
    }
    dispatch!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
}

/// Derivatives of the individual parameters `p_i` w.r.t. η and θ, by central
/// finite difference of `pk_param_fn` at the actual `(θ, η)`. This is the
/// **general** η/θ chain — it makes no log-normal assumption, so it handles
/// log-normal (`p = tv·exp(η)`), logit-normal bioavailability
/// (`F = inv_logit(logit(TVF)+η)`), additive, and any other individual-parameter
/// form, reducing to the closed-form `p·sel` chain in the log-normal case.
/// `pk_param_fn` is cheap (no ODE integration), so the FD cost is negligible
/// next to one dual integration.
struct ParamDerivs {
    /// `∂p_i/∂η_k`.
    dp_deta: Vec<Vec<f64>>,
    /// `∂p_i/∂θ_m`.
    dp_dtheta: Vec<Vec<f64>>,
    /// `∂²p_i/∂η_k∂η_l`.
    d2p_deta2: Vec<Vec<Vec<f64>>>,
    /// `∂²p_i/∂η_k∂θ_m`.
    d2p_detadtheta: Vec<Vec<Vec<f64>>>,
}

fn param_derivatives(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> ParamDerivs {
    let cov = &subject.covariates;
    let slots = &model.pk_indices;
    let n_indiv = slots.len();
    let n_eta = model.n_eta;
    let n_theta = model.n_theta;
    let pval = |th: &[f64], et: &[f64]| -> Vec<f64> {
        let r = (model.pk_param_fn)(th, et, cov);
        slots.iter().map(|&s| r.values[s]).collect()
    };
    let he = 1e-6; // first-derivative step
    let h2 = 1e-4; // second-derivative step (4-point is round-off-prone)
    let base = pval(theta, eta);

    let mut dp_deta = vec![vec![0.0; n_eta]; n_indiv];
    for k in 0..n_eta {
        let mut ep = eta.to_vec();
        ep[k] += he;
        let mut em = eta.to_vec();
        em[k] -= he;
        let (up, dn) = (pval(theta, &ep), pval(theta, &em));
        for i in 0..n_indiv {
            dp_deta[i][k] = (up[i] - dn[i]) / (2.0 * he);
        }
    }

    let mut dp_dtheta = vec![vec![0.0; n_theta]; n_indiv];
    for m in 0..n_theta {
        let s = he * (1.0 + theta[m].abs());
        let mut tp = theta.to_vec();
        tp[m] += s;
        let mut tm = theta.to_vec();
        tm[m] -= s;
        let (up, dn) = (pval(&tp, eta), pval(&tm, eta));
        for i in 0..n_indiv {
            dp_dtheta[i][m] = (up[i] - dn[i]) / (2.0 * s);
        }
    }

    let mut d2p_deta2 = vec![vec![vec![0.0; n_eta]; n_eta]; n_indiv];
    for k in 0..n_eta {
        let mut ep = eta.to_vec();
        ep[k] += h2;
        let mut em = eta.to_vec();
        em[k] -= h2;
        let (up, dn) = (pval(theta, &ep), pval(theta, &em));
        for i in 0..n_indiv {
            d2p_deta2[i][k][k] = (up[i] - 2.0 * base[i] + dn[i]) / (h2 * h2);
        }
        for l in (k + 1)..n_eta {
            let mut pp = eta.to_vec();
            pp[k] += h2;
            pp[l] += h2;
            let mut pm = eta.to_vec();
            pm[k] += h2;
            pm[l] -= h2;
            let mut mp = eta.to_vec();
            mp[k] -= h2;
            mp[l] += h2;
            let mut mm = eta.to_vec();
            mm[k] -= h2;
            mm[l] -= h2;
            let (vpp, vpm, vmp, vmm) = (
                pval(theta, &pp),
                pval(theta, &pm),
                pval(theta, &mp),
                pval(theta, &mm),
            );
            for i in 0..n_indiv {
                let v = (vpp[i] - vpm[i] - vmp[i] + vmm[i]) / (4.0 * h2 * h2);
                d2p_deta2[i][k][l] = v;
                d2p_deta2[i][l][k] = v;
            }
        }
    }

    let mut d2p_detadtheta = vec![vec![vec![0.0; n_theta]; n_eta]; n_indiv];
    for k in 0..n_eta {
        let mut ep = eta.to_vec();
        ep[k] += h2;
        let mut em = eta.to_vec();
        em[k] -= h2;
        for m in 0..n_theta {
            let s = h2 * (1.0 + theta[m].abs());
            let mut tp = theta.to_vec();
            tp[m] += s;
            let mut tm = theta.to_vec();
            tm[m] -= s;
            let (vpp, vpm, vmp, vmm) = (
                pval(&tp, &ep),
                pval(&tm, &ep),
                pval(&tp, &em),
                pval(&tm, &em),
            );
            for i in 0..n_indiv {
                d2p_detadtheta[i][k][m] = (vpp[i] - vpm[i] - vmp[i] + vmm[i]) / (4.0 * h2 * s);
            }
        }
    }

    ParamDerivs {
        dp_deta,
        dp_dtheta,
        d2p_deta2,
        d2p_detadtheta,
    }
}

fn run_subject<const N: usize>(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<SubjectSens> {
    let ode = model.ode_spec.as_ref()?;
    let program = ode.rhs_program.as_ref()?;
    let n_eta = model.n_eta;
    let n_theta = model.n_theta;
    let opts = ode.solver_opts;

    // PK parameter values at (θ, η).
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates);

    // Seed the flat PK-parameter vector: individual parameter i (PK slot
    // `pk_indices[i]`) carries dual dimension i; everything else is constant.
    let mut params_dual: Vec<Dual2<N>> = pk.values.iter().map(|&v| Dual2::constant(v)).collect();
    for (i, &slot) in model.pk_indices.iter().enumerate() {
        params_dual[slot] = Dual2::var(pk.values[slot], i);
    }
    // Individual-parameter η/θ derivatives (general parameterization).
    let pd = param_derivatives(model, subject, theta, eta);

    // Lagtime (a nonzero dose-time shift) is not yet supported over the dual loop.
    if pk.values[PK_IDX_LAGTIME].abs() > 1e-12 {
        return None;
    }
    // Bioavailability F scales the dosed amount/rate (NONMEM F·AMT / F·RATE). F
    // lives at PK_IDX_F (pk_param_fn defaults it to 1 when undeclared); when F is
    // an estimated individual parameter, its derivative flows via `params_dual`.
    let f_bio = if pk.values[PK_IDX_F] > 0.0 {
        params_dual[PK_IDX_F]
    } else {
        Dual2::constant(1.0)
    };

    // Dose-time anchors for TAFD/TAD (constants w.r.t. the parameters).
    let first_dose_time = subject
        .doses
        .iter()
        .map(|d| d.time)
        .fold(f64::INFINITY, f64::min);

    // Integrate the Dual2 state through bolus + infusion events, capturing the
    // full state at each observation time.
    let states = integrate_dual::<N>(
        program,
        ode.n_states,
        subject,
        &params_dual,
        f_bio,
        first_dose_time,
        &opts,
    )?;

    // Apply the readout per observation: the state directly (`ObsCmt`) or the
    // Form C output program (`y = central/V1`) evaluated over the dual state.
    let mut ro_vars: Vec<Dual2<N>> = Vec::new();
    let mut ro_stack: Vec<Dual2<N>> = Vec::new();
    let preds: Vec<Dual2<N>> = states
        .iter()
        .map(|st| match &ode.readout {
            OdeReadout::ObsCmt(idx) => st.get(*idx).copied().unwrap_or(Dual2::constant(0.0)),
            OdeReadout::Single(_) => ode
                .readout_program
                .as_ref()
                .map(|p| p.eval_output_dual::<N>(st, &params_dual, &mut ro_vars, &mut ro_stack))
                .unwrap_or(Dual2::constant(0.0)),
            OdeReadout::PerCmt(_) => Dual2::constant(0.0),
        })
        .collect();

    // Chain ∂f/∂p, ∂²f/∂p² (exact, from the dual) with ∂p/∂η, ∂p/∂θ (general,
    // from `param_derivatives`) → ∂f/∂η, ∂²f/∂η², ∂f/∂θ, ∂²f/∂η∂θ:
    //   ∂f/∂η_k        = Σ_i  g_i · pᵢ,η_k
    //   ∂²f/∂η_k∂η_l   = Σ_ij h_ij · pᵢ,η_k · pⱼ,η_l  +  Σ_i g_i · pᵢ,η_kη_l
    // and likewise with θ in one slot.
    let n_indiv = model.pk_indices.len();
    let mut out = Vec::with_capacity(subject.obs_times.len());
    for fd in &preds {
        let g = &fd.grad; // ∂f/∂p_i
        let h = &fd.hess; // ∂²f/∂p_i∂p_j

        let mut df_deta = vec![0.0; n_eta];
        let mut d2f_deta2 = vec![0.0; n_eta * n_eta];
        let mut df_dtheta = vec![0.0; n_theta];
        let mut d2f_deta_dtheta = vec![0.0; n_eta * n_theta];

        for i in 0..n_indiv {
            for k in 0..n_eta {
                df_deta[k] += g[i] * pd.dp_deta[i][k];
            }
            for m in 0..n_theta {
                df_dtheta[m] += g[i] * pd.dp_dtheta[i][m];
            }
        }
        for k in 0..n_eta {
            for l in 0..n_eta {
                let mut acc = 0.0;
                for i in 0..n_indiv {
                    for j in 0..n_indiv {
                        acc += h[i][j] * pd.dp_deta[i][k] * pd.dp_deta[j][l];
                    }
                    acc += g[i] * pd.d2p_deta2[i][k][l];
                }
                d2f_deta2[k * n_eta + l] = acc;
            }
        }
        for k in 0..n_eta {
            for m in 0..n_theta {
                let mut acc = 0.0;
                for i in 0..n_indiv {
                    for j in 0..n_indiv {
                        acc += h[i][j] * pd.dp_deta[i][k] * pd.dp_dtheta[j][m];
                    }
                    acc += g[i] * pd.d2p_detadtheta[i][k][m];
                }
                d2f_deta_dtheta[k * n_theta + m] = acc;
            }
        }

        out.push(ObsSens {
            f: fd.value,
            df_deta,
            d2f_deta2,
            df_dtheta,
            d2f_deta_dtheta,
        });
    }

    Some(SubjectSens { obs: out })
}

/// Integrate the `Dual2<N>` state through the subject's bolus + infusion events,
/// capturing the full state vector at every observation time. Returns one state
/// vector per observation (parallel to `subject.obs_times`); the caller applies
/// the readout. `f_bio` is the bioavailability (scales bolus amount and infusion
/// rate, carrying its derivative).
#[allow(clippy::too_many_arguments)]
fn integrate_dual<const N: usize>(
    program: &crate::parser::model_parser::OdeRhsProgram,
    n_states: usize,
    subject: &Subject,
    params_dual: &[Dual2<N>],
    f_bio: Dual2<N>,
    first_dose_time: f64,
    opts: &crate::ode::solver::OdeSolverOptions,
) -> Option<Vec<Vec<Dual2<N>>>> {
    let n_obs = subject.obs_times.len();
    let mut states: Vec<Vec<Dual2<N>>> = vec![vec![Dual2::<N>::constant(0.0); n_states]; n_obs];
    let mut recorded = vec![false; n_obs];
    let mut u = vec![Dual2::<N>::constant(0.0); n_states];

    // obs time → all indices sharing it.
    use std::collections::HashMap;
    let mut obs_map: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &t) in subject.obs_times.iter().enumerate() {
        obs_map.entry(t.to_bits()).or_default().push(i);
    }

    // Break the timeline at every dose time and — for infusions — the
    // infusion-end time, so each segment is fully inside or outside every
    // infusion window (the rate forcing is then constant over a segment).
    let t_last = subject.obs_times.iter().cloned().fold(0.0_f64, f64::max);
    let mut break_times: Vec<f64> = vec![0.0];
    for dose in &subject.doses {
        break_times.push(dose.time);
        if dose.is_infusion() {
            break_times.push(dose.time + dose.duration);
        }
    }
    // EVID 3/4 reset times also break the timeline so the state can be zeroed
    // there (the datareader places obs/dose/reset on one absolute timeline).
    for &rt in &subject.reset_times {
        break_times.push(rt);
    }
    break_times.push(t_last);
    break_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    break_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    // Reusable scratch for the RHS evaluation across all stages.
    let vars_cell: RefCell<Vec<Dual2<N>>> = RefCell::new(Vec::new());
    let stack_cell: RefCell<Vec<Dual2<N>>> = RefCell::new(Vec::new());

    for w in 0..(break_times.len() - 1) {
        let t_start = break_times[w];
        let t_end = break_times[w + 1];

        // EVID 3/4 reset: zero the state at this time, *before* the same-time
        // dose (EVID=4 = reset + dose). `init(...)` is gated out, so reset →
        // zero. Infusions from a prior occasion live at earlier absolute times,
        // so they are naturally no longer active after the reset.
        if subject
            .reset_times
            .iter()
            .any(|&rt| (rt - t_start).abs() < 1e-12)
        {
            for s in u.iter_mut() {
                *s = Dual2::constant(0.0);
            }
        }

        // Apply bolus doses (non-infusions) at t_start: u[cmt] += F·amt.
        for dose in &subject.doses {
            if !dose.is_infusion() && (dose.time - t_start).abs() < 1e-12 {
                let cmt_idx = dose.cmt.saturating_sub(1);
                if cmt_idx < n_states {
                    u[cmt_idx] = u[cmt_idx] + f_bio * dose.amt;
                }
            }
        }

        // Record any observation exactly at t_start (after the dose).
        if let Some(idxs) = obs_map.get(&t_start.to_bits()) {
            for &j in idxs {
                if !recorded[j] {
                    states[j].copy_from_slice(&u);
                    recorded[j] = true;
                }
            }
        }

        if (t_end - t_start).abs() < 1e-15 {
            continue;
        }

        // Observation times in (t_start, t_end]; always include t_end so `u`
        // advances for the next segment.
        let mut saveat: Vec<f64> = subject
            .obs_times
            .iter()
            .filter(|&&t| t > t_start + 1e-12 && t <= t_end + 1e-12)
            .cloned()
            .collect();
        if saveat.last().map_or(true, |&l| (l - t_end).abs() > 1e-12) {
            saveat.push(t_end);
        }
        saveat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        saveat.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

        // Infusions spanning this whole segment add a constant rate forcing
        // F·rate to their compartment (the break times guarantee a segment is
        // fully inside or outside each infusion window).
        let active_inf: Vec<(usize, f64)> = subject
            .doses
            .iter()
            .filter(|d| d.is_infusion())
            .filter(|d| d.time <= t_start + 1e-9 && d.time + d.duration >= t_end - 1e-9)
            .map(|d| (d.cmt.saturating_sub(1), d.rate))
            .collect();

        // Last effective dose at or before the segment start, for TAD.
        let last_dose_eff = subject
            .doses
            .iter()
            .map(|d| d.time)
            .filter(|&dt| dt <= t_start + 1e-12)
            .fold(f64::NEG_INFINITY, f64::max);

        let rhs = |us: &[Dual2<N>], ps: &[Dual2<N>], t: f64, du: &mut [Dual2<N>]| {
            let tafd = if first_dose_time.is_finite() {
                t - first_dose_time
            } else {
                f64::NAN
            };
            let tad = if last_dose_eff.is_finite() {
                t - last_dose_eff
            } else {
                f64::NAN
            };
            program.eval_rhs_dual::<N>(
                us,
                ps,
                t,
                tafd,
                tad,
                du,
                &mut vars_cell.borrow_mut(),
                &mut stack_cell.borrow_mut(),
            );
            for &(cmt, rate) in &active_inf {
                if cmt < du.len() {
                    du[cmt] = du[cmt] + f_bio * rate;
                }
            }
        };

        let sol = solve_ode_g(&rhs, &u, (t_start, t_end), params_dual, &saveat, opts);

        // Capture state at the requested observation times; advance u to t_end.
        for pt in &sol {
            if let Some(idxs) = obs_map.get(&pt.t.to_bits()) {
                for &j in idxs {
                    if !recorded[j] {
                        states[j].copy_from_slice(&pt.u);
                        recorded[j] = true;
                    }
                }
            }
            if (pt.t - t_end).abs() < 1e-12 {
                u.copy_from_slice(&pt.u);
            }
        }
    }

    Some(states)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::model_parser::parse_model_string;
    use crate::pk::compute_predictions_with_tv;
    use crate::types::DoseEvent;
    use std::collections::HashMap;

    // 2-cpt IV bolus as a user ODE, with a Form C concentration readout
    // (`y = central / V1`). CL/V1 carry IIV; Q/V2 are fixed individual params.
    const TWOCPT_ODE: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)
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
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    fn bolus_subject(times: &[f64]) -> Subject {
        let n = times.len();
        Subject {
            id: "1".to_string(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: times.to_vec(),
            obs_raw_times: Vec::new(),
            observations: vec![1.0; n],
            obs_cmts: vec![1; n],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n],
            occasions: vec![1; n],
            dose_occasions: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }
    }

    // 2-cpt ODE with an allometric weight covariate on CL and V1 — exercises the
    // covariate path: typical values (and their θ-Jacobian) must fold WT.
    const TWOCPT_ODE_COV: &str = r#"
[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(12.0, 1.0, 500.0)
  theta TVQ(2.0,   0.01, 100.0)
  theta TVV2(25.0, 1.0, 500.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * (WT / 70)^0.75 * exp(ETA_CL)
  V1 = TVV1 * (WT / 70) * exp(ETA_V1)
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
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    fn bolus_subject_wt(times: &[f64], wt: f64) -> Subject {
        let mut s = bolus_subject(times);
        s.covariates.insert("WT".to_string(), wt);
        s
    }

    /// The ODE provider's `f`, `∂f/∂η`, `∂f/∂θ` must match the production
    /// predictor (`compute_predictions_with_tv`) and its finite differences.
    #[test]
    fn ode_provider_2cpt_matches_production() {
        let model = parse_model_string(TWOCPT_ODE).expect("parse");
        assert!(
            ode_analytical_supported(&model),
            "2-cpt ODE with Form C readout should be supported"
        );
        let subject = bolus_subject(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let theta = vec![4.0, 12.0, 2.0, 25.0];
        let eta = vec![0.12, -0.08];

        let sens = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");
        let pred = |e: &[f64], th: &[f64], j: usize| -> f64 {
            compute_predictions_with_tv(&model, &subject, th, e)[j]
        };
        let n_eta = model.n_eta;
        let n_theta = model.n_theta;
        let he = 1e-6;

        for (j, obs) in sens.obs.iter().enumerate() {
            // Value matches the production prediction.
            approx::assert_relative_eq!(
                obs.f,
                pred(&eta, &theta, j),
                max_relative = 1e-6,
                epsilon = 1e-9
            );
            // ∂f/∂η vs central FD.
            for k in 0..n_eta {
                let mut ep = eta.clone();
                ep[k] += he;
                let mut em = eta.clone();
                em[k] -= he;
                let g = (pred(&ep, &theta, j) - pred(&em, &theta, j)) / (2.0 * he);
                approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 1e-3, epsilon = 1e-6);
            }
            // ∂f/∂θ vs central FD.
            for m in 0..n_theta {
                let s = he * (1.0 + theta[m].abs());
                let mut tp = theta.clone();
                tp[m] += s;
                let mut tm = theta.clone();
                tm[m] -= s;
                let g = (pred(&eta, &tp, j) - pred(&eta, &tm, j)) / (2.0 * s);
                approx::assert_relative_eq!(
                    obs.df_dtheta[m],
                    g,
                    max_relative = 1e-3,
                    epsilon = 1e-6
                );
            }
        }
    }

    /// Shared check: provider `f`/`∂f/∂η`/`∂f/∂θ` vs production predictor + FD.
    fn check_vs_production(model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64]) {
        let sens = ode_subject_sensitivities(model, subject, theta, eta).expect("supported");
        let pred = |e: &[f64], th: &[f64], j: usize| -> f64 {
            compute_predictions_with_tv(model, subject, th, e)[j]
        };
        let he = 1e-6;
        for (j, obs) in sens.obs.iter().enumerate() {
            approx::assert_relative_eq!(
                obs.f,
                pred(eta, theta, j),
                max_relative = 1e-6,
                epsilon = 1e-9
            );
            for k in 0..model.n_eta {
                let mut ep = eta.to_vec();
                ep[k] += he;
                let mut em = eta.to_vec();
                em[k] -= he;
                let g = (pred(&ep, theta, j) - pred(&em, theta, j)) / (2.0 * he);
                approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 2e-3, epsilon = 1e-6);
            }
            for m in 0..model.n_theta {
                let s = he * (1.0 + theta[m].abs());
                let mut tp = theta.to_vec();
                tp[m] += s;
                let mut tm = theta.to_vec();
                tm[m] -= s;
                let g = (pred(eta, &tp, j) - pred(eta, &tm, j)) / (2.0 * s);
                approx::assert_relative_eq!(
                    obs.df_dtheta[m],
                    g,
                    max_relative = 2e-3,
                    epsilon = 1e-6
                );
            }
        }
    }

    // 1-cpt oral ODE with estimated, logit-normal bioavailability F — the dose
    // loads `F·AMT` into the depot, so F's derivative must flow through the
    // injection. Mirrors examples/bioavailability_ode.ferx.
    const BIOAV_ODE: &str = r#"
[parameters]
  theta TVCL(5.0,   0.1,  50.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVKA(1.5,  0.05,  20.0)
  theta THETA_F(0.70, 0.001, 0.999)
  omega ETA_CL ~ 0.09
  omega ETA_F  ~ 0.10
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  F  = inv_logit(logit(THETA_F) + ETA_F)
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot / V - CL/V * central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
"#;

    /// Estimated bioavailability F: the provider must propagate F's derivative
    /// through the `F·AMT` depot loading (and the logit/inv_logit individual-F
    /// map), matching the production predictor and its FD.
    #[test]
    fn ode_provider_oral_bioavailability_matches_production() {
        let model = parse_model_string(BIOAV_ODE).expect("parse");
        assert!(
            ode_analytical_supported(&model),
            "estimated F should be in scope"
        );
        let mut subject = bolus_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        check_vs_production(&model, &subject, &[5.0, 50.0, 1.5, 0.70], &[0.15, 0.2]);
    }

    /// Infusion doses (RATE>0): the dual loop must add the rate forcing over the
    /// infusion window and match the production predictor through during- and
    /// post-infusion observations.
    #[test]
    fn ode_provider_2cpt_infusion_matches_production() {
        let model = parse_model_string(TWOCPT_ODE).expect("parse");
        // amt=1000, rate=200 → 5 h infusion into central; obs during and after.
        let mut subject = bolus_subject(&[1.0, 3.0, 5.0, 6.0, 9.0, 24.0]);
        subject.doses = vec![DoseEvent::new(0.0, 1000.0, 1, 200.0, false, 0.0)];
        check_vs_production(&model, &subject, &[4.0, 12.0, 2.0, 25.0], &[0.12, -0.08]);
    }

    /// EVID 3/4 reset: a two-occasion subject (reset + re-dose at t=10) must zero
    /// the dual state at the reset and match the production event-driven path
    /// across both occasions.
    #[test]
    fn ode_provider_2cpt_reset_matches_production() {
        let model = parse_model_string(TWOCPT_ODE).expect("parse");
        let mut subject = bolus_subject(&[1.0, 3.0, 6.0, 11.0, 13.0, 16.0]);
        subject.doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(10.0, 1000.0, 1, 0.0, false, 0.0),
        ];
        subject.reset_times = vec![10.0];
        check_vs_production(&model, &subject, &[4.0, 12.0, 2.0, 25.0], &[0.12, -0.08]);
    }

    /// Covariate models: the provider must fold the subject's covariate-adjusted
    /// typical values (here WT on CL/V1) into both `f` and `∂f/∂θ`. Validated
    /// against the production predictor, which folds WT the same way.
    #[test]
    fn ode_provider_2cpt_covariate_matches_production() {
        let model = parse_model_string(TWOCPT_ODE_COV).expect("parse");
        assert!(ode_analytical_supported(&model));
        // A subject whose weight differs from the 70 kg reference, so the
        // covariate genuinely shifts CL/V1 and their θ-Jacobian.
        let subject = bolus_subject_wt(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0], 90.0);
        let theta = vec![4.0, 12.0, 2.0, 25.0];
        let eta = vec![0.12, -0.08];

        let sens = ode_subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");
        let pred = |e: &[f64], th: &[f64], j: usize| -> f64 {
            compute_predictions_with_tv(&model, &subject, th, e)[j]
        };
        let he = 1e-6;
        for (j, obs) in sens.obs.iter().enumerate() {
            approx::assert_relative_eq!(
                obs.f,
                pred(&eta, &theta, j),
                max_relative = 1e-6,
                epsilon = 1e-9
            );
            for k in 0..model.n_eta {
                let mut ep = eta.clone();
                ep[k] += he;
                let mut em = eta.clone();
                em[k] -= he;
                let g = (pred(&ep, &theta, j) - pred(&em, &theta, j)) / (2.0 * he);
                approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 1e-3, epsilon = 1e-6);
            }
            for m in 0..model.n_theta {
                let s = he * (1.0 + theta[m].abs());
                let mut tp = theta.clone();
                tp[m] += s;
                let mut tm = theta.clone();
                tm[m] -= s;
                let g = (pred(&eta, &tp, j) - pred(&eta, &tm, j)) / (2.0 * s);
                approx::assert_relative_eq!(
                    obs.df_dtheta[m],
                    g,
                    max_relative = 1e-3,
                    epsilon = 1e-6
                );
            }
        }
    }
}
