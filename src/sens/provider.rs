//! Per-observation analytic sensitivities for analytical 1-/2-/3-cpt PK models.
//!
//! Given a [`CompiledModel`], a [`Subject`], population `theta` and a candidate
//! `eta`, [`subject_sensitivities`] returns, for every observation, the exact
//!
//!   * value `f`,
//!   * `∂f/∂η`, `∂²f/∂η²`,
//!   * `∂f/∂θ`, `∂²f/∂η∂θ`
//!
//! built by seeding the PK parameters as [`Dual2`](super::dual2::Dual2) variables
//! through the closed-form PK solution (exact `∂f/∂pk`, `∂²f/∂pk²`) and then
//! applying the **closed-form η/θ chain rule**: the model exposes the relation
//! `pk_i = tv_i·exp(Σ_k sel[i,k]·η_k)`, so
//!
//!   * `∂pk_i/∂η_k        = pk_i · sel[i,k]`,
//!   * `∂²pk_i/∂η_k∂η_l   = pk_i · sel[i,k] · sel[i,l]`,
//!   * `∂pk_i/∂θ_m        = pk_i · ρ[i,m]`,        ρ[i,m] = (∂tv_i/∂θ_m)/tv_i,
//!   * `∂²pk_i/∂η_k∂θ_m   = pk_i · sel[i,k] · ρ[i,m]`,
//!
//! with `∂tv_i/∂θ_m` from a finite difference of the runtime `tv_fn` closure
//! (irreducible — `tv_fn` is user-assembled at parse time). `sel[i,k]`,
//! `pk_indices`, and `tv_fn` all already live on `CompiledModel`.
//!
//! Scope (issue #367): analytical 1-/2-/3-cpt (IV bolus/infusion + oral, incl.
//! steady state — SS infusion only for non-overlapping `T_inf ≤ II`), single
//! endpoint, log-normal η, no output transform (no scaling, no LTBS), no IOV,
//! no time-varying covariates, no resets, no dose lagtime.
//! [`analytical_supported`] (+ per-subject gates in [`subject_sensitivities`])
//! gate exactly that; everything else returns `None` so the caller falls back
//! to the gradient-free path.
#![allow(clippy::needless_range_loop)]

use super::dual2::Dual2;
use super::one_cpt::one_cpt_conc_g;
use super::three_cpt::three_cpt_conc_g;
use super::two_cpt::two_cpt_conc_g;
use crate::types::{
    CompiledModel, DoseEvent, PkModel, ScalingSpec, Subject, PK_IDX_CL, PK_IDX_F, PK_IDX_KA,
    PK_IDX_Q, PK_IDX_Q3, PK_IDX_V, PK_IDX_V2, PK_IDX_V3,
};

/// Exact sensitivities of one observation w.r.t. η and θ. Hessian-shaped fields
/// are stored row-major (`[k*n + l]`).
#[derive(Debug, Clone)]
pub struct ObsSens {
    pub f: f64,
    /// `∂f/∂η_k`, length `n_eta`.
    pub df_deta: Vec<f64>,
    /// `∂²f/∂η_k∂η_l`, row-major `n_eta × n_eta`.
    pub d2f_deta2: Vec<f64>,
    /// `∂f/∂θ_m`, length `n_theta`.
    pub df_dtheta: Vec<f64>,
    /// `∂²f/∂η_k∂θ_m`, row-major `n_eta × n_theta`.
    pub d2f_deta_dtheta: Vec<f64>,
}

/// All observations' sensitivities for one subject, parallel to
/// `subject.obs_times`.
#[derive(Debug, Clone)]
pub struct SubjectSens {
    pub obs: Vec<ObsSens>,
}

/// Map a fixed PK slot to its `Dual2<8>` seed dimension. The analytical 1-/2-/
/// 3-cpt solutions read `CL, V1, Q2, V2, KA, F, Q3, V3` (slots 0,1,2,3,4,5,6,7)
/// — an identity map; any other slot (LAGTIME) is out of scope.
#[inline]
fn slot_to_dim(slot: usize) -> Option<usize> {
    match slot {
        PK_IDX_CL => Some(0),
        PK_IDX_V => Some(1),
        PK_IDX_Q => Some(2),
        PK_IDX_V2 => Some(3),
        PK_IDX_KA => Some(4),
        PK_IDX_F => Some(5),
        PK_IDX_Q3 => Some(6),
        PK_IDX_V3 => Some(7),
        _ => None,
    }
}

/// Number of seeded PK dimensions (`CL, V1, Q2, V2, KA, F, Q3, V3`).
const N_PK: usize = 8;

/// Master switch for the user-ODE sensitivity path (issue #367, Option A). The
/// provider in [`crate::sens::ode_provider`] is complete and tested, but the
/// analytic-sensitivity rollout ships scoped to the analytical PK models first;
/// ODE-model sensitivities are a deferred follow-up. While `false`, ODE models
/// take the prior path (gradient-free outer, AD/FD inner) and the infrastructure
/// stays compiled and exercised by its own tests. Flip to `true` to re-arm.
const ODE_SENS_ENABLED: bool = false;

/// Escape hatch: `FERX_DISABLE_EXPLICIT_SENS=1` forces every subject onto the
/// generic `Dual2<N>` provider path instead of the hand-written explicit-
/// derivative kernels (`*_explicit`). The explicit kernels are the default — they
/// compute the same `(f, ∂f/∂pk, ∂²f/∂pk²)` to ~1e-8 (validated per-kernel
/// against the dual oracle) at a fraction of the cost. This toggle exists to A/B
/// the two on a real fit and as a fallback if a kernel is ever suspected wrong.
fn explicit_sens_disabled() -> bool {
    std::env::var("FERX_DISABLE_EXPLICIT_SENS")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Which explicit-kernel model class serves a subject. A class covers a subset of
/// dose kinds ([`ExKind::covers`]); a subject whose doses are *all* covered takes
/// the explicit path, otherwise the whole subject falls back to `Dual2<N>` (the
/// per-observation chain is identical either way — only `(f, ∂f/∂pk, ∂²f/∂pk²)`
/// is sourced differently).
#[derive(Clone, Copy)]
enum ExKind {
    /// 1-cpt IV: bolus + infusion.
    OneCptIv,
    /// 1-cpt oral: first-order absorption + infusion-into-central.
    OneCptOral,
    /// 2-cpt IV: bolus + infusion.
    TwoCptIv,
    /// 2-cpt oral: first-order absorption + infusion-into-central.
    TwoCptOral,
    /// 3-cpt IV: bolus + infusion.
    ThreeCptIv,
    /// 3-cpt oral: first-order absorption + infusion-into-central.
    ThreeCptOral,
}

impl ExKind {
    /// True when a hand-written kernel covers this dose. Steady state (every
    /// class) is not yet derived, so a subject containing one routes entirely to
    /// the exact `Dual2<N>` path; every other (bolus / infusion / oral) dose is
    /// covered.
    fn covers(self, dose: &DoseEvent) -> bool {
        let ss = dose.ss && dose.ii > 0.0;
        !ss
    }
}

/// True when [`subject_sensitivities`] can serve this model: analytical 1-cpt or
/// 2-cpt, `tv_fn` present, no ODE. Per-subject gates (TV covariates) are checked
/// separately in [`subject_sensitivities`].
pub fn analytical_supported(model: &CompiledModel) -> bool {
    matches!(
        model.pk_model,
        PkModel::OneCptIv
            | PkModel::OneCptOral
            | PkModel::TwoCptIv
            | PkModel::TwoCptOral
            | PkModel::ThreeCptIv
            | PkModel::ThreeCptOral
    ) && model.ode_spec.is_none()
        && model.tv_fn.is_some()
        && model.n_kappa == 0
        // The provider returns the raw concentration `f` and its `∂f`. The
        // production predictor only agrees with that when there is no output
        // transform: no observation scaling and no log-transformed DV (LTBS).
        && matches!(model.scaling, ScalingSpec::None)
        && !model.log_transform
        // Every individual-parameter slot must be one we differentiate. A
        // LAGTIME (slot 8) routes to fall back.
        && model.pk_indices.iter().all(|&s| slot_to_dim(s).is_some())
}

/// True when the exact `sens` outer gradient applies to this model: either the
/// analytical PK provider ([`analytical_supported`]) or the ODE sensitivity
/// provider ([`ode_analytical_supported`](crate::sens::ode_provider::ode_analytical_supported)).
/// Used to gate the gradient dispatch and the Eq. 48 EBE predictor.
pub fn sens_supported(model: &CompiledModel) -> bool {
    analytical_supported(model)
        || (ODE_SENS_ENABLED && crate::sens::ode_provider::ode_analytical_supported(model))
}

/// The per-observation `∂f/∂η` Jacobian (`n_obs × n_eta`, row-major) as a flat
/// vector, or `None` when unsupported. Convenience for the inner loop, whose
/// `h_matrix` is exactly this Jacobian at the converged η̂.
pub fn subject_eta_jacobian(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<Vec<f64>> {
    let sens = subject_sensitivities(model, subject, theta, eta)?;
    let n_eta = model.n_eta;
    let mut jac = Vec::with_capacity(sens.obs.len() * n_eta);
    for obs in &sens.obs {
        jac.extend_from_slice(&obs.df_deta);
    }
    debug_assert_eq!(jac.len(), subject.obs_times.len() * n_eta);
    Some(jac)
}

/// `∂tv_i/∂θ_m` by bound-agnostic central finite difference of `tv_fn`. Returns
/// a row-major `n_tv × n_theta` matrix. `tv_fn` folds covariates and evaluates
/// at η = 0, so this is purely the θ → typical-value Jacobian.
pub(crate) fn tv_theta_jacobian(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
) -> Vec<Vec<f64>> {
    let tv_fn = model
        .tv_fn
        .as_ref()
        .expect("analytical_supported guarantees tv_fn is present");
    let n_theta = model.n_theta;
    let base = tv_fn(theta, &subject.covariates);
    let n_tv = base.len();
    let mut jac = vec![vec![0.0; n_theta]; n_tv];
    let mut th = theta.to_vec();
    for m in 0..n_theta {
        let h = 1e-6 * (1.0 + theta[m].abs());
        let orig = th[m];
        th[m] = orig + h;
        let up = tv_fn(&th, &subject.covariates);
        th[m] = orig - h;
        let dn = tv_fn(&th, &subject.covariates);
        th[m] = orig;
        for i in 0..n_tv {
            jac[i][m] = (up[i] - dn[i]) / (2.0 * h);
        }
    }
    jac
}

/// Fallback `∂p/∂(θ,η)` for the closed-form log-normal parameterization
/// `pk_i = tv_i·exp(Σ_k sel[i,k]·η_k)`, used when the exact `Dual2`-over-program
/// path is unavailable (NN-weight θ or IOV kappa make the program's axis counts
/// disagree with the model's θ/η). The η chain is closed form; the θ chain uses
/// the FD `tv_theta_jacobian` (ρ). Produces the same `ParamDerivs` shape the
/// program path returns so the downstream chain is identical:
///   `∂p_i/∂η_k = pk_i·sel_ik`, `∂p_i/∂θ_m = pk_i·ρ_im`,
///   `∂²p_i/∂η_k∂η_l = pk_i·sel_ik·sel_il`, `∂²p_i/∂η_k∂θ_m = pk_i·sel_ik·ρ_im`.
fn lognormal_param_derivatives(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    pk: &crate::types::PkParams,
) -> crate::sens::ode_provider::ParamDerivs {
    let n_eta = model.n_eta;
    let n_theta = model.n_theta;
    let tv = (model.tv_fn.as_ref().unwrap())(theta, &subject.covariates);
    let tv_jac = tv_theta_jacobian(model, subject, theta);
    let ni = model.pk_indices.len();
    let mut dp_deta = vec![vec![0.0; n_eta]; ni];
    let mut dp_dtheta = vec![vec![0.0; n_theta]; ni];
    let mut d2p_deta2 = vec![vec![vec![0.0; n_eta]; n_eta]; ni];
    let mut d2p_detadtheta = vec![vec![vec![0.0; n_theta]; n_eta]; ni];
    for (i, &slot) in model.pk_indices.iter().enumerate() {
        let pk_val = pk.values[slot];
        let tv_i = tv[i];
        let sel: Vec<f64> = (0..n_eta).map(|k| model.sel_flat[i * n_eta + k]).collect();
        let rho: Vec<f64> = if tv_i.abs() > 0.0 {
            (0..n_theta).map(|m| tv_jac[i][m] / tv_i).collect()
        } else {
            vec![0.0; n_theta]
        };
        for m in 0..n_theta {
            dp_dtheta[i][m] = pk_val * rho[m];
        }
        for k in 0..n_eta {
            dp_deta[i][k] = pk_val * sel[k];
            for l in 0..n_eta {
                d2p_deta2[i][k][l] = pk_val * sel[k] * sel[l];
            }
            for m in 0..n_theta {
                d2p_detadtheta[i][k][m] = pk_val * sel[k] * rho[m];
            }
        }
    }
    crate::sens::ode_provider::ParamDerivs {
        dp_deta,
        dp_dtheta,
        d2p_deta2,
        d2p_detadtheta,
    }
}

/// Compute per-observation analytic sensitivities, or `None` if this
/// model/subject is outside the supported analytical 1-cpt scope (caller falls
/// back to the gradient-free path).
pub fn subject_sensitivities(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<SubjectSens> {
    // ODE models: the ODE sensitivity provider (issue #367, Option A) is complete
    // and tested in `ode_provider`, but the analytic-sensitivity rollout is scoped
    // to the *analytical* PK models for now — user-ODE sensitivities land in a
    // follow-up. Gated off here (not deleted): ODE models fall back to the prior
    // path (gradient-free outer, AD/FD inner). Flip `ODE_SENS_ENABLED` to re-arm.
    if model.ode_spec.is_some() {
        if ODE_SENS_ENABLED {
            return crate::sens::ode_provider::ode_subject_sensitivities(
                model, subject, theta, eta,
            );
        }
        return None;
    }
    if !analytical_supported(model) || subject.has_tv_covariates() {
        return None;
    }
    // EVID=3/4 resets are handled by restricting dose superposition to the
    // current reset segment (see `reset_floor` in the obs loop): for linear PK a
    // reset zeros the compartments, so the prediction at an observation is the
    // superposition of only the doses since the most recent reset — which carries
    // the exact `∂f/∂pk` through the same closed forms. Steady-state doses assume
    // an infinite periodic history that a mid-record reset contradicts, so a
    // subject mixing SS with resets falls back to FD.
    if subject.has_resets() && subject.doses.iter().any(|d| d.ss) {
        return None;
    }
    // Overlapping steady-state infusion (`T_inf > II`) has no single-interval
    // closed form — production returns 0 there too, but rather than match a
    // degenerate zero, fall back to FD. Non-overlapping SS infusion is handled
    // by the `*_infusion_ss_g` closed forms.
    if subject
        .doses
        .iter()
        .any(|d| d.ss && d.ii > 0.0 && d.is_infusion() && d.duration > d.ii)
    {
        return None;
    }

    let n_eta = model.n_eta;
    let n_theta = model.n_theta;
    let oral = matches!(
        model.pk_model,
        PkModel::OneCptOral | PkModel::TwoCptOral | PkModel::ThreeCptOral
    );
    let two_cpt = matches!(model.pk_model, PkModel::TwoCptIv | PkModel::TwoCptOral);
    let three_cpt = matches!(model.pk_model, PkModel::ThreeCptIv | PkModel::ThreeCptOral);

    // PK parameter values at (θ, η): pk_s = tv_s·exp(sel·η). pk_param_fn folds η.
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates);

    // `∂p/∂(θ,η)` (and second order) plus `slots`: the PK slot each `pd` row maps
    // to, in `pd`-row order. Analytical where possible: evaluate the compiled
    // `[individual_parameters]` program over `Dual2` seeded on (θ, η) — exact for
    // ANY parameterization (log-normal, logit-normal F, additive), no finite
    // differences (issue #367). Its rows follow the program's `pk_var_slots` order
    // (analytical: alphabetical by PK name), NOT `pk_indices` (declaration) order.
    // Falls back to the closed-form log-normal `sel` η chain plus the FD
    // `tv_theta_jacobian` θ chain (`ρ`) — whose rows ARE in `pk_indices` order —
    // when the program path is unavailable (NN-weight θ / IOV kappa axis mismatch,
    // or a literal-const PK slot the program omits).
    let (pd, slots): (crate::sens::ode_provider::ParamDerivs, Vec<usize>) = match model
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .filter(|prog| prog.pk_slots().len() == model.pk_indices.len())
        .and_then(|prog| {
            crate::sens::ode_provider::param_derivatives_from_prog(prog, model, subject, theta, eta)
                .map(|pd| (pd, prog.pk_slots()))
        }) {
        Some((pd, slots)) => (pd, slots),
        None => {
            // The closed-form fallback assumes `∂p/∂η = pk·sel` (log-normal). It is
            // only valid when every PK-param eta is LogNormal; for additive / logit
            // / custom etas the exact `∂p/∂η` must come from the program chain
            // above, and if that is unavailable (NN-weight θ or IOV kappa axis
            // mismatch) we fall back to FD rather than mis-apply the log-normal
            // chain (which would be off by a factor of `pk` vs `1`). PR #381 #4.
            if !model
                .eta_param_info
                .iter()
                .all(|e| e.param_type == crate::types::EtaParamType::LogNormal)
            {
                return None;
            }
            let pd = lognormal_param_derivatives(model, subject, theta, &pk);
            (pd, model.pk_indices.clone())
        }
    };

    // Right-size the dual width to the number of differentiated PK parameters
    // (issue #367): seed only those on a compact `0..N`, so a 2-cpt IV model runs
    // `Dual2<4>` (4× fewer Hessian entries per op than the former fixed `Dual2<8>`),
    // 3-cpt `Dual2<6>`, etc. `seed_dim[s]` is the compact dual axis for PK slot `s`
    // (`None` = constant, not differentiated); `pd` row `i` ↔ compact axis `i`.
    let mut seed_dim: [Option<usize>; N_PK] = [None; N_PK];
    for (i, &slot) in slots.iter().enumerate() {
        if slot < N_PK {
            seed_dim[slot] = Some(i);
        }
    }

    // Explicit-kernel fast path (the default): pick the model class, then take it
    // only if a hand-written kernel covers every dose (else the whole subject uses
    // `Dual2<N>`). `FERX_DISABLE_EXPLICIT_SENS=1` forces the dual path everywhere.
    let explicit_kind = if explicit_sens_disabled() {
        None
    } else {
        let kind = match model.pk_model {
            PkModel::OneCptIv => ExKind::OneCptIv,
            PkModel::OneCptOral => ExKind::OneCptOral,
            PkModel::TwoCptIv => ExKind::TwoCptIv,
            PkModel::TwoCptOral => ExKind::TwoCptOral,
            PkModel::ThreeCptIv => ExKind::ThreeCptIv,
            PkModel::ThreeCptOral => ExKind::ThreeCptOral,
        };
        Some(kind).filter(|kind| subject.doses.iter().all(|d| kind.covers(d)))
    };

    // Dispatch on the differentiated-parameter count so the dual width is
    // right-sized. `pk_indices.len()` ≤ `N_PK` (the fixed PK slot table).
    macro_rules! disp {
        ($($n:literal),+) => {
            match slots.len() {
                $($n => Some(SubjectSens {
                    obs: run_obs::<$n>(
                        &seed_dim, &pk, oral, two_cpt, three_cpt, explicit_kind, subject, &pd,
                        n_eta, n_theta,
                    ),
                }),)+
                _ => None,
            }
        };
    }
    disp!(1, 2, 3, 4, 5, 6, 7, 8)
}

/// Per-observation value/grad/Hessian chain at a right-sized dual width `N`
/// (= number of differentiated PK parameters). `seed_dim[s]` is the compact dual
/// axis for PK slot `s` (`None` = constant); `pd` rows are in compact-axis order,
/// so the chain reads `g[i]`/`h[i][j]` directly (identity dims).
#[allow(clippy::too_many_arguments)]
fn run_obs<const N: usize>(
    seed_dim: &[Option<usize>; N_PK],
    pk: &crate::types::PkParams,
    oral: bool,
    two_cpt: bool,
    three_cpt: bool,
    explicit_kind: Option<ExKind>,
    subject: &Subject,
    pd: &crate::sens::ode_provider::ParamDerivs,
    n_eta: usize,
    n_theta: usize,
) -> Vec<ObsSens> {
    let (cl, v1, q, v2, ka, f_bio, q3, v3) = (
        pk.cl(),
        pk.v(),
        pk.q(),
        pk.v2(),
        pk.ka(),
        pk.f_bio(),
        pk.q3(),
        pk.v3(),
    );
    let mut out = Vec::with_capacity(subject.obs_times.len());
    for &t_obs in subject.obs_times.iter() {
        // Reset segment: the most recent EVID=3/4 reset at or before this
        // observation (−∞ when the subject has no resets). Doses before it were
        // zeroed out of the compartments, so they're excluded from the
        // superposition below — mirroring the event-driven walker's
        // `event_reset_floor`. A reset+dose record (EVID=4) sits exactly at the
        // floor and is kept (`dose.time >= reset_floor`).
        let reset_floor = subject
            .reset_times
            .iter()
            .copied()
            .filter(|&r| r <= t_obs)
            .fold(f64::NEG_INFINITY, f64::max);

        // `(f, ∂f/∂pk, ∂²f/∂pk²)` in the compact `0..N` layout — from the explicit
        // kernels when applicable, else the generic `Dual2<N>` path.
        let (fval, g, h): (f64, [f64; N], [[f64; N]; N]) = if let Some(kind) = explicit_kind {
            let mut gv = [0.0; N];
            let mut hv = [[0.0; N]; N];
            let mut val = 0.0;
            for dose in &subject.doses {
                let elapsed = t_obs - dose.time;
                if elapsed < 0.0 || dose.time < reset_floor {
                    continue;
                }
                val += eval_dose_explicit(
                    kind, dose, elapsed, cl, v1, q, v2, ka, f_bio, q3, v3, seed_dim, &mut gv,
                    &mut hv,
                );
            }
            (val, gv, hv)
        } else {
            // Seed only the differentiated PK params as `Dual2<N>` on their compact
            // axes; everything else is a constant. `dv(slot, value)` does the lookup.
            let dv = |slot: usize, value: f64| -> Dual2<N> {
                match seed_dim[slot] {
                    Some(k) => Dual2::<N>::var(value, k),
                    None => Dual2::<N>::constant(value),
                }
            };
            let cl_d = dv(PK_IDX_CL, cl);
            let v1_d = dv(PK_IDX_V, v1);
            let q_d = dv(PK_IDX_Q, q);
            let v2_d = dv(PK_IDX_V2, v2);
            let ka_d = dv(PK_IDX_KA, ka);
            let f_d = dv(PK_IDX_F, f_bio);
            let q3_d = dv(PK_IDX_Q3, q3);
            let v3_d = dv(PK_IDX_V3, v3);

            // Superpose dose contributions: f = Σ conc(dose, t_obs − dose.time),
            // restricted to the current reset segment (`dose.time >= reset_floor`).
            let mut fd = Dual2::<N>::constant(0.0);
            for dose in &subject.doses {
                let elapsed = t_obs - dose.time;
                if elapsed < 0.0 || dose.time < reset_floor {
                    continue;
                }
                let c = if three_cpt {
                    three_cpt_conc_g(
                        dose, elapsed, cl_d, v1_d, q_d, v2_d, q3_d, v3_d, ka_d, f_d, oral,
                    )
                } else if two_cpt {
                    two_cpt_conc_g(dose, elapsed, cl_d, v1_d, q_d, v2_d, ka_d, f_d, oral)
                } else {
                    one_cpt_conc_g(dose, elapsed, cl_d, v1_d, ka_d, f_d, oral)
                };
                fd = fd + c;
            }
            (fd.value, fd.grad, fd.hess)
        };

        // Match production's `conc.max(0.0)` clamp (`pk/mod.rs`): when the closed
        // form goes slightly negative (cancellation at extreme params during a
        // line-search step), the objective uses `f = 0`, so the gradient of
        // `max(f, 0)` is also 0 there. Returning the raw negative `f` and its
        // derivatives would make the analytic gradient inconsistent with the
        // objective it differentiates (PR #381 review finding #5).
        let (fval, g, h) = if fval < 0.0 {
            (0.0, [0.0; N], [[0.0; N]; N])
        } else {
            (fval, g, h)
        };

        let mut df_deta = vec![0.0; n_eta];
        let mut d2f_deta2 = vec![0.0; n_eta * n_eta];
        let mut df_dtheta = vec![0.0; n_theta];
        let mut d2f_deta_dtheta = vec![0.0; n_eta * n_theta];

        // Chain ∂f/∂p, ∂²f/∂p² (exact, from the seeded PK Dual2 in compact layout)
        // with ∂p/∂(θ,η) (from `pd`, analytical or FD fallback):
        //   ∂f/∂η_k      = Σ_i g[i]·pᵢ,η_k
        //   ∂²f/∂η_k∂η_l = Σ_ij H[i][j]·pᵢ,η_k·pⱼ,η_l + Σ_i g[i]·pᵢ,η_kη_l
        // and likewise with θ in one slot. Compact axis `i` ↔ `pd` row `i`.
        for i in 0..N {
            let gi = g[i];
            for k in 0..n_eta {
                df_deta[k] += gi * pd.dp_deta[i][k];
            }
            for m in 0..n_theta {
                df_dtheta[m] += gi * pd.dp_dtheta[i][m];
            }
        }
        for k in 0..n_eta {
            for l in 0..n_eta {
                let mut acc = 0.0;
                for i in 0..N {
                    for j in 0..N {
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
                for i in 0..N {
                    for j in 0..N {
                        acc += h[i][j] * pd.dp_deta[i][k] * pd.dp_dtheta[j][m];
                    }
                    acc += g[i] * pd.d2p_detadtheta[i][k][m];
                }
                d2f_deta_dtheta[k * n_theta + m] = acc;
            }
        }

        out.push(ObsSens {
            f: fval,
            df_deta,
            d2f_deta2,
            df_dtheta,
            d2f_deta_dtheta,
        });
    }
    out
}

/// One dose's explicit-kernel contribution to `(f, ∂f/∂pk, ∂²f/∂pk²)` at the
/// compact axis layout: dispatches on `(kind, dose)` to the right hand-written
/// kernel (bolus / infusion / oral), scatters its `(grad, hess)` into `gv`/`hv`
/// via [`scatter_compact`], and returns the value. Only `kind`-covered doses
/// reach here ([`ExKind::covers`] gates the subject), so the match is total over
/// the covered set; the mirror of [`one_cpt_conc_g`]/[`two_cpt_conc_g`] dispatch.
#[allow(clippy::too_many_arguments)]
fn eval_dose_explicit<const N: usize>(
    kind: ExKind,
    dose: &DoseEvent,
    elapsed: f64,
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
    ka: f64,
    f_bio: f64,
    q3: f64,
    v3: f64,
    seed_dim: &[Option<usize>; N_PK],
    gv: &mut [f64; N],
    hv: &mut [[f64; N]; N],
) -> f64 {
    match kind {
        ExKind::OneCptIv | ExKind::OneCptOral => {
            if dose.is_infusion() {
                let (f, gs, hs) = super::one_cpt_explicit::infusion_explicit(
                    dose.rate,
                    dose.duration,
                    dose.amt,
                    elapsed,
                    cl,
                    v1,
                );
                scatter_compact(gv, hv, &gs, &hs, &[PK_IDX_CL, PK_IDX_V], seed_dim);
                f
            } else if matches!(kind, ExKind::OneCptOral) {
                let (f, gs, hs) =
                    super::one_cpt_explicit::oral_explicit(dose.amt, elapsed, cl, v1, ka, f_bio);
                scatter_compact(
                    gv,
                    hv,
                    &gs,
                    &hs,
                    &[PK_IDX_CL, PK_IDX_V, PK_IDX_KA, PK_IDX_F],
                    seed_dim,
                );
                f
            } else {
                let (f, gs, hs) =
                    super::one_cpt_explicit::iv_bolus_explicit(dose.amt, elapsed, cl, v1);
                scatter_compact(gv, hv, &gs, &hs, &[PK_IDX_CL, PK_IDX_V], seed_dim);
                f
            }
        }
        ExKind::TwoCptIv | ExKind::TwoCptOral => {
            if dose.is_infusion() {
                let (f, gs, hs) = super::two_cpt_explicit::infusion_explicit(
                    dose.rate,
                    dose.duration,
                    dose.amt,
                    elapsed,
                    cl,
                    v1,
                    q,
                    v2,
                );
                scatter_compact(
                    gv,
                    hv,
                    &gs,
                    &hs,
                    &[PK_IDX_CL, PK_IDX_V, PK_IDX_Q, PK_IDX_V2],
                    seed_dim,
                );
                f
            } else if matches!(kind, ExKind::TwoCptOral) {
                let (f, gs, hs) = super::two_cpt_explicit::oral_explicit(
                    dose.amt, elapsed, cl, v1, q, v2, ka, f_bio,
                );
                scatter_compact(
                    gv,
                    hv,
                    &gs,
                    &hs,
                    &[
                        PK_IDX_CL, PK_IDX_V, PK_IDX_Q, PK_IDX_V2, PK_IDX_KA, PK_IDX_F,
                    ],
                    seed_dim,
                );
                f
            } else {
                let (f, gs, hs) =
                    super::two_cpt_explicit::iv_bolus_explicit(dose.amt, elapsed, cl, v1, q, v2);
                scatter_compact(
                    gv,
                    hv,
                    &gs,
                    &hs,
                    &[PK_IDX_CL, PK_IDX_V, PK_IDX_Q, PK_IDX_V2],
                    seed_dim,
                );
                f
            }
        }
        ExKind::ThreeCptIv | ExKind::ThreeCptOral => {
            if dose.is_infusion() {
                let (f, gs, hs) = super::three_cpt_explicit::infusion_explicit(
                    dose.rate,
                    dose.duration,
                    dose.amt,
                    elapsed,
                    cl,
                    v1,
                    q,
                    v2,
                    q3,
                    v3,
                );
                scatter_compact(
                    gv,
                    hv,
                    &gs,
                    &hs,
                    &[
                        PK_IDX_CL, PK_IDX_V, PK_IDX_Q, PK_IDX_V2, PK_IDX_Q3, PK_IDX_V3,
                    ],
                    seed_dim,
                );
                f
            } else if matches!(kind, ExKind::ThreeCptOral) {
                let (f, gs, hs) = super::three_cpt_explicit::oral_explicit(
                    dose.amt, elapsed, cl, v1, q, v2, q3, v3, ka, f_bio,
                );
                scatter_compact(
                    gv,
                    hv,
                    &gs,
                    &hs,
                    &[
                        PK_IDX_CL, PK_IDX_V, PK_IDX_Q, PK_IDX_V2, PK_IDX_Q3, PK_IDX_V3, PK_IDX_KA,
                        PK_IDX_F,
                    ],
                    seed_dim,
                );
                f
            } else {
                let (f, gs, hs) = super::three_cpt_explicit::iv_bolus_explicit(
                    dose.amt, elapsed, cl, v1, q, v2, q3, v3,
                );
                scatter_compact(
                    gv,
                    hv,
                    &gs,
                    &hs,
                    &[
                        PK_IDX_CL, PK_IDX_V, PK_IDX_Q, PK_IDX_V2, PK_IDX_Q3, PK_IDX_V3,
                    ],
                    seed_dim,
                );
                f
            }
        }
    }
}

/// Scatter an explicit kernel's `M`-parameter `(grad, hess)` into the compact
/// `N`-axis layout via `seed_dim` (PK slot → compact axis). `map8[a]` is the PK
/// slot of explicit param `a`; a param whose slot isn't differentiated
/// (`seed_dim = None`, e.g. a literal-const PK value) is dropped — its derivative
/// is not chained.
fn scatter_compact<const M: usize, const N: usize>(
    g: &mut [f64; N],
    h: &mut [[f64; N]; N],
    gs: &[f64; M],
    hs: &[[f64; M]; M],
    map8: &[usize; M],
    seed_dim: &[Option<usize>; N_PK],
) {
    for a in 0..M {
        let ca = match seed_dim[map8[a]] {
            Some(c) => c,
            None => continue,
        };
        g[ca] += gs[a];
        for b in 0..M {
            if let Some(cb) = seed_dim[map8[b]] {
                h[ca][cb] += hs[a][b];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::model_parser::parse_model_string;
    use crate::pk::compute_predictions_with_tv;
    use crate::types::{DoseEvent, Subject};
    use std::collections::HashMap;

    const WARFARIN: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    fn oral_subject(times: &[f64]) -> Subject {
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

    const TWOCPT_IV: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(15.0, 1.0, 100.0)
  theta TVV2(100.0, 10.0, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    const TWOCPT_ORAL: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(15.0, 1.0, 100.0)
  theta TVV2(100.0, 10.0, 1000.0)
  theta TVKA(1.0, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk two_cpt_oral(cl=CL, v=V1, q=Q, v2=V2, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    const THREECPT_IV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.5, 50.0)
  theta TVV1(10.0, 1.0, 100.0)
  theta TVQ2(2.0, 0.1, 20.0)
  theta TVV2(20.0, 2.0, 200.0)
  theta TVQ3(1.5, 0.1, 20.0)
  theta TVV3(30.0, 3.0, 300.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q2 = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
[structural_model]
  pk three_cpt_iv(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    const THREECPT_ORAL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.5, 50.0)
  theta TVV1(10.0, 1.0, 100.0)
  theta TVQ2(2.0, 0.1, 20.0)
  theta TVV2(20.0, 2.0, 200.0)
  theta TVQ3(1.5, 0.1, 20.0)
  theta TVV3(30.0, 3.0, 300.0)
  theta TVKA(1.5, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q2 = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk three_cpt_oral(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    fn subject_with_dose(dose: DoseEvent, times: &[f64]) -> Subject {
        let n = times.len();
        Subject {
            id: "1".to_string(),
            doses: vec![dose],
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

    /// Check the provider's `f` matches the production predictor exactly, and
    /// `∂f/∂η`, `∂f/∂θ` match its finite differences.
    fn check_provider_vs_production(
        model: &CompiledModel,
        subject: &Subject,
        theta: &[f64],
        eta: &[f64],
    ) {
        let sens = subject_sensitivities(model, subject, theta, eta).expect("supported");
        let pred = |e: &[f64], th: &[f64], j: usize| -> f64 {
            compute_predictions_with_tv(model, subject, th, e)[j]
        };
        let n_eta = model.n_eta;
        let n_theta = theta.len();
        let he = 1e-6;
        for (j, obs) in sens.obs.iter().enumerate() {
            // f must equal the production prediction (the closed forms agree).
            approx::assert_relative_eq!(
                obs.f,
                pred(eta, theta, j),
                max_relative = 1e-9,
                epsilon = 1e-10
            );
            for k in 0..n_eta {
                let mut ep = eta.to_vec();
                ep[k] += he;
                let mut em = eta.to_vec();
                em[k] -= he;
                let g = (pred(&ep, theta, j) - pred(&em, theta, j)) / (2.0 * he);
                approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 3e-4, epsilon = 1e-7);
            }
            for m in 0..n_theta {
                let h = he * (1.0 + theta[m].abs());
                let mut tp = theta.to_vec();
                tp[m] += h;
                let mut tm = theta.to_vec();
                tm[m] -= h;
                let g = (pred(eta, &tp, j) - pred(eta, &tm, j)) / (2.0 * h);
                approx::assert_relative_eq!(
                    obs.df_dtheta[m],
                    g,
                    max_relative = 3e-4,
                    epsilon = 1e-7
                );
            }
        }
    }

    #[test]
    fn provider_2cpt_bolus_infusion_oral_match_production() {
        let times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0];
        // 2-cpt IV: bolus (rate=0) and infusion (rate>0, dur=2).
        let iv = parse_model_string(TWOCPT_IV).expect("parse");
        let theta_iv = vec![10.0, 50.0, 15.0, 100.0];
        let eta_iv = vec![0.12, -0.08];
        let bolus = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0), &times);
        let infusion = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 0.0), &times);
        check_provider_vs_production(&iv, &bolus, &theta_iv, &eta_iv);
        check_provider_vs_production(&iv, &infusion, &theta_iv, &eta_iv);

        // 2-cpt oral (first-order absorption).
        let oral_m = parse_model_string(TWOCPT_ORAL).expect("parse");
        let theta_or = vec![10.0, 50.0, 15.0, 100.0, 1.0];
        let eta_or = vec![0.12, -0.08, 0.2];
        let oral_s = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0), &times);
        check_provider_vs_production(&oral_m, &oral_s, &theta_or, &eta_or);
    }

    #[test]
    fn provider_2cpt_steady_state_matches_production() {
        // SS bolus (II=12) and SS oral (II=24) — exercises the *_ss_g branches.
        let times = [0.5, 2.0, 6.0, 11.5];
        let iv = parse_model_string(TWOCPT_IV).expect("parse");
        let ss_bolus = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 0.0, true, 12.0), &times);
        check_provider_vs_production(&iv, &ss_bolus, &[10.0, 50.0, 15.0, 100.0], &[0.1, -0.05]);

        let oral_m = parse_model_string(TWOCPT_ORAL).expect("parse");
        let ss_oral = subject_with_dose(
            DoseEvent::new(0.0, 1000.0, 1, 0.0, true, 24.0),
            &[2.0, 6.0, 12.0, 23.0],
        );
        check_provider_vs_production(
            &oral_m,
            &ss_oral,
            &[10.0, 50.0, 15.0, 100.0, 1.0],
            &[0.1, -0.05, 0.15],
        );
    }

    #[test]
    fn provider_2cpt_ss_infusion_matches_production() {
        // Non-overlapping SS infusion (rate=200, amt=1000 → dur=5; II=12 → dur<II).
        let iv = parse_model_string(TWOCPT_IV).expect("parse");
        let ss_inf = subject_with_dose(
            DoseEvent::new(0.0, 1000.0, 1, 200.0, true, 12.0),
            &[1.0, 4.0, 6.0, 8.0, 11.0],
        );
        check_provider_vs_production(&iv, &ss_inf, &[10.0, 50.0, 15.0, 100.0], &[0.1, -0.05]);
    }

    #[test]
    fn provider_1cpt_ss_infusion_matches_production() {
        let m = parse_model_string(
            "[parameters]\n  theta TVCL(10.0,1.0,100.0)\n  theta TVV(50.0,5.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.09\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n",
        )
        .expect("parse");
        let ss_inf = subject_with_dose(
            DoseEvent::new(0.0, 1000.0, 1, 200.0, true, 12.0),
            &[1.0, 4.0, 6.0, 8.0, 11.0],
        );
        check_provider_vs_production(&m, &ss_inf, &[10.0, 50.0], &[0.1, -0.05]);
    }

    #[test]
    fn provider_3cpt_bolus_infusion_oral_match_production() {
        let times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0];
        let iv = parse_model_string(THREECPT_IV).expect("parse");
        let theta_iv = vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0];
        let eta_iv = vec![0.12, -0.08];
        let bolus = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0), &times);
        let infusion = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 0.0), &times);
        check_provider_vs_production(&iv, &bolus, &theta_iv, &eta_iv);
        check_provider_vs_production(&iv, &infusion, &theta_iv, &eta_iv);

        let oral_m = parse_model_string(THREECPT_ORAL).expect("parse");
        let theta_or = vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 1.5];
        let eta_or = vec![0.12, -0.08, 0.2];
        let oral_s = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0), &times);
        check_provider_vs_production(&oral_m, &oral_s, &theta_or, &eta_or);
    }

    #[test]
    fn provider_3cpt_steady_state_matches_production() {
        // SS bolus (II=12), SS oral (II=24), SS infusion (dur<II) — exercises
        // every *_ss_g branch for 3-cpt.
        let iv = parse_model_string(THREECPT_IV).expect("parse");
        let theta_iv = vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0];
        let ss_bolus = subject_with_dose(
            DoseEvent::new(0.0, 1000.0, 1, 0.0, true, 12.0),
            &[0.5, 2.0, 6.0, 11.5],
        );
        check_provider_vs_production(&iv, &ss_bolus, &theta_iv, &[0.1, -0.05]);

        let ss_inf = subject_with_dose(
            DoseEvent::new(0.0, 1000.0, 1, 200.0, true, 12.0),
            &[1.0, 4.0, 6.0, 8.0, 11.0],
        );
        check_provider_vs_production(&iv, &ss_inf, &theta_iv, &[0.1, -0.05]);

        let oral_m = parse_model_string(THREECPT_ORAL).expect("parse");
        let theta_or = vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 1.5];
        let ss_oral = subject_with_dose(
            DoseEvent::new(0.0, 1000.0, 1, 0.0, true, 24.0),
            &[2.0, 6.0, 12.0, 23.0],
        );
        check_provider_vs_production(&oral_m, &ss_oral, &theta_or, &[0.1, -0.05, 0.15]);
    }

    #[test]
    fn provider_falls_back_on_overlapping_ss_infusion() {
        // Overlapping SS infusion (rate=200, amt=1000 → dur=5; II=2 → dur>II):
        // no single-interval closed form → fall back to FD.
        let iv = parse_model_string(TWOCPT_IV).expect("parse");
        let ss_inf = subject_with_dose(
            DoseEvent::new(0.0, 1000.0, 1, 200.0, true, 2.0),
            &[0.5, 1.0],
        );
        assert!(
            subject_sensitivities(&iv, &ss_inf, &[10.0, 50.0, 15.0, 100.0], &[0.1, -0.05])
                .is_none()
        );
    }

    /// Build a subject carrying explicit doses and EVID=3/4 reset times (no
    /// covariates, no IOV) for the reset-superposition tests.
    fn subject_with_doses_and_resets(
        doses: Vec<DoseEvent>,
        times: &[f64],
        reset_times: Vec<f64>,
    ) -> Subject {
        let n = times.len();
        Subject {
            id: "1".to_string(),
            doses,
            obs_times: times.to_vec(),
            obs_raw_times: Vec::new(),
            observations: vec![1.0; n],
            obs_cmts: vec![1; n],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times,
            cens: vec![0; n],
            occasions: vec![1; n],
            dose_occasions: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }
    }

    /// Two infusion occasions on a 3-cpt IV model separated by an EVID=4 reset:
    /// occasion-2 observations must rebuild from zero (no occasion-1 carryover).
    /// The provider's reset-segment superposition must reproduce the production
    /// event-driven predictor and its FD sensitivities.
    #[test]
    fn provider_3cpt_two_occasion_reset_matches_production() {
        let iv = parse_model_string(THREECPT_IV).expect("parse");
        let theta = vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0];
        let eta = vec![0.12, -0.08];
        // Occasion 1: infusion at t=0 (rate 200, amt 1000 → 5 h). Occasion 2:
        // same infusion at t=120, opened by an EVID=4 reset at t=120.
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 200.0, false, 0.0),
            DoseEvent::new(120.0, 1000.0, 1, 200.0, false, 0.0),
        ];
        let times = [2.0, 4.0, 8.0, 60.0, 122.0, 126.0, 150.0];
        let subject = subject_with_doses_and_resets(doses, &times, vec![120.0]);
        assert!(subject.has_resets(), "fixture must carry a reset");
        check_provider_vs_production(&iv, &subject, &theta, &eta);
    }

    /// A reset that lands mid-infusion (1-cpt IV): the ongoing infusion is turned
    /// off and the compartment zeroed, so post-reset observations see only doses
    /// from the new segment. Exercises the `dose.time < reset_floor` exclusion of
    /// an in-flight infusion.
    #[test]
    fn provider_1cpt_reset_midinfusion_matches_production() {
        let m = parse_model_string(
            "[parameters]\n  theta TVCL(10.0,1.0,100.0)\n  theta TVV(50.0,5.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.09\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n",
        )
        .expect("parse");
        // Infusion 0–8 h (rate 125, amt 1000); reset at t=4 mid-infusion; a fresh
        // bolus opens the new segment at t=4.
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 125.0, false, 0.0),
            DoseEvent::new(4.0, 500.0, 1, 0.0, false, 0.0),
        ];
        let times = [1.0, 3.0, 5.0, 7.0, 10.0];
        let subject = subject_with_doses_and_resets(doses, &times, vec![4.0]);
        check_provider_vs_production(&m, &subject, &[10.0, 50.0], &[0.1, -0.05]);
    }

    /// Provider's exact η/θ sensitivities must match central finite differences
    /// of the production predictor `compute_predictions_with_tv`.
    #[test]
    fn provider_matches_fd_of_production_predictor() {
        let model = parse_model_string(WARFARIN).expect("parse");
        let subject = oral_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let theta = vec![0.2, 10.0, 1.5];
        let eta = vec![0.15, -0.10, 0.25];
        let n_eta = 3;
        let n_theta = 3;

        let sens = subject_sensitivities(&model, &subject, &theta, &eta).expect("supported");

        // FD helpers over the full prediction vector (returns obs j's value).
        let pred = |e: &[f64], th: &[f64], j: usize| -> f64 {
            compute_predictions_with_tv(&model, &subject, th, e)[j]
        };
        let he = 1e-6; // first-derivative step
        let ht = 1e-6;
        let heh = 1e-4; // second-derivative step (4-point central is roundoff-prone)

        for (j, obs) in sens.obs.iter().enumerate() {
            // value
            let f0 = pred(&eta, &theta, j);
            approx::assert_relative_eq!(obs.f, f0, max_relative = 1e-9, epsilon = 1e-12);

            // ∂f/∂η and ∂²f/∂η²
            for k in 0..n_eta {
                let mut ep = eta.clone();
                ep[k] += he;
                let mut em = eta.clone();
                em[k] -= he;
                let g = (pred(&ep, &theta, j) - pred(&em, &theta, j)) / (2.0 * he);
                approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 2e-4, epsilon = 1e-7);
                for l in 0..n_eta {
                    let mut pp = eta.clone();
                    pp[k] += heh;
                    pp[l] += heh;
                    let mut pm = eta.clone();
                    pm[k] += heh;
                    pm[l] -= heh;
                    let mut mp = eta.clone();
                    mp[k] -= heh;
                    mp[l] += heh;
                    let mut mm = eta.clone();
                    mm[k] -= heh;
                    mm[l] -= heh;
                    let hh = (pred(&pp, &theta, j) - pred(&pm, &theta, j) - pred(&mp, &theta, j)
                        + pred(&mm, &theta, j))
                        / (4.0 * heh * heh);
                    approx::assert_relative_eq!(
                        obs.d2f_deta2[k * n_eta + l],
                        hh,
                        max_relative = 3e-3,
                        epsilon = 1e-5
                    );
                }
            }

            // ∂f/∂θ
            for m in 0..n_theta {
                let mut tp = theta.clone();
                tp[m] += ht * (1.0 + theta[m].abs());
                let mut tm = theta.clone();
                tm[m] -= ht * (1.0 + theta[m].abs());
                let step = ht * (1.0 + theta[m].abs());
                let g = (pred(&eta, &tp, j) - pred(&eta, &tm, j)) / (2.0 * step);
                approx::assert_relative_eq!(
                    obs.df_dtheta[m],
                    g,
                    max_relative = 2e-4,
                    epsilon = 1e-7
                );
            }

            // ∂²f/∂η∂θ (mixed 4-point)
            for k in 0..n_eta {
                for m in 0..n_theta {
                    let s = heh * (1.0 + theta[m].abs());
                    let mut ep = eta.clone();
                    ep[k] += heh;
                    let mut em = eta.clone();
                    em[k] -= heh;
                    let mut tp = theta.clone();
                    tp[m] += s;
                    let mut tm = theta.clone();
                    tm[m] -= s;
                    let hh = (pred(&ep, &tp, j) - pred(&ep, &tm, j) - pred(&em, &tp, j)
                        + pred(&em, &tm, j))
                        / (4.0 * heh * s);
                    approx::assert_relative_eq!(
                        obs.d2f_deta_dtheta[k * n_theta + m],
                        hh,
                        max_relative = 3e-3,
                        epsilon = 1e-5
                    );
                }
            }
        }
    }
}
