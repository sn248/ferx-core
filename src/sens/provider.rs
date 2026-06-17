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
    CompiledModel, PkModel, ScalingSpec, Subject, PK_IDX_CL, PK_IDX_F, PK_IDX_KA, PK_IDX_Q,
    PK_IDX_Q3, PK_IDX_V, PK_IDX_V2, PK_IDX_V3,
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

/// Measurement toggle: `FERX_EXPLICIT_SENS=1` routes supported models to the
/// hand-written explicit-derivative kernels (`*_explicit`) instead of the
/// `Dual2<8>` provider path. Default off (Dual2). Used to A/B the per-kernel
/// speedup on a real fit; not a shipped feature.
fn explicit_sens_enabled() -> bool {
    std::env::var("FERX_EXPLICIT_SENS")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Which explicit kernel applies to a subject (model + all-doses-plain).
#[derive(Clone, Copy)]
enum ExKind {
    OneCptBolus,
    OneCptOral,
    TwoCptBolus,
    ThreeCptBolus,
}

/// Accumulate an `M`-parameter explicit `(grad, hess)` into the 8-slot PK layout
/// via `map` (small-array index → PK slot).
fn scatter_explicit<const M: usize>(
    g: &mut [f64; N_PK],
    h: &mut [[f64; N_PK]; N_PK],
    gs: &[f64; M],
    hs: &[[f64; M]; M],
    map: &[usize; M],
) {
    for a in 0..M {
        g[map[a]] += gs[a];
        for b in 0..M {
            h[map[a]][map[b]] += hs[a][b];
        }
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
    analytical_supported(model) || crate::sens::ode_provider::ode_analytical_supported(model)
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

/// Compute per-observation analytic sensitivities, or `None` if this
/// model/subject is outside the supported analytical 1-cpt scope (caller falls
/// back to the gradient-free path).
pub fn subject_sensitivities(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<SubjectSens> {
    // ODE models route to the ODE sensitivity provider (issue #367, Option A):
    // it produces the same `SubjectSens`, so every downstream consumer
    // (`subject_packed_gradient*`, `subject_eta_jacobian`, the inner loop)
    // works unchanged.
    if model.ode_spec.is_some() {
        return crate::sens::ode_provider::ode_subject_sensitivities(model, subject, theta, eta);
    }
    if !analytical_supported(model) || subject.has_tv_covariates() || subject.has_resets() {
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

    // Typical values (η = 0) and their θ Jacobian for the ρ chain.
    let tv = (model.tv_fn.as_ref().unwrap())(theta, &subject.covariates);
    let tv_jac = tv_theta_jacobian(model, subject, theta);

    // Per individual-parameter assignment i: its seed dim, PK value, sel row,
    // and ρ row. These drive the whole chain.
    struct Term {
        dim: usize,
        pk_val: f64,
        sel: Vec<f64>, // length n_eta
        rho: Vec<f64>, // length n_theta
    }
    let mut terms: Vec<Term> = Vec::with_capacity(model.pk_indices.len());
    for (i, &slot) in model.pk_indices.iter().enumerate() {
        let dim = slot_to_dim(slot)?; // guaranteed Some by analytical_supported
        let pk_val = pk.values[slot];
        let sel: Vec<f64> = (0..n_eta).map(|k| model.sel_flat[i * n_eta + k]).collect();
        let tv_i = tv[i];
        let rho: Vec<f64> = if tv_i.abs() > 0.0 {
            (0..n_theta).map(|m| tv_jac[i][m] / tv_i).collect()
        } else {
            vec![0.0; n_theta]
        };
        terms.push(Term {
            dim,
            pk_val,
            sel,
            rho,
        });
    }

    // Explicit-kernel fast path (A/B toggle): only when all doses are plain
    // (non-SS, non-infusion) single inputs for a covered model.
    let explicit_kind =
        if explicit_sens_enabled() && subject.doses.iter().all(|d| !d.ss && !d.is_infusion()) {
            match model.pk_model {
                PkModel::OneCptIv => Some(ExKind::OneCptBolus),
                PkModel::OneCptOral => Some(ExKind::OneCptOral),
                PkModel::TwoCptIv => Some(ExKind::TwoCptBolus),
                PkModel::ThreeCptIv => Some(ExKind::ThreeCptBolus),
                _ => None,
            }
        } else {
            None
        };

    let mut out = Vec::with_capacity(subject.obs_times.len());
    for &t_obs in subject.obs_times.iter() {
        // `(f, ∂f/∂pk, ∂²f/∂pk²)` in the 8-slot layout — from the explicit
        // kernels when applicable, else the generic Dual2<8> path.
        let (fval, g, h): (f64, [f64; N_PK], [[f64; N_PK]; N_PK]) = if let Some(kind) =
            explicit_kind
        {
            let mut gv = [0.0; N_PK];
            let mut hv = [[0.0; N_PK]; N_PK];
            let mut val = 0.0;
            for dose in &subject.doses {
                let elapsed = t_obs - dose.time;
                if elapsed < 0.0 {
                    continue;
                }
                match kind {
                    ExKind::OneCptBolus => {
                        let (f, gs, hs) =
                            super::one_cpt_explicit::iv_bolus_explicit(dose.amt, elapsed, cl, v1);
                        val += f;
                        scatter_explicit(&mut gv, &mut hv, &gs, &hs, &[0, 1]);
                    }
                    ExKind::OneCptOral => {
                        let (f, gs, hs) = super::one_cpt_explicit::oral_explicit(
                            dose.amt, elapsed, cl, v1, ka, f_bio,
                        );
                        val += f;
                        scatter_explicit(&mut gv, &mut hv, &gs, &hs, &[0, 1, 4, 5]);
                    }
                    ExKind::TwoCptBolus => {
                        let (f, gs, hs) = super::two_cpt_explicit::iv_bolus_explicit(
                            dose.amt, elapsed, cl, v1, q, v2,
                        );
                        val += f;
                        scatter_explicit(&mut gv, &mut hv, &gs, &hs, &[0, 1, 2, 3]);
                    }
                    ExKind::ThreeCptBolus => {
                        let (f, gs, hs) = super::three_cpt_explicit::iv_bolus_explicit(
                            dose.amt, elapsed, cl, v1, q, v2, q3, v3,
                        );
                        val += f;
                        // [CL,V1,Q2,V2,Q3,V3] → 8-slot layout (Q3=6, V3=7).
                        scatter_explicit(&mut gv, &mut hv, &gs, &hs, &[0, 1, 2, 3, 6, 7]);
                    }
                }
            }
            (val, gv, hv)
        } else {
            // Seed PK params as Dual2<8> on [CL, V1, Q2, V2, KA, F, Q3, V3].
            // Lower-dimensional solutions ignore the unused dims.
            let cl_d = Dual2::<N_PK>::var(cl, 0);
            let v1_d = Dual2::<N_PK>::var(v1, 1);
            let q_d = Dual2::<N_PK>::var(q, 2);
            let v2_d = Dual2::<N_PK>::var(v2, 3);
            let ka_d = Dual2::<N_PK>::var(ka, 4);
            let f_d = Dual2::<N_PK>::var(f_bio, 5);
            let q3_d = Dual2::<N_PK>::var(q3, 6);
            let v3_d = Dual2::<N_PK>::var(v3, 7);

            // Superpose dose contributions: f = Σ conc(dose, t_obs − dose.time).
            let mut fd = Dual2::<N_PK>::constant(0.0);
            for dose in &subject.doses {
                let elapsed = t_obs - dose.time;
                if elapsed < 0.0 {
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

        let mut df_deta = vec![0.0; n_eta];
        let mut d2f_deta2 = vec![0.0; n_eta * n_eta];
        let mut df_dtheta = vec![0.0; n_theta];
        let mut d2f_deta_dtheta = vec![0.0; n_eta * n_theta];

        // First-order chains: ∂f/∂η_k = Σ_i g[d_i]·pk_i·sel[i,k]; likewise θ.
        for term in &terms {
            let gi = g[term.dim];
            for k in 0..n_eta {
                df_deta[k] += gi * term.pk_val * term.sel[k];
            }
            for m in 0..n_theta {
                df_dtheta[m] += gi * term.pk_val * term.rho[m];
            }
        }

        // Second-order: H term (cross over assignments) + g·(∂²pk) self term.
        // ∂²f/∂η_k∂η_l = Σ_{i,j} H[d_i][d_j]·(pk_i sel_ik)(pk_j sel_jl)
        //              + Σ_i g[d_i]·pk_i·sel_ik·sel_il.
        for k in 0..n_eta {
            for l in 0..n_eta {
                let mut acc = 0.0;
                for ti in &terms {
                    let a = ti.pk_val * ti.sel[k];
                    for tj in &terms {
                        acc += h[ti.dim][tj.dim] * a * (tj.pk_val * tj.sel[l]);
                    }
                    acc += g[ti.dim] * ti.pk_val * ti.sel[k] * ti.sel[l];
                }
                d2f_deta2[k * n_eta + l] = acc;
            }
        }
        // ∂²f/∂η_k∂θ_m = Σ_{i,j} H[d_i][d_j]·(pk_i sel_ik)(pk_j ρ_jm)
        //              + Σ_i g[d_i]·pk_i·sel_ik·ρ_im.
        for k in 0..n_eta {
            for m in 0..n_theta {
                let mut acc = 0.0;
                for ti in &terms {
                    let a = ti.pk_val * ti.sel[k];
                    for tj in &terms {
                        acc += h[ti.dim][tj.dim] * a * (tj.pk_val * tj.rho[m]);
                    }
                    acc += g[ti.dim] * ti.pk_val * ti.sel[k] * ti.rho[m];
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

    Some(SubjectSens { obs: out })
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
