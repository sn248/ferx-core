//! Paper-exact FOCEI outer gradient (Almquist 2015, Eq. 23) for analytical PK
//! models, assembled in closed form from the [`crate::sens`] provider.
//!
//! The per-subject FOCEI Laplace objective is
//!
//! ```text
//!   Fᵢ = ½ Σⱼ (εⱼ²/Rⱼ + ln Rⱼ) + ½ η̂ᵀΩ⁻¹η̂ + ½ ln|Ω| + ½ log|H̃|.
//! ```
//!
//! Its total derivative w.r.t. a population parameter pulls in the EBE response
//! `dη̂/dζ` (Eq. 46). Writing `aⱼ = ∂f/∂η`, `Aⱼ = ∂²f/∂η²`, `bⱼ = ∂f/∂θ`,
//! `Bⱼ = ∂²f/∂η∂θ` — all exact from the provider — and the error-model scalars
//! `R, d = ∂R/∂f, d2 = ∂²R/∂f²`:
//!
//! * `αⱼ = −2ε/R + d(R−ε²)/R²`,  `α'ⱼ = dαⱼ/df`
//! * `pⱼ = 1/R + ½(d/R)²`,        `βⱼ = dpⱼ/df = −d/R² + d·d2/R² − d³/R³`
//! * `H̃ = Σⱼ pⱼ aⱼaⱼᵀ + Ω⁻¹`,    `wⱼ = H̃⁻¹aⱼ`,  `qⱼ = aⱼᵀwⱼ`
//! * true inner Hessian `H = ½ Σⱼ (α'ⱼ aⱼaⱼᵀ + αⱼ Aⱼ) + Ω⁻¹`
//! * mixed `M[:,m] = ½ Σⱼ (α'ⱼ bⱼₘ aⱼ + αⱼ Bⱼ[:,m])`,  `dη̂/dθₘ = −H⁻¹ M[:,m]`
//! * `∂log|H̃|/∂η_l = Σⱼ (βⱼ qⱼ a_{jl} + 2 pⱼ Σₖ w_{jk} A_{jkl})`
//!
//! giving the per-subject θ-gradient
//!
//! ```text
//!   dFᵢ/dθₘ = ½ Σⱼ (αⱼ + βⱼqⱼ) bⱼₘ          (data + a-fixed log|H̃|)
//!           +    Σⱼ pⱼ Σₖ w_{jk} B_{jkm}      (∂²f/∂η∂θ curvature)
//!           + ½ Σ_l (∂log|H̃|/∂η_l) dη̂_l/dθₘ  (Eq. 46 EBE response)
//! ```
//!
//! This is the noise-free closed-form replacement for the previous FD-over-AD
//! curvature/EBE-response path (issue #367). Scope is whatever
//! [`crate::sens::provider::subject_sensitivities`] supports (analytical
//! 1-/2-/3-cpt); callers fall back to the existing FD/Laplace path otherwise.
// Indexed loops index parallel grad/Hessian/Jacobian buffers; clearer than zips.
#![allow(clippy::needless_range_loop)]

use crate::estimation::parameterization::{packed_fixed_mask, theta_packs_log, unpack_params};
use crate::sens::provider::{subject_sensitivities, ObsSens, SubjectSens};
use crate::types::{CompiledModel, ModelParameters, Population, Subject};
use nalgebra::{DMatrix, DVector};

/// Per-observation error-model scalars used throughout the assembly.
struct ErrTerms {
    r: f64,       // Rⱼ
    d: f64,       // dⱼ = ∂R/∂f
    eps: f64,     // εⱼ = y − f
    alpha: f64,   // αⱼ
    alpha_p: f64, // α'ⱼ = dαⱼ/df
    p: f64,       // pⱼ
    beta: f64,    // βⱼ = dpⱼ/df
}

/// Inverse Mills ratio `h = φ(z)/Φ(z)`, evaluated through logs so it stays finite
/// in the far tail (`Φ(z)→0` when the prediction sits well above the LLOQ).
fn inv_mills(z: f64) -> f64 {
    let ln_phi = -0.5 * z * z - 0.5 * std::f64::consts::TAU.ln();
    (ln_phi - crate::stats::special::log_normal_cdf(z)).exp()
}

/// M3 censored-row scalars `(g1, g2) = (∂L/∂f, ∂²L/∂f²)` for the data term
/// `L = −logΦ(z)`, `z = (y−f)/√R`, with `y` the LLOQ, `R(f)` the residual
/// variance, `d = ∂R/∂f`, `d2 = ∂²R/∂f²`. Uses `∂z/∂f = −m` and
/// `dh/dz = −h(h+z)`, where `m = 1/w + (y−f)d/(2w³)`, `w = √R`:
/// ```text
///   g1 = h·m
///   g2 = h(h+z)·m² + h·∂m/∂f,
///   ∂m/∂f = [−2d + (y−f)d2]/(2w³) − 3(y−f)d²/(4w⁵).
/// ```
fn m3_censored_scalars(y: f64, f: f64, r: f64, d: f64, d2: f64) -> (f64, f64) {
    let w = r.sqrt();
    let w3 = r * w; // w³
    let w5 = r * r * w; // w⁵
    let z = (y - f) / w;
    let h = inv_mills(z);
    let m = 1.0 / w + (y - f) * d / (2.0 * w3);
    let g1 = h * m;
    let dm_df = (-2.0 * d + (y - f) * d2) / (2.0 * w3) - 3.0 * (y - f) * d * d / (4.0 * w5);
    let g2 = h * (h + z) * m * m + h * dm_df;
    (g1, g2)
}

fn err_terms(r: f64, d: f64, d2: f64, eps: f64) -> ErrTerms {
    let inv_r = 1.0 / r;
    let inv_r2 = inv_r * inv_r;
    let inv_r3 = inv_r2 * inv_r;
    let alpha = -2.0 * eps * inv_r + d * (r - eps * eps) * inv_r2;
    // α'ⱼ = dαⱼ/df with dε/df = −1, dR/df = d, dd/df = d2:
    //   = 2/R + 2εd/R² + [d2(R−ε²) + d² + 2dε]/R² − 2d²(R−ε²)/R³.
    let alpha_p = 2.0 * inv_r
        + 2.0 * eps * d * inv_r2
        + (d2 * (r - eps * eps) + d * d + 2.0 * d * eps) * inv_r2
        - 2.0 * d * d * (r - eps * eps) * inv_r3;
    let p = inv_r + 0.5 * (d * inv_r) * (d * inv_r);
    let beta = -d * inv_r2 + d * d2 * inv_r2 - d * d * d * inv_r3;
    ErrTerms {
        r,
        d,
        eps,
        alpha,
        alpha_p,
        p,
        beta,
    }
}

/// Shared per-subject quantities the θ/Ω/Σ gradient blocks all consume, built
/// once from the provider sensitivities at the EBE.
struct Prep {
    n_eta: usize,
    n_obs: usize,
    et: Vec<ErrTerms>,
    /// `Ω⁻¹` (copied so blocks don't borrow `params`).
    omega_inv: DMatrix<f64>,
    /// `H̃⁻¹` (first-order FOCEI Hessian inverse).
    htilde_inv: DMatrix<f64>,
    /// `H⁻¹` for the **true** inner Hessian `H = ∂²lᵢ/∂η²` (Eq. 46 denominator).
    h_inner_inv: DMatrix<f64>,
    /// `wⱼ = H̃⁻¹aⱼ`.
    w: Vec<DVector<f64>>,
    /// `qⱼ = aⱼᵀ H̃⁻¹ aⱼ`.
    q: Vec<f64>,
    /// Exact `∂log|H̃|/∂η` (a-fixed part + `∂²f/∂η²` curvature).
    g_eta: Vec<f64>,
    /// Per-observation M3-censored flag. Censored rows enter `H` (true inner
    /// Hessian) and the data gradient but carry `p = β = 0`, so they are excluded
    /// from `H̃` / `log|H̃|` — matching `gaussian_foce_accum`, which accumulates
    /// `hrh`/`ctc` over quantified rows only and adds the censored `−logΦ` to the
    /// data term. Empty `Vec` (all-false) when the subject has no censoring.
    censored: Vec<bool>,
}

fn prepare(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    sens: &SubjectSens,
) -> Option<Prep> {
    let n_eta = model.n_eta;
    let n_obs = subject.observations.len();
    if sens.obs.len() != n_obs {
        return None;
    }
    let sigma = &params.sigma.values;
    let omega_inv = params.omega.inv.clone();

    // H̃ = Σ pⱼ aⱼaⱼᵀ + Ω⁻¹ ; true inner Hessian H = ½Σ(α'ⱼ aⱼaⱼᵀ + αⱼ Aⱼ) + Ω⁻¹.
    let mut htilde = omega_inv.clone();
    let mut h_inner = omega_inv.clone();
    let mut et: Vec<ErrTerms> = Vec::with_capacity(n_obs);

    let m3 = matches!(model.bloq_method, crate::types::BloqMethod::M3);
    let mut censored = vec![false; n_obs];
    let mut any_cens = false;
    for obs in sens.obs.iter() {
        let f = obs.f;
        // obs index → cmt: provider obs are parallel to subject.obs_times.
        let j = et.len();
        let cmt = subject.obs_cmts[j];
        let r = model.error_spec.variance_at(cmt, f, sigma);
        if !(r.is_finite() && r > 0.0) {
            return None;
        }
        let d = model.error_spec.dvar_df(cmt, f, sigma);
        let d2 = model.error_spec.d2var_df2(cmt, sigma);
        let y = subject.observations[j];
        let is_cens = m3 && subject.cens.get(j).copied().unwrap_or(0) != 0;
        // For a censored row the data term is `−logΦ(z)`: store its f-derivatives
        // as `alpha = 2·g1`, `alpha_p = 2·g2` (so the assembly's `½α`, `½α'` recover
        // `∂L/∂f`, `∂²L/∂f²`) and force `p = β = 0` (excluded from `H̃` / `log|H̃|`).
        let t = if is_cens {
            censored[j] = true;
            any_cens = true;
            let (g1, g2) = m3_censored_scalars(y, f, r, d, d2);
            ErrTerms {
                r,
                d,
                eps: y - f,
                alpha: 2.0 * g1,
                alpha_p: 2.0 * g2,
                p: 0.0,
                beta: 0.0,
            }
        } else {
            err_terms(r, d, d2, y - f)
        };

        let a = obs.df_deta.as_slice();
        for k in 0..n_eta {
            for l in 0..n_eta {
                htilde[(k, l)] += t.p * a[k] * a[l];
                h_inner[(k, l)] +=
                    0.5 * (t.alpha_p * a[k] * a[l] + t.alpha * obs.d2f_deta2[k * n_eta + l]);
            }
        }
        et.push(t);
    }
    let censored = if any_cens { censored } else { Vec::new() };

    let htilde_inv = htilde.cholesky()?.inverse();
    let h_inner_inv = h_inner.cholesky()?.inverse();

    let mut w: Vec<DVector<f64>> = Vec::with_capacity(n_obs);
    let mut q = vec![0.0f64; n_obs];
    for (j, obs) in sens.obs.iter().enumerate() {
        let aj = DVector::from_column_slice(&obs.df_deta);
        let wj = &htilde_inv * &aj;
        q[j] = aj.dot(&wj);
        w.push(wj);
    }

    // ∂log|H̃|/∂η_l = Σⱼ (βⱼ qⱼ a_{jl} + 2 pⱼ Σₖ w_{jk} A_{jkl}).
    let mut g_eta = vec![0.0f64; n_eta];
    for l in 0..n_eta {
        let mut s = 0.0;
        for (j, obs) in sens.obs.iter().enumerate() {
            s += et[j].beta * q[j] * obs.df_deta[l];
            let mut curv = 0.0;
            for k in 0..n_eta {
                curv += w[j][k] * obs.d2f_deta2[k * n_eta + l];
            }
            s += 2.0 * et[j].p * curv;
        }
        g_eta[l] = s;
    }

    Some(Prep {
        n_eta,
        n_obs,
        et,
        omega_inv,
        htilde_inv,
        h_inner_inv,
        w,
        q,
        g_eta,
        censored,
    })
}

/// The exact per-subject θ-gradient `dFᵢ/dθ` (length `n_theta`, natural θ
/// space), or `None` when the model/subject is outside the provider's scope.
///
/// `eta_hat` must be the EBE for `params` (the function evaluates the gradient
/// identity at the inner optimum; the envelope theorem and Eq. 46 both assume
/// `∂lᵢ/∂η|_η̂ = 0`).
pub fn subject_theta_gradient(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    eta_hat: &[f64],
) -> Option<Vec<f64>> {
    let sens = subject_sensitivities(model, subject, &params.theta, eta_hat)?;
    let prep = prepare(model, subject, params, &sens)?;
    Some(theta_block(&prep, &sens, params.theta.len()))
}

fn theta_block(prep: &Prep, sens: &SubjectSens, n_theta: usize) -> Vec<f64> {
    let (n_eta, n_obs) = (prep.n_eta, prep.n_obs);
    let mut grad = vec![0.0f64; n_theta];
    for m in 0..n_theta {
        // data + a-fixed log|H̃|:  ½ Σⱼ (αⱼ + βⱼqⱼ) bⱼₘ ; plus ∂²f/∂η∂θ curvature.
        let mut g = 0.0;
        for (j, obs) in sens.obs.iter().enumerate() {
            let bjm = obs.df_dtheta[m];
            g += 0.5 * (prep.et[j].alpha + prep.et[j].beta * prep.q[j]) * bjm;
            let mut curv = 0.0;
            for k in 0..n_eta {
                curv += prep.w[j][k] * obs.d2f_deta_dtheta[k * n_theta + m];
            }
            g += prep.et[j].p * curv;
        }
        // EBE response: ½ g_eta · dη̂/dθₘ,  dη̂/dθₘ = −H⁻¹ M[:,m].
        let m_vec = mixed_eta_theta(&sens.obs, &prep.et, n_eta, n_obs, m);
        let deta = -(&prep.h_inner_inv * m_vec);
        let mut resp = 0.0;
        for l in 0..n_eta {
            resp += prep.g_eta[l] * deta[l];
        }
        grad[m] = g + 0.5 * resp;
    }
    grad
}

/// The exact per-subject Ω-gradient `dFᵢ/dΩ` over the free Ω entries, in the
/// same order the optimizer packs them (diagonal: `(i,i)`; block: lower triangle
/// `(i,j)`, `j ≤ i`), natural variance/covariance scale. `None` when unsupported.
///
/// Per free entry `(r,c)` with `z = Ω⁻¹η̂`, `G = Ω⁻¹H̃⁻¹Ω⁻¹`, `v = Ω⁻¹H⁻¹g_eta`:
/// fixed-η̂ part `½[−zᵀEz + tr(Ω⁻¹E) − tr(GE)]` plus EBE response `½ vᵀEz`,
/// `E = ∂Ω/∂Ω_{rc}` (symmetric).
pub fn subject_omega_gradient(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    eta_hat: &[f64],
) -> Option<Vec<f64>> {
    let sens = subject_sensitivities(model, subject, &params.theta, eta_hat)?;
    let prep = prepare(model, subject, params, &sens)?;
    Some(omega_block(&prep, params, eta_hat))
}

fn omega_block(prep: &Prep, params: &ModelParameters, eta_hat: &[f64]) -> Vec<f64> {
    let n_eta = prep.n_eta;
    let eta = DVector::from_column_slice(eta_hat);
    let z = &prep.omega_inv * &eta;
    let g_mat = &prep.omega_inv * &prep.htilde_inv * &prep.omega_inv;
    let u = &prep.h_inner_inv * DVector::from_column_slice(&prep.g_eta);
    let v = &prep.omega_inv * u;

    let entries: Vec<(usize, usize)> = if params.omega.diagonal {
        (0..n_eta).map(|i| (i, i)).collect()
    } else {
        let mut e = Vec::new();
        for c in 0..n_eta {
            for r in c..n_eta {
                e.push((r, c));
            }
        }
        e
    };

    entries
        .iter()
        .map(|&(r, c)| {
            if r == c {
                // E has a single 1 at (r,r).
                let fixed = 0.5 * (-z[r] * z[r] + prep.omega_inv[(r, r)] - g_mat[(r, r)]);
                let resp = 0.5 * v[r] * z[r];
                fixed + resp
            } else {
                // Symmetric off-diagonal: E has 1 at (r,c) and (c,r).
                let fixed = -z[r] * z[c] + prep.omega_inv[(r, c)] - g_mat[(r, c)];
                let resp = 0.5 * (v[r] * z[c] + v[c] * z[r]);
                fixed + resp
            }
        })
        .collect()
}

/// The exact per-subject Σ-gradient `dFᵢ/dσ` (length `n_sigma`, natural σ
/// scale), or `None` when unsupported. σ enters only through the residual
/// variance, so `∂R/∂σ` and `∂d/∂σ` (`d = ∂R/∂f`) are taken by central FD of the
/// closed-form error functions — exact algebra, well-conditioned, no AD.
///
/// Per σ_k:  `½ Σⱼ Rσⱼ(Rⱼ−εⱼ²)/Rⱼ²` (data + lnR) `+ ½ Σⱼ (∂pⱼ/∂σ) qⱼ` (log|H̃|)
/// `+ ½ g_eta·dη̂/dσ`, with `dη̂/dσ = −H⁻¹ M`, `M[m] = ½ Σⱼ (∂αⱼ/∂σ) a_{jm}`.
pub fn subject_sigma_gradient(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    eta_hat: &[f64],
) -> Option<Vec<f64>> {
    let sens = subject_sensitivities(model, subject, &params.theta, eta_hat)?;
    let prep = prepare(model, subject, params, &sens)?;
    Some(sigma_block(&prep, model, subject, params, &sens))
}

fn sigma_block(
    prep: &Prep,
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    sens: &SubjectSens,
) -> Vec<f64> {
    let n_eta = prep.n_eta;
    let sigma = &params.sigma.values;
    let n_sigma = sigma.len();
    let mut grad = vec![0.0f64; n_sigma];

    for k in 0..n_sigma {
        let h = 1e-6 * (1.0 + sigma[k].abs());
        let mut sp = sigma.clone();
        sp[k] += h;
        let mut sm = sigma.clone();
        sm[k] -= h;

        let mut fixed = 0.0;
        let mut m_vec = DVector::<f64>::zeros(n_eta);
        for (j, obs) in sens.obs.iter().enumerate() {
            let cmt = subject.obs_cmts[j];
            let f = obs.f;
            if prep.censored.get(j).copied().unwrap_or(false) {
                // M3 censored row: data term `−logΦ((y−f)/√R(σ))` (not in `H̃`, so
                // no `log|H̃|` σ-term). `∂L/∂σ` and `∂g1/∂σ = ½∂α/∂σ` by central FD
                // of the closed-form censored scalars, mirroring the Gaussian path.
                let y = subject.observations[j];
                let l_sig = (-crate::stats::special::log_normal_cdf(
                    (y - f) / model.error_spec.variance_at(cmt, f, &sp).sqrt(),
                ) + crate::stats::special::log_normal_cdf(
                    (y - f) / model.error_spec.variance_at(cmt, f, &sm).sqrt(),
                )) / (2.0 * h);
                fixed += l_sig;
                let (g1p, _) = m3_censored_scalars(
                    y,
                    f,
                    model.error_spec.variance_at(cmt, f, &sp),
                    model.error_spec.dvar_df(cmt, f, &sp),
                    model.error_spec.d2var_df2(cmt, &sp),
                );
                let (g1m, _) = m3_censored_scalars(
                    y,
                    f,
                    model.error_spec.variance_at(cmt, f, &sm),
                    model.error_spec.dvar_df(cmt, f, &sm),
                    model.error_spec.d2var_df2(cmt, &sm),
                );
                let dg1 = (g1p - g1m) / (2.0 * h);
                for m in 0..n_eta {
                    m_vec[m] += dg1 * obs.df_deta[m];
                }
                continue;
            }
            let (r, d, eps) = (prep.et[j].r, prep.et[j].d, prep.et[j].eps);
            // ∂R/∂σ_k, ∂d/∂σ_k by central FD of the closed-form error functions.
            let r_sig = (model.error_spec.variance_at(cmt, f, &sp)
                - model.error_spec.variance_at(cmt, f, &sm))
                / (2.0 * h);
            let d_sig = (model.error_spec.dvar_df(cmt, f, &sp)
                - model.error_spec.dvar_df(cmt, f, &sm))
                / (2.0 * h);

            let inv_r = 1.0 / r;
            let inv_r2 = inv_r * inv_r;
            let inv_r3 = inv_r2 * inv_r;

            // data + lnR:  ½ Rσ (R − ε²)/R²
            fixed += 0.5 * r_sig * (r - eps * eps) * inv_r2;
            // log|H̃|:  ½ (∂p/∂σ) q ,  ∂p/∂σ = −Rσ/R² + d·dσ/R² − d²Rσ/R³
            let dp = -r_sig * inv_r2 + d * d_sig * inv_r2 - d * d * r_sig * inv_r3;
            fixed += 0.5 * dp * prep.q[j];

            // ∂α/∂σ = [2ε/R² + d(2ε²−R)/R³] Rσ + [(R−ε²)/R²] dσ
            let dalpha = (2.0 * eps * inv_r2 + d * (2.0 * eps * eps - r) * inv_r3) * r_sig
                + ((r - eps * eps) * inv_r2) * d_sig;
            for m in 0..n_eta {
                m_vec[m] += 0.5 * dalpha * obs.df_deta[m];
            }
        }

        let deta = -(&prep.h_inner_inv * m_vec);
        let mut resp = 0.0;
        for l in 0..n_eta {
            resp += prep.g_eta[l] * deta[l];
        }
        grad[k] = fixed + 0.5 * resp;
    }
    grad
}

/// Per-Ω-Cholesky-entry packed gradient `∂Fᵢ/∂x` in `pack_params` order
/// (diagonal: `ln L_ii`; block: lower-triangle `(i,j)`, off-diagonals raw). The
/// fixed-η̂ part is the existing closed form (the inner factor-2 cancels the
/// outer ½, so it is the *full* ∂NLL/∂x), augmented with the Eq. 46 EBE-response
/// `tᵢ = ½·g_eta·dη̂/dL` mapped into L-space:
/// `t_{L,rc} = ½[(v·z)·s_r + z_r·(s·v)]`, `v = L[:,c]`, `s = Ω⁻¹H⁻¹g_eta`
/// (×`L_kk` for the diagonal log-chain).
fn omega_packed_block(prep: &Prep, params: &ModelParameters, eta_hat: &[f64]) -> Vec<f64> {
    let n_eta = prep.n_eta;
    let l = &params.omega.chol;
    let z = &prep.omega_inv * DVector::from_column_slice(eta_hat);
    let g_mat = &prep.omega_inv * &prep.htilde_inv * &prep.omega_inv;
    let s = &prep.omega_inv * (&prep.h_inner_inv * DVector::from_column_slice(&prep.g_eta));

    let entries: Vec<(usize, usize)> = if params.omega.diagonal {
        (0..n_eta).map(|i| (i, i)).collect()
    } else {
        let mut e = Vec::new();
        for c in 0..n_eta {
            for r in c..n_eta {
                e.push((r, c));
            }
        }
        e
    };

    entries
        .iter()
        .map(|&(row, col)| {
            let v: Vec<f64> = (0..n_eta).map(|r| l[(r, col)]).collect();
            let vz: f64 = v.iter().zip(z.iter()).map(|(a, b)| a * b).sum();
            let gv_row: f64 = (0..n_eta).map(|c| g_mat[(row, c)] * v[c]).sum();
            let sv: f64 = v.iter().zip(s.iter()).map(|(a, b)| a * b).sum();
            let t = 0.5 * (vz * s[row] + z[row] * sv);
            if row == col {
                let l_kk = l[(row, row)];
                (-l_kk * z[row] * vz + 1.0 - l_kk * gv_row) + l_kk * t
            } else {
                (-z[row] * vz - gv_row) + t
            }
        })
        .collect()
}

/// The exact per-subject FOCEI gradient `dFᵢ/dx` in the **packed** optimizer
/// space (log-θ / Cholesky-Ω / log-σ), or `None` when unsupported. `eta_hat`
/// must be the EBE for `unpack_params(x)`.
pub fn subject_packed_gradient(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    eta_hat: &[f64],
) -> Option<Vec<f64>> {
    // M3/BLOQ: censored rows enter through `prepare` (data term `−logΦ`, true
    // inner Hessian; excluded from `H̃`/`log|H̃|` — matching `gaussian_foce_accum`).
    // This is the FOCEI (interaction) path that M3 always promotes to; plain FOCE
    // with M3 still falls back to FD in `subject_packed_gradient_foce`.
    let params = unpack_params(x, template);
    let sens = subject_sensitivities(model, subject, &params.theta, eta_hat)?;
    let prep = prepare(model, subject, &params, &sens)?;

    let n_theta = params.theta.len();
    let n_sigma = params.sigma.values.len();
    let mut g = vec![0.0f64; x.len()];

    // θ: ∂F/∂x = ∂F/∂θ · ∂θ/∂x, ∂θ/∂x = θ (log) or 1 (identity).
    let g_theta = theta_block(&prep, &sens, n_theta);
    for m in 0..n_theta {
        let dtheta_dx = if theta_packs_log(template.theta_lower[m]) {
            params.theta[m]
        } else {
            1.0
        };
        g[m] = g_theta[m] * dtheta_dx;
    }

    // Ω: packed Cholesky-L gradient (already in x-space).
    let omega_start = n_theta;
    let og = omega_packed_block(&prep, &params, eta_hat);
    let n_omega = og.len();
    for (ko, &val) in og.iter().enumerate() {
        g[omega_start + ko] = val;
    }

    // σ: ∂F/∂x = ∂F/∂σ · σ (log-σ chain).
    let sigma_start = omega_start + n_omega;
    let g_sigma = sigma_block(&prep, model, subject, &params, &sens);
    for k in 0..n_sigma {
        g[sigma_start + k] = g_sigma[k] * params.sigma.values[k];
    }

    Some(g)
}

/// The exact analytic population gradient `d(OFV)/dx = 2·Σᵢ dFᵢ/dx` in packed
/// space, or `None` if any subject is unsupported (caller falls back to FD).
/// Fixed coordinates are zeroed. `eta_hats[i]` must be subject `i`'s EBE at `x`.
pub fn population_gradient_sens(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    x: &[f64],
    eta_hats: &[DVector<f64>],
) -> Option<Vec<f64>> {
    let n = x.len();
    let mut grad = vec![0.0f64; n];
    for (i, subject) in population.subjects.iter().enumerate() {
        let gi = subject_packed_gradient(model, subject, template, x, eta_hats[i].as_slice())?;
        for k in 0..n {
            grad[k] += 2.0 * gi[k];
        }
    }
    let fixed = packed_fixed_mask(template);
    for k in 0..n {
        if fixed[k] {
            grad[k] = 0.0;
        }
    }
    Some(grad)
}

/// The exact per-subject **FOCE** (non-interaction) packed gradient `dFᵢ/dx`, or
/// `None` when unsupported. ferx's FOCE objective is the Sheiner–Beal linearized
/// marginal (the algebraic equal of the paper's Laplace FOCE, Eq. 18, with the
/// residual variance independent of η):
///
/// ```text
///   Fᵢ = ½ [ ρᵀ R̃⁻¹ ρ + log|R̃| ],   ρ = y − f0,  f0 = f(η̂) − J·η̂,
///   R̃ = J Ω Jᵀ + diag(R⁰),  J = ∂f/∂η,  R⁰ⱼ = R(fⱼ(η=0)).
/// ```
///
/// The EBE η̂ is the **shared** posterior mode (the inner objective is the same
/// `individual_nll` FOCE and FOCEI both minimise), so the true inner Hessian and
/// the Eq. 46 response `dη̂/dx` are reused verbatim from [`subject_eta_dx`]; the
/// total derivative is `∂Fᵢ/∂x|_η̂ + c·dη̂/dx` with the coupling `c = ∂Fᵢ/∂η̂`.
/// Only the fixed-η̂ marginal partials and `c` are FOCE-specific (computed here).
pub fn subject_packed_gradient_foce(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    eta_hat: &[f64],
) -> Option<Vec<f64>> {
    // Censored rows under M3 are routed to the interaction path by ferx's FOCE
    // dispatcher; fall back rather than mis-model them on the non-interaction path.
    if subject.cens.iter().any(|&c| c != 0) {
        return None;
    }
    let params = unpack_params(x, template);
    let n_eta = model.n_eta;
    let n_obs = subject.observations.len();
    if n_obs == 0 {
        return None;
    }
    let sens = subject_sensitivities(model, subject, &params.theta, eta_hat)?;
    // Residual variance R⁰ is frozen at the η=0 (typical-individual) prediction —
    // ferx's no-interaction semantics. One extra provider pass supplies f(η=0)
    // and ∂f(η=0)/∂θ (for ∂R⁰/∂θ); both reuse the analytic closed forms.
    let zeros = vec![0.0f64; n_eta];
    let sens0 = subject_sensitivities(model, subject, &params.theta, &zeros)?;
    if sens.obs.len() != n_obs || sens0.obs.len() != n_obs {
        return None;
    }

    let sigma = &params.sigma.values;
    let omega = &params.omega.matrix;

    // J = ∂f/∂η (n_obs×n_eta), ρ = y − f0 = ε + J·η̂, R⁰ and d⁰ at f(η=0).
    let mut jmat = DMatrix::<f64>::zeros(n_obs, n_eta);
    let mut rho = DVector::<f64>::zeros(n_obs);
    let mut r0 = vec![0.0f64; n_obs];
    let mut d0 = vec![0.0f64; n_obs];
    for j in 0..n_obs {
        let obs = &sens.obs[j];
        let mut jeta = 0.0;
        for k in 0..n_eta {
            jmat[(j, k)] = obs.df_deta[k];
            jeta += obs.df_deta[k] * eta_hat[k];
        }
        rho[j] = subject.observations[j] - (obs.f - jeta);
        let cmt = subject.obs_cmts[j];
        let f0act = sens0.obs[j].f;
        let r = model.error_spec.variance_at(cmt, f0act, sigma);
        if !(r.is_finite() && r > 0.0) {
            return None;
        }
        r0[j] = r;
        d0[j] = model.error_spec.dvar_df(cmt, f0act, sigma);
    }

    // R̃ = J Ω Jᵀ + diag(R⁰); u = R̃⁻¹ ρ; ΩJᵀ reused throughout.
    let jo = &jmat * omega; // J Ω
    let mut rtilde = &jo * jmat.transpose();
    for j in 0..n_obs {
        rtilde[(j, j)] += r0[j];
    }
    let rtilde_inv = rtilde.cholesky()?.inverse();
    let u = &rtilde_inv * &rho;
    let ojt = omega * jmat.transpose(); // Ω Jᵀ (n_eta×n_obs)

    let n_theta = params.theta.len();
    let n_sigma = sigma.len();
    let mut fixed = vec![0.0f64; x.len()];

    // θ (fixed η̂): ∂F/∂θₘ = u·Qₘ + tr(R̃⁻¹EₘΩJᵀ) − u·(EₘΩJᵀu) + ½Σⱼ ∂R⁰ⱼ/∂θₘ (R̃⁻¹ⱼⱼ − u²ⱼ).
    //   Qₘ[j] = −bⱼₘ + Σₗ Bⱼ[l,m] η̂ₗ,  Eₘ[j,l] = Bⱼ[l,m],  ∂R⁰/∂θ = d⁰·∂f(η=0)/∂θ.
    for m in 0..n_theta {
        let mut qm = DVector::<f64>::zeros(n_obs);
        let mut em = DMatrix::<f64>::zeros(n_obs, n_eta);
        let mut dvar = 0.0;
        for j in 0..n_obs {
            let obs = &sens.obs[j];
            let mut bjeta = 0.0;
            for l in 0..n_eta {
                let bjl_m = obs.d2f_deta_dtheta[l * n_theta + m];
                em[(j, l)] = bjl_m;
                bjeta += bjl_m * eta_hat[l];
            }
            qm[j] = -obs.df_dtheta[m] + bjeta;
            let dr0 = d0[j] * sens0.obs[j].df_dtheta[m];
            dvar += dr0 * (rtilde_inv[(j, j)] - u[j] * u[j]);
        }
        let emojt = &em * &ojt;
        let tr = (&rtilde_inv * &emojt).trace();
        let uemu = u.dot(&(&emojt * &u));
        let nat = u.dot(&qm) + tr - uemu + 0.5 * dvar;
        let dtheta_dx = if theta_packs_log(template.theta_lower[m]) {
            params.theta[m]
        } else {
            1.0
        };
        fixed[m] = nat * dtheta_dx;
    }

    // Ω (fixed η̂, packed Cholesky-L): ∂F/∂L_rc = jvᵀR̃⁻¹jr − (jrᵀu)(jvᵀu),
    //   jr = J[:,row], jv = (J L)[:,col]; ×L_kk for the diagonal log-chain.
    let l = &params.omega.chol;
    let jl = &jmat * l;
    let entries: Vec<(usize, usize)> = if params.omega.diagonal {
        (0..n_eta).map(|i| (i, i)).collect()
    } else {
        let mut e = Vec::new();
        for c in 0..n_eta {
            for r in c..n_eta {
                e.push((r, c));
            }
        }
        e
    };
    let omega_start = n_theta;
    for (ko, &(row, col)) in entries.iter().enumerate() {
        let jr = jmat.column(row);
        let jv = jl.column(col);
        let rinv_jr = &rtilde_inv * jr;
        let fixed_l = jv.dot(&rinv_jr) - jr.dot(&u) * jv.dot(&u);
        let chain = if row == col { l[(row, row)] } else { 1.0 };
        fixed[omega_start + ko] = fixed_l * chain;
    }

    // σ (fixed η̂): ∂F/∂σ_k = ½Σⱼ ∂R⁰ⱼ/∂σ_k (R̃⁻¹ⱼⱼ − u²ⱼ); ×σ for log-σ. ∂R⁰/∂σ
    // by central FD of the closed-form variance at f(η=0).
    let sigma_start = omega_start + entries.len();
    for k in 0..n_sigma {
        let hsig = 1e-6 * (1.0 + sigma[k].abs());
        let mut sp = sigma.clone();
        sp[k] += hsig;
        let mut sm = sigma.clone();
        sm[k] -= hsig;
        let mut dvar = 0.0;
        for j in 0..n_obs {
            let cmt = subject.obs_cmts[j];
            let f0act = sens0.obs[j].f;
            let dr0 = (model.error_spec.variance_at(cmt, f0act, &sp)
                - model.error_spec.variance_at(cmt, f0act, &sm))
                / (2.0 * hsig);
            dvar += dr0 * (rtilde_inv[(j, j)] - u[j] * u[j]);
        }
        fixed[sigma_start + k] = 0.5 * dvar * sigma[k];
    }

    // Coupling c = ∂F/∂η̂:  c_k = u·P_k + tr(R̃⁻¹ Dk ΩJᵀ) − u·(Dk ΩJᵀ u),
    //   P_k[j] = (Aⱼη̂)_k = Σₗ Aⱼ[k,l] η̂ₗ,  Dk[j,l] = Aⱼ[l,k]  (Aⱼ = ∂²f/∂η², symmetric).
    let mut c = DVector::<f64>::zeros(n_eta);
    for k in 0..n_eta {
        let mut pk = DVector::<f64>::zeros(n_obs);
        let mut dk = DMatrix::<f64>::zeros(n_obs, n_eta);
        for j in 0..n_obs {
            let obs = &sens.obs[j];
            let mut s = 0.0;
            for l in 0..n_eta {
                let a_kl = obs.d2f_deta2[k * n_eta + l];
                s += a_kl * eta_hat[l];
                dk[(j, l)] = a_kl; // A symmetric: Aⱼ[l,k] = Aⱼ[k,l]
            }
            pk[j] = s;
        }
        let dkojt = &dk * &ojt;
        let tr = (&rtilde_inv * &dkojt).trace();
        let udku = u.dot(&(&dkojt * &u));
        c[k] = u.dot(&pk) + tr - udku;
    }

    // Total: dFᵢ/dx_k = ∂Fᵢ/∂x_k|_η̂ + c·(dη̂/dx_k). dη̂/dx is interaction-
    // independent (shared inner objective), so it is reused as-is.
    let eta_dx = subject_eta_dx(model, subject, template, x, eta_hat)?;
    let mut g = vec![0.0f64; x.len()];
    for k in 0..x.len() {
        g[k] = fixed[k] + c.dot(&eta_dx[k]);
    }
    Some(g)
}

/// The exact analytic **FOCE** population gradient `d(OFV)/dx = 2·Σᵢ dFᵢ/dx` in
/// packed space, or `None` if any subject is unsupported. Fixed coords zeroed.
pub fn population_gradient_sens_foce(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    x: &[f64],
    eta_hats: &[DVector<f64>],
) -> Option<Vec<f64>> {
    let n = x.len();
    let mut grad = vec![0.0f64; n];
    for (i, subject) in population.subjects.iter().enumerate() {
        let gi = subject_packed_gradient_foce(model, subject, template, x, eta_hats[i].as_slice())?;
        for k in 0..n {
            grad[k] += 2.0 * gi[k];
        }
    }
    let fixed = packed_fixed_mask(template);
    for k in 0..n {
        if fixed[k] {
            grad[k] = 0.0;
        }
    }
    Some(grad)
}

/// Per-packed-coordinate EBE response `dη̂/dx_k` (each a length-`n_eta` vector),
/// for the Almquist Eq. 48 warm-start predictor. Same `H⁻¹·∂²lᵢ/∂η∂x` solves the
/// gradient already forms, chained natural→packed. `None` when unsupported.
pub fn subject_eta_dx(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    eta_hat: &[f64],
) -> Option<Vec<DVector<f64>>> {
    let params = unpack_params(x, template);
    let sens = subject_sensitivities(model, subject, &params.theta, eta_hat)?;
    let prep = prepare(model, subject, &params, &sens)?;
    let n_eta = prep.n_eta;
    let n_theta = params.theta.len();
    let n_sigma = params.sigma.values.len();
    let mut out: Vec<DVector<f64>> = vec![DVector::zeros(n_eta); x.len()];

    // θ coords: dη̂/dx = −H⁻¹ (∂²l/∂η∂θ · ∂θ/∂x).
    for m in 0..n_theta {
        let dtheta_dx = if theta_packs_log(template.theta_lower[m]) {
            params.theta[m]
        } else {
            1.0
        };
        let mvec = mixed_eta_theta(&sens.obs, &prep.et, n_eta, prep.n_obs, m);
        out[m] = -(&prep.h_inner_inv * mvec) * dtheta_dx;
    }

    // Ω coords: M_L = −Ω⁻¹(e_row·(v·z) + v·z_row), v = L[:,col]; ×L_kk for diag-log.
    let z = &prep.omega_inv * DVector::from_column_slice(eta_hat);
    let l = &params.omega.chol;
    let entries: Vec<(usize, usize)> = if params.omega.diagonal {
        (0..n_eta).map(|i| (i, i)).collect()
    } else {
        let mut e = Vec::new();
        for c in 0..n_eta {
            for r in c..n_eta {
                e.push((r, c));
            }
        }
        e
    };
    let omega_start = n_theta;
    for (ko, &(row, col)) in entries.iter().enumerate() {
        let v = DVector::from_iterator(n_eta, (0..n_eta).map(|r| l[(r, col)]));
        let vz = v.dot(&z);
        let oinv_v = &prep.omega_inv * &v;
        let oinv_col_row: DVector<f64> = prep.omega_inv.column(row).into_owned();
        let m_l = -(oinv_col_row * vz + oinv_v * z[row]);
        let chain = if row == col { l[(row, row)] } else { 1.0 };
        out[omega_start + ko] = -(&prep.h_inner_inv * m_l) * chain;
    }

    // σ coords: M_σ = ½ Σⱼ ∂αⱼ/∂σ · aⱼ (∂R/∂σ,∂d/∂σ by FD of closed form); ×σ.
    let sigma_start = omega_start + entries.len();
    let sigma = &params.sigma.values;
    for k in 0..n_sigma {
        let h = 1e-6 * (1.0 + sigma[k].abs());
        let mut sp = sigma.clone();
        sp[k] += h;
        let mut sm = sigma.clone();
        sm[k] -= h;
        let mut mvec = DVector::<f64>::zeros(n_eta);
        for (j, obs) in sens.obs.iter().enumerate() {
            let cmt = subject.obs_cmts[j];
            let f = obs.f;
            let (r, d, eps) = (prep.et[j].r, prep.et[j].d, prep.et[j].eps);
            let r_sig = (model.error_spec.variance_at(cmt, f, &sp)
                - model.error_spec.variance_at(cmt, f, &sm))
                / (2.0 * h);
            let d_sig = (model.error_spec.dvar_df(cmt, f, &sp)
                - model.error_spec.dvar_df(cmt, f, &sm))
                / (2.0 * h);
            let inv_r = 1.0 / r;
            let inv_r2 = inv_r * inv_r;
            let inv_r3 = inv_r2 * inv_r;
            let dalpha = (2.0 * eps * inv_r2 + d * (2.0 * eps * eps - r) * inv_r3) * r_sig
                + ((r - eps * eps) * inv_r2) * d_sig;
            for m in 0..n_eta {
                mvec[m] += 0.5 * dalpha * obs.df_deta[m];
            }
        }
        out[sigma_start + k] = -(&prep.h_inner_inv * mvec) * sigma[k];
    }

    Some(out)
}

/// Per-subject `dη̂/dx` Jacobians for the whole population, or `None` if any
/// subject is unsupported.
pub fn population_eta_dx(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    x: &[f64],
    eta_hats: &[DVector<f64>],
) -> Option<Vec<Vec<DVector<f64>>>> {
    population
        .subjects
        .iter()
        .enumerate()
        .map(|(i, s)| subject_eta_dx(model, s, template, x, eta_hats[i].as_slice()))
        .collect()
}

/// Almquist Eq. 48 warm-start: `η⁰ᵢ = η̂ᵢ + Σₖ (dη̂ᵢ/dx_k)·(x_new−x_prev)_k`.
pub fn predict_warm_etas(
    prev_etas: &[DVector<f64>],
    jacs: &[Vec<DVector<f64>>],
    x_prev: &[f64],
    x_new: &[f64],
) -> Vec<DVector<f64>> {
    // Cap on the L2 norm of a single predicted η warm-start step. The inner solve
    // re-refines from the warm start, so this only needs to keep it inside a sane
    // region: on a large or ill-conditioned outer step the linear Eq.48
    // extrapolation can overshoot the basin, and if the inner BFGS then hits
    // max_iter it can land at a different mode, perturbing the reported OFV.
    // η live on the O(1) random-effects scale, so ~2 (a few IIV SDs) rarely binds
    // on a normal step but blocks a runaway one. PR #381 review finding #8.
    const MAX_PREDICT_STEP_NORM: f64 = 2.0;
    prev_etas
        .iter()
        .zip(jacs.iter())
        .map(|(eta, jac)| {
            let mut step = DVector::zeros(eta.len());
            for (k, jk) in jac.iter().enumerate() {
                let dx = x_new[k] - x_prev[k];
                if dx != 0.0 {
                    step += jk * dx;
                }
            }
            let norm = step.norm();
            if norm > MAX_PREDICT_STEP_NORM {
                step *= MAX_PREDICT_STEP_NORM / norm;
            }
            eta + step
        })
        .collect()
}

/// `M[:,m] = ∂²lᵢ/∂η∂θₘ = ½ Σⱼ (α'ⱼ bⱼₘ aⱼ + αⱼ Bⱼ[:,m])`.
fn mixed_eta_theta(
    obs: &[ObsSens],
    et: &[ErrTerms],
    n_eta: usize,
    n_obs: usize,
    m: usize,
) -> DVector<f64> {
    let n_theta_stride = obs[0].df_dtheta.len();
    let mut mk = DVector::zeros(n_eta);
    for j in 0..n_obs {
        let bjm = obs[j].df_dtheta[m];
        for k in 0..n_eta {
            let b2 = obs[j].d2f_deta_dtheta[k * n_theta_stride + m];
            mk[k] += 0.5 * (et[j].alpha_p * bjm * obs[j].df_deta[k] + et[j].alpha * b2);
        }
    }
    mk
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::inner_optimizer::find_ebe;
    use crate::parser::model_parser::parse_model_string;
    use crate::stats::likelihood::{foce_subject_nll, foce_subject_nll_interaction};
    use crate::types::{DoseEvent, OmegaMatrix, Subject};
    use std::collections::HashMap;

    const TWOCPT: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV1(30.0, 3.0, 300.0)
  theta TVQ(2.0, 0.1, 20.0)
  theta TVV2(50.0, 5.0, 500.0)
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

    const THREECPT: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV1(10.0, 1.0, 100.0)
  theta TVQ2(2.0, 0.1, 20.0)
  theta TVV2(20.0, 2.0, 200.0)
  theta TVQ3(1.5, 0.1, 20.0)
  theta TVV3(30.0, 3.0, 300.0)
  theta TVKA(1.0, 0.05, 20.0)
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

    fn params_with_omega(model: &CompiledModel, theta: &[f64], vars: &[f64]) -> ModelParameters {
        let mut p = model.default_params.clone();
        p.theta = theta.to_vec();
        p.omega = OmegaMatrix::from_diagonal(vars, model.eta_names.clone());
        p
    }

    const WARFARIN: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

    fn subject_with_obs(model: &CompiledModel, theta: &[f64], times: &[f64]) -> Subject {
        // Build observations from the model at a reference eta so residuals are
        // realistic and nonzero (gradient identity holds at any obs).
        let n = times.len();
        let mut subject = Subject {
            id: "1".to_string(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: times.to_vec(),
            obs_raw_times: Vec::new(),
            observations: vec![0.0; n],
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
        };
        let eta_ref = [0.12, -0.08, 0.2];
        let preds = crate::pk::compute_predictions_with_tv(model, &subject, theta, &eta_ref);
        // Perturb by a fixed multiplicative factor so ε ≠ 0.
        subject.observations = preds.iter().map(|p| p * 0.85).collect();
        subject
    }

    /// Precisely locate η̂ via analytic Newton on the inner objective (exact
    /// gradient ½Σαⱼaⱼ + Ω⁻¹η and true Hessian H from the provider), so the
    /// marginal-NLL finite difference is not contaminated by inner-solver
    /// reconvergence noise. Warm-started from `find_ebe`.
    fn precise_ebe(model: &CompiledModel, subject: &Subject, params: &ModelParameters) -> Vec<f64> {
        let warm = find_ebe(model, subject, params, 80, 1e-10, None, None);
        let mut eta: Vec<f64> = warm.eta.iter().copied().collect();
        let n_eta = model.n_eta;
        let sigma = &params.sigma.values;
        let omega_inv = &params.omega.inv;
        for _ in 0..50 {
            let sens =
                crate::sens::provider::subject_sensitivities(model, subject, &params.theta, &eta)
                    .unwrap();
            let mut grad = nalgebra::DVector::<f64>::from_column_slice(
                &(omega_inv * nalgebra::DVector::from_column_slice(&eta))
                    .iter()
                    .copied()
                    .collect::<Vec<_>>(),
            );
            let mut hess = omega_inv.clone();
            let m3 = matches!(model.bloq_method, crate::types::BloqMethod::M3);
            for (j, obs) in sens.obs.iter().enumerate() {
                let f = obs.f;
                let cmt = subject.obs_cmts[j];
                let r = model.error_spec.variance_at(cmt, f, sigma);
                let d = model.error_spec.dvar_df(cmt, f, sigma);
                let d2 = model.error_spec.d2var_df2(cmt, sigma);
                let y = subject.observations[j];
                // (g1, g2) = (∂L/∂f, ∂²L/∂f²): the censored `−logΦ` scalars for an
                // M3 BLOQ row, else the Gaussian `½α`, `½α'`.
                let (g1, g2) = if m3 && subject.cens.get(j).copied().unwrap_or(0) != 0 {
                    m3_censored_scalars(y, f, r, d, d2)
                } else {
                    let t = err_terms(r, d, d2, y - f);
                    (0.5 * t.alpha, 0.5 * t.alpha_p)
                };
                let a = obs.df_deta.as_slice();
                for k in 0..n_eta {
                    grad[k] += g1 * a[k];
                    for l in 0..n_eta {
                        hess[(k, l)] += g2 * a[k] * a[l] + g1 * obs.d2f_deta2[k * n_eta + l];
                    }
                }
            }
            let step = hess.cholesky().unwrap().solve(&grad);
            for k in 0..n_eta {
                eta[k] -= step[k];
            }
            if step.norm() < 1e-13 {
                break;
            }
        }
        eta
    }

    /// Per-subject Laplace NLL Fᵢ at a *given* η̂ (no reconvergence).
    fn marginal_nll_at(
        model: &CompiledModel,
        subject: &Subject,
        params: &ModelParameters,
        eta: &[f64],
    ) -> f64 {
        let eta_v = nalgebra::DVector::from_column_slice(eta);
        let ipreds = crate::pk::compute_predictions_with_tv(model, subject, &params.theta, eta);
        let jac = crate::sens::provider::subject_eta_jacobian(model, subject, &params.theta, eta)
            .unwrap();
        let h_matrix =
            nalgebra::DMatrix::from_row_slice(subject.obs_times.len(), model.n_eta, &jac);
        foce_subject_nll_interaction(
            subject,
            &ipreds,
            &eta_v,
            &h_matrix,
            &params.omega,
            &params.sigma.values,
            &model.error_spec,
            model.bloq_method,
            &[],
        )
    }

    /// Reconverged marginal NLL Fᵢ(θ) at the precisely-located EBE.
    fn marginal_nll(model: &CompiledModel, subject: &Subject, params: &ModelParameters) -> f64 {
        let eta = precise_ebe(model, subject, params);
        marginal_nll_at(model, subject, params, &eta)
    }

    /// Per-subject **FOCE** (non-interaction) marginal NLL at a given η̂ — ferx's
    /// Sheiner–Beal linearized objective via `foce_subject_nll(.., interaction=false)`.
    fn marginal_nll_foce_at(
        model: &CompiledModel,
        subject: &Subject,
        params: &ModelParameters,
        eta: &[f64],
    ) -> f64 {
        let eta_v = nalgebra::DVector::from_column_slice(eta);
        let jac = crate::sens::provider::subject_eta_jacobian(model, subject, &params.theta, eta)
            .unwrap();
        let h_matrix =
            nalgebra::DMatrix::from_row_slice(subject.obs_times.len(), model.n_eta, &jac);
        foce_subject_nll(
            model,
            subject,
            &params.theta,
            &eta_v,
            &h_matrix,
            &params.omega,
            &params.sigma.values,
            false,
        )
    }

    /// Reconverged FOCE marginal NLL at the precisely-located (shared) EBE.
    fn marginal_nll_foce(
        model: &CompiledModel,
        subject: &Subject,
        params: &ModelParameters,
    ) -> f64 {
        let eta = precise_ebe(model, subject, params);
        marginal_nll_foce_at(model, subject, params, &eta)
    }

    /// FOCE analog of [`run_population_packed_gradient_check`]: the analytic FOCE
    /// packed gradient must match the reconverged-FD of ferx's FOCE OFV.
    fn run_packed_check_foce(model: &CompiledModel, theta: &[f64]) {
        use crate::estimation::parameterization::pack_params;
        use crate::types::Population;

        let s1 = subject_with_obs(model, theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let s2 = subject_with_obs(model, theta, &[0.25, 1.5, 3.0, 6.0, 12.0, 36.0, 72.0]);
        let pop = Population {
            subjects: vec![s1, s2],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let ehs: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|s| DVector::from_vec(precise_ebe(model, s, &params)))
            .collect();

        let analytic =
            population_gradient_sens_foce(model, &pop, &template, &x, &ehs).expect("supported");

        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            2.0 * pop
                .subjects
                .iter()
                .map(|s| marginal_nll_foce(model, s, &p))
                .sum::<f64>()
        };
        let fd_at = |k: usize, h: f64| -> f64 {
            let mut xp = x.clone();
            xp[k] += h;
            let mut xm = x.clone();
            xm[k] -= h;
            (ofv(&xp) - ofv(&xm)) / (2.0 * h)
        };
        for k in 0..x.len() {
            let h = 1e-4 * (1.0 + x[k].abs());
            let f1 = fd_at(k, h);
            let f2 = fd_at(k, h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0;
            eprintln!(
                "x[{k}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[k],
                fd,
                (analytic[k] - fd).abs() / fd.abs().max(1e-12)
            );
            approx::assert_relative_eq!(analytic[k], fd, max_relative = 3e-3, epsilon = 1e-5);
        }
    }

    #[test]
    fn theta_gradient_matches_reconverged_fd() {
        let model = parse_model_string(WARFARIN).expect("parse");
        let theta = vec![0.22, 11.0, 1.4];
        let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
        let subject = subject_with_obs(&model, &theta, &times);

        let mut params = model.default_params.clone();
        params.theta = theta.clone();

        // Precisely-located EBE at the base point.
        let eta_hat = precise_ebe(&model, &subject, &params);

        let analytic =
            subject_theta_gradient(&model, &subject, &params, &eta_hat).expect("supported");

        // Richardson-extrapolated reconverged central FD of the marginal NLL
        // (cancels the O(h²) truncation; EBE is located analytically so there is
        // no inner-solver noise floor).
        let fd_at = |m: usize, h: f64| -> f64 {
            let mut pp = params.clone();
            pp.theta[m] += h;
            let mut pm = params.clone();
            pm.theta[m] -= h;
            (marginal_nll(&model, &subject, &pp) - marginal_nll(&model, &subject, &pm)) / (2.0 * h)
        };
        let n_theta = theta.len();
        for m in 0..n_theta {
            let h = 1e-4 * (1.0 + theta[m].abs());
            let f1 = fd_at(m, h);
            let f2 = fd_at(m, h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0; // Richardson
            eprintln!(
                "theta[{m}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[m],
                fd,
                (analytic[m] - fd).abs() / fd.abs().max(1e-12)
            );
            approx::assert_relative_eq!(analytic[m], fd, max_relative = 1e-3, epsilon = 1e-6);
        }
    }

    #[test]
    fn omega_gradient_matches_reconverged_fd() {
        let model = parse_model_string(WARFARIN).expect("parse");
        let theta = vec![0.22, 11.0, 1.4];
        let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
        let subject = subject_with_obs(&model, &theta, &times);

        let vars = vec![0.09, 0.04, 0.30];
        let params = params_with_omega(&model, &theta, &vars);
        let eta_hat = precise_ebe(&model, &subject, &params);

        let analytic =
            subject_omega_gradient(&model, &subject, &params, &eta_hat).expect("supported");

        // Richardson reconverged FD over each natural variance entry.
        let fd_at = |i: usize, h: f64| -> f64 {
            let mut vp = vars.clone();
            vp[i] += h;
            let mut vm = vars.clone();
            vm[i] -= h;
            let pp = params_with_omega(&model, &theta, &vp);
            let pm = params_with_omega(&model, &theta, &vm);
            (marginal_nll(&model, &subject, &pp) - marginal_nll(&model, &subject, &pm)) / (2.0 * h)
        };
        for i in 0..vars.len() {
            let h = 1e-4 * (1.0 + vars[i].abs());
            let f1 = fd_at(i, h);
            let f2 = fd_at(i, h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0;
            eprintln!(
                "omega[{i}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[i],
                fd,
                (analytic[i] - fd).abs() / fd.abs().max(1e-12)
            );
            approx::assert_relative_eq!(analytic[i], fd, max_relative = 1e-3, epsilon = 1e-6);
        }
    }

    #[test]
    fn sigma_gradient_matches_reconverged_fd() {
        let model = parse_model_string(WARFARIN).expect("parse");
        let theta = vec![0.22, 11.0, 1.4];
        let times = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
        let subject = subject_with_obs(&model, &theta, &times);

        let mut params = model.default_params.clone();
        params.theta = theta.clone();
        let eta_hat = precise_ebe(&model, &subject, &params);

        let analytic =
            subject_sigma_gradient(&model, &subject, &params, &eta_hat).expect("supported");

        let sig0 = params.sigma.values.clone();
        let fd_at = |k: usize, h: f64| -> f64 {
            let mut pp = params.clone();
            pp.sigma.values[k] += h;
            let mut pm = params.clone();
            pm.sigma.values[k] -= h;
            (marginal_nll(&model, &subject, &pp) - marginal_nll(&model, &subject, &pm)) / (2.0 * h)
        };
        for k in 0..sig0.len() {
            let h = 1e-4 * (1.0 + sig0[k].abs());
            let f1 = fd_at(k, h);
            let f2 = fd_at(k, h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0;
            eprintln!(
                "sigma[{k}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[k],
                fd,
                (analytic[k] - fd).abs() / fd.abs().max(1e-12)
            );
            approx::assert_relative_eq!(analytic[k], fd, max_relative = 2e-3, epsilon = 1e-6);
        }
    }

    #[test]
    fn eta_dx_matches_fd() {
        use crate::estimation::parameterization::pack_params;
        let model = parse_model_string(WARFARIN).expect("parse");
        let theta = vec![0.22, 11.0, 1.4];
        let subject = subject_with_obs(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let mut template = model.default_params.clone();
        template.theta = theta.clone();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let eta_hat = precise_ebe(&model, &subject, &params);

        let jac = subject_eta_dx(&model, &subject, &template, &x, &eta_hat).expect("supported");
        let n_eta = model.n_eta;
        for k in 0..x.len() {
            let h = 1e-5 * (1.0 + x[k].abs());
            let mut xp = x.clone();
            xp[k] += h;
            let mut xm = x.clone();
            xm[k] -= h;
            let ep = precise_ebe(&model, &subject, &unpack_params(&xp, &template));
            let em = precise_ebe(&model, &subject, &unpack_params(&xm, &template));
            for l in 0..n_eta {
                let fd = (ep[l] - em[l]) / (2.0 * h);
                approx::assert_relative_eq!(jac[k][l], fd, max_relative = 2e-3, epsilon = 1e-6);
            }
        }
    }

    fn run_population_packed_gradient_check(model: &CompiledModel, theta: &[f64]) {
        use crate::estimation::parameterization::pack_params;
        use crate::types::Population;

        let s1 = subject_with_obs(model, theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let s2 = subject_with_obs(model, theta, &[0.25, 1.5, 3.0, 6.0, 12.0, 36.0, 72.0]);
        let pop = Population {
            subjects: vec![s1, s2],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        let x = pack_params(&template);

        let params = unpack_params(&x, &template);
        let ehs: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|s| DVector::from_vec(precise_ebe(model, s, &params)))
            .collect();

        let analytic =
            population_gradient_sens(model, &pop, &template, &x, &ehs).expect("supported");

        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            2.0 * pop
                .subjects
                .iter()
                .map(|s| marginal_nll(model, s, &p))
                .sum::<f64>()
        };
        let fd_at = |k: usize, h: f64| -> f64 {
            let mut xp = x.clone();
            xp[k] += h;
            let mut xm = x.clone();
            xm[k] -= h;
            (ofv(&xp) - ofv(&xm)) / (2.0 * h)
        };
        for k in 0..x.len() {
            let h = 1e-4 * (1.0 + x[k].abs());
            let f1 = fd_at(k, h);
            let f2 = fd_at(k, h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0;
            eprintln!(
                "x[{k}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[k],
                fd,
                (analytic[k] - fd).abs() / fd.abs().max(1e-12)
            );
            approx::assert_relative_eq!(analytic[k], fd, max_relative = 3e-3, epsilon = 1e-5);
        }
    }

    #[test]
    fn population_packed_gradient_matches_reconverged_fd() {
        let model = parse_model_string(WARFARIN).expect("parse");
        run_population_packed_gradient_check(&model, &[0.22, 11.0, 1.4]);
    }

    #[test]
    fn population_packed_gradient_2cpt_matches_fd() {
        let model = parse_model_string(TWOCPT).expect("parse");
        run_population_packed_gradient_check(&model, &[5.0, 30.0, 2.0, 50.0, 1.0]);
    }

    /// The analytic FOCEI **M3** packed gradient (censored rows enter `prepare`'s
    /// M3 branch: data `−logΦ` + true-inner-Hessian, excluded from `H̃`) must match
    /// the reconverged-FD of ferx's M3 FOCEI objective (`foce_subject_nll_interaction`
    /// with `bloq_term`). Each subject carries both quantified and censored rows.
    #[test]
    fn population_packed_gradient_m3_matches_fd() {
        use crate::estimation::parameterization::pack_params;
        use crate::types::{BloqMethod, Population};

        let mut model = parse_model_string(WARFARIN).expect("parse");
        model.bloq_method = BloqMethod::M3;
        let theta = [0.22, 11.0, 1.4];

        // Build subjects, then mark the last two observations of each as censored
        // (CENS=1, the obs cell carries the LLOQ) so every subject mixes quantified
        // and BLOQ rows. Leaves z moderate (LLOQ ≈ 0.85·f_ref), away from the tail.
        let mut s1 = subject_with_obs(&model, &theta, &[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        let mut s2 = subject_with_obs(&model, &theta, &[0.25, 1.5, 3.0, 6.0, 12.0, 36.0, 72.0]);
        for s in [&mut s1, &mut s2] {
            let n = s.observations.len();
            s.cens[n - 1] = 1;
            s.cens[n - 2] = 1;
        }
        assert!(s1.cens.iter().any(|&c| c != 0) && s2.cens.iter().any(|&c| c != 0));

        let pop = Population {
            subjects: vec![s1, s2],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let mut template = model.default_params.clone();
        template.theta = theta.to_vec();
        let x = pack_params(&template);
        let params = unpack_params(&x, &template);
        let ehs: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
            .collect();

        let analytic =
            population_gradient_sens(&model, &pop, &template, &x, &ehs).expect("M3 supported");

        // marginal_nll uses foce_subject_nll_interaction with model.bloq_method = M3,
        // so the OFV carries the censored −2logΦ term; precise_ebe is M3-aware.
        let ofv = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, &template);
            2.0 * pop
                .subjects
                .iter()
                .map(|s| marginal_nll(&model, s, &p))
                .sum::<f64>()
        };
        let fd_at = |k: usize, h: f64| -> f64 {
            let mut xp = x.clone();
            xp[k] += h;
            let mut xm = x.clone();
            xm[k] -= h;
            (ofv(&xp) - ofv(&xm)) / (2.0 * h)
        };
        for k in 0..x.len() {
            let h = 1e-4 * (1.0 + x[k].abs());
            let f1 = fd_at(k, h);
            let f2 = fd_at(k, h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0;
            eprintln!(
                "x[{k}]: analytic={:.8}  fd={:.8}  rel={:.2e}",
                analytic[k],
                fd,
                (analytic[k] - fd).abs() / fd.abs().max(1e-12)
            );
            approx::assert_relative_eq!(analytic[k], fd, max_relative = 5e-3, epsilon = 1e-5);
        }
    }

    #[test]
    fn population_packed_gradient_3cpt_matches_fd() {
        let model = parse_model_string(THREECPT).expect("parse");
        run_population_packed_gradient_check(&model, &[5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 1.0]);
    }

    // --- FOCE (non-interaction, Sheiner–Beal linearized marginal) ---

    #[test]
    fn population_packed_gradient_foce_matches_fd() {
        let model = parse_model_string(WARFARIN).expect("parse");
        run_packed_check_foce(&model, &[0.22, 11.0, 1.4]);
    }

    #[test]
    fn population_packed_gradient_foce_2cpt_matches_fd() {
        let model = parse_model_string(TWOCPT).expect("parse");
        run_packed_check_foce(&model, &[5.0, 30.0, 2.0, 50.0, 1.0]);
    }

    #[test]
    fn population_packed_gradient_foce_3cpt_matches_fd() {
        let model = parse_model_string(THREECPT).expect("parse");
        run_packed_check_foce(&model, &[5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 1.0]);
    }

    // --- Eq. 48 EBE warm-start predictor: is it correct & better than plain warm? ---

    /// The Eq. 48 predictor `η⁰ = η̂_prev + (dη̂/dx)·Δx` is a first-order Taylor
    /// extrapolation of the EBE as the packed parameters move x_prev → x_new. So
    /// against the *converged* EBE at x_new it must beat the plain warm-start
    /// (reuse η̂_prev): the prediction error is `O(‖Δx‖²)` while the warm-start
    /// error is `O(‖Δx‖)`. This walks several step sizes in a representative
    /// direction and checks (a) prediction strictly beats warm for small steps,
    /// and (b) the prediction/warm error ratio shrinks ∝ ‖Δx‖ (second order).
    #[test]
    fn eta_predictor_beats_warm_start() {
        use crate::estimation::parameterization::pack_params;

        let model = parse_model_string(TWOCPT).expect("parse");
        let theta = vec![5.0, 30.0, 2.0, 50.0, 1.0];
        let times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0];
        let subjects = [
            subject_with_obs(&model, &theta, &times),
            subject_with_obs(&model, &theta, &[0.5, 1.5, 3.0, 6.0, 12.0, 36.0]),
        ];

        let mut template = model.default_params.clone();
        template.theta = theta.clone();
        let x0 = pack_params(&template);
        let n = x0.len();

        // A fixed, representative outer direction (unit-norm in packed space).
        let mut dir: Vec<f64> = (0..n).map(|k| 0.5 + 0.1 * k as f64).collect();
        let dnorm = dir.iter().map(|d| d * d).sum::<f64>().sqrt();
        for d in dir.iter_mut() {
            *d /= dnorm;
        }

        // Base EBEs and dη̂/dx at x0.
        let p0 = unpack_params(&x0, &template);
        let eta0: Vec<DVector<f64>> = subjects
            .iter()
            .map(|s| DVector::from_vec(precise_ebe(&model, s, &p0)))
            .collect();
        let jac: Vec<Vec<DVector<f64>>> = subjects
            .iter()
            .enumerate()
            .map(|(i, s)| subject_eta_dx(&model, s, &template, &x0, eta0[i].as_slice()).unwrap())
            .collect();

        eprintln!("  step    warm_err   pred_err   ratio");
        let mut prev_ratio: Option<f64> = None;
        for &s in &[0.20_f64, 0.10, 0.05, 0.025] {
            let x1: Vec<f64> = (0..n).map(|k| x0[k] + s * dir[k]).collect();
            let p1 = unpack_params(&x1, &template);

            let pred = predict_warm_etas(&eta0, &jac, &x0, &x1);

            let mut warm_err = 0.0;
            let mut pred_err = 0.0;
            for (i, subj) in subjects.iter().enumerate() {
                let eta1 = DVector::from_vec(precise_ebe(&model, subj, &p1));
                warm_err += (&eta0[i] - &eta1).norm();
                pred_err += (&pred[i] - &eta1).norm();
            }
            let ratio = pred_err / warm_err.max(1e-300);
            eprintln!("  {s:>5.3}  {warm_err:>9.2e}  {pred_err:>9.2e}  {ratio:>6.3}");

            // (a) the predictor must be a real improvement on warm-start.
            assert!(
                pred_err < 0.5 * warm_err,
                "predictor (err {pred_err:.3e}) should beat warm-start (err {warm_err:.3e}) at step {s}"
            );
            // (b) halving the step should shrink the ratio (second-order error).
            if let Some(pr) = prev_ratio {
                assert!(
                    ratio < pr + 1e-9,
                    "pred/warm ratio should not grow as the step shrinks ({ratio:.3} vs {pr:.3})"
                );
            }
            prev_ratio = Some(ratio);
        }
    }
}
