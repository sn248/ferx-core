//! One-compartment analytic sensitivities via `Dual2` (issue #367 clean slate).
//!
//! The closed-form solutions are written **once**, generic over [`PkNum`]:
//! evaluating in `f64` gives the prediction, evaluating in `Dual2<N>` (seeded on
//! the PK parameters) gives `f`, `∂f/∂pk`, `∂²f/∂pk²` exactly, no hand-derivation.
//! The `*_hardcoded` form is kept only as an independent cross-check in tests.

use super::dual2::Dual2;
use super::num::PkNum;
use crate::types::DoseEvent;

// ── Generic closed-form single-dose solutions ────────────────────────────────

/// 1-cpt IV bolus: `C(t) = (D/V)·exp(−(CL/V)·t)`.
pub fn one_cpt_iv_bolus_g<T: PkNum>(amt: f64, t: f64, cl: T, v: T) -> T {
    if t < 0.0 || v.val() <= 0.0 || cl.val() <= 0.0 {
        return T::from_f64(0.0);
    }
    let k = cl / v;
    (T::from_f64(amt) / v) * (-(k * T::from_f64(t))).exp()
}

/// 1-cpt oral (first-order absorption), with the `KA ≈ k` L'Hôpital limit.
pub fn one_cpt_oral_g<T: PkNum>(amt: f64, t: f64, cl: T, v: T, ka: T, f_bio: T) -> T {
    if t < 0.0 || v.val() <= 0.0 || cl.val() <= 0.0 || ka.val() <= 0.0 {
        return T::from_f64(0.0);
    }
    let k = cl / v;
    let d = f_bio * T::from_f64(amt);
    let tt = T::from_f64(t);
    if (ka.val() - k.val()).abs() < 1e-6 {
        (d * ka / v) * tt * (-(k * tt)).exp()
    } else {
        (d * ka / (v * (ka - k))) * ((-(k * tt)).exp() - (-(ka * tt)).exp())
    }
}

/// 1-cpt infusion (zero-order input of duration `dur`, rate `rate`).
pub fn one_cpt_infusion_g<T: PkNum>(rate: f64, dur: f64, amt: f64, t: f64, cl: T, v: T) -> T {
    if t < 0.0 || v.val() <= 0.0 || cl.val() <= 0.0 {
        return T::from_f64(0.0);
    }
    if dur <= 0.0 {
        return one_cpt_iv_bolus_g(amt, t, cl, v);
    }
    let k = cl / v;
    let r_cl = T::from_f64(rate) / cl;
    let one = T::from_f64(1.0);
    if t <= dur {
        r_cl * (one - (-(k * T::from_f64(t))).exp())
    } else {
        r_cl * (one - (-(k * T::from_f64(dur))).exp()) * (-(k * T::from_f64(t - dur))).exp()
    }
}

/// 1-cpt IV bolus at steady state (interval `ii`).
pub fn one_cpt_iv_bolus_ss_g<T: PkNum>(amt: f64, t: f64, ii: f64, cl: T, v: T) -> T {
    if t < 0.0 || v.val() <= 0.0 || cl.val() <= 0.0 || ii <= 0.0 {
        return T::from_f64(0.0);
    }
    let k = cl / v;
    let denom = T::from_f64(1.0) - (-(k * T::from_f64(ii))).exp();
    if denom.val() <= 0.0 {
        return T::from_f64(0.0);
    }
    (T::from_f64(amt) / v) * (-(k * T::from_f64(t))).exp() / denom
}

/// 1-cpt oral at steady state (interval `ii`), with the `KA ≈ k` L'Hôpital limit.
pub fn one_cpt_oral_ss_g<T: PkNum>(amt: f64, t: f64, ii: f64, cl: T, v: T, ka: T, f_bio: T) -> T {
    if t < 0.0 || v.val() <= 0.0 || cl.val() <= 0.0 || ka.val() <= 0.0 || ii <= 0.0 {
        return T::from_f64(0.0);
    }
    let k = cl / v;
    let d = f_bio * T::from_f64(amt);
    let tt = T::from_f64(t);
    if (ka.val() - k.val()).abs() < 1e-6 {
        // Σ_{n≥0} (τ+nII)·e^{-k(τ+nII)} = e^{-kτ}·[τ/(1-x) + II·x/(1-x)²], x=e^{-kII}.
        let x = (-(k * T::from_f64(ii))).exp();
        let omx = T::from_f64(1.0) - x;
        if omx.val() <= 0.0 {
            return T::from_f64(0.0);
        }
        let s = tt / omx + (x * T::from_f64(ii)) / (omx * omx);
        (d * ka / v) * (-(k * tt)).exp() * s
    } else {
        let denom_k = T::from_f64(1.0) - (-(k * T::from_f64(ii))).exp();
        let denom_ka = T::from_f64(1.0) - (-(ka * T::from_f64(ii))).exp();
        if denom_k.val() <= 0.0 || denom_ka.val() <= 0.0 {
            return T::from_f64(0.0);
        }
        (d * ka / (v * (ka - k))) * ((-(k * tt)).exp() / denom_k - (-(ka * tt)).exp() / denom_ka)
    }
}

/// 1-cpt infusion at steady state (interval `ii`). Closed form requires
/// `dur ≤ ii` (non-overlapping); returns 0 otherwise (matches the production
/// `one_cpt_infusion_ss`, which routes the overlapping case to the ODE solver).
pub fn one_cpt_infusion_ss_g<T: PkNum>(
    rate: f64,
    dur: f64,
    amt: f64,
    t: f64,
    ii: f64,
    cl: T,
    v: T,
) -> T {
    if t < 0.0 || v.val() <= 0.0 || cl.val() <= 0.0 || ii <= 0.0 {
        return T::from_f64(0.0);
    }
    if dur <= 0.0 {
        return one_cpt_iv_bolus_ss_g(amt, t, ii, cl, v);
    }
    if dur > ii {
        return T::from_f64(0.0);
    }
    let k = cl / v;
    let denom = T::from_f64(1.0) - (-(k * T::from_f64(ii))).exp();
    if denom.val() <= 0.0 {
        return T::from_f64(0.0);
    }
    let r_over_cl = T::from_f64(rate) / cl;
    let one_minus_e_kt_inf = T::from_f64(1.0) - (-(k * T::from_f64(dur))).exp();
    // Past pulses (n ≥ 1) are always "after-infusion".
    let past = r_over_cl * one_minus_e_kt_inf * (-(k * T::from_f64(t + ii - dur))).exp() / denom;
    if t <= dur {
        r_over_cl * (T::from_f64(1.0) - (-(k * T::from_f64(t))).exp()) + past
    } else {
        r_over_cl * one_minus_e_kt_inf * (-(k * T::from_f64(t - dur))).exp() / denom
    }
}

/// Concentration contribution of a single dose at elapsed time `t` since the
/// dose, dispatching bolus / infusion / oral and their steady-state variants —
/// the generic counterpart to [`crate::pk::one_cpt_predict`]. `oral` selects the
/// absorption route (the PK model is per-subject-static, so this is a flag, not
/// per-dose). `cl`/`v`/`ka`/`f_bio` are the seeded PK params.
pub fn one_cpt_conc_g<T: PkNum>(
    dose: &DoseEvent,
    t: f64,
    cl: T,
    v: T,
    ka: T,
    f_bio: T,
    oral: bool,
) -> T {
    if dose.ss && dose.ii > 0.0 {
        if dose.is_infusion() {
            one_cpt_infusion_ss_g(dose.rate, dose.duration, dose.amt, t, dose.ii, cl, v)
        } else if oral {
            one_cpt_oral_ss_g(dose.amt, t, dose.ii, cl, v, ka, f_bio)
        } else {
            one_cpt_iv_bolus_ss_g(dose.amt, t, dose.ii, cl, v)
        }
    } else if dose.is_infusion() {
        one_cpt_infusion_g(dose.rate, dose.duration, dose.amt, t, cl, v)
    } else if oral {
        one_cpt_oral_g(dose.amt, t, cl, v, ka, f_bio)
    } else {
        one_cpt_iv_bolus_g(dose.amt, t, cl, v)
    }
}

// ── Sensitivity extraction (seed the active PK params as Dual2 variables) ─────

/// `(f, ∂f/∂[CL,V], ∂²f/∂[CL,V]²)` for the IV bolus.
pub fn iv_bolus_sens(amt: f64, t: f64, cl: f64, v: f64) -> (f64, [f64; 2], [[f64; 2]; 2]) {
    let f = one_cpt_iv_bolus_g::<Dual2<2>>(amt, t, Dual2::var(cl, 0), Dual2::var(v, 1));
    (f.value, f.grad, f.hess)
}

/// `(f, ∂f/∂[CL,V,KA,F], ∂²f/∂[CL,V,KA,F]²)` for oral.
pub fn oral_sens(
    amt: f64,
    t: f64,
    cl: f64,
    v: f64,
    ka: f64,
    f_bio: f64,
) -> (f64, [f64; 4], [[f64; 4]; 4]) {
    let f = one_cpt_oral_g::<Dual2<4>>(
        amt,
        t,
        Dual2::var(cl, 0),
        Dual2::var(v, 1),
        Dual2::var(ka, 2),
        Dual2::var(f_bio, 3),
    );
    (f.value, f.grad, f.hess)
}

/// Independent hand-derived IV-bolus derivatives, used only to cross-check the
/// dual path in tests (`k=CL/V`, `f=(D/V)e^{−kt}`).
#[cfg(test)]
fn iv_bolus_hardcoded(amt: f64, t: f64, cl: f64, v: f64) -> (f64, [f64; 2], [[f64; 2]; 2]) {
    if t < 0.0 || v <= 0.0 || cl <= 0.0 {
        return (0.0, [0.0; 2], [[0.0; 2]; 2]);
    }
    let k = cl / v;
    let f = (amt / v) * (-k * t).exp();
    let v2 = v * v;
    let grad = [f * (-t / v), f * (k * t - 1.0) / v];
    let hess = [
        [f * t * t / v2, f * t * (2.0 - k * t) / v2],
        [
            f * t * (2.0 - k * t) / v2,
            f * (k * k * t * t - 4.0 * k * t + 2.0) / v2,
        ],
    ];
    (f, grad, hess)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Central FD grad + 4-point Hessian of an `N`-arg value closure.
    fn fd<const N: usize>(p: [f64; N], val: impl Fn([f64; N]) -> f64) -> ([f64; N], [[f64; N]; N]) {
        let h: [f64; N] = std::array::from_fn(|i| 1e-5 * (1.0 + p[i].abs()));
        let bump = |d: [f64; N]| {
            let mut q = p;
            for i in 0..N {
                q[i] += d[i];
            }
            val(q)
        };
        let mut g = [0.0; N];
        let mut hh = [[0.0; N]; N];
        for i in 0..N {
            let mut u = [0.0; N];
            u[i] = h[i];
            let mut dn = [0.0; N];
            dn[i] = -h[i];
            g[i] = (bump(u) - bump(dn)) / (2.0 * h[i]);
        }
        for i in 0..N {
            for j in 0..N {
                let mut pp = [0.0; N];
                pp[i] += h[i];
                pp[j] += h[j];
                let mut pm = [0.0; N];
                pm[i] += h[i];
                pm[j] -= h[j];
                let mut mp = [0.0; N];
                mp[i] -= h[i];
                mp[j] += h[j];
                let mut mm = [0.0; N];
                mm[i] -= h[i];
                mm[j] -= h[j];
                hh[i][j] = (bump(pp) - bump(pm) - bump(mp) + bump(mm)) / (4.0 * h[i] * h[j]);
            }
        }
        (g, hh)
    }

    #[test]
    fn iv_bolus_dual_matches_hardcoded_and_fd() {
        for &(amt, t, cl, v) in &[(100.0, 0.5, 3.0, 30.0), (50.0, 9.0, 5.0, 40.0)] {
            let (fd_v, gd, hd) = iv_bolus_sens(amt, t, cl, v);
            let (fh, gh, hh) = iv_bolus_hardcoded(amt, t, cl, v);
            approx::assert_relative_eq!(fd_v, fh, max_relative = 1e-12);
            for i in 0..2 {
                approx::assert_relative_eq!(gd[i], gh[i], max_relative = 1e-10);
                for j in 0..2 {
                    approx::assert_relative_eq!(hd[i][j], hh[i][j], max_relative = 1e-10);
                }
            }
        }
    }

    #[test]
    fn oral_dual_matches_fd() {
        let (amt, t, cl, v, ka, fb) = (100.0, 2.0, 1.2, 12.0, 0.8, 0.9);
        let (_, gd, hd) = oral_sens(amt, t, cl, v, ka, fb);
        let (gfd, hfd) = fd([cl, v, ka, fb], |p| {
            one_cpt_oral_g::<f64>(amt, t, p[0], p[1], p[2], p[3])
        });
        for i in 0..4 {
            approx::assert_relative_eq!(gd[i], gfd[i], max_relative = 1e-4, epsilon = 1e-9);
            for j in 0..4 {
                approx::assert_relative_eq!(
                    hd[i][j],
                    hfd[i][j],
                    max_relative = 3e-3,
                    epsilon = 1e-8
                );
            }
        }
    }

    /// Overhead of value + gradient + Hessian (`Dual2`) vs the bare `f64` value.
    #[test]
    #[ignore = "bench: run with -- --ignored --nocapture"]
    fn dual2_overhead() {
        use std::time::Instant;
        let n = 5_000_000u64;
        let (amt, cl, v, ka, fb) = (100.0, 1.2, 12.0, 0.8, 0.9);

        let t0 = Instant::now();
        let mut a = 0.0;
        for i in 0..n {
            let t = (i % 24) as f64 * 0.5;
            a += one_cpt_oral_g::<f64>(amt, t, cl, v, ka, fb);
        }
        let f64_ns = t0.elapsed().as_nanos() as f64 / n as f64;

        let t1 = Instant::now();
        let mut b = 0.0;
        for i in 0..n {
            let t = (i % 24) as f64 * 0.5;
            let (f, _, _) = oral_sens(amt, t, cl, v, ka, fb);
            b += f;
        }
        let dual_ns = t1.elapsed().as_nanos() as f64 / n as f64;

        eprintln!(
            "1-cpt oral (4 params): f64 value = {f64_ns:.1} ns; Dual2<4> value+grad+hess = {dual_ns:.1} ns; overhead = {:.1}x",
            dual_ns / f64_ns
        );
        std::hint::black_box((a, b));
    }
}
