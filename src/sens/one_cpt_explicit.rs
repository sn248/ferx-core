//! Option B (explicit symbolic derivatives) for the 1-cpt analytical solutions.
//!
//! Where [`super::one_cpt`] obtains `f`, `∂f/∂pk`, `∂²f/∂pk²` by evaluating the
//! closed form over [`Dual2`](super::dual2::Dual2) (generic forward 2nd-order,
//! `O(N²)` per op), this module writes the derivatives out by hand in scalar
//! `f64`. It computes only the entries that exist — no width padding, no dual
//! bookkeeping — so it is the speed ceiling for the per-observation sensitivity
//! kernel. The Dual2 path is the correctness oracle: every function here is
//! checked against it to ~1e-10 in tests.
//!
//! Scope: the common single-dose 1-cpt models (IV bolus, oral). Genuinely
//! awkward branches (the `ka ≈ CL/V` L'Hôpital limit) fall back to the dual
//! path — a measure-zero case where hand-derivation buys nothing.

use super::dual2::Dual2;
use super::one_cpt::one_cpt_oral_g;

/// `(f, ∂f/∂[CL,V], ∂²f/∂[CL,V]²)` for the 1-cpt IV bolus `C=(D/V)e^{−(CL/V)t}`.
pub fn iv_bolus_explicit(amt: f64, t: f64, cl: f64, v: f64) -> (f64, [f64; 2], [[f64; 2]; 2]) {
    if t < 0.0 || v <= 0.0 || cl <= 0.0 {
        return (0.0, [0.0; 2], [[0.0; 2]; 2]);
    }
    let k = cl / v;
    let f = (amt / v) * (-k * t).exp();
    let v2 = v * v;
    // ∂f/∂CL = f·(−t/V); ∂f/∂V = f·(kt−1)/V.
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

/// `(f, ∂f/∂[CL,V,KA,F], ∂²f/∂[CL,V,KA,F]²)` for 1-cpt oral.
///
/// With `D = amt`, `k = CL/V`, `Δ = ka − k`, `S = e^{−kt} − e^{−ka·t}`, the
/// `F = 1` response is `g = h/V`, `h = D·ka·S/Δ`. F factors out linearly
/// (`f = F·g`), and CL/V enter only through `k` (plus the explicit `1/V` in
/// `g`), so the whole 4×4 reduces to `h` and its `k`/`ka` derivatives chained by
/// `k_CL = 1/V`, `k_V = −k/V`. The `ka ≈ k` limit routes to the dual path.
pub fn oral_explicit(
    amt: f64,
    t: f64,
    cl: f64,
    v: f64,
    ka: f64,
    f_bio: f64,
) -> (f64, [f64; 4], [[f64; 4]; 4]) {
    if t < 0.0 || v <= 0.0 || cl <= 0.0 || ka <= 0.0 {
        return (0.0, [0.0; 4], [[0.0; 4]; 4]);
    }
    let k = cl / v;
    if (ka - k).abs() < 1e-6 {
        // L'Hôpital limit — rare; let the dual path handle it exactly.
        let d = one_cpt_oral_g::<Dual2<4>>(
            amt,
            t,
            Dual2::var(cl, 0),
            Dual2::var(v, 1),
            Dual2::var(ka, 2),
            Dual2::var(f_bio, 3),
        );
        return (d.value, d.grad, d.hess);
    }

    let dd = amt; // D (F factored out below)
    let ek = (-k * t).exp();
    let eka = (-ka * t).exp();
    let s = ek - eka;
    let s_k = -t * ek; // ∂S/∂k
    let s_ka = t * eka; // ∂S/∂ka
    let s_kk = t * t * ek; // ∂²S/∂k²
    let s_kaka = -t * t * eka; // ∂²S/∂ka²
    let delta = ka - k;
    let d2 = delta * delta;
    let d3 = d2 * delta;

    // a = S_k·Δ + S ; b = S_ka·Δ − S ; a_ka = S_k + S_ka.
    let a = s_k * delta + s;
    let b = s_ka * delta - s;
    let a_ka = s_k + s_ka;

    // h = D·ka·S/Δ and its k / ka derivatives.
    let h = dd * ka * s / delta;
    let h_k = dd * ka * a / d2;
    let h_ka = dd * (s / delta + ka * b / d2);
    let h_kk = dd * ka * (s_kk * d2 + 2.0 * a) / d3;
    let h_kka = dd * (a / d2 + ka * (a_ka * delta - 2.0 * a) / d3);
    let h_kaka = dd * (2.0 * b / d2 + ka * (s_kaka * d2 - 2.0 * b) / d3);

    let v2 = v * v;
    let v3 = v2 * v;

    // g = h/V, gradient + Hessian over [CL, V, KA] (k_CL = 1/V, k_V = −k/V).
    let g = h / v;
    let g_cl = h_k / v2;
    let g_v = -(k * h_k + h) / v2;
    let g_ka = h_ka / v;
    let g_clcl = h_kk / v3;
    let g_clv = -(k * h_kk + 2.0 * h_k) / v3;
    let g_clka = h_kka / v2;
    let g_vv = (k * k * h_kk + 4.0 * k * h_k + 2.0 * h) / v3;
    let g_vka = -(k * h_kka + h_ka) / v2;
    let g_kaka = h_kaka / v;

    // f = F·g over [CL, V, KA, F]; F is linear so its column is the F=1 gradient.
    let f = f_bio * g;
    let grad = [f_bio * g_cl, f_bio * g_v, f_bio * g_ka, g];
    let hess = [
        [f_bio * g_clcl, f_bio * g_clv, f_bio * g_clka, g_cl],
        [f_bio * g_clv, f_bio * g_vv, f_bio * g_vka, g_v],
        [f_bio * g_clka, f_bio * g_vka, f_bio * g_kaka, g_ka],
        [g_cl, g_v, g_ka, 0.0],
    ];
    (f, grad, hess)
}

/// `(f, ∂f/∂[CL,V], ∂²f/∂[CL,V]²)` for the 1-cpt infusion (rate `rate`, duration
/// `dur`). The response shape `f = (R/CL)·G(k)` has a single disposition rate
/// `k = CL/V`, so the whole 2×2 reduces to the **1-D** derivatives `G'(k)`,
/// `G''(k)` chained through `k(CL,V)` and the `A = R/CL` prefactor:
///
/// * during the infusion (`t ≤ dur`):  `G = 1 − e^{−kt}`
/// * after (`t > dur`, `Δ = t−dur`):   `G = e^{−kΔ} − e^{−kt}`
///
/// with `k_CL = 1/V`, `k_V = −k/V`, `k_VV = 2k/V²`, `k_CL,V = −1/V²`.
pub fn infusion_explicit(
    rate: f64,
    dur: f64,
    amt: f64,
    t: f64,
    cl: f64,
    v: f64,
) -> (f64, [f64; 2], [[f64; 2]; 2]) {
    if t < 0.0 || v <= 0.0 || cl <= 0.0 {
        return (0.0, [0.0; 2], [[0.0; 2]; 2]);
    }
    if dur <= 0.0 {
        return iv_bolus_explicit(amt, t, cl, v);
    }
    let k = cl / v;
    // G(k), G'(k), G''(k) for the infusion shape.
    let (g, g1, g2) = if t <= dur {
        let e = (-k * t).exp();
        (1.0 - e, t * e, -t * t * e)
    } else {
        let dt = t - dur;
        let edt = (-k * dt).exp();
        let et = (-k * t).exp();
        (edt - et, -dt * edt + t * et, dt * dt * edt - t * t * et)
    };
    // A = R/CL (depends only on CL); k = CL/V.
    let a = rate / cl;
    let ac = -rate / (cl * cl);
    let acc = 2.0 * rate / (cl * cl * cl);
    let kc = 1.0 / v;
    let kv = -k / v;
    let kcv = -1.0 / (v * v);
    let kvv = 2.0 * k / (v * v);
    let f = a * g;
    // f = A(CL)·G(k(CL,V)); product + chain rule. k_CL,CL = 0.
    let fc = ac * g + a * g1 * kc;
    let fv = a * g1 * kv;
    let fcc = acc * g + 2.0 * ac * g1 * kc + a * (g2 * kc * kc);
    let fcv = ac * g1 * kv + a * (g2 * kc * kv + g1 * kcv);
    let fvv = a * (g2 * kv * kv + g1 * kvv);
    (f, [fc, fv], [[fcc, fcv], [fcv, fvv]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sens::one_cpt::{one_cpt_infusion_g, one_cpt_iv_bolus_g, one_cpt_oral_g};

    fn dual_bolus(amt: f64, t: f64, cl: f64, v: f64) -> (f64, [f64; 2], [[f64; 2]; 2]) {
        let d = one_cpt_iv_bolus_g::<Dual2<2>>(amt, t, Dual2::var(cl, 0), Dual2::var(v, 1));
        (d.value, d.grad, d.hess)
    }
    fn dual_oral(
        amt: f64,
        t: f64,
        cl: f64,
        v: f64,
        ka: f64,
        fb: f64,
    ) -> (f64, [f64; 4], [[f64; 4]; 4]) {
        let d = one_cpt_oral_g::<Dual2<4>>(
            amt,
            t,
            Dual2::var(cl, 0),
            Dual2::var(v, 1),
            Dual2::var(ka, 2),
            Dual2::var(fb, 3),
        );
        (d.value, d.grad, d.hess)
    }

    #[test]
    fn iv_bolus_explicit_matches_dual() {
        for &(amt, t, cl, v) in &[(100.0, 0.5, 3.0, 30.0), (50.0, 9.0, 5.0, 40.0)] {
            let (fe, ge, he) = iv_bolus_explicit(amt, t, cl, v);
            let (fd, gd, hd) = dual_bolus(amt, t, cl, v);
            approx::assert_relative_eq!(fe, fd, max_relative = 1e-12);
            for i in 0..2 {
                approx::assert_relative_eq!(ge[i], gd[i], max_relative = 1e-10);
                for j in 0..2 {
                    approx::assert_relative_eq!(he[i][j], hd[i][j], max_relative = 1e-10);
                }
            }
        }
    }

    fn dual_infusion(
        rate: f64,
        dur: f64,
        amt: f64,
        t: f64,
        cl: f64,
        v: f64,
    ) -> (f64, [f64; 2], [[f64; 2]; 2]) {
        let d =
            one_cpt_infusion_g::<Dual2<2>>(rate, dur, amt, t, Dual2::var(cl, 0), Dual2::var(v, 1));
        (d.value, d.grad, d.hess)
    }

    #[test]
    fn infusion_explicit_matches_dual() {
        // dur = amt/rate; cover both during (t ≤ dur) and after (t > dur).
        for &(rate, amt, t, cl, v) in &[
            (100.0, 1000.0, 5.0, 10.0, 100.0),  // during (dur=10)
            (100.0, 1000.0, 15.0, 10.0, 100.0), // after
            (200.0, 1000.0, 5.0, 3.0, 30.0),    // after (dur=5)
            (50.0, 500.0, 2.0, 0.5, 8.0),       // during (dur=10)
        ] {
            let dur = amt / rate;
            let (fe, ge, he) = infusion_explicit(rate, dur, amt, t, cl, v);
            let (fd, gd, hd) = dual_infusion(rate, dur, amt, t, cl, v);
            approx::assert_relative_eq!(fe, fd, max_relative = 1e-11, epsilon = 1e-12);
            for i in 0..2 {
                approx::assert_relative_eq!(ge[i], gd[i], max_relative = 1e-9, epsilon = 1e-12);
                for j in 0..2 {
                    approx::assert_relative_eq!(
                        he[i][j],
                        hd[i][j],
                        max_relative = 1e-8,
                        epsilon = 1e-12
                    );
                }
            }
        }
    }

    #[test]
    fn oral_explicit_matches_dual() {
        // A spread of (CL, V, KA) avoiding the ka≈k limit, plus F≠1.
        for &(amt, t, cl, v, ka, fb) in &[
            (100.0, 2.0, 1.2, 12.0, 0.8, 0.9),
            (100.0, 0.5, 0.2, 10.0, 1.5, 1.0),
            (50.0, 8.0, 5.0, 40.0, 2.0, 0.75),
            (100.0, 24.0, 0.13, 7.7, 0.81, 1.0), // warfarin-ish
        ] {
            let (fe, ge, he) = oral_explicit(amt, t, cl, v, ka, fb);
            let (fd, gd, hd) = dual_oral(amt, t, cl, v, ka, fb);
            approx::assert_relative_eq!(fe, fd, max_relative = 1e-10, epsilon = 1e-12);
            for i in 0..4 {
                approx::assert_relative_eq!(ge[i], gd[i], max_relative = 1e-9, epsilon = 1e-12);
                for j in 0..4 {
                    approx::assert_relative_eq!(
                        he[i][j],
                        hd[i][j],
                        max_relative = 1e-8,
                        epsilon = 1e-12
                    );
                }
            }
        }
    }

    /// Option B (explicit) vs Dual2 for the real per-observation kernel.
    #[test]
    #[ignore = "bench: run with -- --ignored --nocapture"]
    fn explicit_vs_dual_bench() {
        use std::time::Instant;
        let n = 20_000_000u64;
        let (amt, cl, v, ka, fb) = (100.0, 1.2, 12.0, 0.8, 0.9);
        let run = |label: &str, f: &dyn Fn(f64) -> f64| {
            let t0 = Instant::now();
            let mut acc = 0.0;
            for i in 0..n {
                acc += f((i % 24) as f64 * 0.5);
            }
            let ns = t0.elapsed().as_nanos() as f64 / n as f64;
            std::hint::black_box(acc);
            eprintln!("  {label:<32} {ns:6.2} ns/eval");
            ns
        };
        eprintln!("1-cpt oral f+grad+hess:");
        let exp = run("Option B (explicit f64)", &|t| {
            let (f, g, h) = oral_explicit(amt, t, cl, v, ka, fb);
            f + g.iter().sum::<f64>() + h.iter().flatten().sum::<f64>()
        });
        let d4 = run("Dual2<4> (minimal width)", &|t| {
            let d = one_cpt_oral_g::<Dual2<4>>(
                amt,
                t,
                Dual2::var(cl, 0),
                Dual2::var(v, 1),
                Dual2::var(ka, 2),
                Dual2::var(fb, 3),
            );
            d.value + d.grad.iter().sum::<f64>() + d.hess.iter().flatten().sum::<f64>()
        });
        let d8 = run("Dual2<8> (provider width)", &|t| {
            let d = one_cpt_oral_g::<Dual2<8>>(
                amt,
                t,
                Dual2::var(cl, 0),
                Dual2::var(v, 1),
                Dual2::var(ka, 2),
                Dual2::var(fb, 3),
            );
            d.value + d.grad.iter().sum::<f64>() + d.hess.iter().flatten().sum::<f64>()
        });
        eprintln!(
            "  → explicit is {:.1}x faster than Dual2<4>, {:.1}x faster than Dual2<8>",
            d4 / exp,
            d8 / exp
        );
    }
}
