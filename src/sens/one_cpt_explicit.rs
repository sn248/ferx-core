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
//! Scope: the 1-cpt models — IV bolus, infusion, oral, and their steady-state
//! variants. Genuinely awkward branches (the `ka ≈ CL/V` L'Hôpital limit) fall
//! back to the dual path — a measure-zero case where hand-derivation buys
//! nothing.
//!
//! The steady-state forms add a geometric SS factor that is a transcendental
//! function of the single disposition rate `k = CL/V` (and, for oral, also of
//! `ka`). Rather than hand-differentiate those, we evaluate the `k`-dependence
//! over a **1-D second-order jet** [`D1`] (value + `d/dk` + `d²/dk²`) and then
//! chain `k(CL,V)` to the `[CL,V]` derivatives in closed form via [`chain1`].
//! This stays scalar/cheap (a 1-D jet, not the `O(N²)` `Dual2<N>`) while letting
//! the jet carry the messy `k`-calculus — so a kernel only transcribes the
//! algebraic shape.

use super::dual2::Dual2;
use super::one_cpt::{one_cpt_oral_g, one_cpt_oral_ss_g};

/// A 1-D second-order jet over a single scalar variable (here the disposition
/// rate `k`): `v` value, `d1 = d/dk`, `d2 = d²/dk²`. Cheap (3 `f64`s, scalar
/// ops); used to obtain `G(k), G'(k), G''(k)` for the steady-state shapes
/// without hand-differentiation, before [`chain1`] maps `k(CL,V)`.
#[derive(Clone, Copy)]
struct D1 {
    v: f64,
    d1: f64,
    d2: f64,
}

impl D1 {
    #[inline]
    fn cst(v: f64) -> Self {
        D1 {
            v,
            d1: 0.0,
            d2: 0.0,
        }
    }
    /// The variable itself (`d/dk = 1`).
    #[inline]
    fn var(v: f64) -> Self {
        D1 {
            v,
            d1: 1.0,
            d2: 0.0,
        }
    }
    #[inline]
    fn sub(self, o: Self) -> Self {
        D1 {
            v: self.v - o.v,
            d1: self.d1 - o.d1,
            d2: self.d2 - o.d2,
        }
    }
    #[inline]
    fn add(self, o: Self) -> Self {
        D1 {
            v: self.v + o.v,
            d1: self.d1 + o.d1,
            d2: self.d2 + o.d2,
        }
    }
    #[inline]
    fn mul(self, o: Self) -> Self {
        D1 {
            v: self.v * o.v,
            d1: self.v * o.d1 + self.d1 * o.v,
            d2: self.v * o.d2 + 2.0 * self.d1 * o.d1 + self.d2 * o.v,
        }
    }
    /// `1/self`.
    #[inline]
    fn recip(self) -> Self {
        let inv = 1.0 / self.v;
        let inv2 = inv * inv;
        D1 {
            v: inv,
            d1: -self.d1 * inv2,
            d2: -self.d2 * inv2 + 2.0 * self.d1 * self.d1 * inv2 * inv,
        }
    }
    /// `exp(self)`.
    #[inline]
    fn exp(self) -> Self {
        let e = self.v.exp();
        D1 {
            v: e,
            d1: e * self.d1,
            d2: e * (self.d2 + self.d1 * self.d1),
        }
    }
    /// Scale by a plain constant `c` applied to the jet's variable as `c·k`
    /// (i.e. an affine reparam): returns the jet of `c·k`. Used to form `−t·k`,
    /// `−II·k`, etc. before `exp`.
    #[inline]
    fn scale(self, c: f64) -> Self {
        D1 {
            v: self.v * c,
            d1: self.d1 * c,
            d2: self.d2 * c,
        }
    }
}

/// Chain a prefactor `A(CL,V)` and a shape `G(k)` (given as value/`G'`/`G''`,
/// `k = CL/V`) into `(f, ∂f/∂[CL,V], ∂²f/∂[CL,V]²)` for `f = A·G(k)`. `A` is
/// supplied by its own partials `(a, a_CL, a_V, a_CLCL, a_CLV, a_VV)`; the
/// `k(CL,V)` chain (`k_CL=1/V`, `k_V=−k/V`, `k_CLV=−1/V²`, `k_VV=2k/V²`,
/// `k_CLCL=0`) is applied in closed form. Generalises the IV-bolus (`A=D/V`) and
/// infusion (`A=R/CL`) kernels and is reused by their steady-state variants.
#[inline]
#[allow(clippy::too_many_arguments)]
fn chain1(
    a: f64,
    ac: f64,
    av: f64,
    acc: f64,
    acv: f64,
    avv: f64,
    g: f64,
    g1: f64,
    g2: f64,
    cl: f64,
    v: f64,
) -> (f64, [f64; 2], [[f64; 2]; 2]) {
    let k = cl / v;
    let kc = 1.0 / v;
    let kv = -k / v;
    let kcv = -1.0 / (v * v);
    let kvv = 2.0 * k / (v * v);
    let f = a * g;
    let fc = ac * g + a * g1 * kc;
    let fv = av * g + a * g1 * kv;
    let fcc = acc * g + 2.0 * ac * g1 * kc + a * (g2 * kc * kc);
    let fcv = acv * g + ac * g1 * kv + av * g1 * kc + a * (g2 * kc * kv + g1 * kcv);
    let fvv = avv * g + 2.0 * av * g1 * kv + a * (g2 * kv * kv + g1 * kvv);
    (f, [fc, fv], [[fcc, fcv], [fcv, fvv]])
}

/// `(A, A_CL, A_V, A_CLCL, A_CLV, A_VV)` for `A = D/V` (IV-bolus / SS-bolus
/// prefactor; depends on `V` only).
#[inline]
fn amt_over_v(amt: f64, v: f64) -> (f64, f64, f64, f64, f64, f64) {
    (
        amt / v,
        0.0,
        -amt / (v * v),
        0.0,
        0.0,
        2.0 * amt / (v * v * v),
    )
}

/// `(A, A_CL, A_V, A_CLCL, A_CLV, A_VV)` for `A = R/CL` (infusion / SS-infusion
/// prefactor; depends on `CL` only).
#[inline]
fn rate_over_cl(rate: f64, cl: f64) -> (f64, f64, f64, f64, f64, f64) {
    (
        rate / cl,
        -rate / (cl * cl),
        0.0,
        2.0 * rate / (cl * cl * cl),
        0.0,
        0.0,
    )
}

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

    // Non-SS Bateman: S = e^{−kt} − e^{−ka·t}; ∂S/∂k∂ka = 0 (separates).
    let ek = (-k * t).exp();
    let eka = (-ka * t).exp();
    oral_chain(
        amt,
        k,
        ka,
        v,
        f_bio,
        ek - eka,
        -t * ek,
        t * eka,
        t * t * ek,
        -t * t * eka,
    )
}

/// Shared `[CL,V,KA,F]` assembly for 1-cpt oral and oral-SS. Given the Bateman
/// numerator `S` and its `k`/`ka` partials (`s_k, s_ka, s_kk, s_kaka`; the cross
/// `∂²S/∂k∂ka` is zero for both the non-SS `e^{−kt}−e^{−ka·t}` and the SS
/// `P(k)−Q(ka)` shapes), chains `h = D·ka·S/Δ`, `g = h/V`, then `f = F·g` (F
/// linear). `Δ = ka − k` (caller guards `ka ≈ k`).
#[inline]
#[allow(clippy::too_many_arguments)]
fn oral_chain(
    dd: f64,
    k: f64,
    ka: f64,
    v: f64,
    f_bio: f64,
    s: f64,
    s_k: f64,
    s_ka: f64,
    s_kk: f64,
    s_kaka: f64,
) -> (f64, [f64; 4], [[f64; 4]; 4]) {
    let delta = ka - k;
    let d2 = delta * delta;
    let d3 = d2 * delta;

    // a = S_k·Δ + S ; b = S_ka·Δ − S ; a_ka = S_k + S_ka (since S_k,ka = 0).
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

/// `(f, ∂f/∂[CL,V,KA,F], ∂²f/∂[...]²)` for 1-cpt oral at steady state. Same
/// `h = D·ka·S/Δ` assembly as the non-SS oral, but with the SS Bateman numerator
/// `S = P(k) − Q(ka)`, `P(λ)=e^{−λt}/(1−e^{−λ·II})` (each evaluated over the 1-D
/// `k`/`ka` jet so the SS factor's derivatives come for free). The `ka ≈ k`
/// L'Hôpital limit routes to the dual path.
pub fn oral_ss_explicit(
    amt: f64,
    t: f64,
    ii: f64,
    cl: f64,
    v: f64,
    ka: f64,
    f_bio: f64,
) -> (f64, [f64; 4], [[f64; 4]; 4]) {
    if t < 0.0 || v <= 0.0 || cl <= 0.0 || ka <= 0.0 || ii <= 0.0 {
        return (0.0, [0.0; 4], [[0.0; 4]; 4]);
    }
    let k = cl / v;
    let fallback = || {
        let d = one_cpt_oral_ss_g::<Dual2<4>>(
            amt,
            t,
            ii,
            Dual2::var(cl, 0),
            Dual2::var(v, 1),
            Dual2::var(ka, 2),
            Dual2::var(f_bio, 3),
        );
        (d.value, d.grad, d.hess)
    };
    if (ka - k).abs() < 1e-6 {
        return fallback();
    }
    // P(λ) = e^{−λt}/(1−e^{−λ·II}) over the 1-D jet; S = P(k) − Q(ka).
    let one = D1::cst(1.0);
    let ss_shape = |lambda: f64| -> D1 {
        let lj = D1::var(lambda);
        lj.scale(-t).exp().mul(one.sub(lj.scale(-ii).exp()).recip())
    };
    let p = ss_shape(k);
    let q = ss_shape(ka);
    // Denominators must be positive (matches the generic SS guard).
    if (1.0 - (-k * ii).exp()) <= 0.0 || (1.0 - (-ka * ii).exp()) <= 0.0 {
        return (0.0, [0.0; 4], [[0.0; 4]; 4]);
    }
    oral_chain(amt, k, ka, v, f_bio, p.v - q.v, p.d1, -q.d1, p.d2, -q.d2)
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

/// `(f, ∂f/∂[CL,V], ∂²f/∂[CL,V]²)` for the 1-cpt IV-bolus at steady state
/// (interval `ii`): `C = (D/V)·e^{−kt}/(1−e^{−k·II})`. `A=D/V`; the SS shape
/// `G(k)=e^{−kt}/(1−e^{−k·II})` is evaluated over the 1-D `k`-jet then chained.
pub fn iv_bolus_ss_explicit(
    amt: f64,
    t: f64,
    ii: f64,
    cl: f64,
    v: f64,
) -> (f64, [f64; 2], [[f64; 2]; 2]) {
    if t < 0.0 || v <= 0.0 || cl <= 0.0 || ii <= 0.0 {
        return (0.0, [0.0; 2], [[0.0; 2]; 2]);
    }
    let k = cl / v;
    let kj = D1::var(k);
    let one = D1::cst(1.0);
    let denom = one.sub(kj.scale(-ii).exp());
    if denom.v <= 0.0 {
        return (0.0, [0.0; 2], [[0.0; 2]; 2]);
    }
    let gg = kj.scale(-t).exp().mul(denom.recip());
    let (a, ac, av, acc, acv, avv) = amt_over_v(amt, v);
    chain1(a, ac, av, acc, acv, avv, gg.v, gg.d1, gg.d2, cl, v)
}

/// `(f, ∂f/∂[CL,V], ∂²f/∂[CL,V]²)` for the 1-cpt infusion at steady state
/// (non-overlapping `dur ≤ II`). `A=R/CL`; the during/after + past-pulse shape of
/// [`one_cpt_infusion_ss_g`] is evaluated over the 1-D `k`-jet then chained.
pub fn infusion_ss_explicit(
    rate: f64,
    dur: f64,
    amt: f64,
    t: f64,
    ii: f64,
    cl: f64,
    v: f64,
) -> (f64, [f64; 2], [[f64; 2]; 2]) {
    if t < 0.0 || v <= 0.0 || cl <= 0.0 || ii <= 0.0 {
        return (0.0, [0.0; 2], [[0.0; 2]; 2]);
    }
    if dur <= 0.0 {
        return iv_bolus_ss_explicit(amt, t, ii, cl, v);
    }
    if dur > ii {
        // Overlapping SS infusion has no single-interval closed form (production
        // returns 0 / routes to ODE); match the generic zero.
        return (0.0, [0.0; 2], [[0.0; 2]; 2]);
    }
    let k = cl / v;
    let kj = D1::var(k);
    let one = D1::cst(1.0);
    let denom = one.sub(kj.scale(-ii).exp());
    if denom.v <= 0.0 {
        return (0.0, [0.0; 2], [[0.0; 2]; 2]);
    }
    let inv_denom = denom.recip();
    let omekt_inf = one.sub(kj.scale(-dur).exp());
    // Past pulses (n ≥ 1) are always "after-infusion".
    let past = omekt_inf
        .mul(kj.scale(-(t + ii - dur)).exp())
        .mul(inv_denom);
    let gg = if t <= dur {
        one.sub(kj.scale(-t).exp()).add(past)
    } else {
        omekt_inf.mul(kj.scale(-(t - dur)).exp()).mul(inv_denom)
    };
    let (a, ac, av, acc, acv, avv) = rate_over_cl(rate, cl);
    chain1(a, ac, av, acc, acv, avv, gg.v, gg.d1, gg.d2, cl, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sens::one_cpt::{
        one_cpt_infusion_g, one_cpt_infusion_ss_g, one_cpt_iv_bolus_g, one_cpt_iv_bolus_ss_g,
        one_cpt_oral_g,
    };

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

    fn dual_bolus_ss(amt: f64, t: f64, ii: f64, cl: f64, v: f64) -> (f64, [f64; 2], [[f64; 2]; 2]) {
        let d = one_cpt_iv_bolus_ss_g::<Dual2<2>>(amt, t, ii, Dual2::var(cl, 0), Dual2::var(v, 1));
        (d.value, d.grad, d.hess)
    }

    #[test]
    fn iv_bolus_ss_explicit_matches_dual() {
        for &(amt, t, ii, cl, v) in &[
            (100.0, 2.0, 24.0, 3.0, 30.0),
            (100.0, 12.0, 24.0, 3.0, 30.0),
            (50.0, 0.5, 12.0, 5.0, 40.0),
        ] {
            let (fe, ge, he) = iv_bolus_ss_explicit(amt, t, ii, cl, v);
            let (fd, gd, hd) = dual_bolus_ss(amt, t, ii, cl, v);
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

    fn dual_infusion_ss(
        rate: f64,
        dur: f64,
        amt: f64,
        t: f64,
        ii: f64,
        cl: f64,
        v: f64,
    ) -> (f64, [f64; 2], [[f64; 2]; 2]) {
        let d = one_cpt_infusion_ss_g::<Dual2<2>>(
            rate,
            dur,
            amt,
            t,
            ii,
            Dual2::var(cl, 0),
            Dual2::var(v, 1),
        );
        (d.value, d.grad, d.hess)
    }

    #[test]
    fn infusion_ss_explicit_matches_dual() {
        // dur ≤ ii (non-overlapping); during (t ≤ dur) and after (t > dur).
        for &(rate, dur, amt, t, ii, cl, v) in &[
            (100.0, 2.0, 200.0, 1.0, 12.0, 3.0, 30.0), // during
            (100.0, 2.0, 200.0, 6.0, 12.0, 3.0, 30.0), // after
            (50.0, 4.0, 200.0, 10.0, 24.0, 0.5, 8.0),  // after
        ] {
            let (fe, ge, he) = infusion_ss_explicit(rate, dur, amt, t, ii, cl, v);
            let (fd, gd, hd) = dual_infusion_ss(rate, dur, amt, t, ii, cl, v);
            approx::assert_relative_eq!(fe, fd, max_relative = 1e-10, epsilon = 1e-12);
            for i in 0..2 {
                approx::assert_relative_eq!(ge[i], gd[i], max_relative = 1e-8, epsilon = 1e-11);
                for j in 0..2 {
                    approx::assert_relative_eq!(
                        he[i][j],
                        hd[i][j],
                        max_relative = 1e-7,
                        epsilon = 1e-10
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

    #[allow(clippy::too_many_arguments)]
    fn dual_oral_ss(
        amt: f64,
        t: f64,
        ii: f64,
        cl: f64,
        v: f64,
        ka: f64,
        fb: f64,
    ) -> (f64, [f64; 4], [[f64; 4]; 4]) {
        let d = one_cpt_oral_ss_g::<Dual2<4>>(
            amt,
            t,
            ii,
            Dual2::var(cl, 0),
            Dual2::var(v, 1),
            Dual2::var(ka, 2),
            Dual2::var(fb, 3),
        );
        (d.value, d.grad, d.hess)
    }

    #[test]
    fn oral_ss_explicit_matches_dual() {
        for &(amt, t, ii, cl, v, ka, fb) in &[
            (100.0, 2.0, 24.0, 1.2, 12.0, 0.8, 0.9),
            (100.0, 12.0, 24.0, 0.2, 10.0, 1.5, 1.0),
            (50.0, 4.0, 12.0, 5.0, 40.0, 2.0, 0.75),
            (100.0, 18.0, 24.0, 0.13, 7.7, 0.81, 1.0), // warfarin-ish
        ] {
            let (fe, ge, he) = oral_ss_explicit(amt, t, ii, cl, v, ka, fb);
            let (fd, gd, hd) = dual_oral_ss(amt, t, ii, cl, v, ka, fb);
            approx::assert_relative_eq!(fe, fd, max_relative = 1e-10, epsilon = 1e-12);
            for i in 0..4 {
                approx::assert_relative_eq!(ge[i], gd[i], max_relative = 1e-8, epsilon = 1e-11);
                for j in 0..4 {
                    approx::assert_relative_eq!(
                        he[i][j],
                        hd[i][j],
                        max_relative = 1e-7,
                        epsilon = 1e-10
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
