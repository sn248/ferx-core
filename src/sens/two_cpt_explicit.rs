//! Option B (explicit symbolic derivatives) for the 2-cpt IV-bolus solution.
//!
//! Unlike 1-cpt, the 2-cpt response routes through the macro-rate eigenvalues
//! `α, β` (roots of `λ² − sλ + d = 0`, `s = k10+k12+k21`, `d = k10·k21`). The
//! expensive part for a `Dual2<N>` would be differentiating the `√(s²−4d)` and
//! the `β = d/α` division through the dual rules. Here we instead get the
//! eigenvalue first/second derivatives in **closed form** by implicit
//! differentiation of Vieta's relations (`α+β = s`, `αβ = d`):
//!
//! ```text
//!   α'ᵢ = (α·s'ᵢ − d'ᵢ)/Δ,            β'ᵢ = (d'ᵢ − β·s'ᵢ)/Δ,      Δ = α−β
//!   α''ᵢⱼ = [(α'ⱼ s'ᵢ + α s''ᵢⱼ − d''ᵢⱼ)Δ − (α s'ᵢ − d'ᵢ)(α'ⱼ−β'ⱼ)] / Δ²
//!   β''ᵢⱼ = [(d''ᵢⱼ − β'ⱼ s'ᵢ − β s''ᵢⱼ)Δ − (d'ᵢ − β s'ᵢ)(α'ⱼ−β'ⱼ)] / Δ²
//! ```
//!
//! and propagate the coefficient/exponential assembly with a small 4-variable
//! second-order jet. Seeds are `[CL, V1, Q, V2]`. Validated against
//! [`Dual2<4>`](super::dual2::Dual2) to ~1e-8; the near-degenerate (`Δ≈0`) and
//! invalid cases fall back to the dual path.

use super::dual2::Dual2;
use super::two_cpt::{two_cpt_infusion_g, two_cpt_iv_bolus_g};

/// A second-order jet over the four PK parameters `[CL, V1, Q, V2]`.
#[derive(Clone, Copy)]
struct J4 {
    v: f64,
    g: [f64; 4],
    h: [[f64; 4]; 4],
}

impl J4 {
    #[inline]
    fn cst(v: f64) -> Self {
        J4 {
            v,
            g: [0.0; 4],
            h: [[0.0; 4]; 4],
        }
    }
    #[inline]
    fn add(self, o: Self) -> Self {
        let mut r = J4::cst(self.v + o.v);
        for i in 0..4 {
            r.g[i] = self.g[i] + o.g[i];
            for j in 0..4 {
                r.h[i][j] = self.h[i][j] + o.h[i][j];
            }
        }
        r
    }
    #[inline]
    fn sub(self, o: Self) -> Self {
        let mut r = J4::cst(self.v - o.v);
        for i in 0..4 {
            r.g[i] = self.g[i] - o.g[i];
            for j in 0..4 {
                r.h[i][j] = self.h[i][j] - o.h[i][j];
            }
        }
        r
    }
    /// Multiply by a plain scalar (no derivatives of the scalar).
    #[inline]
    fn scale(self, k: f64) -> Self {
        let mut r = J4::cst(self.v * k);
        for i in 0..4 {
            r.g[i] = self.g[i] * k;
            for j in 0..4 {
                r.h[i][j] = self.h[i][j] * k;
            }
        }
        r
    }
    /// Leibniz product: `(ab)ᵢ = a bᵢ + aᵢ b`, `(ab)ᵢⱼ = a bᵢⱼ + aᵢbⱼ + aⱼbᵢ + aᵢⱼ b`.
    #[inline]
    fn mul(self, o: Self) -> Self {
        let (a, b) = (self.v, o.v);
        let mut r = J4::cst(a * b);
        for i in 0..4 {
            r.g[i] = a * o.g[i] + self.g[i] * b;
            for j in 0..4 {
                r.h[i][j] =
                    a * o.h[i][j] + self.g[i] * o.g[j] + self.g[j] * o.g[i] + self.h[i][j] * b;
            }
        }
        r
    }
    /// `1/self`: `u' = −b'/b²`, `u'' = −b''/b² + 2 b'⊗b'/b³`.
    #[inline]
    fn recip(self) -> Self {
        let inv = 1.0 / self.v;
        let inv2 = inv * inv;
        let inv3 = inv2 * inv;
        let mut r = J4::cst(inv);
        for i in 0..4 {
            r.g[i] = -self.g[i] * inv2;
            for j in 0..4 {
                r.h[i][j] = -self.h[i][j] * inv2 + 2.0 * self.g[i] * self.g[j] * inv3;
            }
        }
        r
    }
    /// `exp(self)`: `u' = u·x'`, `u'' = u·(x'' + x'⊗x')`.
    #[inline]
    fn exp(self) -> Self {
        let e = self.v.exp();
        let mut r = J4::cst(e);
        for i in 0..4 {
            r.g[i] = e * self.g[i];
            for j in 0..4 {
                r.h[i][j] = e * (self.h[i][j] + self.g[i] * self.g[j]);
            }
        }
        r
    }
}

/// The macro-rate eigenvalue jets `(α, β, k21)` over `[CL,V1,Q,V2]`, or `None`
/// when the disposition is degenerate (`disc≈0`, `α≈0`, or `α≈β`) and the caller
/// should fall back to the dual path. Obtained by implicit differentiation of
/// Vieta's relations (closed form, no `√`-jet); see the module header.
fn macro_rate_jets(cl: f64, v1: f64, q: f64, v2: f64) -> Option<(J4, J4, J4)> {
    // Micro-rates as jets (closed-form sparse grad/hess).
    let mut k10 = J4::cst(cl / v1);
    k10.g = [1.0 / v1, -cl / (v1 * v1), 0.0, 0.0];
    k10.h[0][1] = -1.0 / (v1 * v1);
    k10.h[1][0] = -1.0 / (v1 * v1);
    k10.h[1][1] = 2.0 * cl / (v1 * v1 * v1);

    let mut k12 = J4::cst(q / v1);
    k12.g = [0.0, -q / (v1 * v1), 1.0 / v1, 0.0];
    k12.h[1][2] = -1.0 / (v1 * v1);
    k12.h[2][1] = -1.0 / (v1 * v1);
    k12.h[1][1] = 2.0 * q / (v1 * v1 * v1);

    let mut k21 = J4::cst(q / v2);
    k21.g = [0.0, 0.0, 1.0 / v2, -q / (v2 * v2)];
    k21.h[2][3] = -1.0 / (v2 * v2);
    k21.h[3][2] = -1.0 / (v2 * v2);
    k21.h[3][3] = 2.0 * q / (v2 * v2 * v2);

    // s = k10 + k12 + k21 ; d = k10·k21.
    let s = k10.add(k12).add(k21);
    let d = k10.mul(k21);

    // Eigenvalues via Vieta + implicit differentiation (closed form, no √-jet).
    let disc_sq = s.v * s.v - 4.0 * d.v;
    if disc_sq <= 1e-300 {
        return None;
    }
    let disc = disc_sq.sqrt();
    let av = 0.5 * (s.v + disc);
    if av <= 1e-300 {
        return None;
    }
    let bv = d.v / av;
    let delta = av - bv;
    if delta.abs() < 1e-12 {
        return None;
    }
    let inv_d = 1.0 / delta;
    let inv_d2 = inv_d * inv_d;

    let mut alpha = J4::cst(av);
    let mut beta = J4::cst(bv);
    // First derivatives.
    for i in 0..4 {
        alpha.g[i] = (av * s.g[i] - d.g[i]) * inv_d;
        beta.g[i] = (d.g[i] - bv * s.g[i]) * inv_d;
    }
    // Second derivatives (see module header). Symmetrised for the jet invariant.
    for i in 0..4 {
        for j in 0..4 {
            let dg = alpha.g[j] - beta.g[j];
            let a_ij = ((alpha.g[j] * s.g[i] + av * s.h[i][j] - d.h[i][j]) * delta
                - (av * s.g[i] - d.g[i]) * dg)
                * inv_d2;
            let b_ij = ((d.h[i][j] - beta.g[j] * s.g[i] - bv * s.h[i][j]) * delta
                - (d.g[i] - bv * s.g[i]) * dg)
                * inv_d2;
            alpha.h[i][j] = a_ij;
            beta.h[i][j] = b_ij;
        }
    }
    // Symmetrise (the formula evaluates [i][j] and [j][i] via different but
    // mathematically equal expressions; average to kill round-off asymmetry).
    for i in 0..4 {
        for j in (i + 1)..4 {
            let a = 0.5 * (alpha.h[i][j] + alpha.h[j][i]);
            alpha.h[i][j] = a;
            alpha.h[j][i] = a;
            let b = 0.5 * (beta.h[i][j] + beta.h[j][i]);
            beta.h[i][j] = b;
            beta.h[j][i] = b;
        }
    }
    Some((alpha, beta, k21))
}

/// `R/V1` (or `amt/V1`) as a jet: depends on `V1` only (axis 1).
#[inline]
fn over_v1(num: f64, v1: f64) -> J4 {
    let mut j = J4::cst(num / v1);
    j.g[1] = -num / (v1 * v1);
    j.h[1][1] = 2.0 * num / (v1 * v1 * v1);
    j
}

/// `(f, ∂f/∂[CL,V1,Q,V2], ∂²f/∂[CL,V1,Q,V2]²)` for the 2-cpt IV bolus.
pub fn iv_bolus_explicit(
    amt: f64,
    t: f64,
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
) -> (f64, [f64; 4], [[f64; 4]; 4]) {
    let fallback = || {
        let d = two_cpt_iv_bolus_g::<Dual2<4>>(
            amt,
            t,
            Dual2::var(cl, 0),
            Dual2::var(v1, 1),
            Dual2::var(q, 2),
            Dual2::var(v2, 3),
        );
        (d.value, d.grad, d.hess)
    };
    if t < 0.0 || v1 <= 0.0 || v2 <= 0.0 || cl <= 0.0 || q < 0.0 {
        return (0.0, [0.0; 4], [[0.0; 4]; 4]);
    }
    let (alpha, beta, k21) = match macro_rate_jets(cl, v1, q, v2) {
        Some(x) => x,
        None => return fallback(),
    };

    // Coefficients: a = (amt/V1)(α−k21)/Δ, b = (amt/V1)(k21−β)/Δ, Δ = α−β.
    let amt_v1 = over_v1(amt, v1);
    let diff = alpha.sub(beta);
    let inv_diff = diff.recip();
    let a = amt_v1.mul(alpha.sub(k21)).mul(inv_diff);
    let b = amt_v1.mul(k21.sub(beta)).mul(inv_diff);

    // C = a·e^{−αt} + b·e^{−βt}.
    let e_a = alpha.scale(-t).exp();
    let e_b = beta.scale(-t).exp();
    let c = a.mul(e_a).add(b.mul(e_b));
    (c.v, c.g, c.h)
}

/// `(f, ∂f/∂[CL,V1,Q,V2], ∂²f/∂[CL,V1,Q,V2]²)` for the 2-cpt infusion (rate
/// `rate`, duration `dur`). Same eigenvalue jets as the bolus; the coefficients
/// carry an extra `1/α`, `1/β` (zero-order input), and the response is the
/// during/after piecewise of [`two_cpt_infusion_g`].
pub fn infusion_explicit(
    rate: f64,
    dur: f64,
    amt: f64,
    t: f64,
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
) -> (f64, [f64; 4], [[f64; 4]; 4]) {
    let fallback = || {
        let d = two_cpt_infusion_g::<Dual2<4>>(
            rate,
            dur,
            amt,
            t,
            Dual2::var(cl, 0),
            Dual2::var(v1, 1),
            Dual2::var(q, 2),
            Dual2::var(v2, 3),
        );
        (d.value, d.grad, d.hess)
    };
    if t < 0.0 || v1 <= 0.0 || v2 <= 0.0 || cl <= 0.0 || q < 0.0 {
        return (0.0, [0.0; 4], [[0.0; 4]; 4]);
    }
    if dur <= 0.0 {
        return iv_bolus_explicit(amt, t, cl, v1, q, v2);
    }
    let (alpha, beta, k21) = match macro_rate_jets(cl, v1, q, v2) {
        Some(x) => x,
        None => return fallback(),
    };
    // The coefficients divide by α and β; bail to the dual path if either is
    // near-zero (the generic form returns 0 there, but FD-matching that
    // degenerate zero buys nothing).
    if alpha.v.abs() < 1e-12 || beta.v.abs() < 1e-12 {
        return fallback();
    }

    // a = (R/V1)(α−k21)/(Δ·α), b = (R/V1)(k21−β)/(Δ·β), Δ = α−β.
    let r_v1 = over_v1(rate, v1);
    let diff = alpha.sub(beta);
    let inv_diff = diff.recip();
    let a_coeff = r_v1.mul(alpha.sub(k21)).mul(inv_diff).mul(alpha.recip());
    let b_coeff = r_v1.mul(k21.sub(beta)).mul(inv_diff).mul(beta.recip());

    let one = J4::cst(1.0);
    let c = if t <= dur {
        let e_a = alpha.scale(-t).exp();
        let e_b = beta.scale(-t).exp();
        a_coeff.mul(one.sub(e_a)).add(b_coeff.mul(one.sub(e_b)))
    } else {
        let e_ad = alpha.scale(-dur).exp();
        let e_bd = beta.scale(-dur).exp();
        let e_adt = alpha.scale(-(t - dur)).exp();
        let e_bdt = beta.scale(-(t - dur)).exp();
        a_coeff
            .mul(one.sub(e_ad))
            .mul(e_adt)
            .add(b_coeff.mul(one.sub(e_bd)).mul(e_bdt))
    };
    (c.v, c.g, c.h)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dual_bolus(
        amt: f64,
        t: f64,
        cl: f64,
        v1: f64,
        q: f64,
        v2: f64,
    ) -> (f64, [f64; 4], [[f64; 4]; 4]) {
        let d = two_cpt_iv_bolus_g::<Dual2<4>>(
            amt,
            t,
            Dual2::var(cl, 0),
            Dual2::var(v1, 1),
            Dual2::var(q, 2),
            Dual2::var(v2, 3),
        );
        (d.value, d.grad, d.hess)
    }

    #[test]
    fn two_cpt_iv_bolus_explicit_matches_dual() {
        for &(amt, t, cl, v1, q, v2) in &[
            (1000.0, 0.25, 10.0, 50.0, 15.0, 100.0),
            (1000.0, 2.0, 10.0, 50.0, 15.0, 100.0),
            (1000.0, 24.0, 10.0, 50.0, 15.0, 100.0),
            (500.0, 4.0, 5.0, 30.0, 2.0, 50.0),
            (1000.0, 1.0, 4.41, 15.5, 3.14, 29.3), // 2-cpt NONMEM-fit-ish
        ] {
            let (fe, ge, he) = iv_bolus_explicit(amt, t, cl, v1, q, v2);
            let (fd, gd, hd) = dual_bolus(amt, t, cl, v1, q, v2);
            approx::assert_relative_eq!(fe, fd, max_relative = 1e-10, epsilon = 1e-12);
            for i in 0..4 {
                approx::assert_relative_eq!(ge[i], gd[i], max_relative = 1e-8, epsilon = 1e-12);
                for j in 0..4 {
                    approx::assert_relative_eq!(
                        he[i][j],
                        hd[i][j],
                        max_relative = 1e-7,
                        epsilon = 1e-11
                    );
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
        v1: f64,
        q: f64,
        v2: f64,
    ) -> (f64, [f64; 4], [[f64; 4]; 4]) {
        let d = two_cpt_infusion_g::<Dual2<4>>(
            rate,
            dur,
            amt,
            t,
            Dual2::var(cl, 0),
            Dual2::var(v1, 1),
            Dual2::var(q, 2),
            Dual2::var(v2, 3),
        );
        (d.value, d.grad, d.hess)
    }

    #[test]
    fn two_cpt_infusion_explicit_matches_dual() {
        // dur = amt/rate; cover both during (t ≤ dur) and after (t > dur).
        for &(rate, amt, t, cl, v1, q, v2) in &[
            (500.0, 1000.0, 1.0, 10.0, 50.0, 15.0, 100.0), // during (dur=2)
            (500.0, 1000.0, 6.0, 10.0, 50.0, 15.0, 100.0), // after
            (250.0, 1000.0, 2.0, 5.0, 30.0, 2.0, 50.0),    // during (dur=4)
            (250.0, 1000.0, 10.0, 5.0, 30.0, 2.0, 50.0),   // after
            (1000.0, 1000.0, 0.5, 4.41, 15.5, 3.14, 29.3), // during (dur=1), fit-ish
            (1000.0, 1000.0, 3.0, 4.41, 15.5, 3.14, 29.3), // after
        ] {
            let dur = amt / rate;
            let (fe, ge, he) = infusion_explicit(rate, dur, amt, t, cl, v1, q, v2);
            let (fd, gd, hd) = dual_infusion(rate, dur, amt, t, cl, v1, q, v2);
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

    #[test]
    #[ignore = "bench: run with -- --ignored --nocapture"]
    fn two_cpt_explicit_vs_dual_bench() {
        use std::time::Instant;
        let n = 20_000_000u64;
        let (amt, cl, v1, q, v2) = (1000.0, 10.0, 50.0, 15.0, 100.0);
        let run = |label: &str, f: &dyn Fn(f64) -> f64| {
            let t0 = Instant::now();
            let mut acc = 0.0;
            for i in 0..n {
                acc += f((i % 24) as f64 * 0.5 + 0.25);
            }
            let ns = t0.elapsed().as_nanos() as f64 / n as f64;
            std::hint::black_box(acc);
            eprintln!("  {label:<34} {ns:6.2} ns/eval");
            ns
        };
        eprintln!("2-cpt IV bolus f+grad+hess:");
        let exp = run("Option B (explicit, closed-form λ)", &|t| {
            let (f, g, h) = iv_bolus_explicit(amt, t, cl, v1, q, v2);
            f + g.iter().sum::<f64>() + h.iter().flatten().sum::<f64>()
        });
        let d4 = run("Dual2<4> (minimal width)", &|t| {
            let d = two_cpt_iv_bolus_g::<Dual2<4>>(
                amt,
                t,
                Dual2::var(cl, 0),
                Dual2::var(v1, 1),
                Dual2::var(q, 2),
                Dual2::var(v2, 3),
            );
            d.value + d.grad.iter().sum::<f64>() + d.hess.iter().flatten().sum::<f64>()
        });
        let d8 = run("Dual2<8> (provider width)", &|t| {
            let d = two_cpt_iv_bolus_g::<Dual2<8>>(
                amt,
                t,
                Dual2::var(cl, 0),
                Dual2::var(v1, 1),
                Dual2::var(q, 2),
                Dual2::var(v2, 3),
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
