//! `Dual2<N>` — a small, explicit forward-mode second-order multivariate dual
//! number for analytic PK-parameter sensitivities.
//!
//! This is **not** Enzyme / compile-time AD: it is a plain value-plus-jet struct
//! (value, gradient, Hessian over `N` seeded variables) with hand-written
//! operator rules. Evaluating a closed-form PK solution generic over `Dual2<N>`
//! yields `f`, `∂f/∂pkᵢ`, and `∂²f/∂pkᵢ∂pkⱼ` exactly, in one pass, reusing the
//! solution code instead of hand-deriving each model. It is the
//! [`crate::sens`] counterpart to the hand-written closed-form derivatives, and
//! the natural seed of the future num-dual path.
//!
//! `N` is the number of differentiated inputs (PK parameters), a small const
//! (≤ 8: CL, V/V1, Q2, V2, KA, F, Q3, V3). The Hessian is symmetric; rules
//! maintain that.
// Indexed loops over the `[f64; N]` grad / `[[f64; N]; N]` hess jets read more
// clearly than zipped iterators here, and `Div` is defined as `× recip` (the
// `suspicious_arithmetic_impl` lint is a false positive for that identity).
#![allow(clippy::needless_range_loop, clippy::suspicious_arithmetic_impl)]

/// `scalar_div(k, d) = k / d` — re-exported from the [`DualMixed`] module so the
/// `crate::sens::dual2::scalar_div` path (and `Dual2`'s own tests) keep working after
/// the dual-number rules moved there (#448 review #3).
pub use super::dual_mixed::scalar_div;
use super::dual_mixed::DualMixed;

/// `Dual2<N>` is the **square-Hessian** case of [`DualMixed`]: the full `N×N`
/// Hessian, every seeded input differentiated to second order. Since #448 it is a
/// type alias, not a second hand-written copy of the dual-number rules — the
/// rectangular [`DualMixed<NA, N>`](DualMixed) (which drops the IIV-free Hessian
/// block, #445) is the single source, so the two cannot drift (#448 review #3).
pub type Dual2<const N: usize> = DualMixed<N, N>;

#[cfg(test)]
mod tests {
    use super::*;

    /// `sqrt` at a zero (or rounded-negative) argument with a seeded gradient must
    /// return finite, zero derivatives — not `inf`/`NaN` from the singular
    /// `1/(2√x)` / `1/(4x^{3/2})` factors.
    #[test]
    fn dual2_sqrt_zero_argument_is_finite() {
        let x = Dual2::<2>::var(0.0, 0); // value 0, grad[0] = 1
        let r = x.sqrt();
        assert_eq!(r.value, 0.0);
        assert!(r.grad.iter().all(|g| g.is_finite()));
        assert!(r.hess.iter().flatten().all(|h| h.is_finite()));
        assert!(r.grad.iter().all(|&g| g == 0.0));
        // A slightly-negative discriminant (rounding) must also stay finite.
        let neg = Dual2::<2>::var(-1e-18, 0).sqrt();
        assert!(neg.value.is_finite() && neg.grad.iter().all(|g| g.is_finite()));
    }

    /// `f(x,y) = (1/y)·exp(-x/y)` (the 1-cpt IV-bolus shape, amt=1) at a point,
    /// cross-checked against analytic derivatives.
    #[test]
    fn dual2_matches_hand_derivatives_on_iv_shape() {
        let (x, y) = (3.0_f64, 5.0_f64); // x≈CL, y≈V
        let xd = Dual2::<2>::var(x, 0);
        let yd = Dual2::<2>::var(y, 1);
        let f = scalar_div(1.0, yd) * (-(xd / yd)).exp();

        let k = x / y;
        let val = (1.0 / y) * (-k).exp();
        approx::assert_relative_eq!(f.value, val, max_relative = 1e-12);
        // ∂f/∂x = val·(-1/y); ∂f/∂y = val·(k-1)/y  (t=1 here)
        approx::assert_relative_eq!(f.grad[0], val * (-1.0 / y), max_relative = 1e-10);
        approx::assert_relative_eq!(f.grad[1], val * (k - 1.0) / y, max_relative = 1e-10);
        // ∂²f/∂x² = val/y²; symmetric Hessian.
        approx::assert_relative_eq!(f.hess[0][0], val / (y * y), max_relative = 1e-10);
        approx::assert_relative_eq!(f.hess[0][1], f.hess[1][0], max_relative = 1e-12);
    }

    /// Validate a `Dual2<2>` expression's grad/Hessian against central finite
    /// differences of its value, at `(x, y)`.
    fn fd_check<F>(f: F, x: f64, y: f64, gtol: f64, htol: f64)
    where
        F: Fn(Dual2<2>, Dual2<2>) -> Dual2<2>,
    {
        let d = f(Dual2::var(x, 0), Dual2::var(y, 1));
        let v = |a: f64, b: f64| f(Dual2::var(a, 0), Dual2::var(b, 1)).value;
        let h = 1e-5;
        let gx = (v(x + h, y) - v(x - h, y)) / (2.0 * h);
        let gy = (v(x, y + h) - v(x, y - h)) / (2.0 * h);
        approx::assert_relative_eq!(d.grad[0], gx, max_relative = gtol, epsilon = 1e-8);
        approx::assert_relative_eq!(d.grad[1], gy, max_relative = gtol, epsilon = 1e-8);

        let hh = 1e-4;
        let hxx = (v(x + hh, y) - 2.0 * v(x, y) + v(x - hh, y)) / (hh * hh);
        let hyy = (v(x, y + hh) - 2.0 * v(x, y) + v(x, y - hh)) / (hh * hh);
        let hxy = (v(x + hh, y + hh) - v(x + hh, y - hh) - v(x - hh, y + hh) + v(x - hh, y - hh))
            / (4.0 * hh * hh);
        approx::assert_relative_eq!(d.hess[0][0], hxx, max_relative = htol, epsilon = 1e-5);
        approx::assert_relative_eq!(d.hess[1][1], hyy, max_relative = htol, epsilon = 1e-5);
        approx::assert_relative_eq!(d.hess[0][1], hxy, max_relative = htol, epsilon = 1e-5);
        // Hessian symmetry is a structural invariant.
        approx::assert_relative_eq!(d.hess[0][1], d.hess[1][0], max_relative = 1e-12);
    }

    #[test]
    fn ln_matches_fd() {
        fd_check(|x, y| (x * y).ln(), 2.0, 3.0, 1e-6, 1e-3);
        fd_check(|x, y| (x + y * y).ln(), 1.5, 0.7, 1e-6, 1e-3);
    }

    #[test]
    fn pow_constant_exponent_matches_fd() {
        // aⁿ power rule, including a non-integer exponent (positive base).
        fd_check(
            |x, y| x.powd(Dual2::constant(2.5)) * y,
            1.7,
            0.9,
            1e-6,
            1e-3,
        );
        // Integer exponent works for a negative base.
        fd_check(
            |x, y| (x - y).powd(Dual2::constant(3.0)),
            1.0,
            2.5,
            1e-6,
            1e-3,
        );
    }

    #[test]
    fn pow_variable_exponent_matches_fd() {
        // General aᵇ = exp(b·ln a) form (both vary; base > 0).
        fd_check(|x, y| x.powd(y), 1.7, 0.8, 1e-6, 1e-3);
        fd_check(|x, y| (x * y).powd(x), 1.3, 1.1, 1e-6, 2e-3);
    }

    #[test]
    fn abs_matches_fd_away_from_kink() {
        // Positive branch: |x²−y| = x²−y.
        fd_check(|x, y| (x * x - y).abs(), 2.0, 1.0, 1e-6, 1e-3);
        // Negative branch: |x²−y| = −(x²−y).
        fd_check(|x, y| (x * x - y).abs(), 1.0, 3.0, 1e-6, 1e-3);
    }

    #[test]
    fn inv_logit_matches_fd() {
        fd_check(|x, y| (x - y).inv_logit(), 0.6, 0.2, 1e-6, 1e-3);
        fd_check(|x, y| (x * y).inv_logit(), 1.2, 0.5, 1e-6, 1e-3);
    }

    #[test]
    fn logit_matches_fd() {
        // Argument must lie in (0, 1).
        fd_check(|x, y| (x * y).logit(), 0.3, 0.9, 1e-6, 1e-3);
        fd_check(|x, _y| x.logit(), 0.42, 0.0, 1e-6, 1e-3);
    }

    /// `logit` and `inv_logit` are inverses: `inv_logit(logit(p)) = p`, with the
    /// jet propagating consistently.
    #[test]
    fn logit_inv_logit_roundtrip() {
        let p = Dual2::<2>::var(0.37, 0);
        let r = p.logit().inv_logit();
        approx::assert_relative_eq!(r.value, 0.37, max_relative = 1e-12);
        approx::assert_relative_eq!(r.grad[0], 1.0, max_relative = 1e-9);
        approx::assert_relative_eq!(r.hess[0][0], 0.0, epsilon = 1e-7);
    }
}
