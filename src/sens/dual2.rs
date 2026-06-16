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

use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug)]
pub struct Dual2<const N: usize> {
    pub value: f64,
    pub grad: [f64; N],
    pub hess: [[f64; N]; N],
}

impl<const N: usize> Dual2<N> {
    /// A constant (zero gradient and Hessian).
    pub fn constant(value: f64) -> Self {
        Dual2 {
            value,
            grad: [0.0; N],
            hess: [[0.0; N]; N],
        }
    }

    /// Seed input variable `i`: `value` with `∂/∂xᵢ = 1`, all else zero.
    pub fn var(value: f64, i: usize) -> Self {
        let mut grad = [0.0; N];
        grad[i] = 1.0;
        Dual2 {
            value,
            grad,
            hess: [[0.0; N]; N],
        }
    }

    /// `exp(self)`. With `u = e^x`: `u' = u·x'`, `u'' = u·(x'' + x'⊗x')`.
    pub fn exp(self) -> Self {
        let v = self.value.exp();
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        for i in 0..N {
            grad[i] = v * self.grad[i];
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = v * (self.hess[i][j] + self.grad[i] * self.grad[j]);
            }
        }
        Dual2 {
            value: v,
            grad,
            hess,
        }
    }

    /// `sqrt(self)`. With `u = √x`: `u' = x'/(2u)`, `u'' = x''/(2u) − x'⊗x'/(4u³)`.
    pub fn sqrt(self) -> Self {
        let v = self.value.sqrt();
        let inv2u = 0.5 / v;
        let inv4u3 = 0.25 / (v * v * v);
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        for i in 0..N {
            grad[i] = inv2u * self.grad[i];
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = inv2u * self.hess[i][j] - inv4u3 * self.grad[i] * self.grad[j];
            }
        }
        Dual2 {
            value: v,
            grad,
            hess,
        }
    }

    /// `cos(self)`. With `u = cos(x)`: `u' = −sin(x)·x'`,
    /// `u'' = −cos(x)·x'⊗x' − sin(x)·x''`.
    pub fn cos(self) -> Self {
        let (s, c) = self.value.sin_cos();
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        for i in 0..N {
            grad[i] = -s * self.grad[i];
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = -s * self.hess[i][j] - c * self.grad[i] * self.grad[j];
            }
        }
        Dual2 {
            value: c,
            grad,
            hess,
        }
    }

    /// `acos(self)`. With `u = acos(x)`: `u' = g₁·x'`, `u'' = g₁·x'' + g₂·x'⊗x'`,
    /// where `g₁ = −1/√(1−x²)` and `g₂ = −x·(1−x²)^{−3/2}`.
    pub fn acos(self) -> Self {
        let v = self.value;
        let one_minus = 1.0 - v * v;
        let s = one_minus.sqrt();
        let g1 = -1.0 / s;
        let g2 = -v / (one_minus * s);
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        for i in 0..N {
            grad[i] = g1 * self.grad[i];
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = g1 * self.hess[i][j] + g2 * self.grad[i] * self.grad[j];
            }
        }
        Dual2 {
            value: v.acos(),
            grad,
            hess,
        }
    }

    /// `1/self`. With `u = 1/b`: `u' = −b'/b²`, `u'' = −b''/b² + 2·b'⊗b'/b³`.
    pub fn recip(self) -> Self {
        let b = self.value;
        let inv = 1.0 / b;
        let inv2 = inv * inv;
        let inv3 = inv2 * inv;
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        for i in 0..N {
            grad[i] = -self.grad[i] * inv2;
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = -self.hess[i][j] * inv2 + 2.0 * self.grad[i] * self.grad[j] * inv3;
            }
        }
        Dual2 {
            value: inv,
            grad,
            hess,
        }
    }
}

impl<const N: usize> Add for Dual2<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        for i in 0..N {
            grad[i] = self.grad[i] + rhs.grad[i];
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = self.hess[i][j] + rhs.hess[i][j];
            }
        }
        Dual2 {
            value: self.value + rhs.value,
            grad,
            hess,
        }
    }
}

impl<const N: usize> Sub for Dual2<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl<const N: usize> Neg for Dual2<N> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        for i in 0..N {
            grad[i] = -self.grad[i];
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = -self.hess[i][j];
            }
        }
        Dual2 {
            value: -self.value,
            grad,
            hess,
        }
    }
}

impl<const N: usize> Mul for Dual2<N> {
    type Output = Self;
    /// Leibniz: `(ab)ᵢ = a·bᵢ + aᵢ·b`, `(ab)ᵢⱼ = a·bᵢⱼ + aᵢ·bⱼ + aⱼ·bᵢ + aᵢⱼ·b`.
    fn mul(self, rhs: Self) -> Self {
        let (a, b) = (self.value, rhs.value);
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        for i in 0..N {
            grad[i] = a * rhs.grad[i] + self.grad[i] * b;
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = a * rhs.hess[i][j]
                    + self.grad[i] * rhs.grad[j]
                    + self.grad[j] * rhs.grad[i]
                    + self.hess[i][j] * b;
            }
        }
        Dual2 {
            value: a * b,
            grad,
            hess,
        }
    }
}

impl<const N: usize> Div for Dual2<N> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self * rhs.recip()
    }
}

// ── Scalar (f64) conveniences ────────────────────────────────────────────────

impl<const N: usize> Mul<f64> for Dual2<N> {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        for i in 0..N {
            grad[i] = self.grad[i] * s;
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = self.hess[i][j] * s;
            }
        }
        Dual2 {
            value: self.value * s,
            grad,
            hess,
        }
    }
}

impl<const N: usize> Add<f64> for Dual2<N> {
    type Output = Self;
    fn add(self, s: f64) -> Self {
        Dual2 {
            value: self.value + s,
            ..self
        }
    }
}

/// `s / dual` (scalar numerator).
pub fn scalar_div<const N: usize>(s: f64, d: Dual2<N>) -> Dual2<N> {
    d.recip() * s
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
