//! A small fixed-width second-order jet (value + gradient + Hessian) over `N`
//! parameters, used by the explicit-derivative PK kernels (`*_explicit`).
//!
//! It is the same algebra as [`Dual2`](super::dual2::Dual2) minus the
//! transcendental ops (`sqrt`, `acos`, `ln`, …): the explicit kernels obtain the
//! disposition eigenvalues' derivatives in **closed form** (Vieta / implicit
//! differentiation of the characteristic polynomial) and then propagate the
//! coefficient/exponential assembly through this jet, whose `mul`/`recip`/`exp`
//! carry the chain rule automatically. So a kernel transcribes the algebraic
//! closed form and the jet does the calculus — no hand-derived grad/Hessian
//! except for the eigenvalues themselves.
//!
//! `N` is the number of differentiated PK parameters for the model (1-cpt IV = 2,
//! 2-cpt IV = 4, 2-cpt oral = 6, 3-cpt IV = 6, 3-cpt oral = 8). Seeds use fixed
//! axes shared across kernels: `CL@0, V1@1, Q2@2, V2@3, Q3@4, V3@5` for the IV
//! disposition; oral adds `KA, F` on the next two axes.

/// Second-order jet over `N` parameters: `v` (value), `g` (gradient `∂/∂xᵢ`),
/// `h` (Hessian `∂²/∂xᵢ∂xⱼ`, kept symmetric).
#[derive(Clone, Copy)]
pub(crate) struct Jet<const N: usize> {
    pub v: f64,
    pub g: [f64; N],
    pub h: [[f64; N]; N],
}

impl<const N: usize> Jet<N> {
    /// A constant (zero derivatives).
    #[inline]
    pub fn cst(v: f64) -> Self {
        Jet {
            v,
            g: [0.0; N],
            h: [[0.0; N]; N],
        }
    }

    /// An independent variable seeded on axis `i` (`∂/∂xᵢ = 1`).
    #[inline]
    pub fn var(v: f64, i: usize) -> Self {
        let mut j = Jet::cst(v);
        j.g[i] = 1.0;
        j
    }

    /// `num/den` where `num` is the seed at axis `ni` and `den` the seed at axis
    /// `di` (both plain variables, no cross terms among other seeds). Closed form:
    /// `xₙ=1/d`, `x_d=−n/d²`, `x_{nd}=−1/d²`, `x_{dd}=2n/d³`, `x_{nn}=0`.
    #[inline]
    pub fn ratio(num: f64, ni: usize, den: f64, di: usize) -> Self {
        let inv = 1.0 / den;
        let inv2 = inv * inv;
        let mut r = Jet::cst(num * inv);
        r.g[ni] = inv;
        r.g[di] = -num * inv2;
        r.h[ni][di] = -inv2;
        r.h[di][ni] = -inv2;
        r.h[di][di] = 2.0 * num * inv2 * inv;
        r
    }

    #[inline]
    pub fn add(self, o: Self) -> Self {
        let mut r = Jet::cst(self.v + o.v);
        for i in 0..N {
            r.g[i] = self.g[i] + o.g[i];
            for j in 0..N {
                r.h[i][j] = self.h[i][j] + o.h[i][j];
            }
        }
        r
    }

    #[inline]
    pub fn sub(self, o: Self) -> Self {
        let mut r = Jet::cst(self.v - o.v);
        for i in 0..N {
            r.g[i] = self.g[i] - o.g[i];
            for j in 0..N {
                r.h[i][j] = self.h[i][j] - o.h[i][j];
            }
        }
        r
    }

    /// Multiply by a plain scalar (no derivatives of the scalar).
    #[inline]
    pub fn scale(self, k: f64) -> Self {
        let mut r = Jet::cst(self.v * k);
        for i in 0..N {
            r.g[i] = self.g[i] * k;
            for j in 0..N {
                r.h[i][j] = self.h[i][j] * k;
            }
        }
        r
    }

    /// Leibniz product: `(ab)ᵢ = a bᵢ + aᵢ b`, `(ab)ᵢⱼ = a bᵢⱼ + aᵢbⱼ + aⱼbᵢ + aᵢⱼ b`.
    #[inline]
    pub fn mul(self, o: Self) -> Self {
        let (a, b) = (self.v, o.v);
        let mut r = Jet::cst(a * b);
        for i in 0..N {
            r.g[i] = a * o.g[i] + self.g[i] * b;
            for j in 0..N {
                r.h[i][j] =
                    a * o.h[i][j] + self.g[i] * o.g[j] + self.g[j] * o.g[i] + self.h[i][j] * b;
            }
        }
        r
    }

    /// `1/self`: `u' = −b'/b²`, `u'' = −b''/b² + 2 b'⊗b'/b³`.
    #[inline]
    pub fn recip(self) -> Self {
        let inv = 1.0 / self.v;
        let inv2 = inv * inv;
        let inv3 = inv2 * inv;
        let mut r = Jet::cst(inv);
        for i in 0..N {
            r.g[i] = -self.g[i] * inv2;
            for j in 0..N {
                r.h[i][j] = -self.h[i][j] * inv2 + 2.0 * self.g[i] * self.g[j] * inv3;
            }
        }
        r
    }

    /// `exp(self)`: `u' = u·x'`, `u'' = u·(x'' + x'⊗x')`.
    #[inline]
    pub fn exp(self) -> Self {
        let e = self.v.exp();
        let mut r = Jet::cst(e);
        for i in 0..N {
            r.g[i] = e * self.g[i];
            for j in 0..N {
                r.h[i][j] = e * (self.h[i][j] + self.g[i] * self.g[j]);
            }
        }
        r
    }

    /// Steady-state geometric factor `1/(1 − e^{−λ·ii})` as a jet, with `λ = self`.
    /// `None` when the denominator is non-positive (degenerate `λ·ii`; caller
    /// falls back to the dual path).
    #[inline]
    pub fn ss_coeff(self, ii: f64) -> Option<Self> {
        let denom = Jet::<N>::cst(1.0).sub(self.scale(-ii).exp());
        if denom.v <= 0.0 {
            return None;
        }
        Some(denom.recip())
    }

    /// Average `h[i][j]` and `h[j][i]` to kill round-off asymmetry after a step
    /// (e.g. implicit eigenvalue differentiation) that fills the two via
    /// mathematically-equal-but-distinct expressions.
    #[inline]
    pub fn symmetrise(&mut self) {
        for i in 0..N {
            for j in (i + 1)..N {
                let a = 0.5 * (self.h[i][j] + self.h[j][i]);
                self.h[i][j] = a;
                self.h[j][i] = a;
            }
        }
    }
}
