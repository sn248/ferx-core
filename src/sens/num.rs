//! `PkNum` — the small numeric interface a closed-form PK solution needs, so a
//! single generic implementation serves both the scalar prediction (`T = f64`)
//! and its exact sensitivities (`T = Dual2<N>`). No Enzyme, no codegen: the same
//! source is monomorphised for each numeric type.

use super::dual2::Dual2;
use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait PkNum:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Lift a constant into the numeric type (zero derivatives for duals).
    fn from_f64(x: f64) -> Self;
    /// The underlying value — for guards / branch conditions only.
    fn val(self) -> f64;
    /// `exp`.
    fn exp(self) -> Self;
    /// `ln`.
    fn ln(self) -> Self;
    /// `sqrt`.
    fn sqrt(self) -> Self;
    /// `self^e` (binary power; the bytecode `Op::Pow`).
    fn pow(self, e: Self) -> Self;
    /// `|self|`.
    fn abs(self) -> Self;
    /// `inv_logit(self) = 1/(1+e^{−x})` (logistic sigmoid).
    fn inv_logit(self) -> Self;
    /// `logit(self) = ln(x/(1−x))`.
    fn logit(self) -> Self;
    /// `cos` (3-cpt trigonometric cubic root solve).
    fn cos(self) -> Self;
    /// `acos` (3-cpt trigonometric cubic root solve).
    fn acos(self) -> Self;
    /// Lower-clamp the value to `lo` (branch on `.val()`), used by the bytecode
    /// VM to reproduce the `Op::Ln`/`Op::Sqrt` domain guards (`v.max(lo)`) before
    /// the transcendental call. For duals the clamped region is flat (zero jet).
    fn guard_floor(self, lo: f64) -> Self;
}

impl PkNum for f64 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
    #[inline]
    fn val(self) -> f64 {
        self
    }
    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        f64::ln(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn pow(self, e: Self) -> Self {
        f64::powf(self, e)
    }
    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }
    #[inline]
    fn inv_logit(self) -> Self {
        1.0 / (1.0 + f64::exp(-self))
    }
    #[inline]
    fn logit(self) -> Self {
        f64::ln(self / (1.0 - self))
    }
    #[inline]
    fn cos(self) -> Self {
        f64::cos(self)
    }
    #[inline]
    fn acos(self) -> Self {
        f64::acos(self)
    }
    #[inline]
    fn guard_floor(self, lo: f64) -> Self {
        self.max(lo)
    }
}

impl<const N: usize> PkNum for Dual2<N> {
    #[inline]
    fn from_f64(x: f64) -> Self {
        Dual2::constant(x)
    }
    #[inline]
    fn val(self) -> f64 {
        self.value
    }
    #[inline]
    fn exp(self) -> Self {
        Dual2::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        Dual2::ln(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        Dual2::sqrt(self)
    }
    #[inline]
    fn pow(self, e: Self) -> Self {
        Dual2::powd(self, e)
    }
    #[inline]
    fn abs(self) -> Self {
        Dual2::abs(self)
    }
    #[inline]
    fn inv_logit(self) -> Self {
        Dual2::inv_logit(self)
    }
    #[inline]
    fn logit(self) -> Self {
        Dual2::logit(self)
    }
    #[inline]
    fn cos(self) -> Self {
        Dual2::cos(self)
    }
    #[inline]
    fn acos(self) -> Self {
        Dual2::acos(self)
    }
    #[inline]
    fn guard_floor(self, lo: f64) -> Self {
        if self.value < lo {
            Dual2::constant(lo)
        } else {
            self
        }
    }
}
