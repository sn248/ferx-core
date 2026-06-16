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
    /// `sqrt`.
    fn sqrt(self) -> Self;
    /// `cos` (3-cpt trigonometric cubic root solve).
    fn cos(self) -> Self;
    /// `acos` (3-cpt trigonometric cubic root solve).
    fn acos(self) -> Self;
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
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn cos(self) -> Self {
        f64::cos(self)
    }
    #[inline]
    fn acos(self) -> Self {
        f64::acos(self)
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
    fn sqrt(self) -> Self {
        Dual2::sqrt(self)
    }
    #[inline]
    fn cos(self) -> Self {
        Dual2::cos(self)
    }
    #[inline]
    fn acos(self) -> Self {
        Dual2::acos(self)
    }
}
