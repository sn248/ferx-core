//! Analytic PK-parameter sensitivities for the closed-form (analytical) PK
//! models — the clean-slate replacement for Enzyme AD / FD-of-Jacobian in the
//! FOCEI outer gradient (issue #367).
//!
//! Goal: provide `f`, `∂f/∂pk`, `∂²f/∂pk²` for each built-in PK solution exactly
//! and noise-free, so the gradient assembly can chain them to `∂f/∂η`, `∂f/∂θ`,
//! `∂²f/∂η²`, `∂²f/∂η∂θ` via closed-form chain rule (η enters only as
//! `pk = tv·exp(sel·η)`; θ through `tv_fn`).
//!
//! Two implementation styles are being compared on the 1-cpt model before we
//! commit to one for 2-/3-cpt — see [`one_cpt`]:
//!   * hand-written closed-form derivatives, and
//!   * the same solution evaluated over the [`dual2::Dual2`] forward-2nd-order
//!     dual number.

pub mod dual2;
pub mod num;
pub mod one_cpt;
pub mod provider;
pub mod three_cpt;
pub mod two_cpt;
