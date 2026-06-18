//! Generic (`PkNum`) compartment-state propagators — the `Dual2`-differentiable
//! mirror of the `f64` event-driven propagators in [`crate::pk::event_driven`].
//!
//! The closed-form *superposition* provider ([`super::provider`]) sums
//! independent per-dose contributions using a single per-subject `pk`. That works
//! when the PK parameters are constant over the subject's timeline. **IOV** and
//! **time-varying covariates** break it: the parameters *switch mid-decay* (a dose
//! given in one occasion carries over into the next and continues decaying with
//! that occasion's clearance — NONMEM #104 semantics). Representing that exactly
//! needs an *event walk* that carries the compartment **amounts** across occasion
//! boundaries and swaps the parameters at each boundary — exactly what the
//! production `f64` walker does.
//!
//! These propagators evolve the **dual** state over one inter-event interval, so
//! the walk can be run over `Dual2` and yield exact `∂(amount)/∂(η,κ)` (and second
//! order) — closed-form per segment, no numerical ODE integration. This module is
//! the foundation; the dual event walk + readout + the per-occasion block-Ω outer
//! assembly build on top (issue #367).

use super::num::PkNum;

/// 1-cpt IV/central propagator: evolve `state[0]` (central amount) over `dt` with
/// a constant input `rate` into the central compartment. Mirror of
/// [`crate::pk::event_driven::propagate_one_cpt`].
pub fn propagate_one_cpt_g<T: PkNum>(state: &mut [T], dt: f64, cl: T, v: T, rate: T) {
    if v.val() <= 0.0 || cl.val() <= 0.0 {
        // Degenerate params: skip (the outer optimizer sees a poor OFV and steps
        // away), matching the production propagator.
        return;
    }
    let ke = cl / v;
    let exp_term = (-(ke * T::from_f64(dt))).exp();
    state[0] = exp_term * state[0] + (rate / ke) * (T::from_f64(1.0) - exp_term);
}

/// 1-cpt oral propagator. `state = [A_depot, A_central]`; the depot drains into
/// the central compartment at absorption rate `ka` (bolus dosing — doses are added
/// to the depot by the event handler). Mirror of
/// [`crate::pk::event_driven::propagate_one_cpt_oral`], including the `ka ≈ ke`
/// L'Hôpital limit.
pub fn propagate_one_cpt_oral_g<T: PkNum>(state: &mut [T], dt: f64, cl: T, v: T, ka: T) {
    if v.val() <= 0.0 || cl.val() <= 0.0 || ka.val() <= 0.0 {
        return;
    }
    let ke = cl / v;
    let dtt = T::from_f64(dt);
    let e_ka = (-(ka * dtt)).exp();
    let e_ke = (-(ke * dtt)).exp();
    let a_d_0 = state[0];
    let a_c_0 = state[1];

    // Depot decays exponentially (decoupled).
    state[0] = a_d_0 * e_ka;

    // Central: homogeneous decay of A_c(0) plus the depot-driven Bateman term,
    // with the `ka ≈ ke` L'Hôpital fallback (branch on `.val()`).
    if (ka.val() - ke.val()).abs() < 1e-9 {
        state[1] = a_c_0 * e_ke + ka * a_d_0 * dtt * e_ke;
    } else {
        state[1] = a_c_0 * e_ke + (ka * a_d_0 / (ke - ka)) * (e_ka - e_ke);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sens::dual2::Dual2;
    use crate::types::{PkParams, PK_IDX_CL, PK_IDX_KA, PK_IDX_V};

    fn pk_of(cl: f64, v: f64, ka: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[PK_IDX_CL] = cl;
        p.values[PK_IDX_V] = v;
        p.values[PK_IDX_KA] = ka;
        p
    }

    /// The `f64` instantiation of the generic propagator must reproduce the
    /// production `f64` propagator bit-for-bit (same closed form, one source of
    /// truth in disguise).
    #[test]
    fn one_cpt_g_matches_production_f64() {
        for &(cl, v, rate, dt, s0) in &[
            (3.0, 30.0, 0.0, 2.5, 10.0),
            (5.0, 40.0, 8.0, 1.0, 0.0),
            (1.2, 12.0, 2.0, 6.0, 3.5),
        ] {
            let mut s_g = [s0];
            propagate_one_cpt_g::<f64>(&mut s_g, dt, cl, v, rate);
            let mut s_p = [s0];
            crate::pk::event_driven::propagate_one_cpt(&mut s_p, dt, &pk_of(cl, v, 1.0), rate);
            approx::assert_relative_eq!(s_g[0], s_p[0], max_relative = 1e-12);
        }
    }

    #[test]
    fn one_cpt_oral_g_matches_production_f64() {
        for &(cl, v, ka, dt, ad, ac) in &[
            (3.0, 30.0, 1.5, 2.5, 50.0, 5.0),
            (1.2, 12.0, 0.8, 4.0, 20.0, 1.0),
            // ka ≈ ke (L'Hôpital): ke = cl/v = 0.1, ka = 0.1.
            (3.0, 30.0, 0.1, 3.0, 40.0, 2.0),
        ] {
            let mut s_g = [ad, ac];
            propagate_one_cpt_oral_g::<f64>(&mut s_g, dt, cl, v, ka);
            let mut s_p = [ad, ac];
            crate::pk::event_driven::propagate_one_cpt_oral(&mut s_p, dt, &pk_of(cl, v, ka));
            approx::assert_relative_eq!(s_g[0], s_p[0], max_relative = 1e-12);
            approx::assert_relative_eq!(s_g[1], s_p[1], max_relative = 1e-12);
        }
    }

    /// Central FD grad + 4-point Hessian of a 2-arg `f64` closure.
    fn fd2(p: [f64; 2], val: impl Fn([f64; 2]) -> f64) -> ([f64; 2], [[f64; 2]; 2]) {
        let h = [1e-6 * (1.0 + p[0].abs()), 1e-6 * (1.0 + p[1].abs())];
        let hh = [1e-4 * (1.0 + p[0].abs()), 1e-4 * (1.0 + p[1].abs())];
        let mut g = [0.0; 2];
        for i in 0..2 {
            let mut up = p;
            up[i] += h[i];
            let mut dn = p;
            dn[i] -= h[i];
            g[i] = (val(up) - val(dn)) / (2.0 * h[i]);
        }
        let mut he = [[0.0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                let mut pp = p;
                pp[i] += hh[i];
                pp[j] += hh[j];
                let mut pm = p;
                pm[i] += hh[i];
                pm[j] -= hh[j];
                let mut mp = p;
                mp[i] -= hh[i];
                mp[j] += hh[j];
                let mut mm = p;
                mm[i] -= hh[i];
                mm[j] -= hh[j];
                he[i][j] = (val(pp) - val(pm) - val(mp) + val(mm)) / (4.0 * hh[i] * hh[j]);
            }
        }
        (g, he)
    }

    /// The `Dual2` instantiation's `∂(amount)/∂(cl,v)` (and Hessian) must match
    /// finite differences of the `f64` propagator — the propagator differentiates
    /// the compartment amount w.r.t. the PK parameters exactly.
    #[test]
    fn one_cpt_g_dual_matches_fd() {
        let (cl, v, rate, dt, s0) = (3.0, 30.0, 4.0, 2.5, 10.0);
        let mut sd = [Dual2::<2>::constant(s0)];
        propagate_one_cpt_g::<Dual2<2>>(
            &mut sd,
            dt,
            Dual2::var(cl, 0),
            Dual2::var(v, 1),
            Dual2::constant(rate),
        );
        let (g, he) = fd2([cl, v], |p| {
            let mut s = [s0];
            propagate_one_cpt_g::<f64>(&mut s, dt, p[0], p[1], rate);
            s[0]
        });
        for i in 0..2 {
            approx::assert_relative_eq!(sd[0].grad[i], g[i], max_relative = 1e-4, epsilon = 1e-8);
            for j in 0..2 {
                approx::assert_relative_eq!(
                    sd[0].hess[i][j],
                    he[i][j],
                    max_relative = 3e-3,
                    epsilon = 1e-6
                );
            }
        }
    }

    /// Same for 1-cpt oral, validating the central-compartment amount's `∂/∂(cl,v)`
    /// through the Bateman term.
    #[test]
    fn one_cpt_oral_g_dual_matches_fd() {
        let (cl, v, ka, dt, ad, ac) = (1.2, 12.0, 0.8, 4.0, 50.0, 3.0);
        let mut sd = [Dual2::<2>::constant(ad), Dual2::<2>::constant(ac)];
        propagate_one_cpt_oral_g::<Dual2<2>>(
            &mut sd,
            dt,
            Dual2::var(cl, 0),
            Dual2::var(v, 1),
            Dual2::constant(ka),
        );
        let (g, he) = fd2([cl, v], |p| {
            let mut s = [ad, ac];
            propagate_one_cpt_oral_g::<f64>(&mut s, dt, p[0], p[1], ka);
            s[1] // central amount
        });
        for i in 0..2 {
            approx::assert_relative_eq!(sd[1].grad[i], g[i], max_relative = 1e-4, epsilon = 1e-8);
            for j in 0..2 {
                approx::assert_relative_eq!(
                    sd[1].hess[i][j],
                    he[i][j],
                    max_relative = 3e-3,
                    epsilon = 1e-6
                );
            }
        }
    }
}
