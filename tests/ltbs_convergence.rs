//! Slow convergence test for the log-transform-both-sides (LTBS) error model
//! (issue #120).
//!
//! Fits the log-scale warfarin model (`examples/warfarin_ltbs.ferx`,
//! `log(DV) ~ additive(ADD_LOG)`) on the warfarin dataset and checks that the
//! full FOCEI LTBS pipeline — DV log-transform at load, log-wrapped predictions,
//! additive-on-log likelihood — converges to a sensible minimum and recovers
//! plausible PK parameters.
//!
//! Gate: skipped in the default PR job.
//!
//!   cargo test --features slow-tests --test ltbs_convergence
//!
//! ## NONMEM cross-check
//!
//! The log-scale model corresponds to NONMEM's
//!
//! ```text
//! $ERROR
//!   IPRED = LOG(F)
//!   Y     = IPRED + EPS(1)
//! ```
//!
//! with the warfarin DV column log-transformed. A NONMEM 7.5.1 FOCEI reference
//! fit (`tests/nonmem/warfarin_ltbs.{ctl,lst}`, run over `data/warfarin.csv`
//! with DV log-transformed) gives an essentially exact cross-engine match —
//! ferx fitting the natural-scale data with `log(DV) ~ additive` recovers the
//! same MLE as NONMEM fitting log-DV data with `Y = LOG(F) + EPS(1)`:
//!
//! | Parameter   | ferx       | NONMEM 7.5.1 |
//! |-------------|------------|--------------|
//! | OFV         | −675.302   | −675.302     |
//! | TVCL        | 0.132698   | 0.132697     |
//! | TVV         | 7.7381     | 7.7383       |
//! | TVKA        | 0.81085    | 0.81096      |
//! | ADD_LOG (SD)| 0.010564   | 0.010564     |
//! | ω²(CL)      | 0.028588   | 0.028594     |
//! | ω²(V)       | 0.009601   | 0.009604     |
//! | ω²(KA)      | 0.335861   | 0.335945     |
//!
//! The assertions below pin ferx to NONMEM's MLE so a regression in the LTBS
//! wiring (log-wrap, DV transform, or the additive-on-log likelihood) is caught.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{fit, read_nonmem_csv, FitOptions};
use std::path::Path;

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn ltbs_warfarin_fit_converges_and_recovers_pk() {
    let model = parse_model_file(Path::new("examples/warfarin_ltbs.ferx"))
        .expect("LTBS warfarin model must parse");
    assert!(model.log_transform, "model must be flagged LTBS");
    assert!(
        !model.dv_pre_logged,
        "log(DV) ~ additive logs DV in-engine (case 2)"
    );

    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");

    let mut opts = FitOptions::default();
    opts.outer_maxiter = 300;
    opts.run_covariance_step = false;
    opts.verbose = false;

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("LTBS FOCEI fit must succeed");

    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );
    assert!(result.converged, "LTBS fit should converge");

    // Cross-check against the NONMEM 7.5.1 FOCEI reference (see module docs and
    // tests/nonmem/warfarin_ltbs.lst). Tolerances absorb optimizer/engine noise.
    const NM_OFV: f64 = -675.3016;
    const NM_TVCL: f64 = 0.132697;
    const NM_TVV: f64 = 7.73826;
    const NM_TVKA: f64 = 0.810965;
    const NM_ADD_LOG_SD: f64 = 0.010564; // sqrt(SIGMA(1,1) = 1.11601e-4)
    const NM_OM_CL: f64 = 0.0285942;
    const NM_OM_V: f64 = 0.00960426;
    const NM_OM_KA: f64 = 0.335945;

    assert!(
        (result.ofv - NM_OFV).abs() < 0.5,
        "OFV {:.4} differs from NONMEM {NM_OFV:.4} by > 0.5",
        result.ofv
    );

    let rel = |got: f64, nm: f64| (got - nm).abs() / nm.abs();
    assert!(
        rel(result.theta[0], NM_TVCL) < 0.02,
        "TVCL {} vs NM {NM_TVCL}",
        result.theta[0]
    );
    assert!(
        rel(result.theta[1], NM_TVV) < 0.02,
        "TVV {} vs NM {NM_TVV}",
        result.theta[1]
    );
    assert!(
        rel(result.theta[2], NM_TVKA) < 0.02,
        "TVKA {} vs NM {NM_TVKA}",
        result.theta[2]
    );
    assert!(
        rel(result.sigma[0], NM_ADD_LOG_SD) < 0.05,
        "ADD_LOG SD {} vs NM {NM_ADD_LOG_SD}",
        result.sigma[0]
    );

    // OMEGA diagonal (variances), parallel to eta order CL, V, KA.
    let om: Vec<f64> = (0..3).map(|i| result.omega[(i, i)]).collect();
    assert!(
        rel(om[0], NM_OM_CL) < 0.10,
        "ω²(CL) {} vs NM {NM_OM_CL}",
        om[0]
    );
    assert!(
        rel(om[1], NM_OM_V) < 0.10,
        "ω²(V) {} vs NM {NM_OM_V}",
        om[1]
    );
    assert!(
        rel(om[2], NM_OM_KA) < 0.10,
        "ω²(KA) {} vs NM {NM_OM_KA}",
        om[2]
    );
}
