//! NONMEM cross-check for the issue #261 acceptance model (FeRx-NLME/ferx-r#154).
//!
//! #261 made an undefined `[structural_model]` parameter reference a hard parse
//! error (covered by the fast unit tests in `parser/model_parser.rs`). This
//! guards the *other* half of that issue's acceptance: the **corrected** model —
//! the same benchmark with the missing `CL = KE * V` line restored — fits
//! cleanly and lands on the NONMEM objective rather than one of the two broken
//! regimes the issue surfaced.
//!
//! ## Provenance / reference
//!
//! Model + data come from FeRx-NLME/ferx-r#154 (an external NONMEM/OpenPMX/ferx
//! benchmark; the dataset is the PAGE blind-analysis "DATASIM" set — 1-cpt oral,
//! DV already on the log scale, hence `DV ~ log_additive`). `data/datsim_oral.csv`
//! is that dataset with the `EVID` column already present (dose rows
//! `EVID=1, AMT=10000, MDV=1`), so it does not depend on the no-EVID dose
//! inference tracked separately in #262.
//!
//! Reference objective values on this exact model + data:
//!   - NONMEM 7.x `#OBJV`            : 70.85  (without the 2pi constant)
//!   - ferx-core `main`, autodiff    : 71.28
//!   - ferx-core `main`, FD (`ci`)   : ~72.14 (the gradient path shifts the
//!                                     ill-conditioned minimum slightly)
//! and the two broken regimes #261/#154 surfaced, which this test must exclude:
//!   - `CL` undefined → predictions floored → OFV ~2400 (now a parse error)
//!   - pre-Almquist FOCEI-INTER marginal → variance collapse → OFV ~ -29.5
//!
//! The band is therefore **two-sided** and centered on the NONMEM/ferx
//! agreement — a one-sided "lower is better" check (as in `gn_convergence.rs`)
//! would wrongly *pass* the -29.5 collapse. It is deliberately generous (±4):
//! the model is ill-conditioned (NONMEM itself only reached MINIMIZATION
//! TERMINATED), the FD-vs-autodiff minimum differs by ~1 OFV, and slow-tests do
//! not run on PR CI, so there is no pre-merge feedback loop to tune against.
//! Either broken regime is ~100+ OFV away and still fails decisively. Re-anchor
//! `EXPECTED_OFV` if a deliberate marginal-likelihood change moves the minimum
//! (cf. the "Baseline history" note in `gn_convergence.rs`).

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

/// The #154 benchmark model, corrected by restoring the `CL = KE * V` line whose
/// omission is the subject of #261. `[fit_options]` are set programmatically
/// below so the fit configuration does not depend on string parsing.
const MODEL_SRC: &str = r"
[parameters]
  theta TVV (3.40, 1.61, 4.1)   # V  (log scale)
  theta TVKE(-1.2, -4.6, 0.7)   # KE (log scale)
  theta TVKA(-0.7, -4.6, 0.7)   # KA (log scale, excess over KE)

  block_omega (ETA_V, ETA_KE, ETA_KA) = [
    1.0,
    0.001, 1.0,
    0.001, 0.001, 1.0
  ]

  sigma ADD_ERR ~ 1.0           # match NONMEM $SIGMA 1

[individual_parameters]
  V  = exp(TVV  + ETA_V)
  KE = exp(TVKE + ETA_KE)
  KA = KE + exp(TVKA + ETA_KA)
  CL = KE * V                   # <-- the line whose omission is issue #261

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ log_additive(ADD_ERR)
";

/// FOCEI fit of the corrected #154 model must land on the NONMEM objective,
/// within a band that excludes both documented broken regimes.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored #261 / ferx-r#154 acceptance: opt in with --features slow-tests"
)]
fn corrected_structural_model_matches_nonmem_ofv() {
    const EXPECTED_OFV: f64 = 71.28; // ferx-core main; NONMEM #OBJV 70.85
    const TOLERANCE: f64 = 4.0;

    let model = parse_model_string(MODEL_SRC).expect("corrected #154 model must parse");
    let pop = read_nonmem_csv(Path::new("data/datsim_oral.csv"), None, None)
        .expect("datsim_oral data must load");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.outer_maxiter = 300;
    opts.run_covariance_step = false;
    opts.verbose = false;

    let result =
        fit(&model, &pop, &model.default_params, &opts).expect("corrected #154 fit must run");

    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );
    assert!(
        (result.ofv - EXPECTED_OFV).abs() < TOLERANCE,
        "OFV {:.3} is outside the NONMEM-anchored band {:.2} ± {:.1} \
         (NONMEM #OBJV 70.85); a value near -29.5 (marginal collapse) or ~2400 \
         (floored predictions) is the #261 / ferx-r#154 regression this guards",
        result.ofv,
        EXPECTED_OFV,
        TOLERANCE,
    );
}
