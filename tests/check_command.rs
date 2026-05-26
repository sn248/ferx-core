//! Integration tests for `validate_model_file` (the engine behind `ferx check`).
//!
//! These exercise the public validation API end-to-end: parse a real example,
//! catch a missing block, catch a data/model covariate mismatch, and prove the
//! refactor kept `fit()`'s error string byte-identical to the diagnostic.
//! All return immediately (no fit convergence), so they belong in Tier 2.

use ferx_core::{fit, parse_full_model_file, read_nonmem_csv, validate_model_file, FitOptions};
use std::path::Path;

/// Write `content` to a uniquely-named temp `.ferx` file and return its path.
fn temp_model(tag: &str, content: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "ferx_check_{}_{}_{}.ferx",
        tag,
        std::process::id(),
        n
    ));
    std::fs::write(&path, content).expect("write temp model");
    path
}

const COV_MODEL: &str = "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL) * (WGT / 70.0)
  V  = TVV

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=1.0)

[error_model]
  DV ~ proportional(PROP_ERR)
";

#[test]
fn valid_example_passes_with_no_diagnostics() {
    let report = validate_model_file("examples/warfarin_bobyqa.ferx", None);
    assert!(
        report.valid,
        "unexpected diagnostics: {:?}",
        report.diagnostics
    );
    assert_eq!(report.error_count(), 0);
    assert_eq!(report.model, "warfarin_bobyqa");
}

#[test]
fn missing_block_is_reported_as_e_missing_block() {
    // No [error_model] block.
    let model = temp_model(
        "missing_block",
        "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=1.0)
",
    );
    let report = validate_model_file(model.to_str().unwrap(), None);
    assert!(!report.valid);
    let d = &report.diagnostics[0];
    assert_eq!(d.code, "E_MISSING_BLOCK");
    assert_eq!(d.block.as_deref(), Some("error_model"));
    let _ = std::fs::remove_file(&model);
}

#[test]
fn missing_covariate_is_reported_with_data() {
    // bioavailability.csv carries no covariate columns, but the model references WGT.
    let model = temp_model("missing_cov", COV_MODEL);
    let report = validate_model_file(model.to_str().unwrap(), Some("data/bioavailability.csv"));
    assert!(!report.valid);
    let d = report
        .diagnostics
        .iter()
        .find(|d| d.code == "E_MISSING_COVARIATE")
        .expect("expected E_MISSING_COVARIATE");
    assert!(d.message.contains("WGT"));
    assert!(d.suggestion.is_some());
    let _ = std::fs::remove_file(&model);
}

#[test]
fn no_data_means_no_covariate_check() {
    // Same model, but without --data the covariate check does not run, so the
    // model is structurally valid.
    let model = temp_model("no_data", COV_MODEL);
    let report = validate_model_file(model.to_str().unwrap(), None);
    assert!(
        report.valid,
        "unexpected diagnostics: {:?}",
        report.diagnostics
    );
    let _ = std::fs::remove_file(&model);
}

/// Regression guard: the message `fit()` produces for a missing covariate must
/// stay byte-identical to the diagnostic `validate_model_file` reports — both
/// now flow through the shared `check_covariates`.
#[test]
fn fit_error_matches_check_diagnostic_for_missing_covariate() {
    let model_path = temp_model("fit_regression", COV_MODEL);
    let report = validate_model_file(
        model_path.to_str().unwrap(),
        Some("data/bioavailability.csv"),
    );
    let diag_msg = report
        .diagnostics
        .iter()
        .find(|d| d.code == "E_MISSING_COVARIATE")
        .expect("diagnostic present")
        .message
        .clone();

    let model = parse_full_model_file(&model_path).unwrap().model;
    let pop = read_nonmem_csv(Path::new("data/bioavailability.csv"), None, None).unwrap();
    let fit_err = fit(&model, &pop, &model.default_params, &FitOptions::default())
        .expect_err("fit must reject the missing covariate before fitting");

    assert_eq!(diag_msg, fit_err);
    let _ = std::fs::remove_file(&model_path);
}

/// Block-level line numbers are recorded on the parsed model.
#[test]
fn parser_records_block_header_lines() {
    let model_path = temp_model("block_lines", COV_MODEL);
    let parsed = parse_full_model_file(&model_path).unwrap();
    // `[parameters]` is line 1; `[individual_parameters]` line 7; `[error_model]` line 14.
    assert_eq!(parsed.block_lines.get("parameters"), Some(&1));
    assert_eq!(parsed.block_lines.get("individual_parameters"), Some(&7));
    assert_eq!(parsed.block_lines.get("error_model"), Some(&14));
    let _ = std::fs::remove_file(&model_path);
}

/// The report serializes to JSON with the documented shape.
#[test]
fn report_serializes_to_json() {
    let report = validate_model_file("examples/warfarin_bobyqa.ferx", None);
    let json = serde_json::to_string(&report).unwrap();
    assert!(json.contains("\"valid\":true"));
    assert!(json.contains("\"model\":\"warfarin_bobyqa\""));
    assert!(json.contains("\"diagnostics\":[]"));
}
