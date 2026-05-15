//! Standalone SIR — run Sampling Importance Resampling against an existing
//! `FitResult` without re-fitting.
//!
//! Mirrors the SIR step that [`fit()`](crate::api::fit) runs inline when
//! `options.sir = true`, but lets callers drive SIR after a fit has completed
//! (potentially from a different session, loaded via `.fitrx`).
//!
//! Hash-verification rules:
//! - If the caller supplies `model` / `population` directly, those are used
//!   as-is. We cannot verify against `fit.model_hash` / `fit.data_hash`
//!   because the in-memory values don't carry their source bytes.
//! - If the caller passes `None`, we re-read from `fit.model_path` /
//!   `fit.data_path`. If a stored hash exists, a mismatch is a **hard error**
//!   — the whole point of `run_sir` is to refuse stale data.

use crate::estimation::uncertainty_samples::fitted_params_from_result;
use crate::io::hash::sha256_file;
use crate::types::*;
use nalgebra::DVector;
use std::path::Path;

/// Run SIR against an existing fit. Returns a new `FitResult` that is a clone
/// of `fit` with the `sir_*` fields populated (and an extra warning when the
/// underlying `run_sir` reported one).
///
/// # Arguments
/// - `fit`: the maximum-likelihood fit to SIR-refine. Must carry a
///   `covariance_matrix` (i.e. the original fit ran with `covariance = true`).
/// - `model`: pre-compiled model. When `None`, re-parsed from `fit.model_path`.
/// - `population`: dataset. When `None`, re-read from `fit.data_path`.
/// - `options`: SIR-relevant fields read are `sir_samples`, `sir_resamples`,
///   `sir_seed`, `sir_keep_samples`, plus the inner-loop settings
///   (`inner_maxiter`, `inner_tol`, `interaction`, `mu_referencing`,
///   `verbose`, `cancel`). Other fields (e.g. `method`) are ignored.
pub fn run_sir(
    fit: &FitResult,
    model: Option<&CompiledModel>,
    population: Option<&Population>,
    options: &FitOptions,
) -> Result<FitResult, String> {
    // Hash verification runs before the covariance check so a stale-input
    // error wins over a missing-cov error. A user pointing at the wrong
    // model or dataset should hear about that first; the cov-missing case
    // is downstream and only matters once the inputs are confirmed.

    // --- Resolve model -----------------------------------------------------
    let model_owned: Option<CompiledModel>;
    let model_ref: &CompiledModel = match model {
        Some(m) => m,
        None => {
            let path = fit.model_path.as_deref().ok_or_else(|| {
                "run_sir: no model supplied and fit.model_path is None. \
                 Either pass `model = Some(&model)` or re-fit via fit_from_files \
                 so the path is recorded."
                    .to_string()
            })?;
            if let Some(expected) = &fit.model_hash {
                let actual = sha256_file(Path::new(path))?;
                if &actual != expected {
                    return Err(format!(
                        "run_sir: model hash mismatch for {}. Stored: {}, current: {}. \
                         The .ferx file has changed since the fit was produced — refusing \
                         to run SIR against stale source.",
                        path, expected, actual
                    ));
                }
            }
            let parsed = crate::parser::model_parser::parse_model_file(Path::new(path))?;
            model_owned = Some(parsed);
            model_owned.as_ref().unwrap()
        }
    };

    // --- Resolve population -----------------------------------------------
    let pop_owned: Option<Population>;
    let pop_ref: &Population = match population {
        Some(p) => p,
        None => {
            let path = fit.data_path.as_deref().ok_or_else(|| {
                "run_sir: no population supplied and fit.data_path is None. \
                 Either pass `population = Some(&pop)` or re-fit via fit_from_files \
                 so the path is recorded."
                    .to_string()
            })?;
            if let Some(expected) = &fit.data_hash {
                let actual = sha256_file(Path::new(path))?;
                if &actual != expected {
                    return Err(format!(
                        "run_sir: data hash mismatch for {}. Stored: {}, current: {}. \
                         The dataset has changed since the fit was produced — refusing \
                         to run SIR against stale data.",
                        path, expected, actual
                    ));
                }
            }
            let p = crate::io::datareader::read_nonmem_csv(Path::new(path), None, None)?;
            pop_owned = Some(p);
            pop_owned.as_ref().unwrap()
        }
    };

    // --- Sanity-check dimensions ------------------------------------------
    if model_ref.n_eta != fit.omega.nrows() {
        return Err(format!(
            "run_sir: model has n_eta = {} but fit.omega is {}×{}. \
             Model file may have been edited in a way that doesn't change its hash \
             coverage — verify you supplied the same model used for the fit.",
            model_ref.n_eta,
            fit.omega.nrows(),
            fit.omega.ncols()
        ));
    }
    if !fit.subjects.is_empty() && fit.subjects[0].eta.len() != model_ref.n_eta {
        return Err(format!(
            "run_sir: fit.subjects[0] has eta dim {} but model has n_eta = {}. \
             Subject EBEs are inconsistent with the supplied model.",
            fit.subjects[0].eta.len(),
            model_ref.n_eta
        ));
    }

    // --- Reconstruct ModelParameters and eta_hats -------------------------
    let params = fitted_params_from_result(fit, model_ref);
    let eta_hats: Vec<DVector<f64>> = fit.subjects.iter().map(|s| s.eta.clone()).collect();

    // --- Now require a covariance matrix to seed the proposal -------------
    let cov = fit.covariance_matrix.as_ref().ok_or_else(|| {
        "run_sir requires fit.covariance_matrix; re-run the original fit \
             with covariance = true."
            .to_string()
    })?;

    // --- Run SIR (identical to the inline path in fit()) ------------------
    let sir = crate::estimation::sir::run_sir_core(
        model_ref, pop_ref, &params, &eta_hats, cov, fit.ofv, options,
    )?;

    // --- Build the augmented FitResult ------------------------------------
    let mut out = fit.clone();
    out.sir_ci_theta = Some(sir.ci_theta);
    out.sir_ci_omega = Some(sir.ci_omega);
    out.sir_ci_sigma = Some(sir.ci_sigma);
    out.sir_ess = Some(sir.effective_sample_size);
    out.sir_resamples_packed = sir.resamples_packed;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::fit_from_files;

    // Use the in-tree warfarin example + data. They live at repo paths
    // `examples/warfarin.ferx` and `data/warfarin.csv` (see CLAUDE.md);
    // tests run from the crate root, so relative paths work directly.
    const MODEL_PATH: &str = "examples/warfarin.ferx";
    const DATA_PATH: &str = "data/warfarin.csv";

    fn copy_example_to_tempdir(dir: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf) {
        // Hash-mismatch tests need to mutate the source files. Copy them to a
        // tempdir so we don't touch the checked-in examples.
        let model = dir.join("model.ferx");
        let data = dir.join("data.csv");
        std::fs::copy(MODEL_PATH, &model).unwrap();
        std::fs::copy(DATA_PATH, &data).unwrap();
        (model, data)
    }

    fn quick_opts() -> FitOptions {
        // Small SIR settings so the test stays under a few seconds.
        FitOptions {
            verbose: false,
            run_covariance_step: true,
            sir_samples: 8,
            sir_resamples: 4,
            sir_seed: Some(1),
            ..FitOptions::default()
        }
    }

    #[test]
    fn paths_and_hashes_are_populated_by_fit_from_files() {
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let fit = fit_from_files(
            model_path.to_str().unwrap(),
            data_path.to_str().unwrap(),
            None,
            Some(quick_opts()),
        )
        .expect("fit must converge");

        assert_eq!(
            fit.model_path.as_deref(),
            Some(model_path.to_str().unwrap())
        );
        assert_eq!(fit.data_path.as_deref(), Some(data_path.to_str().unwrap()));
        assert_eq!(fit.model_hash.as_deref().map(|s| s.len()), Some(64));
        assert_eq!(fit.data_hash.as_deref().map(|s| s.len()), Some(64));
        assert_eq!(
            fit.model_hash.as_deref(),
            Some(sha256_file(&model_path).unwrap().as_str())
        );
        assert_eq!(
            fit.data_hash.as_deref(),
            Some(sha256_file(&data_path).unwrap().as_str())
        );
    }

    #[test]
    fn run_sir_rejects_when_no_covariance() {
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let opts = FitOptions {
            verbose: false,
            run_covariance_step: false,
            ..FitOptions::default()
        };
        let fit = fit_from_files(
            model_path.to_str().unwrap(),
            data_path.to_str().unwrap(),
            None,
            Some(opts.clone()),
        )
        .expect("fit must converge");

        assert!(fit.covariance_matrix.is_none());
        let err = run_sir(&fit, None, None, &opts).unwrap_err();
        assert!(
            err.contains("covariance_matrix"),
            "expected cov-missing message, got: {}",
            err
        );
    }

    #[test]
    fn run_sir_detects_modified_model_file() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let opts = quick_opts();
        let fit = fit_from_files(
            model_path.to_str().unwrap(),
            data_path.to_str().unwrap(),
            None,
            Some(opts.clone()),
        )
        .expect("fit must converge");

        // Tamper with the model file (append whitespace — enough to flip the
        // SHA-256). The next run_sir call must refuse.
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&model_path)
            .unwrap();
        writeln!(f, "  ").unwrap();
        drop(f);

        let err = run_sir(&fit, None, None, &opts).unwrap_err();
        assert!(
            err.contains("model hash mismatch"),
            "expected hash-mismatch message, got: {}",
            err
        );
    }
}
