use crate::estimation::outer_optimizer::optimize_population;
use crate::estimation::saem;
use crate::io::datareader::read_nonmem_csv;
use crate::io::output;
use crate::pk;
use crate::stats::likelihood::{
    compute_cwres, foce_subject_nll, foce_subject_nll_iov, split_obs_by_occasion,
};
use crate::stats::residual_error::compute_iwres;
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use std::path::Path;
use std::time::Instant;

/// Route predictions through the TV-aware dispatcher. Used by post-processing
/// (sdtab generation) and the public `predict` API. Same semantics as
/// `stats::likelihood::model_predictions` — passes `(theta, eta)` so the
/// dispatcher can build per-event PK params when the subject has TV covariates.
fn model_preds(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Vec<f64> {
    pk::compute_predictions_with_tv(model, subject, theta, eta)
}

/// Run a model file with a NONMEM-format CSV dataset.
/// Returns (FitResult, Population) so caller can write sdtab.
pub fn run_model_with_data(
    model_path: &str,
    data_path: &str,
) -> Result<(FitResult, Population), String> {
    use crate::parser::model_parser::parse_full_model_file;

    let mut parsed = parse_full_model_file(Path::new(model_path))?;
    set_model_name(&mut parsed.model, model_path);

    eprintln!("Model: {}", parsed.model.name);

    let iov_col = parsed.fit_options.iov_column.as_deref();
    let population = read_nonmem_csv(Path::new(data_path), None, iov_col)?;
    eprintln!(
        "Data:  {} subjects, {} observations from {}",
        population.subjects.len(),
        population.n_obs(),
        data_path
    );

    let init_params = build_init_params(&parsed);
    let result = fit(
        &parsed.model,
        &population,
        &init_params,
        &parsed.fit_options,
    )?;
    Ok((result, population))
}

/// Run a model file with simulated data (from [simulation] block).
/// Returns (FitResult, Population) so caller can write sdtab.
pub fn run_model_simulate(model_path: &str) -> Result<(FitResult, Population), String> {
    use crate::parser::model_parser::parse_full_model_file;
    use std::collections::HashMap;

    let mut parsed = parse_full_model_file(Path::new(model_path))?;
    let sim_spec = parsed
        .simulation
        .clone()
        .ok_or("Model file has no [simulation] block — use --data instead")?;
    set_model_name(&mut parsed.model, model_path);

    eprintln!("Model: {}", parsed.model.name);

    // Build template population
    let subjects: Vec<Subject> = (1..=sim_spec.n_subjects)
        .map(|i| Subject {
            id: format!("{}", i),
            doses: vec![DoseEvent::new(
                0.0,
                sim_spec.dose_amt,
                sim_spec.dose_cmt,
                0.0,
                false,
                0.0,
            )],
            obs_times: sim_spec.obs_times.clone(),
            observations: vec![0.0; sim_spec.obs_times.len()],
            obs_cmts: vec![1; sim_spec.obs_times.len()],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            cens: vec![0; sim_spec.obs_times.len()],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
        })
        .collect();
    let template = Population {
        subjects,
        covariate_names: vec![],
        dv_column: "dv".into(),
    };

    // Simulate
    eprintln!(
        "Simulating {} subjects (seed={})...",
        sim_spec.n_subjects, sim_spec.seed
    );
    let sim_results = simulate_with_seed(
        &parsed.model,
        &template,
        &parsed.model.default_params,
        1,
        sim_spec.seed,
    );

    let mut population = template;
    for subject in &mut population.subjects {
        let sims: Vec<_> = sim_results.iter().filter(|s| s.id == subject.id).collect();
        for (j, s) in sims.iter().enumerate() {
            if j < subject.observations.len() {
                subject.observations[j] = s.dv_sim.max(0.001);
            }
        }
    }

    eprintln!(
        "Loaded {} subjects, {} observations",
        population.subjects.len(),
        population.n_obs()
    );

    let init_params = build_init_params(&parsed);
    let result = fit(
        &parsed.model,
        &population,
        &init_params,
        &parsed.fit_options,
    )?;
    Ok((result, population))
}

/// Legacy alias
pub fn run_from_file(path: &str) -> Result<FitResult, String> {
    run_model_simulate(path).map(|(r, _)| r)
}

fn set_model_name(model: &mut CompiledModel, path: &str) {
    if model.name == "Unnamed" {
        if let Some(stem) = Path::new(path).file_stem().and_then(|s| s.to_str()) {
            model.name = stem.to_string();
        }
    }
}

fn build_init_params(parsed: &ParsedModel) -> ModelParameters {
    parsed.model.default_params.clone()
}

/// Fail early if the model references covariates that the data doesn't carry.
/// Case-sensitive: `CRCL` and `crcl` are distinct names. Historically a missing
/// covariate silently evaluated to zero, which left fits stuck at the initial
/// estimates with no visible diagnostic (see commit introducing this check).
fn validate_covariates(model: &CompiledModel, population: &Population) -> Result<(), String> {
    let missing: Vec<&str> = model
        .referenced_covariates
        .iter()
        .filter(|name| !population.covariate_names.iter().any(|n| n == *name))
        .map(|s| s.as_str())
        .collect();

    if missing.is_empty() {
        return Ok(());
    }

    let available = if population.covariate_names.is_empty() {
        "(none)".to_string()
    } else {
        population.covariate_names.join(", ")
    };
    Err(format!(
        "Model references covariate(s) not found in data (case-sensitive): {}. \
         Available covariate columns: {}.",
        missing.join(", "),
        available
    ))
}

/// High-level fit: model file path + data file path → FitResult
pub fn fit_from_files(
    model_path: &str,
    data_path: &str,
    covariate_columns: Option<&[&str]>,
    options: Option<FitOptions>,
) -> Result<FitResult, String> {
    let mut model = crate::parser::model_parser::parse_model_file(Path::new(model_path))?;
    let population = read_nonmem_csv(Path::new(data_path), covariate_columns, None)?;
    let opts = options.unwrap_or_default();
    model.bloq_method = opts.bloq_method;
    model.gradient_method = opts.gradient_method;
    fit(&model, &population, &model.default_params, &opts)
}

/// Main fit entry point: CompiledModel + Population → FitResult.
///
/// When `options.threads` is `Some(n)`, the fit runs inside a scoped rayon
/// pool of `n` workers, so this setting is per-call (different fits in the
/// same process can use different thread counts). When `None`, rayon's
/// global pool is used (one worker per logical CPU).
pub fn fit(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> Result<FitResult, String> {
    validate_covariates(model, population)?;
    match options.threads {
        Some(n) if n > 0 => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(|e| format!("failed to build rayon pool with {} threads: {}", n, e))?;
            pool.install(|| fit_inner(model, population, init_params, options))
        }
        _ => fit_inner(model, population, init_params, options),
    }
}

/// Probe whether NLopt CRS2-LM (used for global_search) is available.
fn probe_nlopt_algorithms() -> Vec<String> {
    fn dummy_obj(_x: &[f64], _grad: Option<&mut [f64]>, _data: &mut ()) -> f64 {
        0.0
    }
    let available = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _opt = nlopt::Nlopt::new(
            nlopt::Algorithm::Crs2Lm,
            1,
            dummy_obj,
            nlopt::Target::Minimize,
            (),
        );
    }));
    if available.is_err() {
        vec![
            "NLopt CRS2-LM not available in this build — global_search = true will fail. \
             Install a full NLopt build: brew install nlopt / apt install libnlopt-dev"
                .to_string(),
        ]
    } else {
        vec![]
    }
}

fn fit_inner(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> Result<FitResult, String> {
    let fit_start = Instant::now();
    let chain = options.method_chain();
    // Compute up-front so we can both surface the warnings before the fit
    // starts (a long-running fit shouldn't bury a "this option is unused"
    // notice at the end) and carry them through into FitResult.warnings.
    let unsupported_warnings = options.unsupported_keys_warnings();

    // Capture thread count before chain runs (current_num_threads() reports
    // whichever Rayon pool is active — scoped pool when threads=Some, else global).
    let n_threads_used = rayon::current_num_threads();

    // Initialise the per-iteration optimizer trace if requested.
    if options.optimizer_trace {
        let pid = std::process::id();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let path = format!("/tmp/ferx_trace_{}_{}.csv", pid, ts);
        if let Err(e) = crate::estimation::trace::init(path.clone()) {
            eprintln!("[ferx] warning: could not open trace file {}: {}", path, e);
        } else {
            eprintln!("[ferx] optimizer trace → {}", path);
        }
    }

    // Reset gradient timing counters for this fit so FERX_TIME_GRADIENTS
    // readouts are per-call rather than cumulative across a long R session.
    let time_gradients = std::env::var("FERX_TIME_GRADIENTS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if time_gradients {
        crate::estimation::inner_optimizer::GRADIENT_TIMINGS.reset();
    }
    if options.verbose {
        let chain_str: Vec<&str> = chain.iter().map(|m| m.label()).collect();
        // rayon::current_num_threads() reports whichever pool par_iter would use
        // from the current call — the scoped pool when options.threads is Some,
        // otherwise the global pool. So this stays accurate in both paths.
        let n_threads = rayon::current_num_threads();
        let thread_word = if n_threads == 1 { "thread" } else { "threads" };
        if !unsupported_warnings.is_empty() {
            eprintln!("--- Warnings ---");
            for w in &unsupported_warnings {
                eprintln!("  * {}", w);
            }
            eprintln!();
        }
        eprintln!(
            "Starting estimation (chain: {}) on {} {}...",
            chain_str.join(" → "),
            n_threads,
            thread_word
        );
        eprintln!(
            "  {} subjects, {} observations",
            population.subjects.len(),
            population.n_obs()
        );
        eprintln!(
            "  {} thetas, {} etas, {} sigmas",
            model.n_theta, model.n_eta, model.n_epsilon
        );
    }

    // Guard: SAEM does not support IOV (kappas are not sampled in the SAEM
    // stochastic approximation loop).  Fail early with a clear message.
    if model.n_kappa > 0 && chain.iter().any(|&m| m == EstimationMethod::Saem) {
        return Err(
            "method = saem does not support IOV (n_kappa > 0). \
             Use method = foce or method = focei for models with kappa declarations."
                .to_string(),
        );
    }

    // Guard: the trust-region outer optimizer does not yet thread kappas through
    // its OFV evaluation (see trust_region.rs::FoceiProblem::ofv_fixed which
    // passes &[] for kappas). Running it on an IOV model would silently produce
    // a wrong OFV. Fail early until the trust-region path supports IOV.
    if model.n_kappa > 0 && options.optimizer == Optimizer::TrustRegion {
        return Err(
            "optimizer = trust_region does not support IOV (n_kappa > 0). \
             Use optimizer = bobyqa, slsqp, lbfgs, nlopt_lbfgs, mma, or bfgs \
             for models with kappa declarations."
                .to_string(),
        );
    }

    // Pre-compute n_params (uses init_params, available before chain runs).
    let fixed_mask = crate::estimation::parameterization::packed_fixed_mask(init_params);
    let n_params_pre = fixed_mask.iter().filter(|&&b| !b).count();

    // Probe NLopt algorithm availability only when global_search will actually
    // run — otherwise the CRS2-LM warning is misleading for users who never
    // requested it.
    let nlopt_missing = if options.global_search {
        probe_nlopt_algorithms()
    } else {
        Vec::new()
    };

    // Covariance step cost warning: fire before chain so user sees it
    // immediately. Use checked_mul so an absurd parameter count cannot wrap
    // and produce a bogus estimate; on overflow we still warn but suppress
    // the numeric estimate.
    let covariance_n_evals_estimated = if options.run_covariance_step && n_params_pre > 30 {
        n_params_pre.checked_mul(n_params_pre)
    } else {
        None
    };

    // Run each stage in sequence, feeding params forward.
    let n_stages = chain.len();
    let mut stage_params: ModelParameters = init_params.clone();
    let mut result: Option<crate::estimation::outer_optimizer::OuterResult> = None;
    let mut accumulated_warnings: Vec<String> = model.parse_warnings.clone();
    accumulated_warnings.extend(unsupported_warnings);

    // Emit NLopt / covariance warnings before any work starts.
    accumulated_warnings.extend(nlopt_missing.iter().cloned());

    // TV-covariate AD downgrade notice — only fires for the *unsupported*
    // structural models (oral / 3-cpt). Supported analytical models
    // (1- and 2-cpt IV bolus + infusion) now run the event-driven AD path
    // for TV-cov subjects via `crate::ad::event_driven_ad`. ODE models
    // never had an AD path to start with so the message would be noise.
    let want_ad = matches!(
        model.gradient_method,
        GradientMethod::Ad | GradientMethod::Auto
    );
    // All analytical PK models are now covered by event-driven AD.
    let event_driven_ad_supported = !matches!(model.pk_model, _ if model.ode_spec.is_some())
        && matches!(
            model.pk_model,
            PkModel::OneCptIvBolus
                | PkModel::OneCptInfusion
                | PkModel::OneCptOral
                | PkModel::TwoCptIvBolus
                | PkModel::TwoCptInfusion
                | PkModel::TwoCptOral
                | PkModel::ThreeCptIvBolus
                | PkModel::ThreeCptInfusion
                | PkModel::ThreeCptOral
        );
    let tv_unsupported_subjects = if want_ad
        && model.tv_fn.is_some()
        && model.ode_spec.is_none()
        && !event_driven_ad_supported
    {
        population
            .subjects
            .iter()
            .filter(|s| s.has_tv_covariates())
            .count()
    } else {
        0
    };
    if tv_unsupported_subjects > 0 {
        // All analytical PkModel variants are currently in the supported
        // list, so this path only fires if a future variant is added
        // without AD coverage. Kept as a forward-looking guard.
        accumulated_warnings.push(format!(
            "AD gradients disabled for {}/{} subjects with time-varying covariates \
             on this structural model. Falling back to FD.",
            tv_unsupported_subjects,
            population.subjects.len()
        ));
    }
    if options.run_covariance_step && n_params_pre > 30 {
        if let Some(n_evals) = covariance_n_evals_estimated {
            accumulated_warnings.push(format!(
                "Covariance step: {} parameters → {} OFV evaluations \
                 (finite-difference Hessian). This may take several minutes \
                 on complex models.",
                n_params_pre, n_evals
            ));
        } else {
            // n_params² overflowed usize — warn without the (wrapped) number.
            accumulated_warnings.push(format!(
                "Covariance step: {} parameters → n² OFV evaluations \
                 (finite-difference Hessian). Estimate exceeds usize range; \
                 expect this to be very slow.",
                n_params_pre
            ));
        }
    }

    let mut total_iterations: usize = 0;

    for (stage_idx, &method) in chain.iter().enumerate() {
        if crate::cancel::is_cancelled(&options.cancel) {
            return Err("cancelled by user".to_string());
        }
        let is_last = stage_idx + 1 == n_stages;
        let mut stage_opts = options.clone();
        stage_opts.method = method;
        stage_opts.methods = Vec::new();
        // Per-stage interaction flag: FOCEI=on, FOCE=off, others inherit from user options.
        match method {
            EstimationMethod::FoceI => stage_opts.interaction = true,
            EstimationMethod::Foce => stage_opts.interaction = false,
            _ => {}
        }
        // Only run the covariance step on the final stage to avoid wasted work.
        if !is_last {
            stage_opts.run_covariance_step = false;
            stage_opts.sir = false;
        }

        if options.verbose && n_stages > 1 {
            eprintln!(
                "\n── Stage {}/{}: {} ──",
                stage_idx + 1,
                n_stages,
                method.label()
            );
        }

        let stage_result = match method {
            EstimationMethod::Saem => {
                saem::run_saem(model, population, &stage_params, &stage_opts)?
            }
            EstimationMethod::FoceGn | EstimationMethod::FoceGnHybrid => {
                crate::estimation::gauss_newton::run_foce_gn(
                    model,
                    population,
                    &stage_params,
                    &stage_opts,
                )
            }
            _ => optimize_population(model, population, &stage_params, &stage_opts),
        };

        stage_params = stage_result.params.clone();
        total_iterations += stage_result.n_iterations;
        for w in &stage_result.warnings {
            accumulated_warnings.push(if n_stages > 1 {
                format!("[{}] {}", method.label(), w)
            } else {
                w.clone()
            });
        }
        result = Some(stage_result);
    }

    if crate::cancel::is_cancelled(&options.cancel) {
        return Err("cancelled by user".to_string());
    }

    let mut result = result.expect("method chain must have at least one stage");
    // Overwrite with chain-aware totals
    result.n_iterations = total_iterations;
    result.warnings = accumulated_warnings;

    // Thread efficiency warnings (post-chain, uses n_threads_used captured above).
    let n_subjects = population.subjects.len();
    if n_subjects > 0 && n_threads_used > n_subjects {
        // `threads = 0` is not a valid Rayon pool size, so for n_subjects = 1
        // we still suggest a 1-thread pool.
        let suggested = n_subjects.max(1);
        result.warnings.push(format!(
            "{} threads configured but only {} subject(s) — consider threads = {} to reduce \
             scheduling overhead (no speed benefit beyond n_subjects)",
            n_threads_used, n_subjects, suggested
        ));
    }
    // SAEM-specific: MH scheduling has higher per-subject overhead than FOCE.
    // Skip when n_subjects < 2 (n_subjects/2 = 0 is meaningless and the prior
    // warning already covers the n_threads > n_subjects case).
    if chain.iter().any(|&m| m == EstimationMethod::Saem) && n_subjects >= 2 {
        let suggested = (n_subjects / 2).max(1);
        if n_threads_used > suggested {
            result.warnings.push(format!(
                "SAEM with more threads than subjects/2 may be slower due to MH scheduling \
                 overhead. Consider threads = {} for SAEM.",
                suggested
            ));
        }
    }

    // Compute per-subject diagnostics
    let subjects = compute_subject_results(
        model,
        population,
        &result.params,
        &result.eta_hats,
        &result.h_matrices,
        &result.kappas,
        options.interaction,
    );

    let n_obs = population.n_obs();
    let n_params = n_params_pre;

    let ofv = result.ofv;
    let aic = ofv + 2.0 * n_params as f64;
    let bic = ofv + n_params as f64 * (n_obs as f64).ln();

    // Extract SEs from covariance matrix using converged parameter values
    let (se_theta, se_omega, se_sigma, se_kappa) =
        extract_standard_errors(&result.covariance_matrix, &result.params);

    // Optional SIR step
    let mut warnings = result.warnings;

    // Report detected mu-referencing relationships (only when feature is enabled)
    if options.mu_referencing && !model.mu_refs.is_empty() {
        let mut names: Vec<&String> = model.mu_refs.keys().collect();
        names.sort();
        warnings.push(format!(
            "mu-ref: {}",
            names
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    // When M3 BLOQ is combined with non-interaction FOCE, mixing linearized
    // Gaussian residuals with non-linearized log Φ terms gives inconsistent
    // OFVs near the LLOQ boundary. The FOCE dispatcher routes affected
    // subjects through FOCEI internally — surface the promotion to the user.
    if matches!(model.bloq_method, BloqMethod::M3)
        && matches!(
            options.method,
            EstimationMethod::Foce | EstimationMethod::FoceGn
        )
        && !options.interaction
        && population.subjects.iter().any(|s| s.has_bloq())
    {
        warnings.push(
            "M3 BLOQ handling requires FOCEI semantics; subjects with CENS=1 \
             rows were evaluated with η-interaction. Set method=focei explicitly \
             to silence this notice."
                .to_string(),
        );
    }
    let sir_result = if options.sir && !crate::cancel::is_cancelled(&options.cancel) {
        if let Some(ref cov) = result.covariance_matrix {
            if options.verbose {
                eprintln!("\nRunning SIR...");
            }
            match crate::estimation::sir::run_sir(
                model,
                population,
                &result.params,
                &result.eta_hats,
                cov,
                result.ofv,
                options,
            ) {
                Ok(sir) => Some(sir),
                Err(e) => {
                    warnings.push(format!("SIR failed: {}", e));
                    None
                }
            }
        } else {
            warnings.push(
                "SIR requested but covariance matrix is not available. \
                 Enable covariance = true in [fit_options]."
                    .to_string(),
            );
            None
        }
    } else {
        None
    };

    let final_method = *chain.last().expect("chain non-empty");
    let grad_inner =
        crate::build_info::gradient_method_inner(&crate::build_info::BUILD_INFO, model);
    let grad_outer = crate::build_info::gradient_method_outer(
        &crate::build_info::BUILD_INFO,
        final_method,
        options.optimizer,
    );

    // Flush and close the trace file; capture path for FitResult.
    let trace_path = crate::estimation::trace::finish();

    // Shrinkage
    let shrinkage_eta = compute_eta_shrinkage(&subjects, &result.params.omega.matrix);
    let shrinkage_eps = compute_eps_shrinkage(&subjects);

    // Covariance status
    let covariance_status = if !options.run_covariance_step {
        CovarianceStatus::NotRequested
    } else if result.covariance_matrix.is_some() {
        CovarianceStatus::Computed
    } else {
        CovarianceStatus::Failed
    };

    let wall_time_secs = fit_start.elapsed().as_secs_f64();

    let fit_result = FitResult {
        method: final_method,
        method_chain: chain.clone(),
        converged: result.converged,
        ofv,
        aic,
        bic,
        theta: result.params.theta.clone(),
        theta_names: result.params.theta_names.clone(),
        eta_names: result.params.omega.eta_names.clone(),
        omega: result.params.omega.matrix.clone(),
        sigma: result.params.sigma.values.clone(),
        sigma_names: result.params.sigma.names.clone(),
        error_model: model.error_model,
        covariance_matrix: result.covariance_matrix,
        se_theta,
        se_omega,
        se_sigma,
        theta_fixed: result.params.theta_fixed.clone(),
        omega_fixed: result.params.omega_fixed.clone(),
        sigma_fixed: result.params.sigma_fixed.clone(),
        subjects,
        n_obs,
        n_subjects: population.subjects.len(),
        n_parameters: n_params,
        n_iterations: result.n_iterations,
        interaction: options.interaction,
        warnings,
        sir_ci_theta: sir_result.as_ref().map(|s| s.ci_theta.clone()),
        sir_ci_omega: sir_result.as_ref().map(|s| s.ci_omega.clone()),
        sir_ci_sigma: sir_result.as_ref().map(|s| s.ci_sigma.clone()),
        sir_ess: sir_result.as_ref().map(|s| s.effective_sample_size),
        omega_iov: result.params.omega_iov.as_ref().map(|m| m.matrix.clone()),
        kappa_names: model.kappa_names.clone(),
        kappa_fixed: result.params.kappa_fixed.clone(),
        se_kappa,
        shrinkage_kappa: Vec::new(),
        ebe_kappas: result.kappas.clone(),
        saem_mu_ref_m_step_evals_saved: result.saem_mu_ref_m_step_evals_saved,
        gradient_method_inner: grad_inner.as_str().to_string(),
        gradient_method_outer: grad_outer.as_str().to_string(),
        uses_ode_solver: model.is_ode_based(),
        n_threads_used,
        nlopt_missing_algorithms: nlopt_missing,
        covariance_n_evals_estimated,
        trace_path,
        ebe_convergence_warnings: result.ebe_convergence_warnings,
        max_unconverged_subjects: result.max_unconverged_subjects,
        total_ebe_fallbacks: result.total_ebe_fallbacks,
        covariance_status,
        shrinkage_eta,
        shrinkage_eps,
        wall_time_secs,
        model_name: model.name.clone(),
        ferx_version: env!("CARGO_PKG_VERSION").to_string(),
    };

    if options.verbose {
        output::print_results(&fit_result);
    }

    if time_gradients {
        let (ad_c, ad_n, fd_c, fd_n, jac_ad_c, jac_ad_n, jac_fd_c, jac_fd_n) =
            crate::estimation::inner_optimizer::GRADIENT_TIMINGS.snapshot();
        let ms = |n: u64| (n as f64) / 1_000_000.0;
        let avg_us = |n: u64, c: u64| {
            if c == 0 {
                0.0
            } else {
                (n as f64) / (c as f64) / 1_000.0
            }
        };
        eprintln!("--- Gradient timings (FERX_TIME_GRADIENTS=1) ---");
        eprintln!(
            "  BFGS (AD):  {:>8} calls, {:>10.2} ms total, {:>8.2} µs/call",
            ad_c,
            ms(ad_n),
            avg_us(ad_n, ad_c)
        );
        eprintln!(
            "  BFGS (FD):  {:>8} calls, {:>10.2} ms total, {:>8.2} µs/call",
            fd_c,
            ms(fd_n),
            avg_us(fd_n, fd_c)
        );
        eprintln!(
            "  Jac  (AD):  {:>8} calls, {:>10.2} ms total, {:>8.2} µs/call",
            jac_ad_c,
            ms(jac_ad_n),
            avg_us(jac_ad_n, jac_ad_c)
        );
        eprintln!(
            "  Jac  (FD):  {:>8} calls, {:>10.2} ms total, {:>8.2} µs/call",
            jac_fd_c,
            ms(jac_fd_n),
            avg_us(jac_fd_n, jac_fd_c)
        );
    }

    Ok(fit_result)
}

/// Compute per-subject diagnostics (IPRED, PRED, IWRES, CWRES)
fn compute_subject_results(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas_per_subject: &[Vec<DVector<f64>>],
    interaction: bool,
) -> Vec<SubjectResult> {
    population
        .subjects
        .iter()
        .enumerate()
        .map(|(i, subject)| {
            let eta = &eta_hats[i];
            let h = &h_matrices[i];
            let kappas: &[DVector<f64>] = if i < kappas_per_subject.len() {
                kappas_per_subject[i].as_slice()
            } else {
                &[]
            };

            // Individual predictions: f(eta_hat), with occasion-specific kappas for IOV.
            let ipred = if !kappas.is_empty() {
                let occ_groups = split_obs_by_occasion(subject);
                let mut ipreds = vec![0.0; subject.obs_times.len()];
                for (k, (_, obs_indices)) in occ_groups.iter().enumerate() {
                    let kap: &[f64] =
                        if k < kappas.len() { kappas[k].as_slice() } else { &[] };
                    let combined: Vec<f64> =
                        eta.iter().copied().chain(kap.iter().copied()).collect();
                    let all_preds = model_preds(model, subject, &params.theta, &combined);
                    for &j in obs_indices {
                        ipreds[j] = all_preds[j];
                    }
                }
                ipreds
            } else {
                model_preds(model, subject, &params.theta, eta.as_slice())
            };

            // Population predictions: f(eta = 0, kappa = 0).
            let zero_eta = vec![0.0_f64; model.n_eta + model.n_kappa];
            let pred = model_preds(model, subject, &params.theta, &zero_eta);

            // IWRES (NaN on BLOQ rows — see compute_cwres for CWRES handling).
            let mut iwres = compute_iwres(
                &subject.observations,
                &ipred,
                model.error_model,
                &params.sigma.values,
            );
            for (j, c) in subject.cens.iter().enumerate() {
                if *c != 0 {
                    iwres[j] = f64::NAN;
                }
            }

            // CWRES
            let cwres = compute_cwres(
                subject,
                &ipred,
                eta,
                h,
                &params.omega,
                &params.sigma.values,
                model.error_model,
            );

            // OFV contribution
            let ofv_i = if !kappas.is_empty() {
                let omega_iov = params
                    .omega_iov
                    .as_ref()
                    .expect("omega_iov present when kappas non-empty");
                foce_subject_nll_iov(
                    model,
                    subject,
                    &params.theta,
                    eta,
                    h,
                    &params.omega,
                    &params.sigma.values,
                    interaction,
                    kappas,
                    omega_iov,
                )
            } else {
                foce_subject_nll(
                    model,
                    subject,
                    &params.theta,
                    eta,
                    h,
                    &params.omega,
                    &params.sigma.values,
                    interaction,
                )
            };

            SubjectResult {
                id: subject.id.clone(),
                eta: eta.clone(),
                ipred,
                pred,
                iwres,
                cwres,
                ofv_contribution: 2.0 * ofv_i,
                cens: subject.cens.clone(),
                n_obs: subject.observations.len(),
            }
        })
        .collect()
}

/// ETA shrinkage: `1 - SD(eta_hat_k) / sqrt(omega_kk)` for each random effect k.
pub(crate) fn compute_eta_shrinkage(subjects: &[SubjectResult], omega: &DMatrix<f64>) -> Vec<f64> {
    let n_eta = omega.nrows();
    let n_subj = subjects.len();
    if n_subj < 2 || n_eta == 0 {
        return vec![f64::NAN; n_eta];
    }
    (0..n_eta)
        .map(|k| {
            let omega_var = omega[(k, k)];
            if omega_var <= 0.0 {
                return f64::NAN;
            }
            let omega_sd = omega_var.sqrt();
            let vals: Vec<f64> = subjects.iter().map(|s| s.eta[k]).collect();
            let mean = vals.iter().sum::<f64>() / n_subj as f64;
            let var =
                vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n_subj - 1) as f64;
            1.0 - var.sqrt() / omega_sd
        })
        .collect()
}

/// EPS shrinkage: `1 - SD(IWRES)` across all valid (non-NaN) residuals.
pub(crate) fn compute_eps_shrinkage(subjects: &[SubjectResult]) -> f64 {
    let vals: Vec<f64> = subjects
        .iter()
        .flat_map(|s| s.iwres.iter().copied())
        .filter(|v| v.is_finite())
        .collect();
    let n = vals.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = vals.iter().sum::<f64>() / n as f64;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    1.0 - var.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    fn make_subject(eta: Vec<f64>, iwres: Vec<f64>) -> SubjectResult {
        let n = iwres.len();
        SubjectResult {
            id: "1".to_string(),
            eta: DVector::from_vec(eta),
            ipred: vec![0.0; n],
            pred: vec![0.0; n],
            iwres,
            cwres: vec![0.0; n],
            ofv_contribution: 0.0,
            cens: vec![0; n],
            n_obs: n,
        }
    }

    #[test]
    fn test_eta_shrinkage_zero_when_eta_matches_omega_sd() {
        // If SD(eta_hat) == sqrt(omega), shrinkage = 0.
        // With n=2 subjects: eta = [+s, -s] => SD = s * sqrt(2/(n-1)) for n=2 => SD = s*sqrt(2)
        // For shrinkage=0: SD(eta_hat) = sqrt(omega). So pick omega = 2.0, eta = [+1, -1].
        let omega = DMatrix::from_diagonal_element(1, 1, 2.0);
        let subjects = vec![
            make_subject(vec![1.0], vec![0.0]),
            make_subject(vec![-1.0], vec![0.0]),
        ];
        let sh = compute_eta_shrinkage(&subjects, &omega);
        assert_eq!(sh.len(), 1);
        // SD([1.0, -1.0]) = sqrt(((1-0)^2 + (-1-0)^2) / 1) = sqrt(2) ≈ 1.414
        // shrinkage = 1 - sqrt(2) / sqrt(2) = 0.0
        assert!((sh[0]).abs() < 1e-10, "expected ~0 shrinkage, got {}", sh[0]);
    }

    #[test]
    fn test_eta_shrinkage_positive_when_etas_shrunk() {
        // Etas close to zero → shrinkage > 0
        let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
        let subjects: Vec<SubjectResult> = (0..10)
            .map(|_| make_subject(vec![0.01], vec![0.0]))
            .collect();
        let sh = compute_eta_shrinkage(&subjects, &omega);
        assert!(sh[0] > 0.5, "expected high shrinkage, got {}", sh[0]);
    }

    #[test]
    fn test_eta_shrinkage_nan_when_omega_zero() {
        let omega = DMatrix::zeros(1, 1);
        let subjects = vec![
            make_subject(vec![0.1], vec![0.0]),
            make_subject(vec![-0.1], vec![0.0]),
        ];
        let sh = compute_eta_shrinkage(&subjects, &omega);
        assert!(sh[0].is_nan(), "expected NaN when omega=0");
    }

    #[test]
    fn test_eta_shrinkage_nan_when_fewer_than_2_subjects() {
        let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
        let subjects = vec![make_subject(vec![0.5], vec![0.0])];
        let sh = compute_eta_shrinkage(&subjects, &omega);
        assert!(sh[0].is_nan(), "expected NaN with only 1 subject");
    }

    #[test]
    fn test_eps_shrinkage_near_zero_for_unit_normal_iwres() {
        // IWRES with sample SD = 1 => shrinkage = 0.
        // For n=2: SD([a, -a]) = a*sqrt(2); set a = 1/sqrt(2) so SD = 1.
        let a = 1.0_f64 / 2.0_f64.sqrt();
        let subjects = vec![
            make_subject(vec![0.0], vec![a]),
            make_subject(vec![0.0], vec![-a]),
        ];
        let sh = compute_eps_shrinkage(&subjects);
        assert!((sh).abs() < 1e-10, "expected ~0 eps shrinkage, got {}", sh);
    }

    #[test]
    fn test_eps_shrinkage_nan_for_fewer_than_2_residuals() {
        let subjects = vec![make_subject(vec![0.0], vec![0.5])];
        assert!(compute_eps_shrinkage(&subjects).is_nan());
    }

    #[test]
    fn test_eps_shrinkage_ignores_nan_iwres() {
        // BLOQ rows have NaN IWRES — they must be filtered out.
        // After filtering, two valid values with SD=1 remain => shrinkage = 0.
        let a = 1.0_f64 / 2.0_f64.sqrt();
        let subjects = vec![
            make_subject(vec![0.0], vec![a, f64::NAN]),
            make_subject(vec![0.0], vec![-a, f64::NAN]),
        ];
        let sh = compute_eps_shrinkage(&subjects);
        assert!((sh).abs() < 1e-10, "NaN IWRES not filtered, got {}", sh);
    }
}

/// Extract standard errors from covariance matrix on the packed parameter scale,
/// then transform back to the original scale via delta method.
fn extract_standard_errors(
    cov: &Option<DMatrix<f64>>,
    template: &ModelParameters,
) -> (
    Option<Vec<f64>>,
    Option<Vec<f64>>,
    Option<Vec<f64>>,
    Option<Vec<f64>>,
) {
    let cov = match cov {
        Some(c) => c,
        None => return (None, None, None, None),
    };

    let n = cov.nrows();
    let n_theta = template.theta.len();
    let n_eta = template.omega.dim();
    let n_sigma = template.sigma.values.len();

    // SE on packed scale
    let se_packed: Vec<f64> = (0..n)
        .map(|i| {
            let v = cov[(i, i)];
            if v > 0.0 {
                v.sqrt()
            } else {
                0.0
            }
        })
        .collect();

    // Theta: SE on original scale via delta method
    // If x = log(theta), then SE(theta) = theta * SE(x)
    let se_theta: Vec<f64> = (0..n_theta)
        .map(|i| template.theta[i] * se_packed[i])
        .collect();

    // Omega: SE for diagonal variances
    // omega_ii = L_ii^2, so SE(omega_ii) ≈ 2*L_ii * SE(L_ii)
    // L_ii = exp(x_i), SE(L_ii) = L_ii * SE(x_i)
    // SE(omega_ii) = 2 * L_ii^2 * SE(x_i) = 2 * omega_ii * SE(x_i)
    let omega_start = n_theta;
    let se_omega: Vec<f64> = (0..n_eta)
        .map(|i| {
            let idx = if template.omega.diagonal {
                omega_start + i
            } else {
                // L[i,i] in column-major lower-triangle packing (see `pack_params`):
                // packed entries before column j are sum_{k<j} (n_eta - k), so the
                // i-th diagonal sits at i*n_eta - i*(i-1)/2.  The previous formula
                // i*(i+1)/2 + i was row-major and gave the wrong index for n_eta ≥ 3
                // (e.g. picked L[2,0] instead of L[1,1] for n_eta=3).
                omega_start + i * n_eta - i * i.saturating_sub(1) / 2
            };
            if idx < n {
                2.0 * template.omega.matrix[(i, i)] * se_packed[idx]
            } else {
                0.0
            }
        })
        .collect();

    // Sigma: SE via delta method (log-transformed)
    let sigma_start = omega_start
        + if template.omega.diagonal {
            n_eta
        } else {
            n_eta * (n_eta + 1) / 2
        };
    let se_sigma: Vec<f64> = (0..n_sigma)
        .map(|i| {
            let idx = sigma_start + i;
            if idx < n {
                template.sigma.values[i] * se_packed[idx]
            } else {
                0.0
            }
        })
        .collect();

    // IOV (kappa): SE for diagonal variances of omega_iov.
    //
    // The packed Cholesky layout is column-major (see `pack_params`):
    // L[i,i] sits at offset `i*n - i*(i-1)/2` within the IOV block.
    // Same delta-method approximation as `se_omega`: SE(var_i) ≈ 2 * var_i * SE(log L_ii),
    // which is exact for diagonal IOV and a first-order approximation for block_kappa.
    // Off-diagonal covariance SEs are not currently reported (matches BSV omega).
    let kappa_start = sigma_start + n_sigma;
    let se_kappa: Option<Vec<f64>> = template.omega_iov.as_ref().map(|iov| {
        let n_kappa = iov.dim();
        (0..n_kappa)
            .map(|i| {
                let idx = if iov.diagonal {
                    kappa_start + i
                } else {
                    kappa_start + i * n_kappa - i * (i.saturating_sub(1)) / 2
                };
                if idx < n {
                    2.0 * iov.matrix[(i, i)] * se_packed[idx]
                } else {
                    0.0
                }
            })
            .collect()
    });

    (Some(se_theta), Some(se_omega), Some(se_sigma), se_kappa)
}

/// Simulate observations from a model with given parameters (random seed).
pub fn simulate(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
) -> Vec<SimulationResult> {
    use rand::prelude::*;
    simulate_inner(model, population, params, n_sim, &mut thread_rng())
}

/// Simulate with a fixed seed for reproducibility.
pub fn simulate_with_seed(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    seed: u64,
) -> Vec<SimulationResult> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    simulate_inner(model, population, params, n_sim, &mut rng)
}

fn simulate_inner<R: rand::Rng>(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    rng: &mut R,
) -> Vec<SimulationResult> {
    use rand_distr::Normal;

    let normal = Normal::new(0.0, 1.0).unwrap();
    let n_eta = model.n_eta;

    let mut results = Vec::new();

    for sim_idx in 0..n_sim {
        for subject in &population.subjects {
            // Sample eta from N(0, Omega); append zero kappas for IOV models.
            let z: Vec<f64> = (0..n_eta).map(|_| rng.sample(normal)).collect();
            let z_vec = DVector::from_column_slice(&z);
            let eta = &params.omega.chol * z_vec;
            let mut eta_slice: Vec<f64> = eta.iter().copied().collect();
            eta_slice.resize(n_eta + model.n_kappa, 0.0);

            // Predict concentrations (TV-cov-aware dispatcher).
            let ipreds = model_preds(model, subject, &params.theta, &eta_slice);

            // Add residual error
            for (j, &ipred) in ipreds.iter().enumerate() {
                let var = crate::stats::residual_error::residual_variance(
                    model.error_model,
                    ipred,
                    &params.sigma.values,
                );
                let eps: f64 = rng.sample(normal);
                let dv_sim = ipred + var.sqrt() * eps;

                results.push(SimulationResult {
                    sim: sim_idx + 1,
                    id: subject.id.clone(),
                    time: subject.obs_times[j],
                    ipred,
                    dv_sim,
                });
            }
        }
    }

    results
}

/// A single simulated observation
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub sim: usize,
    pub id: String,
    pub time: f64,
    pub ipred: f64,
    pub dv_sim: f64,
}

/// Predict concentrations for a population using given parameters (no random effects).
pub fn predict(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
) -> Vec<PredictionResult> {
    let zero_eta = vec![0.0_f64; model.n_eta + model.n_kappa];
    let mut results = Vec::new();

    for subject in &population.subjects {
        let preds = model_preds(model, subject, &params.theta, &zero_eta);

        for (j, &pred) in preds.iter().enumerate() {
            results.push(PredictionResult {
                id: subject.id.clone(),
                time: subject.obs_times[j],
                pred,
            });
        }
    }

    results
}

/// A single prediction
#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub id: String,
    pub time: f64,
    pub pred: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
//  IOV integration tests
//
//  Each test builds a minimal warfarin-like 1-cpt IV model with a single kappa
//  for CL, simulates a small population (4 subjects × 2 occasions × 3 obs),
//  and verifies that `fit()` completes without panicking and returns meaningful
//  IOV estimates.  Tests run under `--features ci` (no autodiff required).
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod iov_integration {
    use super::fit;
    use crate::types::*;

    use std::collections::HashMap;

    // ── Model ────────────────────────────────────────────────────────────────
    fn make_iov_model() -> CompiledModel {
        let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
        let omega_iov = OmegaMatrix::from_diagonal(&[0.04], vec!["KAPPA_CL".into()]);
        let default_params = ModelParameters {
            theta: vec![5.0, 50.0],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            theta_lower: vec![0.1, 5.0],
            theta_upper: vec![50.0, 500.0],
            theta_fixed: vec![false; 2],
            omega,
            omega_fixed: vec![false],
            sigma: SigmaVector { values: vec![0.05], names: vec!["PROP_ERR".into()] },
            sigma_fixed: vec![false],
            omega_iov: Some(omega_iov),
            kappa_fixed: vec![false],
        };
        CompiledModel {
            name: "iov_test".into(),
            pk_model: PkModel::OneCptIvBolus,
            error_model: ErrorModel::Proportional,
            // pk_param_fn: eta[0]=BSV for CL, eta[1]=KAPPA_CL (appended by IOV path)
            pk_param_fn: Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>| {
                let mut p = PkParams::default();
                let kappa = if eta.len() > 1 { eta[1] } else { 0.0 };
                p.values[0] = theta[0] * (eta[0] + kappa).exp(); // CL = TVCL * exp(ETA_CL + KAPPA_CL)
                p.values[1] = theta[1];                           // V
                p
            }),
            n_theta: 2,
            n_eta: 1,
            n_epsilon: 1,
            n_kappa: 1,
            kappa_names: vec!["KAPPA_CL".into()],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            eta_names: vec!["ETA_CL".into()],
            default_params,
            mu_refs: HashMap::new(),
            tv_fn: None,
            pk_indices: vec![0, 1],
            eta_map: vec![0],
            pk_idx_f64: vec![0.0, 1.0],
            sel_flat: vec![1.0, 0.0],
            ode_spec: None,
            bloq_method: BloqMethod::Drop,
            referenced_covariates: Vec::new(),
            gradient_method: GradientMethod::Fd,
            parse_warnings: Vec::new(),
        }
    }

    // ── Population ───────────────────────────────────────────────────────────
    // 4 subjects, each with 2 occasions.  Times 1–3 = occasion 1, times 4–6 = occ 2.
    // Observations are plausible IV-bolus concentrations (dose=100).
    fn make_iov_population() -> Population {
        let obs_times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let occasions = vec![1u32, 1, 1, 2, 2, 2];
        let dose_occ = vec![1u32, 2]; // two dose rows: one per occasion
        let subject_data: &[(&str, Vec<f64>)] = &[
            ("S1", vec![36.0, 28.0, 21.0, 34.0, 26.0, 19.0]),
            ("S2", vec![40.0, 32.0, 24.0, 38.0, 29.0, 22.0]),
            ("S3", vec![33.0, 25.0, 19.0, 31.0, 24.0, 18.0]),
            ("S4", vec![42.0, 33.0, 25.0, 39.0, 30.0, 23.0]),
        ];

        let subjects: Vec<Subject> = subject_data
            .iter()
            .map(|(id, obs)| Subject {
                id: id.to_string(),
                doses: vec![
                    DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                    DoseEvent::new(3.5, 100.0, 1, 0.0, false, 0.0),
                ],
                obs_times: obs_times.clone(),
                observations: obs.clone(),
                obs_cmts: vec![1; 6],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
                cens: vec![0; 6],
                occasions: occasions.clone(),
                dose_occasions: dose_occ.clone(),
            })
            .collect();
        Population {
            subjects,
            covariate_names: Vec::new(),
            dv_column: "DV".to_string(),
        }
    }

    fn fast_opts(method: EstimationMethod, optimizer: Optimizer, mu_referencing: bool) -> FitOptions {
        FitOptions {
            method,
            methods: Vec::new(),
            outer_maxiter: 60,
            outer_gtol: 1e-3,
            inner_maxiter: 50,
            inner_tol: 1e-4,
            run_covariance_step: false,
            interaction: method == EstimationMethod::FoceI,
            mu_referencing,
            optimizer,
            lbfgs_memory: 5,
            verbose: false,
            ..FitOptions::default()
        }
    }

    // ── Helper ───────────────────────────────────────────────────────────────
    fn assert_iov_fit_ok(result: &FitResult) {
        assert!(result.ofv.is_finite(), "OFV must be finite");
        assert!(result.omega_iov.is_some(), "omega_iov must be populated");
        let iov_diag = result.omega_iov.as_ref().unwrap()[(0, 0)];
        assert!(iov_diag > 0.0, "omega_iov diagonal must be positive, got {iov_diag}");
        assert_eq!(result.kappa_names, vec!["KAPPA_CL"], "kappa name mismatch");
        assert_eq!(result.ebe_kappas.len(), 4, "expected kappas for 4 subjects");
        for (i, subj_kappas) in result.ebe_kappas.iter().enumerate() {
            assert_eq!(subj_kappas.len(), 2, "subject {i} should have 2 occasions");
        }
    }

    // ── Tests: FOCE + all outer optimizers ───────────────────────────────────

    #[test]
    fn test_iov_foce_bobyqa() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_foce_slsqp() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Slsqp, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_foce_lbfgs() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Lbfgs, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_foce_nlopt_lbfgs() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::NloptLbfgs, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_foce_mma() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Mma, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_foce_bfgs() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bfgs, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    // ── Tests: FOCEI ─────────────────────────────────────────────────────────

    #[test]
    fn test_iov_focei_bobyqa() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::FoceI, Optimizer::Bobyqa, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    // ── Tests: mu-referencing ─────────────────────────────────────────────────

    #[test]
    fn test_iov_foce_mu_referencing_on() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, true);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit with mu_referencing should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_focei_mu_referencing_on() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::FoceI, Optimizer::Bobyqa, true);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit with mu_referencing should succeed");
        assert_iov_fit_ok(&result);
    }

    // ── Tests: GN and GN_Hybrid ───────────────────────────────────────────────

    #[test]
    fn test_iov_gn() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::FoceGn, Optimizer::Bobyqa, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("GN fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_gn_hybrid() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::FoceGnHybrid, Optimizer::Bobyqa, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("GN hybrid fit should succeed");
        assert_iov_fit_ok(&result);
    }

    // ── Test: SAEM + IOV must return Err ──────────────────────────────────────

    #[test]
    fn test_iov_saem_returns_err() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Saem, Optimizer::Bobyqa, false);
        let result = fit(&model, &pop, &model.default_params, &opts);
        assert!(result.is_err(), "SAEM with IOV must return an error");
        let msg = result.unwrap_err();
        assert!(
            msg.contains("saem") && msg.contains("IOV"),
            "error message should mention saem and IOV, got: {msg}"
        );
    }

    // ── Test: SAEM in a chained methods sequence + IOV must also Err ──────────
    // The guard checks the full chain, not just `method`; this locks in that
    // behaviour so a future refactor can't accidentally drop the chain check.
    #[test]
    fn test_iov_saem_in_methods_chain_returns_err() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let mut opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
        opts.methods = vec![EstimationMethod::Saem, EstimationMethod::Foce];
        let result = fit(&model, &pop, &model.default_params, &opts);
        assert!(result.is_err(), "SAEM in methods chain with IOV must return an error");
        let msg = result.unwrap_err();
        assert!(
            msg.contains("saem") && msg.contains("IOV"),
            "error message should mention saem and IOV, got: {msg}"
        );
    }

    // ── Test: trust-region optimizer + IOV must return Err ────────────────────
    // trust_region.rs currently passes `&[]` for kappas to pop_nll, which would
    // silently route the OFV through the non-IOV path. Guard at api.rs blocks
    // that before any wrong number escapes.
    #[test]
    fn test_iov_trust_region_returns_err() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::TrustRegion, false);
        let result = fit(&model, &pop, &model.default_params, &opts);
        assert!(result.is_err(), "trust_region with IOV must return an error");
        let msg = result.unwrap_err();
        assert!(
            msg.contains("trust_region") && msg.contains("IOV"),
            "error message should mention trust_region and IOV, got: {msg}"
        );
    }
}

/// End-to-end checks for time-varying covariate handling: the fit pipeline
/// must accept per-event covariate snapshots, route through the event-driven
/// PK path, surface the AD-downgrade warning, and return finite OFVs.
#[cfg(test)]
mod tv_cov_integration {
    use super::fit;
    use crate::types::*;

    use std::collections::HashMap;

    /// 1-cpt IV bolus model where CL = TVCL * (CR / 1.0) * exp(ETA_CL).
    /// Covariate `CR` is what changes within a subject in the test population.
    fn make_tv_cov_model() -> CompiledModel {
        let omega = OmegaMatrix::from_diagonal(&[0.04], vec!["ETA_CL".into()]);
        let default_params = ModelParameters {
            theta: vec![5.0, 50.0],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            theta_lower: vec![0.1, 5.0],
            theta_upper: vec![50.0, 500.0],
            theta_fixed: vec![false; 2],
            omega,
            omega_fixed: vec![false],
            sigma: SigmaVector { values: vec![0.05], names: vec!["PROP_ERR".into()] },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: Vec::new(),
        };
        CompiledModel {
            name: "tv_cov_test".into(),
            pk_model: PkModel::OneCptIvBolus,
            error_model: ErrorModel::Proportional,
            pk_param_fn: Box::new(|theta: &[f64], eta: &[f64], cov: &HashMap<String, f64>| {
                let mut p = PkParams::default();
                let cr = cov.get("CR").copied().unwrap_or(1.0);
                p.values[0] = theta[0] * cr * eta[0].exp(); // CL = TVCL * CR * exp(ETA_CL)
                p.values[1] = theta[1]; // V
                p
            }),
            n_theta: 2,
            n_eta: 1,
            n_epsilon: 1,
            n_kappa: 0,
            kappa_names: Vec::new(),
            theta_names: vec!["TVCL".into(), "TVV".into()],
            eta_names: vec!["ETA_CL".into()],
            default_params,
            mu_refs: HashMap::new(),
            tv_fn: None, // forces FD; no AD path needed for the test
            pk_indices: vec![0, 1],
            eta_map: vec![0],
            pk_idx_f64: vec![0.0, 1.0],
            sel_flat: vec![1.0, 0.0],
            ode_spec: None,
            bloq_method: BloqMethod::Drop,
            referenced_covariates: vec!["CR".into()],
            gradient_method: GradientMethod::Fd,
            parse_warnings: Vec::new(),
        }
    }

    /// Build a 4-subject population where CR doubles between obs 3 and obs 4.
    /// This exercises the per-event LOCF snapshot path and the event-driven
    /// analytical PK propagator.
    fn make_tv_cov_population() -> Population {
        let mk_cov = |cr: f64| {
            let mut h = HashMap::new();
            h.insert("CR".to_string(), cr);
            h
        };
        let dose_covs = vec![mk_cov(1.0)];
        let obs_covs = vec![mk_cov(1.0), mk_cov(1.0), mk_cov(1.0), mk_cov(2.0), mk_cov(2.0), mk_cov(2.0)];
        let obs_times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let subject_data: &[(&str, Vec<f64>)] = &[
            ("S1", vec![18.0, 16.0, 14.0, 11.0, 9.0, 7.0]),
            ("S2", vec![20.0, 18.0, 16.0, 12.5, 10.0, 8.0]),
            ("S3", vec![17.0, 15.5, 13.5, 10.5, 8.5, 6.5]),
            ("S4", vec![21.0, 19.0, 17.0, 13.0, 10.5, 8.5]),
        ];

        let subjects: Vec<Subject> = subject_data
            .iter()
            .map(|(id, obs)| Subject {
                id: id.to_string(),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: obs_times.clone(),
                observations: obs.clone(),
                obs_cmts: vec![1; 6],
                covariates: mk_cov(1.0),
                dose_covariates: dose_covs.clone(),
                obs_covariates: obs_covs.clone(),
                cens: vec![0; 6],
                occasions: Vec::new(),
                dose_occasions: Vec::new(),
            })
            .collect();
        Population {
            subjects,
            covariate_names: vec!["CR".into()],
            dv_column: "DV".to_string(),
        }
    }

    fn fast_opts(method: EstimationMethod) -> FitOptions {
        FitOptions {
            method,
            methods: Vec::new(),
            outer_maxiter: 30,
            outer_gtol: 1e-3,
            inner_maxiter: 50,
            inner_tol: 1e-4,
            run_covariance_step: false,
            interaction: method == EstimationMethod::FoceI,
            mu_referencing: false,
            optimizer: Optimizer::Bobyqa,
            verbose: false,
            ..FitOptions::default()
        }
    }

    #[test]
    fn tv_cov_foce_runs_and_produces_finite_ofv() {
        let model = make_tv_cov_model();
        let pop = make_tv_cov_population();
        // Sanity check the population was built with TV cov on every subject.
        assert!(pop.subjects.iter().all(|s| s.has_tv_covariates()));

        let opts = fast_opts(EstimationMethod::Foce);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("FOCE fit should succeed");
        assert!(result.ofv.is_finite(), "OFV must be finite, got {}", result.ofv);
        assert!(result.theta[0] > 0.0, "TVCL should remain positive");
    }

    #[test]
    fn tv_cov_focei_runs_and_produces_finite_ofv() {
        let model = make_tv_cov_model();
        let pop = make_tv_cov_population();
        let opts = fast_opts(EstimationMethod::FoceI);
        let result =
            fit(&model, &pop, &model.default_params, &opts).expect("FOCEI fit should succeed");
        assert!(result.ofv.is_finite(), "OFV must be finite, got {}", result.ofv);
    }

    #[test]
    fn tv_cov_saem_runs_and_produces_finite_ofv() {
        let model = make_tv_cov_model();
        let pop = make_tv_cov_population();
        let mut opts = fast_opts(EstimationMethod::Saem);
        opts.saem_n_exploration = 10;
        opts.saem_n_convergence = 20;
        let result =
            fit(&model, &pop, &model.default_params, &opts).expect("SAEM fit should succeed");
        assert!(result.ofv.is_finite(), "OFV must be finite, got {}", result.ofv);
    }

    /// Sanity: when the population has zero TV-cov subjects the AD-downgrade
    /// warning must NOT fire. This is the no-op branch and should keep the
    /// existing behavior unchanged.
    #[test]
    fn no_tv_cov_population_does_not_emit_ad_downgrade_warning() {
        let model = make_tv_cov_model();
        // Strip per-event covariates to flip every subject back to "no TV".
        let mut pop = make_tv_cov_population();
        for s in &mut pop.subjects {
            s.dose_covariates.clear();
            s.obs_covariates.clear();
        }
        assert!(pop.subjects.iter().all(|s| !s.has_tv_covariates()));

        let opts = fast_opts(EstimationMethod::Foce);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert!(
            !result
                .warnings
                .iter()
                .any(|w| w.contains("AD gradients disabled")),
            "no TV cov → no AD downgrade warning; got warnings: {:?}",
            result.warnings
        );
    }

    /// Supported analytical models with TV covariates take the event-driven
    /// AD path — no downgrade warning. (Pre-AD-fast-path version of this
    /// test asserted the opposite; updated when event-driven AD landed.)
    #[test]
    fn tv_cov_supported_model_with_ad_does_not_warn() {
        let mut model = make_tv_cov_model();
        model.tv_fn = Some(Box::new(|theta: &[f64], _cov: &HashMap<String, f64>| {
            vec![theta[0], theta[1]]
        }));
        model.gradient_method = GradientMethod::Auto;
        let pop = make_tv_cov_population();

        let opts = fast_opts(EstimationMethod::Foce);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert!(
            !result
                .warnings
                .iter()
                .any(|w| w.contains("AD gradients disabled")),
            "1-cpt IV bolus + TV cov is supported by event-driven AD; should not warn. \
             Warnings: {:?}",
            result.warnings
        );
    }

    // ── Per-model TV-cov fit smoke tests. Each variant builds a tiny
    //   model + population just rich enough to exercise the per-event
    //   covariate path and the new analytical / AD propagator. We assert
    //   the fit returns a finite OFV — that's enough to catch panics
    //   in the propagator math and the dispatcher wiring. ────────────

    fn build_tv_model(pk_model: PkModel) -> CompiledModel {
        // CL = TVCL · CR · exp(ETA_CL); other PK params constants suitable
        // for the model variant. tv_fn is None to keep the test
        // path FD-only (avoids requiring nightly+enzyme to run unit tests).
        let omega = OmegaMatrix::from_diagonal(&[0.04], vec!["ETA_CL".into()]);
        let (theta, theta_names): (Vec<f64>, Vec<String>) = match pk_model {
            PkModel::OneCptIvBolus | PkModel::OneCptInfusion => {
                (vec![5.0, 50.0], vec!["TVCL".into(), "TVV".into()])
            }
            PkModel::OneCptOral => (
                vec![5.0, 50.0, 1.5],
                vec!["TVCL".into(), "TVV".into(), "TVKA".into()],
            ),
            PkModel::TwoCptIvBolus | PkModel::TwoCptInfusion => (
                vec![5.0, 30.0, 2.0, 50.0],
                vec!["TVCL".into(), "TVV1".into(), "TVQ".into(), "TVV2".into()],
            ),
            PkModel::TwoCptOral => (
                vec![5.0, 30.0, 2.0, 50.0, 1.5],
                vec![
                    "TVCL".into(),
                    "TVV1".into(),
                    "TVQ".into(),
                    "TVV2".into(),
                    "TVKA".into(),
                ],
            ),
            PkModel::ThreeCptIvBolus | PkModel::ThreeCptInfusion => (
                vec![5.0, 20.0, 2.0, 30.0, 0.5, 100.0],
                vec![
                    "TVCL".into(),
                    "TVV1".into(),
                    "TVQ".into(),
                    "TVV2".into(),
                    "TVQ3".into(),
                    "TVV3".into(),
                ],
            ),
            PkModel::ThreeCptOral => (
                vec![5.0, 20.0, 2.0, 30.0, 0.5, 100.0, 1.5],
                vec![
                    "TVCL".into(),
                    "TVV1".into(),
                    "TVQ".into(),
                    "TVV2".into(),
                    "TVQ3".into(),
                    "TVV3".into(),
                    "TVKA".into(),
                ],
            ),
        };
        let n_theta = theta.len();
        let default_params = ModelParameters {
            theta: theta.clone(),
            theta_names: theta_names.clone(),
            theta_lower: vec![0.01; n_theta],
            theta_upper: vec![1000.0; n_theta],
            theta_fixed: vec![false; n_theta],
            omega,
            omega_fixed: vec![false],
            sigma: SigmaVector { values: vec![0.1], names: vec!["PROP_ERR".into()] },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: Vec::new(),
        };

        CompiledModel {
            name: format!("tv_cov_{:?}", pk_model),
            pk_model,
            error_model: ErrorModel::Proportional,
            pk_param_fn: Box::new(move |theta: &[f64], eta: &[f64], cov: &HashMap<String, f64>| {
                let mut p = PkParams::default();
                let cr = cov.get("CR").copied().unwrap_or(1.0);
                // Slot order in PkParams: 0=CL, 1=V, 2=Q, 3=V2, 4=KA, 5=F, 6=Q3, 7=V3.
                p.values[0] = theta[0] * cr * eta[0].exp(); // CL
                p.values[1] = theta[1]; // V
                match pk_model {
                    PkModel::OneCptOral => p.values[4] = theta[2],
                    PkModel::TwoCptIvBolus | PkModel::TwoCptInfusion => {
                        p.values[2] = theta[2];
                        p.values[3] = theta[3];
                    }
                    PkModel::TwoCptOral => {
                        p.values[2] = theta[2];
                        p.values[3] = theta[3];
                        p.values[4] = theta[4];
                    }
                    PkModel::ThreeCptIvBolus | PkModel::ThreeCptInfusion => {
                        p.values[2] = theta[2];
                        p.values[3] = theta[3];
                        p.values[6] = theta[4];
                        p.values[7] = theta[5];
                    }
                    PkModel::ThreeCptOral => {
                        p.values[2] = theta[2];
                        p.values[3] = theta[3];
                        p.values[6] = theta[4];
                        p.values[7] = theta[5];
                        p.values[4] = theta[6];
                    }
                    _ => {}
                }
                p
            }),
            n_theta,
            n_eta: 1,
            n_epsilon: 1,
            n_kappa: 0,
            kappa_names: Vec::new(),
            theta_names,
            eta_names: vec!["ETA_CL".into()],
            default_params,
            mu_refs: HashMap::new(),
            tv_fn: None,
            pk_indices: vec![0, 1],
            eta_map: vec![0],
            pk_idx_f64: vec![0.0, 1.0],
            sel_flat: vec![1.0, 0.0],
            ode_spec: None,
            bloq_method: BloqMethod::Drop,
            referenced_covariates: vec!["CR".into()],
            gradient_method: GradientMethod::Fd,
            parse_warnings: Vec::new(),
        }
    }

    fn build_tv_population(infusion: bool) -> Population {
        let mk_cov = |cr: f64| {
            let mut h = HashMap::new();
            h.insert("CR".to_string(), cr);
            h
        };
        let obs_times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dose = if infusion {
            DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0) // 100mg over 2h
        } else {
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)
        };
        let dose_covs = vec![mk_cov(1.0)];
        let obs_covs = vec![mk_cov(1.0), mk_cov(1.0), mk_cov(1.0), mk_cov(2.0), mk_cov(2.0), mk_cov(2.0)];

        let subject_data: &[(&str, Vec<f64>)] = &[
            ("S1", vec![1.5, 2.0, 1.8, 1.5, 1.2, 1.0]),
            ("S2", vec![1.7, 2.1, 1.9, 1.6, 1.3, 1.1]),
            ("S3", vec![1.4, 1.9, 1.7, 1.4, 1.1, 0.9]),
            ("S4", vec![1.8, 2.2, 2.0, 1.7, 1.4, 1.2]),
        ];
        let subjects: Vec<Subject> = subject_data
            .iter()
            .map(|(id, obs)| Subject {
                id: id.to_string(),
                doses: vec![dose.clone()],
                obs_times: obs_times.clone(),
                observations: obs.clone(),
                obs_cmts: vec![1; 6],
                covariates: mk_cov(1.0),
                dose_covariates: dose_covs.clone(),
                obs_covariates: obs_covs.clone(),
                cens: vec![0; 6],
                occasions: Vec::new(),
                dose_occasions: Vec::new(),
            })
            .collect();
        Population {
            subjects,
            covariate_names: vec!["CR".into()],
            dv_column: "DV".to_string(),
        }
    }

    fn assert_tv_fit_finite(pk_model: PkModel, infusion: bool) {
        let model = build_tv_model(pk_model);
        let pop = build_tv_population(infusion);
        let opts = fast_opts(EstimationMethod::Foce);
        let result = fit(&model, &pop, &model.default_params, &opts)
            .unwrap_or_else(|e| panic!("fit failed for {:?}: {}", pk_model, e));
        assert!(
            result.ofv.is_finite(),
            "OFV should be finite for {:?}, got {}",
            pk_model,
            result.ofv
        );
        assert!(
            pop.subjects.iter().all(|s| s.has_tv_covariates()),
            "test population must carry TV covariates"
        );
    }

    #[test]
    fn tv_cov_one_cpt_oral_fits() {
        assert_tv_fit_finite(PkModel::OneCptOral, false);
    }

    #[test]
    fn tv_cov_two_cpt_oral_fits() {
        assert_tv_fit_finite(PkModel::TwoCptOral, false);
    }

    #[test]
    fn tv_cov_three_cpt_iv_bolus_fits() {
        assert_tv_fit_finite(PkModel::ThreeCptIvBolus, false);
    }

    #[test]
    fn tv_cov_three_cpt_infusion_fits() {
        assert_tv_fit_finite(PkModel::ThreeCptInfusion, true);
    }

    #[test]
    fn tv_cov_three_cpt_oral_fits() {
        assert_tv_fit_finite(PkModel::ThreeCptOral, false);
    }

    /// All analytical PK models are now covered by the event-driven AD
    /// path, so the downgrade warning shouldn't fire for any of them
    /// even on TV-cov data. This locks in the new behavior so a future
    /// refactor can't silently flip a PkModel variant to FD-fallback
    /// without updating the AD coverage list.
    #[test]
    fn tv_cov_oral_with_ad_does_not_warn() {
        let mut model = make_tv_cov_model();
        model.pk_model = PkModel::OneCptOral;
        model.tv_fn = Some(Box::new(|theta: &[f64], _cov: &HashMap<String, f64>| {
            vec![theta[0], theta[1]]
        }));
        model.gradient_method = GradientMethod::Auto;
        let pop = make_tv_cov_population();

        let opts = fast_opts(EstimationMethod::Foce);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert!(
            !result
                .warnings
                .iter()
                .any(|w| w.contains("AD gradients disabled")),
            "1-cpt oral + TV cov is now supported by event-driven AD; \
             should not warn. Warnings: {:?}",
            result.warnings
        );
    }
}

#[cfg(test)]
mod extract_se_tests {
    use super::extract_standard_errors;
    use crate::types::*;
    use nalgebra::DMatrix;

    /// Helper for tests where the BSV omega is a single diagonal eta and only
    /// the IOV block varies. BSV-omega tests below build their own template
    /// inline because they need a 3-eta block omega.
    fn make_template(omega_iov: Option<OmegaMatrix>, kappa_fixed: Vec<bool>) -> ModelParameters {
        let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
        ModelParameters {
            theta: vec![5.0],
            theta_names: vec!["TVCL".into()],
            theta_lower: vec![0.1],
            theta_upper: vec![50.0],
            theta_fixed: vec![false],
            omega,
            omega_fixed: vec![false],
            sigma: SigmaVector { values: vec![0.05], names: vec!["PROP_ERR".into()] },
            sigma_fixed: vec![false],
            omega_iov,
            kappa_fixed,
        }
    }

    // ── BSV omega ────────────────────────────────────────────────────────────

    /// Block omega with n_eta = 3.  The packed Cholesky layout is column-major
    /// (see `pack_params`): L[0,0]=0, L[1,0]=1, L[2,0]=2, L[1,1]=3, L[2,1]=4,
    /// L[2,2]=5.  The previous index formula `i*(i+1)/2 + i` was row-major and
    /// returned offsets 0, 2, 5 — picking L[2,0] for the L[1,1] slot.
    /// For n_eta ≤ 2 the row- and column-major formulas coincide, which is why
    /// this regressed silently.
    #[test]
    fn test_se_omega_block_n3_uses_column_major_indexing() {
        let mut mat = DMatrix::<f64>::zeros(3, 3);
        mat[(0, 0)] = 0.04;
        mat[(1, 1)] = 0.09;
        mat[(2, 2)] = 0.16;
        let _chol = mat.clone().cholesky().unwrap().l();
        let omega = OmegaMatrix::from_matrix(mat, vec!["E1".into(), "E2".into(), "E3".into()], false);
        let template = ModelParameters {
            theta: vec![5.0],
            theta_names: vec!["TVCL".into()],
            theta_lower: vec![0.1],
            theta_upper: vec![50.0],
            theta_fixed: vec![false],
            omega,
            omega_fixed: vec![false; 3],
            sigma: SigmaVector { values: vec![0.05], names: vec!["PROP_ERR".into()] },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: vec![],
        };
        // Packed layout: theta(1) + omega_block(6) + sigma(1) = 8.
        // Within the omega block (start = 1): L[0,0] at idx 1, L[1,1] at idx 4,
        // L[2,2] at idx 6.  Use distinct cov diagonals so we can tell which one
        // each SE pulls from.
        let n = 8;
        let mut cov = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            cov[(i, i)] = ((i + 1) as f64).powi(2); // se_packed[i] = i + 1
        }
        let (_, se_omega, _, _) = extract_standard_errors(&Some(cov), &template);
        let se = se_omega.unwrap();
        // L[0,0] at packed idx 1 → se_packed = 2 → se_omega[0] = 2 * 0.04 * 2 = 0.16
        assert!((se[0] - 2.0 * 0.04 * 2.0).abs() < 1e-12, "got {}", se[0]);
        // L[1,1] at packed idx 4 → se_packed = 5 → se_omega[1] = 2 * 0.09 * 5 = 0.90
        // Pre-fix this would have used idx 3 (= L[2,0]) → 2 * 0.09 * 4 = 0.72.
        assert!((se[1] - 2.0 * 0.09 * 5.0).abs() < 1e-12, "got {}", se[1]);
        // L[2,2] at packed idx 6 → se_packed = 7 → se_omega[2] = 2 * 0.16 * 7 = 2.24
        assert!((se[2] - 2.0 * 0.16 * 7.0).abs() < 1e-12, "got {}", se[2]);
    }

    /// Diagonal omega path is unaffected by the fix; this guards the simple case.
    #[test]
    fn test_se_omega_diagonal_unchanged() {
        let omega = OmegaMatrix::from_diagonal(
            &[0.04, 0.09],
            vec!["E1".into(), "E2".into()],
        );
        let template = ModelParameters {
            theta: vec![5.0],
            theta_names: vec!["TVCL".into()],
            theta_lower: vec![0.1],
            theta_upper: vec![50.0],
            theta_fixed: vec![false],
            omega,
            omega_fixed: vec![false; 2],
            sigma: SigmaVector { values: vec![0.05], names: vec!["PROP_ERR".into()] },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: vec![],
        };
        // Packed layout: theta(1) + omega_diag(2) + sigma(1) = 4. Identity cov.
        let cov = Some(DMatrix::<f64>::identity(4, 4));
        let (_, se_omega, _, _) = extract_standard_errors(&cov, &template);
        let se = se_omega.unwrap();
        assert!((se[0] - 2.0 * 0.04).abs() < 1e-12);
        assert!((se[1] - 2.0 * 0.09).abs() < 1e-12);
    }

    // ── IOV (kappa) ──────────────────────────────────────────────────────────

    /// Identity covariance matrix gives unit SEs on the packed scale.  After
    /// the delta-method transform `SE(var_i) = 2 * var_i * SE(log L_ii)`, the
    /// returned se_kappa[i] equals 2 * variance_i for diagonal IOV.
    #[test]
    fn test_se_kappa_diagonal_uses_correct_index() {
        let iov = OmegaMatrix::from_diagonal(
            &[0.04, 0.09],
            vec!["KAPPA_CL".into(), "KAPPA_V".into()],
        );
        let template = make_template(Some(iov), vec![false, false]);
        // Packed layout: [theta, omega(1), sigma, kappa_1, kappa_2] → 5 entries.
        // Identity cov means se_packed = [1, 1, 1, 1, 1].
        let cov = Some(DMatrix::<f64>::identity(5, 5));
        let (_, _, _, se_kappa) = extract_standard_errors(&cov, &template);
        let se = se_kappa.expect("se_kappa should be Some when omega_iov is set");
        assert_eq!(se.len(), 2);
        assert!((se[0] - 2.0 * 0.04).abs() < 1e-12);
        assert!((se[1] - 2.0 * 0.09).abs() < 1e-12);
    }

    /// Block kappa with n=3: column-major Cholesky packing places L[1,1] at
    /// offset 3 and L[2,2] at offset 5 within the IOV block.  Mirrors the BSV
    /// regression test above.
    #[test]
    fn test_se_kappa_block_n3_column_major_indexing() {
        let mut mat = DMatrix::<f64>::zeros(3, 3);
        mat[(0, 0)] = 0.04;
        mat[(1, 1)] = 0.09;
        mat[(2, 2)] = 0.16;
        let _chol = mat.clone().cholesky().unwrap().l();
        let iov = OmegaMatrix::from_matrix(mat, vec!["K1".into(), "K2".into(), "K3".into()], false);
        let template = make_template(Some(iov), vec![false; 3]);
        // Packed layout: theta(1) + omega_diag(1) + sigma(1) + kappa_block(6) = 9
        // Within the kappa block (start = 3): L11=0, L21=1, L31=2, L22=3, L32=4, L33=5
        // → diagonals at packed indices 3, 6, 8.
        // Build a cov matrix with distinct diagonal entries so we can verify
        // which index each SE pulls from.
        let n = 9;
        let mut cov = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            cov[(i, i)] = ((i + 1) as f64).powi(2); // se_packed[i] = i + 1
        }
        let (_, _, _, se_kappa) = extract_standard_errors(&Some(cov), &template);
        let se = se_kappa.unwrap();
        // L[0,0] at idx 3 → se_packed = 4 → se_kappa[0] = 2 * 0.04 * 4 = 0.32
        assert!((se[0] - 2.0 * 0.04 * 4.0).abs() < 1e-12, "got {}", se[0]);
        // L[1,1] at idx 6 → se_packed = 7 → se_kappa[1] = 2 * 0.09 * 7 = 1.26
        assert!((se[1] - 2.0 * 0.09 * 7.0).abs() < 1e-12, "got {}", se[1]);
        // L[2,2] at idx 8 → se_packed = 9 → se_kappa[2] = 2 * 0.16 * 9 = 2.88
        assert!((se[2] - 2.0 * 0.16 * 9.0).abs() < 1e-12, "got {}", se[2]);
    }

    #[test]
    fn test_se_kappa_none_when_no_iov() {
        let template = make_template(None, vec![]);
        let cov = Some(DMatrix::<f64>::identity(3, 3));
        let (_, _, _, se_kappa) = extract_standard_errors(&cov, &template);
        assert!(se_kappa.is_none());
    }

    #[test]
    fn test_se_kappa_none_when_no_cov() {
        let iov = OmegaMatrix::from_diagonal(&[0.04], vec!["KAPPA_CL".into()]);
        let template = make_template(Some(iov), vec![false]);
        let (_, _, _, se_kappa) = extract_standard_errors(&None, &template);
        assert!(se_kappa.is_none());
    }
}
