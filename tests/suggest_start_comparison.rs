//! Comparison test: suggest_start (Option A) vs suggest_start_thorough (Option B)
//! across analytical and ODE model types.
//!
//! Run with:
//!   cargo test --no-default-features --features ci suggest_start_comparison -- --nocapture

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{read_nonmem_csv, suggest_start, suggest_start_thorough};
use std::path::Path;

struct Case {
    label: &'static str,
    model_path: &'static str,
    data_path: &'static str,
    /// True / expected values for each theta (None = no ground truth available)
    truth: Vec<(&'static str, f64)>,
}

fn run_case(case: &Case) {
    let model = parse_model_file(Path::new(case.model_path))
        .unwrap_or_else(|e| panic!("parse {}: {e}", case.model_path));
    let population = read_nonmem_csv(Path::new(case.data_path), None, None)
        .unwrap_or_else(|e| panic!("data {}: {e}", case.data_path));

    let a = suggest_start(&model, &population);
    let b = suggest_start_thorough(&model, &population);

    println!("\n══ {} ══", case.label);
    println!("  Theta              Default    OptA       OptB       Truth");
    println!("  {:-<60}", "");

    for (i, name) in a.params.theta_names.iter().enumerate() {
        let default = model.default_params.theta[i];
        let opt_a = a.params.theta[i];
        let opt_b = b.params.theta[i];
        let truth_str = case
            .truth
            .iter()
            .find(|(n, _)| *n == name.as_str())
            .map(|(_, v)| format!("{v:>10.3}"))
            .unwrap_or_else(|| "          ".to_string());
        let changed_a = (opt_a - default).abs() > 1e-10;
        let changed_b = (opt_b - opt_a).abs() > 1e-10;
        let tag_a = if changed_a { "*" } else { " " };
        let tag_b = if changed_b { "†" } else { " " };
        println!(
            "  {name:<18} {default:>10.3} {opt_a:>10.3}{tag_a} {opt_b:>10.3}{tag_b}{truth_str}"
        );
    }

    println!("  (* = Option A changed from default; † = Option B changed from Option A)");

    if !a.warnings.is_empty() {
        println!("  Option A warnings:");
        for w in &a.warnings {
            println!("    [A] {w}");
        }
    }
    if !b.warnings.is_empty() {
        let b_only: Vec<_> = b
            .warnings
            .iter()
            .filter(|w| !a.warnings.contains(w))
            .collect();
        if !b_only.is_empty() {
            println!("  Option B additional warnings:");
            for w in b_only {
                println!("    [B] {w}");
            }
        }
    }
}

#[test]
fn suggest_start_comparison_all_models() {
    let cases = vec![
        // ── Analytical models ────────────────────────────────────────────────
        Case {
            label: "1-cpt IV bolus (analytical)",
            model_path: "examples/one_cpt_iv.ferx",
            data_path: "data/one_cpt_iv.csv",
            truth: vec![("TVCL", 5.0), ("TVV", 50.0)],
        },
        Case {
            label: "1-cpt oral (analytical, warfarin)",
            model_path: "examples/warfarin.ferx",
            data_path: "data/warfarin.csv",
            truth: vec![("TVCL", 0.134), ("TVV", 8.0), ("TVKA", 1.0)],
        },
        Case {
            label: "2-cpt IV bolus (analytical)",
            model_path: "examples/two_cpt_iv.ferx",
            data_path: "data/two_cpt_iv.csv",
            truth: vec![("TVCL", 4.0), ("TVV1", 12.0), ("TVQ", 2.0), ("TVV2", 25.0)],
        },
        Case {
            label: "2-cpt oral (analytical, with covariates)",
            model_path: "examples/two_cpt_oral_cov.ferx",
            data_path: "data/two_cpt_oral_cov.csv",
            truth: vec![("TVCL", 4.0), ("TVV1", 40.0), ("TVKA", 1.0)],
        },
        Case {
            label: "3-cpt IV bolus (analytical)",
            model_path: "examples/three_cpt_iv.ferx",
            data_path: "data/three_cpt_iv.csv",
            truth: vec![
                ("TVCL", 5.0),
                ("TVV1", 10.0),
                ("TVQ2", 2.0),
                ("TVV2", 20.0),
                ("TVQ3", 1.5),
                ("TVV3", 30.0),
            ],
        },
        Case {
            label: "1-cpt oral with lag time (analytical)",
            model_path: "examples/oral_with_lagtime.ferx",
            data_path: "data/one_cpt_oral_lagtime.csv",
            truth: vec![
                ("TVCL", 5.0),
                ("TVV", 50.0),
                ("TVKA", 1.5),
                ("TVLAGTIME", 0.75),
            ],
        },
        Case {
            label: "1-cpt oral with bioavailability F (warfarin_logit_f)",
            model_path: "examples/warfarin_logit_f.ferx",
            data_path: "data/warfarin_logit_f.csv",
            // THETA_F default 0.80; NCA CL/V are apparent (CL/F, V/F) and are
            // corrected by F_default so TVCL/TVV start in the true-parameter space.
            // F itself stays at default for Option A; Option B sweeps it.
            truth: vec![
                ("TVCL", 0.134),
                ("TVV", 8.1),
                ("TVKA", 1.0),
                ("THETA_F", 0.80),
            ],
        },
        Case {
            label: "1-cpt IV infusion (analytical)",
            model_path: "examples/one_cpt_infusion.ferx",
            data_path: "data/one_cpt_infusion.csv",
            truth: vec![("TVCL", 5.0), ("TVV", 50.0)],
        },
        // ── ODE models ───────────────────────────────────────────────────────
        Case {
            label: "1-cpt IV bolus (ODE)",
            model_path: "examples/one_cpt_iv_ode.ferx",
            data_path: "data/one_cpt_iv.csv",
            truth: vec![("TVCL", 5.0), ("TVV", 50.0)],
        },
        Case {
            label: "1-cpt oral (ODE, warfarin)",
            model_path: "examples/warfarin_ode.ferx",
            data_path: "data/warfarin.csv",
            truth: vec![("TVCL", 0.134), ("TVV", 8.0), ("TVKA", 1.0)],
        },
        Case {
            label: "2-cpt IV bolus (ODE)",
            model_path: "examples/two_cpt_iv_ode.ferx",
            data_path: "data/two_cpt_iv.csv",
            truth: vec![("TVCL", 4.0), ("TVV1", 12.0), ("TVQ", 2.0), ("TVV2", 25.0)],
        },
        Case {
            label: "2-cpt oral (ODE)",
            model_path: "examples/two_cpt_oral_ode.ferx",
            data_path: "data/two_cpt_oral_cov.csv",
            truth: vec![("TVCL", 4.0), ("TVV1", 40.0), ("TVKA", 1.0)],
        },
        Case {
            label: "3-cpt IV bolus (ODE)",
            model_path: "examples/three_cpt_iv_ode.ferx",
            data_path: "data/three_cpt_iv.csv",
            truth: vec![("TVCL", 5.0), ("TVV1", 10.0)],
        },
        // ── Name-independence & non-standard model types ─────────────────────
        Case {
            label: "1-cpt IV bolus (renamed: CLEARANCE/DISTRIB)",
            model_path: "examples/one_cpt_iv_renamed.ferx",
            data_path: "data/one_cpt_iv.csv",
            // Thetas are named CLEARANCE and DISTRIB — find_theta_for_slot must
            // resolve them through mu_refs rather than canonical name search.
            truth: vec![("CLEARANCE", 5.0), ("DISTRIB", 50.0)],
        },
        Case {
            label: "MM IV bolus (Michaelis-Menten, ODE)",
            model_path: "examples/mm_iv.ferx",
            data_path: "data/mm_iv.csv",
            // VMAX and KM do not match any canonical PK slot name —
            // Option A returns model defaults; Option B sweeps all thetas via
            // 1D coordinate sweeps.  No CL/V ridge, so 1D sweeps are appropriate.
            truth: vec![("TVVMAX", 5.0), ("TVKM", 2.0), ("TVV", 10.0)],
        },
    ];

    for case in &cases {
        run_case(case);
    }

    println!("\nLegend: * = Option A wrote value; † = Option B refined over Option A");
}
