//! Regression tests for subject-relative internal TIME handling.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{predict, read_nonmem_csv};
use std::io::Write;
use tempfile::NamedTempFile;

const TIME_DEPENDENT_ODE: &str = r"
[parameters]
  theta TVRIN(1.0, 0.0, 10.0)
  omega ETA_RIN ~ 0.1
  sigma ADD_ERR ~ 1.0

[individual_parameters]
  RIN = TVRIN * exp(ETA_RIN)

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = if (TIME < 2.0) RIN else 0.0

[error_model]
  DV ~ additive(ADD_ERR)
";

#[test]
fn ode_time_builtin_uses_subject_relative_time_not_raw_clock() {
    let mut f = NamedTempFile::new().unwrap();
    f.write_all(
        b"ID,TIME,DV,EVID,MDV,AMT,CMT\n\
          1,10,.,2,1,0,1\n\
          1,11,0,0,0,0,1\n\
          1,15,0,0,0,0,1\n",
    )
    .unwrap();

    let model = parse_model_string(TIME_DEPENDENT_ODE).expect("model parses");
    let population = read_nonmem_csv(f.path(), None, None).expect("data reads");
    let preds = predict(&model, &population, &model.default_params);

    assert_eq!(preds.len(), 2);
    assert_eq!(preds[0].time, 11.0);
    assert_eq!(preds[1].time, 15.0);
    assert!(
        (preds[0].pred - 1.0).abs() < 1e-6,
        "first observation should see elapsed TIME=1, got {}",
        preds[0].pred
    );
    assert!(
        (preds[1].pred - 2.0).abs() < 3e-2,
        "second observation should accumulate only until elapsed TIME=2, got {}",
        preds[1].pred
    );
}
