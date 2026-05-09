# Error Model

The `[error_model]` block defines the residual error structure, specifying how observed data (DV) relates to model predictions.

## Syntax

```
DV ~ ERROR_TYPE(SIGMA_PARAMS)
```

## Available Error Models

### Additive

```
DV ~ additive(SIGMA_NAME)
```

The residual variance is constant across all predictions:

\\[ \text{Var}(DV) = \sigma^2 \\]

Use when measurement error is independent of concentration (e.g., assay with fixed precision).

### Proportional

```
DV ~ proportional(SIGMA_NAME)
```

The residual variance scales with the predicted value:

\\[ \text{Var}(DV) = (\sigma \cdot f)^2 \\]

where \\( f \\) is the model prediction. Use when measurement error increases with concentration (most common in PK).

### Combined

```
DV ~ combined(SIGMA_PROP, SIGMA_ADD)
```

Combines proportional and additive components:

\\[ \text{Var}(DV) = (\sigma_1 \cdot f)^2 + \sigma_2^2 \\]

Use when both proportional and additive error sources are present. Requires two sigma parameters defined in `[parameters]`.

## Sigma scale

**All sigma parameters are estimated and reported on the standard-deviation scale**, not the variance scale. This is true for both proportional and additive components, and for both elements of a combined error model.

In particular:

| Error model       | Initial value `sigma = X` means …                                       |
|-------------------|--------------------------------------------------------------------------|
| `proportional`    | The residual SD scales as `X · f`, i.e. **CV% = `X · 100`**             |
| `additive`        | The residual SD is `X` in the units of `DV` (no CV interpretation)     |
| `combined`        | First sigma is proportional (CV-style), second is additive (units of `DV`) |

So `sigma PROP_ERR ~ 0.1` for a proportional model is a **10% CV** initial value, not 1%. Likewise, the SE on a proportional sigma is on the SD scale — multiply by 100 for an SE in CV-percentage points.

The fitted YAML emits both the SD (`estimate`) and the variance (`variance: estimate²`); for proportional components it also emits `cv_pct = estimate · 100` so downstream tooling does not have to re-derive it.

## Examples

Proportional error (most common):
```
[parameters]
  sigma PROP_ERR ~ 0.01

[error_model]
  DV ~ proportional(PROP_ERR)
```

Additive error:
```
[parameters]
  sigma ADD_ERR ~ 1.0

[error_model]
  DV ~ additive(ADD_ERR)
```

Combined error:
```
[parameters]
  sigma PROP_ERR ~ 0.1
  sigma ADD_ERR  ~ 0.5

[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
```

## Impact on Estimation

The error model affects:
- **Individual weighted residuals (IWRES)**: `(DV - IPRED) / sqrt(Var)`
- **Conditional weighted residuals (CWRES)**: Accounts for uncertainty in random effect estimates
- **Objective function value (OFV)**: The likelihood includes `log(Var)` terms, so the error model structure directly influences parameter estimates
