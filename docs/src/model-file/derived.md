# Derived Columns (`[derived]`)

The optional `[derived]` block defines extra columns that are computed after the fit and appended to the sdtab output. Each line has the form:

```
name = expression
```

where `name` becomes a new sdtab column and `expression` is evaluated for every observation row.

## Quick example

```
[derived]
  CLi  = CL                          # individual CL echoed directly
  AUC  = integral(IPRED, from=0, to=24)   # AUC 0-24 from predicted curve
  Cmax = max(IPRED)                   # subject-level maximum prediction
  Tmax = tmax(IPRED)                  # time of maximum
```

## Available variables

| Name | Meaning |
|------|---------|
| Individual-parameter names (`CL`, `V`, …) | EBE-derived value for the subject |
| Theta names (`TVCL`, `TVV`, …) | Final population estimate |
| Eta names (`ETA_CL`, …) | Subject EBE |
| Covariate names (`WT`, `AGE`, …) | Subject covariate |
| `IPRED` | Individual prediction at the row's time point |
| `DV` | Observed value |
| `TAFD` | Time after first dose |
| `TAD` | Time after most recent dose |
| `TIME` | Nominal time |

## Operators and functions

Standard arithmetic (`+`, `-`, `*`, `/`, `%`), comparison (`<`, `>`, `<=`, `>=`, `==`, `!=`), and logical (`&&`, `||`, `!`) operators are supported, as well as:

| Function | Description |
|----------|-------------|
| `exp(x)` | Exponential |
| `log(x)` | Natural logarithm |
| `sqrt(x)` | Square root |
| `abs(x)` | Absolute value |
| `floor(x)` / `ceil(x)` / `round(x)` | Rounding |
| `pow(x, y)` | Power |
| `max(expr)` | Subject-level maximum of `expr` over all observations |
| `min(expr)` | Subject-level minimum |
| `tmax(expr)` | Time at which `expr` is maximised |
| `integral(expr, from=t0, to=t1)` | AUC from `t0` to `t1` using the trapezoidal rule |
| `integral(expr, from=t0, to=t1, step=dt)` | Grid-based integral with step `dt` (IPRED-based only; ignored for DV) |

The constant `MACHEPS` (machine epsilon, ≈ 2.22 × 10⁻¹⁶) is also available.

## Computation kinds

### Per-row (`PerRow`)

The default: the expression is evaluated independently at each observation time point and written to the corresponding sdtab row.

```
CLi = CL        # same value every row for a subject; still PerRow
```

### Aggregate (`Aggregate`)

Using `max(expr)`, `min(expr)`, or `tmax(expr)` produces a single value per subject that is broadcast to all of that subject's rows.

```
Cmax = max(IPRED)
Tmax = tmax(IPRED)
```

A filter can be applied with `if condition`:

```
CmaxAfter12 = max(IPRED) if TIME > 12
```

The filter uses the same expression language. If no observation passes the filter the result is `NaN`.

### Integral (`Integral`)

`integral(expr, from=t0, to=t1)` computes the trapezoidal area under `expr` over the half-open interval `[t0, t1]`.

- When `expr` references `DV`, the observed values at the matching observation times are used.
- When `expr` references `IPRED`, values are evaluated at the observation times within the window. A `step=dt` keyword argument causes the integrand to be evaluated on a uniform grid of spacing `dt` using nearest-neighbour IPRED approximation; the `step=` argument is silently ignored for DV-based integrals (warning `W_DERIVED_STEP_IGNORED`).

```
AUC24 = integral(IPRED, from=0, to=24)
AUC24_obs = integral(DV, from=0, to=24)
AUC24_grid = integral(IPRED, from=0, to=24, step=0.5)
```

## Naming rules

A `[derived]` name must **not** clash with:

- Built-in sdtab columns: `ID`, `TIME`, `DV`, `IPRED`, `CWRES`, `IWRES`, `ETA1`, `ETA2`, …, `TAFD`, `TAD`
- Any theta, eta, or individual-parameter name in the model

Shadowing a **covariate** name is permitted (the covariate column is replaced) but will emit a `W_DERIVED_COVARIATE_SHADOW` warning.

## Referencing earlier derived columns

A derived expression may reference the name of any **preceding** derived column defined in the same `[derived]` block (sequential scoping). Forward references are not allowed.

```
[derived]
  CLi  = CL
  VDSS = CLi / 0.693    # ok — CLi defined on the line above
```
