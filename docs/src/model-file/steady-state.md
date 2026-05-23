# Steady-State Doses (SS=1)

A dose record with `SS=1` and `II > 0` tells ferx-core that, at the time
of the record, the compartmental state is at steady state under repeated
dosing of that amount/rate every `II` time units. The record itself is
treated as one of the pulses in the train.

This matches the NONMEM `SS=1` semantic: the compartments are initialised
to the value that would arise from an infinite-past pulse train at
interval `II`. Subsequent observations decay from that loaded state under
the model's normal dynamics; the SS train is **not** implicitly continued
past the SS dose record. To continue dosing forward in time, add explicit
dose records (the typical clinical pattern is one `SS=1` "loading" row
followed by any number of probe observations).

## Dataset columns

A steady-state row uses two NONMEM-format columns: `SS` and `II`.

```
ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,II,SS
1,0,.,1,100,1,0,1,24,1     # SS=1: at SS under q24h dosing of AMT=100
1,1,4.18,0,.,1,0,0,.,.
1,4,3.86,0,.,1,0,0,.,.
1,12,1.78,0,.,1,0,0,.,.
1,23,0.59,0,.,1,0,0,.,.
```

`II` is the dosing interval. For a bolus row (`RATE=0`) and an
infusion row (`RATE>0`), `SS=1` works the same way — the steady-state
state is computed from the corresponding single-dose response repeated
every `II`.

## Supported prediction paths

Every prediction path in ferx-core honours `SS=1`:

| Path                                              | Where                                                                     | How                                                |
| ------------------------------------------------- | ------------------------------------------------------------------------- | -------------------------------------------------- |
| Analytical (1-/2-/3-cpt, no TV covariates)        | `predict_concentration` in `src/pk/mod.rs`                                | Closed-form geometric-series                       |
| Analytical (1-/2-/3-cpt, time-varying covariates) | `event_driven_predictions_with_schedule` in `src/pk/event_driven.rs`      | Numerical pre-equilibration via the propagator     |
| ODE (`[odes]`-block models)                       | `ode_predictions` / `ode_predictions_event_driven` in `src/ode/predictions.rs` | Numerical pre-equilibration via the RK45 solver |

The closed-form path uses the identity

```
C_ss(τ) = Σ_{n=0}^∞ C_single(τ + n·II)
```

evaluated in closed form per disposition eigenvalue: for every
exponential `A·exp(-λ·t)` in the single-dose response, the steady-state
amplitude becomes `A·exp(-λ·τ) / (1 - exp(-λ·II))`. For oral models with
`KA ≈ λ` a separate L'Hopital limit applies; ferx handles it
automatically.

The numerical paths (ODE and event-driven analytical) run the same
identity but expand the sum as `N = 50` cycles of `(apply dose;
propagate II)`. Fifty cycles puts the truncation tail below `1e-9` of
the steady-state value for any realistic PK, so users won't see the
distinction in fit output.

## DSL example

No new DSL syntax is required — the model file is identical to a
single-dose model. SS is a property of the **dataset**:

```
[parameters]
  theta TVCL(2.5, 0.01, 50.0)
  theta TVV(15.0, 0.5, 200.0)
  theta TVKA(1.0, 0.05, 20.0)

  omega ETA_CL ~ 0.05
  omega ETA_V  ~ 0.05
  omega ETA_KA ~ 0.1

  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method  = foce
  maxiter = 100
```

Combine with the SS=1 dataset above and run:

```
cargo run --release -- examples/ss_oral_q24.ferx --data data/ss_oral_q24.csv
```

A runnable example pair ships in the repository at
`examples/ss_oral_q24.ferx` + `data/ss_oral_q24.csv`.

## Combining with `LAGTIME`

`LAGTIME` shifts each pulse in the SS train, including the SS pulse
itself. Under linear superposition this satisfies the identity

```
C_ss(τ, L) = C_ss(τ - L, 0)   for τ > L
```

i.e. the lagged SS curve is just the un-lagged SS curve shifted in time.
ferx-core enforces this exactly — declaring `LAGTIME` on a steady-state
model is fully supported across all paths.

## Limitations

The SS code paths return `0` and emit a warning in
`FitResult.warnings` for the following malformed cases:

- **`SS=1` with `II ≤ 0`** — interval is required for SS predictions;
  set `II` in the dataset or remove the `SS=1` flag.
- **`SS=1` infusion with `T_inf > II`** (overlapping pulses) — no
  closed-form solution covers this; use a shorter infusion or remove
  `SS=1`. The user can equivalently model overlapping infusions with
  an `[odes]` block and explicit periodic dose records (no SS).

The `SS=2` flag (NONMEM "add to the existing train without
re-equilibrating") is **not** supported. Datasets that use `SS=2` should
be converted to explicit dose records.

## NONMEM equivalence

- `SS=1`, `II=24`, `AMT=100`, `RATE=0` → matches NONMEM SS=1 bolus.
- `SS=1`, `II=24`, `AMT=100`, `RATE=25` → matches NONMEM SS=1 infusion
  of duration `4` (= `AMT/RATE`), repeated every 24 time units.
- The SS state is computed assuming the same `CL`, `V`, etc. as the
  current PK record — i.e. ferx uses the **current** subject's
  parameters, not a separate "SS parameters" set. This matches
  NONMEM's `$PK` evaluation convention.

## Validation

Closed-form SS expressions are unit-tested against 200- to 400-term
numerical pulse sums at 1e-9 relative tolerance (see
`src/pk/one_compartment.rs::tests::test_ss_*`, similarly for 2-cpt and
3-cpt). The ODE and event-driven paths are cross-checked against the
analytical closed forms in their own test modules. The end-to-end fit
path is covered by `tests/ss_fit_smoke.rs`.
