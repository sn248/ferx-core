# TTE Exponential — reference / hand-off kit

Self-contained package to validate ferx's Phase 1 standalone Exponential TTE
estimator against reference software. **ferx + base-R `survreg` columns are done;
the nlmixr2 and NONMEM columns need a machine with those tools** (NONMEM needs a
licence) — that is the hand-off.

## Files

| File | What it is | Who runs it |
|---|---|---|
| `tte_exp.csv` | The dataset everything fits — 100 subjects, λ=0.1, ω²=0.25, censored at t=24 (82 events, 18% censored). Columns `ID,TIME,DV,EVID,CMT,MDV`; `DV` 1=event 0=censored. | — |
| `simulate.R` | Regenerates `tte_exp.csv` (base R only, `set.seed(42)`). | done |
| `survreg.R` | Fixed-effects exponential MLE via base-R `survival::survreg`. | ✅ done |
| `nlmixr2.R` | Mixed-effects FOCEI fit. Needs `nlmixr2` + `rxode2`. | **hand-off** |
| `nonmem.ctl` | Mixed-effects `LAPLACIAN INTERACTION` fit. Needs NONMEM (licence). | **hand-off** |
| `expected.md` | Filled comparison table + acceptance status. | paste results here |

## How to run the hand-off pieces

**NONMEM** (any 7.x):

```bash
nmfe75 nonmem.ctl run.lst      # writes run.lst + tte_exp.sdtab
```

From `run.lst` report, for the final estimation step:
- `THETA(1)` and its SE  → `log(lambda)` (compare `exp(THETA(1))` to ferx `TVLAMBDA`)
- `OMEGA(1,1)`           → `omega^2` (compare to ferx `omega^2`)
- `OBJECTIVE FUNCTION VALUE` → OFV (the `-2LL`)

**nlmixr2** (`install.packages("nlmixr2")`):

```bash
Rscript nlmixr2.R              # prints OFV, AIC, log_lambda (+SE), omega
```

Paste both into the **Mixed-effects** table in `expected.md` (and, optionally, the
matching table in `docs/src/estimation/tte.md`).

## What ferx + survreg already give (the targets to compare against)

Fixed-effects (n_eta=0) — **exact** anchor, no licence needed:

| Quantity | ferx (fixed) | survreg | agreement |
|---|---|---|---|
| λ (rate) | 0.074506 | 0.074506 | exact |
| OFV / −2logLik | 589.888 | 589.888 | exact |

Mixed-effects FOCEI (ferx, fitting `tte_exp.csv`): `TVLAMBDA=0.0768`,
`omega^2=0.290`, `OFV=588.93`. nlmixr2 and NONMEM should land close to these (they
maximise the same Laplace marginal); OFV may differ by an additive constant only.

Reproduce the ferx numbers:

```bash
cargo test --features survival,slow-tests --test tte_convergence -- --nocapture
```

## Parameterisation cheat-sheet

ferx `[event_model] scale = TVLAMBDA*exp(ETA_LAMBDA)` → `TVLAMBDA` is the **rate**.
NONMEM/nlmixr2 use `log(lambda)` (`THETA(1)` / `log_lambda`). So
`TVLAMBDA == exp(THETA(1)) == exp(log_lambda)`, and `omega^2` is the same quantity in
all three (variance of `log lambda`).
