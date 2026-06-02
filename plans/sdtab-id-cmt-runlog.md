# sdtab ID / CMT / runlog fields — Implementation Plan

Read `CLAUDE.md` first. Complete steps in order; each step is self-contained and
can be PRed independently. Later steps build on the struct additions from earlier
ones, so merge order matters.

Before touching any file, follow the three-phase workflow: **Read → Reconcile →
Implement**. The descriptions below are accurate as of 2026-06-02; verify each
claim against the live code before writing anything.

---

## Repository split

Changes land in two repositories. Do **ferx-core first**; only start ferx-r
work after the relevant ferx-core PR has merged to `main`.

| Step | ferx-core | ferx-r |
|------|-----------|--------|
| 1 — sdtab ID | Fix `sdtab()` in `src/io/output.rs` | Update any R code that reads the sdtab ID column and previously worked around the 1-based index bug |
| 2 — sdtab CMT | Fix `sdtab()` in `src/io/output.rs` | `ferx_sdtab()` gains a CMT column; update per-CMT diagnostic plots to split by CMT |
| 3 — model text | Add `FitResult.model_text`; populate in all file-based entry points and `load_fit()` | `ferx_runlog()`: use `fit$model_text` for the model-source section of the `.lst` equivalent |
| 4 — initial estimates | Add `theta_init / omega_init / sigma_init` to `FitResult`; wire through `.fitrx` | `ferx_runlog()`: initial vs final parameter table (two-column layout matching NONMEM `.lst`) |
| 5 — obs stats | Add `obs_time_range` to `FitResult`; wire through `.fitrx` | `ferx_runlog()`: data-summary header ("N subjects / N obs, time range MIN–MAX") |
| 6 — final gradient | Add `final_gradient` to `OuterResult` + `FitResult`; wire through `.fitrx` | `ferx_runlog()`: FINAL GRADIENT section; convergence check (`all(abs(grad) < tol)`) |

Steps 1–2 are pure ferx-core bugs with no ferx-r follow-up beyond defensive
reads. Steps 3–6 each require a paired ferx-r PR after the ferx-core change
lands.

---

## Background

Three related gaps block a complete `ferx_runlog()` in the R layer and cause
silent data-integrity failures for real-world datasets:

1. **sdtab ID is wrong** — `sdtab()` writes a 1-based loop index instead of the
   subject's original numeric ID. Any join back to the source data by ID
   silently mismatches for non-consecutive IDs (the normal case in clinical data).

2. **CMT absent from sdtab** — The `Subject.obs_cmts` field is populated by the
   data reader for every dataset but is never read inside `sdtab()`. Per-CMT
   (multi-endpoint) models produce sdtab rows with no way to identify which
   endpoint each residual belongs to.

3. **Four fields missing from `FitResult`** — `ferx_runlog()` needs model text,
   initial estimates, per-subject observation counts + time range, and the final
   gradient vector to reproduce the information content of a NONMEM `.lst` file.
   None of the four are currently stored on `FitResult`.

---

## Status Legend

- ✅ DONE
- ❌ NOT STARTED

---

## Step 1 — Fix sdtab ID column

**Status:** ❌ NOT STARTED

### What and why

`src/io/output.rs:488`:

```rust
ids.push(si as f64 + 1.0);   // BUG: loop index, not subject ID
```

`si` is the 0-based `enumerate` index over `result.subjects`. For a dataset
with subjects 101, 202, 303 the sdtab writes 1.0, 2.0, 3.0. Any downstream
join by ID silently mismatches.

`sr` (a `&SubjectResult`) is already in scope at that line and carries
`sr.id: String` — the original ID string exactly as it appeared in the
NONMEM CSV. NONMEM IDs are always numeric. Parse it as `f64`; fall back to
the 1-based index only if parsing fails (non-numeric ID, should never happen
for valid NONMEM data).

### Exact change

**File:** `src/io/output.rs`, line 488 — inside `sdtab()`, inner loop body.

```rust
// Before
ids.push(si as f64 + 1.0);

// After
ids.push(sr.id.parse::<f64>().unwrap_or(si as f64 + 1.0));
```

That is the entire code change for this step.

### Test

Add a Tier 1 unit test in the `#[cfg(test)] mod tests` block at the bottom of
`src/io/output.rs`. The test must:

1. Construct a minimal `FitResult` and `Population` with three subjects whose
   string IDs are `"101"`, `"202"`, `"303"` (non-consecutive).
2. Give each subject one observation so `sdtab()` produces three rows.
3. Call `sdtab(&result, &population)`.
4. Find the `"ID"` column in the returned vec.
5. Assert `id_col == vec![101.0, 202.0, 303.0]`, **not** `[1.0, 2.0, 3.0]`.

Look at the existing `tests_sdtab_tv_cov` module (around line 4198 of `api.rs`)
for the pattern used to build synthetic `FitResult` + `Population` fixtures.
The unit test here should be simpler — no estimation needed, just the
`SubjectResult` and `Subject` structs populated directly.

### ferx-r follow-up

If the R side has any workaround that renumbers IDs to be consecutive (to
compensate for the bug), remove it. The sdtab ID column now matches the input
data directly.

### Files touched

| Repo | File | Change |
|------|------|--------|
| ferx-core | `src/io/output.rs` | 1-line fix + unit test |

---

## Step 2 — Add CMT column to sdtab

**Status:** ❌ NOT STARTED

### What and why

`Subject.obs_cmts: Vec<usize>` (declared at `src/types.rs:183`) is populated
by the data reader for every dataset. When no CMT column is present in the CSV
it defaults to 1 for every observation (confirmed in `src/io/datareader.rs:344`).
`sdtab()` in `src/io/output.rs` never reads this field — the column simply does
not appear in the output.

For a multi-endpoint model (e.g. PK + PD measured simultaneously, each on a
different CMT) every sdtab row looks identical in structure; the analyst cannot
tell which endpoint produced which CWRES/IWRES without the CMT column.

### Approach

Follow the same conditional-inclusion pattern already used for `CENS` and `OCC`:
build the column unconditionally in the loop, then only append it to `cols`
when at least one observation has `CMT != 1`. This keeps sdtab clean for the
common single-endpoint case (all `obs_cmts == 1`, or no CMT column in the CSV)
while automatically populating the column for multi-endpoint models.

### Exact changes

**File:** `src/io/output.rs`, inside `fn sdtab()`.

**1. Before the loop — add guard and column buffer** (after the existing
`any_occ` line, around line 471):

```rust
let any_multicmt = population
    .subjects
    .iter()
    .any(|s| s.obs_cmts.iter().any(|&c| c != 1));
let mut cmt_col = Vec::with_capacity(n_total);
```

**2. Inside the loop body** — push immediately after the `occ_col.push(...)` line
(around line 492). The existing loop variable for the current subject is `subj`
(confirmed at `output.rs:486: let subj = &population.subjects[si]`):

```rust
cmt_col.push(subj.obs_cmts.get(j).copied().unwrap_or(1) as f64);
```

The `.get(j).copied().unwrap_or(1)` guard is defensive: `obs_cmts` is
always parallel to `obs_times` for data-reader-produced subjects, but
synthetic test fixtures may leave `obs_cmts` empty.

**3. In the `cols` assembly block** — insert between the OCC block and the
`PRED` block (after the `if any_occ` push, around line 510):

```rust
if any_multicmt {
    cols.push(("CMT".to_string(), cmt_col));
}
```

### Column position in output

The column order after this change will be:
`ID, TIME, DV, [CENS], [OCC], [CMT], PRED, IPRED, CWRES, IWRES, EBE_OFV, N_OBS`

CMT is placed after OCC and before PRED because it identifies the observation
kind (which endpoint) — logically it sits with the observation descriptor
columns, not the prediction columns.

### Test

Add a Tier 1 unit test in `src/io/output.rs`:

1. **Multi-CMT case**: build a `Population` where one subject has
   `obs_cmts = vec![1, 2]` and a matching two-observation `FitResult`. Call
   `sdtab()`. Assert a `"CMT"` column is present and its values are
   `[1.0, 2.0]`.

2. **Single-CMT case**: same fixture but with `obs_cmts = vec![1, 1]`. Assert
   no `"CMT"` column appears.

### ferx-r follow-up

Update `ferx_sdtab()` / any diagnostic-plot helpers to split CWRES/IWRES panels
by CMT when the column is present. This is additive; existing single-CMT
behaviour is unchanged.

### Files touched

| Repo | File | Change |
|------|------|--------|
| ferx-core | `src/io/output.rs` | ~12 lines + unit test |

---

## Step 3 — Add model text to `FitResult`

**Status:** ❌ NOT STARTED

### What and why

`FitResult` stores `model_path: Option<String>` and `model_hash: Option<String>`
but not the file's content. `ferx_runlog()` in R needs the verbatim `.ferx`
source to include the model-text section of the `.lst` equivalent. Re-reading
from disk at report time breaks for archived/remote runs.

Note: the `.fitrx` bundle already stores `model.ferx` as a zip entry and
surfaces it as `LoadedFit.model_source: String`. This step adds the same
content to `FitResult` so it is accessible without a save/load round-trip.

**Second disk read is unavoidable.** `parse_full_model_file()` reads the file
into a local `content: String`, parses it into `ParsedModel`, then drops the
raw text — `ParsedModel` carries no `model_text` field. The
`std::fs::read_to_string(model_path)` call in Step 3C is therefore a second
read of the same file. A future clean-up could add `source: String` to
`ParsedModel` and read once; that is out of scope here.

### Exact changes

**A. `src/types.rs`** — add one field to `FitResult`, after the `model_hash`
field (around line 1724):

```rust
/// Verbatim content of the `.ferx` model file. `Some` when the fit was
/// launched via `fit_from_files` / CLI or loaded from a `.fitrx` bundle;
/// `None` for in-memory `fit()` callers who never had a file path.
pub model_text: Option<String>,
```

**B. `src/api.rs`** — initialise to `None` in the `FitResult { ... }` literal
inside `fit_inner()` (around line 1643, alongside `model_path: None`):

```rust
model_text: None,
```

**C. `src/api.rs`** — populate in **all three** file-based entry points, after
the existing `result.model_hash = ...` line in each:

- `run_model_with_data_inits()` (~line 132)
- `run_model_simulate()` (~line 232)
- `fit_from_files()` (~line 731)

```rust
result.model_text = std::fs::read_to_string(model_path).ok();
```

All three set `result.model_path` and `result.model_hash` today. All three
must also set `model_text`. Missing any one of them silently leaves
`model_text: None` for that call path — `fit_from_files()` is the primary
public file-based API that the R layer uses, so omitting it defeats the purpose.

**D. `src/io/fitrx.rs`** — in `load_fit()`, the `FitResult` is returned by
`wire_to_fit_result()` and bound as `let fit`. Change the binding to
`let mut fit` so the post-call assignment compiles:

```rust
// load_fit(), ~line 915
let mut fit = wire_to_fit_result(wire, subjects, ebe_kappas)?;
fit.model_text = Some(model_source.clone());
Ok(LoadedFit { fit, model_source, population, manifest })
```

No change to `FitWire` or `build_fit_wire` is needed — the model text is
already stored as a separate `model.ferx` zip entry and read back in
`load_fit()` as `model_source`.

**E. `save_fit()` consistency** — `save_fit()` takes `model_source: &str`
separately and embeds that string in the zip, ignoring `result.model_text`.
After Step 3 the two can silently diverge. Update `save_fit()` to prefer
`result.model_text` when present:

```rust
// In save_fit(), replace the direct use of model_source with:
let effective_source = result.model_text.as_deref().unwrap_or(model_source);
// then use effective_source wherever model_source was written to the zip
```

**F. Search for all `FitResult { ... }` literals** by grepping
`grep -n "FitResult {" src/`. Every literal must include `model_text: None`
or the build will fail with a struct-exhaustiveness error. The main hits are:

- `src/api.rs` — the primary `fit_inner` literal (~line 1564) ← covered above
- `src/api.rs` — the `synthetic_fit` test helper (~line 3639)
- Any fixture-building helpers in test modules

Update each with `model_text: None`.

### No YAML output change needed

`write_estimates_yaml` is a human-readable summary file; embedding the full
model text there would make it unreadable. The field is for programmatic
consumers (the R layer). No change to `output.rs`.

### Test

No dedicated test required — the field is pure data threading. `cargo check
--tests` after the change confirms all struct literals are updated.

### ferx-r follow-up

In `ferx_runlog()`, access `fit$model_text` (a character string or `NULL`) and
render it as the "Model file" section of the output document. Guard against
`NULL` for in-memory fits.

### Files touched

| Repo | File | Change |
|------|------|--------|
| ferx-core | `src/types.rs` | +1 field on `FitResult` |
| ferx-core | `src/api.rs` | init `None` in literal; populate in **all three** file-based entry points; update synthetic fixtures |
| ferx-core | `src/io/fitrx.rs` | `let mut fit` in `load_fit()`; assign `model_text`; update `save_fit()` to prefer `result.model_text` |

---

## Step 4 — Add initial estimates to `FitResult`

**Status:** ❌ NOT STARTED

### What and why

`FitResult` carries only final theta/omega/sigma. `ferx_runlog()` needs both
the initial and final columns (like the NONMEM `.lst` parameter table). The
initial values live in `init_params` inside `fit_inner()` (`src/api.rs:1015`),
which is a clone of `CompiledModel.default_params`. They are never written to
`FitResult`.

### Multi-start caveat

For `n_starts > 1`, `fit_inner()` is called with perturbed initial parameters
for starts k > 0 (`perturb_init(init_params, k, ...)` at api.rs:904). The
`theta_init` captured inside `fit_inner` for the winning start therefore
reflects the **perturbed** starting point, not the user's original values.
This is acceptable for diagnostic purposes (it shows what the winning start
actually began from), but the R layer should note that `theta_init` may differ
from the model file's declared initials when `n_starts > 1`. If the user's
original initials are needed unconditionally, capture them in `fit()` before
the perturbation loop and thread them down separately — that is out of scope
for this step.

### IOV note

`FitResult.omega_iov` carries the final IOV kappa omega, and `ModelParameters`
has a corresponding `omega_iov: Option<OmegaMatrix>`. For completeness, an
`omega_iov_init: Option<DMatrix<f64>>` field would let `ferx_runlog()` show
the initial vs final kappa variance table for IOV models. This is left as a
follow-on: add it alongside `omega_init` if IOV runlog output is needed, using
the same `Option<MatrixWire>` wire pattern.

### Exact changes

**A. `src/types.rs`** — add three fields to `FitResult`, after `theta` /
before `theta_names` (around line 1545):

```rust
/// Initial theta values as supplied to the optimizer (from
/// `CompiledModel.default_params`), parallel to `theta` and `theta_names`.
pub theta_init: Vec<f64>,
/// Initial omega matrix (variance scale), same layout as `omega`.
pub omega_init: DMatrix<f64>,
/// Initial sigma values, parallel to `sigma` and `sigma_names`.
pub sigma_init: Vec<f64>,
```

**B. `src/api.rs`** — inside `fit_inner()`, capture initial values
**before** the chain runs (around line 1148, after
`let mut stage_params: ModelParameters = init_params.clone()`):

```rust
let theta_init = init_params.theta.clone();
let omega_init = init_params.omega.matrix.clone();
let sigma_init = init_params.sigma.values.clone();
```

**C. `src/api.rs`** — populate in the `FitResult { ... }` literal
(around line 1564, alongside `theta`):

```rust
theta_init,
omega_init,
sigma_init,
```

**D.** Find every other `FitResult { ... }` literal (see Step 3 §F for the
grep) and add the three fields. For synthetic test helpers where initial values
are not meaningful, use the final values or empty vecs as appropriate.

**E. `src/io/fitrx.rs`** — add three fields to `FitWire` (the JSON wire
struct, around line 248) with `#[serde(default)]` for backward compatibility
with bundles written before this change:

```rust
#[serde(default)]
theta_init: Vec<f64>,
#[serde(default)]
omega_init: Option<MatrixWire>,    // None on old bundles
#[serde(default)]
sigma_init: Vec<f64>,
```

In `build_fit_wire()`:
```rust
theta_init: r.theta_init.clone(),
omega_init: Some(MatrixWire::from(&r.omega_init)),
sigma_init: r.sigma_init.clone(),
```

In `wire_to_fit_result()`, convert with an explicit fallback dimension.
`omega_init: Option<MatrixWire>` is `None` on old bundles; the fallback must
be a correctly-sized zero matrix, not a 0×0 one (which would panic on R-side
indexing). Use the already-deserialized `omega` to derive the size:

```rust
omega_init: w.omega_init
    .map(|m| m.into_dmatrix())
    .transpose()?
    .unwrap_or_else(|| {
        let n = w.omega.matrix.rows;
        DMatrix::zeros(n, n)
    }),
```

### Test

Add a Tier 1 unit test in `src/api.rs` (in the existing inline test module):

1. Build a minimal model with known initial thetas (e.g. `[1.0, 2.0]`).
2. Call `fit()`.
3. Assert `result.theta_init == model.default_params.theta`.
4. Assert `result.theta_init.len() == result.theta.len()` and all values are
   finite (avoids a fragile "estimates must have moved" check).

### ferx-r follow-up

In `ferx_runlog()`, render the parameter table with two columns: `INITIAL` and
`FINAL`, parallel to `theta_names`. For omega use `result$omega_init` and
`result$omega`. Flag fixed parameters (from `theta_fixed`) with `FIXED` instead
of a numeric initial.

### Files touched

| Repo | File | Change |
|------|------|--------|
| ferx-core | `src/types.rs` | +3 fields on `FitResult` |
| ferx-core | `src/api.rs` | capture initials before chain; populate literal; update fixtures |
| ferx-core | `src/io/fitrx.rs` | +3 fields on `FitWire` with `serde(default)`; wire conversion with explicit fallback |

---

## Step 5 — Add time range to `FitResult`

**Status:** ❌ NOT STARTED

### What and why

`FitResult` already carries `n_obs: usize` (total) and `n_subjects: usize`.
`ferx_runlog()` needs the observation time range to reproduce the data-summary
header at the top of every NONMEM `.lst`:

```
 TOT. NO. OF OBS RECS:     332   NO. OF INDIVIDUALS:      32
 (AND OBSERVATIONS TIMES 0.250 TO 96.0)
```

**Per-subject observation counts are already available.** `SubjectResult.n_obs:
usize` (types.rs:1341) stores the count for each subject and is already
serialised in `ebes.csv` inside `.fitrx` bundles. Adding a parallel
`obs_per_subject: Vec<usize>` would duplicate this information and create a
consistency hazard. The R layer can use `sapply(result$subjects, \(sr) sr$n_obs)`
instead. This step adds only `obs_time_range`.

### Exact changes

**A. `src/types.rs`** — add one field to `FitResult` (after `n_subjects`,
around line 1582):

```rust
/// `(min_time, max_time)` across all observation records. `None` only when
/// there are no observations at all.
pub obs_time_range: Option<(f64, f64)>,
```

**B. `src/api.rs`** — compute from `population` in `fit_inner()`, before
constructing `FitResult` (anywhere after `population` is in scope):

```rust
let obs_time_range: Option<(f64, f64)> = {
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for s in &population.subjects {
        for &t in &s.obs_times {
            if t < mn { mn = t; }
            if t > mx { mx = t; }
        }
    }
    if mn.is_finite() { Some((mn, mx)) } else { None }
};
```

Do **not** use `f64::min()` / `f64::max()` — use explicit comparisons (see
`CLAUDE.md` "Autodiff-Safe Code").

**C. `src/api.rs`** — populate in the `FitResult { ... }` literal.

**D.** Update all other `FitResult { ... }` literals with
`obs_time_range: None`.

**E. `src/io/fitrx.rs`** — add to `FitWire` with `#[serde(default)]`.
`Option<(f64, f64)>` serialises as a JSON array or null; with `#[serde(default)]`
a missing key deserialises as `None` on old bundles:

```rust
#[serde(default)]
obs_time_range: Option<(f64, f64)>,
```

Populate in `build_fit_wire()` and `wire_to_fit_result()`.

### Test

No dedicated test required beyond compile-check. The value is trivially derived
from `population` and verifiable by inspection on any test model.

### ferx-r follow-up

In `ferx_runlog()`, render the data-summary line using `result$obs_time_range`
(a length-2 numeric vector or `NULL`). For per-subject observation counts, use
`sapply(result$subjects, \(sr) sr$n_obs)` — no new field needed.

### Files touched

| Repo | File | Change |
|------|------|--------|
| ferx-core | `src/types.rs` | +1 field on `FitResult` |
| ferx-core | `src/api.rs` | compute + populate in literal; update fixtures |
| ferx-core | `src/io/fitrx.rs` | +1 wire field with `serde(default)` |

---

## Step 6 — Expose final gradient vector on `FitResult`

**Status:** ❌ NOT STARTED

### What and why

NONMEM `.lst` outputs the "FINAL GRADIENT" — the gradient of the objective
at the converged parameter estimates. This gives the analyst a direct
convergence diagnostic: if any component is large, the optimizer may not have
converged.

The outer optimizer computes the gradient inside the NLopt objective callback
(`src/estimation/outer_optimizer.rs`, around line 702) but discards it after
each evaluation. `OuterResult` has no `final_gradient` field and neither does
`FitResult`.

The gradient is only available for NLopt gradient-based algorithms (SLSQP,
L-BFGS, MMA). BOBYQA is derivative-free and never calls `population_gradient`.
The built-in BFGS path (`optimize_bfgs`) computes gradients internally but
does not expose them. `final_gradient` stays `None` for both.

### What to store

Store `grad_raw` (the output of `population_gradient()`), **not** the
NLopt-scaled `g` values. `grad_raw` is the gradient in the packed parameter
space (log-theta, Cholesky-omega, log-sigma) — the natural estimation space
that the R layer understands. The additional `scale[k]` multiplier applied
before handing to NLopt is an optimizer-internal detail.

### Gradient capture strategy — anchor to best_seen

**Do not** capture the gradient at the last gradient-requesting call. Two
failure modes make this wrong:

1. **Stagnation early-return** (outer_optimizer.rs:624–632): when
   `stagnation_stopped` is latched, the closure returns at line 632 — before
   the `population_gradient()` call at ~line 702. For SLSQP runs that
   converge via stagnation (the common path), the final several calls all
   hit the early-return and no gradient is stored. The `Arc<Mutex>` would
   hold a gradient from a pre-stagnation iteration.

2. **Line-search overshoot**: after the best point, the optimizer may evaluate
   at points with higher OFV. The last gradient call may not be at the minimum.

**Correct approach**: store `grad_raw` only inside the block that updates
`best_seen`, gated on `ofv < best_so_far`. This mirrors exactly how `best_seen`
itself is managed and guarantees the stored gradient corresponds to the
minimum-OFV parameter vector:

```rust
// Inside objective, after state.best_ofv is updated (around line 736):
if ofv < state.best_ofv {
    state.best_ofv = ofv;
    *last_gradient_cl.lock().unwrap() = Some(grad_raw.clone());
    // existing verbose eprintln...
}
```

Note: `grad_raw` is only computed when `grad.is_some()`. Guard the store
with a check that `grad_raw` is available (it is a local variable in scope
only inside `if let Some(g) = grad { ... }`). Move the store into that block,
still gated on `ofv < state.best_ofv`:

```rust
// Inside `if let Some(g) = grad { ... }`, after the g[k] loop:
if ofv < state.best_ofv {          // grad_raw is in scope here
    *last_gradient_cl.lock().unwrap() = Some(grad_raw.clone());
}
```

The stagnation path zeroes `g` and returns before `grad_raw` is ever computed,
so no special handling is needed for stagnation — `last_gradient_cl` simply
retains whatever was stored at the last improving gradient step.

### Exact changes

**A. `src/estimation/outer_optimizer.rs`**

1. Add `Arc<Mutex>` shared state before the `objective` closure (around
   line 607), following the existing `best_seen` / `n_evals_outer` pattern:

   ```rust
   let last_gradient: Arc<Mutex<Option<Vec<f64>>>> = Arc::new(Mutex::new(None));
   let last_gradient_cl = Arc::clone(&last_gradient);
   ```

2. Inside `objective`, inside `if let Some(g) = grad { ... }`, after the
   `g[k]` loop, add the gated store:

   ```rust
   if ofv < state.best_ofv {
       *last_gradient_cl.lock().unwrap() = Some(grad_raw.clone());
   }
   ```

3. For the SLSQP fallback closure (`objective2`, around line 903), clone the
   same Arc before `objective2` is defined:

   ```rust
   let last_gradient_cl2 = Arc::clone(&last_gradient);
   ```

   Apply the same gated store inside `objective2` using `last_gradient_cl2`.
   The two closures share one `Arc<Mutex>` so the final stored value is always
   the gradient at the best OFV across both the primary run and the fallback.

4. After both optimizer runs conclude, read the captured gradient:

   ```rust
   let final_gradient = last_gradient.lock().unwrap().clone();
   ```

5. Add `final_gradient: Option<Vec<f64>>` to `OuterResult` (around line 12):

   ```rust
   pub final_gradient: Option<Vec<f64>>,
   ```

6. Populate `final_gradient` when constructing `OuterResult` at the return
   point of `optimize_nlopt()`. For `optimize_bfgs()`, set
   `final_gradient: None`.

**B. `src/types.rs`** — add to `FitResult` (after `n_iterations`, around
line 1584):

```rust
/// Gradient of the objective function at the best-OFV parameter point,
/// in the packed parameter space (log-theta, Cholesky-omega, log-sigma).
/// `Some` only for NLopt gradient-based runs (SLSQP, L-BFGS, MMA) when at
/// least one gradient-requesting iteration improved the OFV; `None` for
/// BOBYQA (derivative-free), built-in BFGS, and SAEM.
pub final_gradient: Option<Vec<f64>>,
```

**C. `src/api.rs`** — populate from the last stage's result in `fit_inner()`:

```rust
final_gradient: result.final_gradient.clone(),
```

Update all other `FitResult { ... }` literals with `final_gradient: None`.

**D. `src/io/fitrx.rs`** — add to `FitWire`:

```rust
#[serde(default, skip_serializing_if = "Option::is_none")]
final_gradient: Option<Vec<f64>>,
```

Populate in `build_fit_wire()` and `wire_to_fit_result()`.

### Callers of `OuterResult` to update

Every place that constructs an `OuterResult` must be updated. Run the grep
first — do not rely on this list alone:

```bash
grep -rn "OuterResult {" src/
```

As of 2026-06-02 the confirmed constructors are:

| File | Line | Path | `final_gradient` value |
|------|------|------|----------------------|
| `src/estimation/outer_optimizer.rs` | ~1202 | `optimize_nlopt` main return | from `Arc<Mutex>` |
| `src/estimation/outer_optimizer.rs` | ~1533 | `optimize_bfgs` return | `None` |
| `src/estimation/trust_region.rs` | 428 | `optimize_trust_region` return | `None` |
| `src/estimation/gauss_newton.rs` | 384 | GN early-return | `None` |
| `src/estimation/gauss_newton.rs` | 487 | GN normal return | `None` |
| `src/estimation/saem.rs` | 1799 | SAEM return | `None` |

Any test helper constructors found by the grep also need `final_gradient: None`.
Missing any one of these produces a struct-exhaustiveness compile error.

### Test

No standalone unit test required. Add `assert!` lines in an existing
integration test:

1. After a SLSQP fit: `result.final_gradient.is_some()`.
2. After a BOBYQA fit: `result.final_gradient.is_none()`.

### ferx-r follow-up

In `ferx_runlog()`:
- Print the FINAL GRADIENT section using `result$final_gradient` (a numeric
  vector or `NULL`).
- Add a convergence check: `all(abs(result$final_gradient) < tol)` where
  `tol` is e.g. `0.01`; warn if any component exceeds it.
- Guard against `NULL` (BOBYQA / BFGS runs) with a "gradient not available
  for this optimizer" note rather than an error.

### Files touched

| Repo | File | Change |
|------|------|--------|
| ferx-core | `src/estimation/outer_optimizer.rs` | `Arc<Mutex>` capture gated on OFV improvement; `OuterResult.final_gradient`; populate on return |
| ferx-core | `src/types.rs` | +1 field on `FitResult` |
| ferx-core | `src/api.rs` | wire through from `OuterResult`; update literals |
| ferx-core | `src/io/fitrx.rs` | +1 wire field with `serde(default)` |

---

## Ordering and PR strategy

Each step produces a focused, reviewable diff and can be merged independently.
**Merge in step order** (1 → 2 → 3 → 4 → 5 → 6) because Steps 3–6 all add
fields to `FitResult` and the struct-literal exhaustiveness check in `api.rs`
must be satisfied at each merge.

After each ferx-core step:

```bash
cargo check --tests
```

This validates struct-literal completeness across all constructors without
running the full compilation (Enzyme toolchain not available locally — push to
CI for full build and tests).

After the ferx-core PR merges, open the matching ferx-r PR. The `[patch]` in
`ferx-r/src/rust/.cargo/config.toml` points to the local ferx-core checkout,
so the R-side build picks up the new fields automatically.

---

## Quick reference: affected struct constructors

Every `FitResult { ... }` literal must be updated when new fields are added.
Find them all before starting each step:

```bash
grep -n "FitResult {" src/api.rs src/io/fitrx.rs
```

As of 2026-06-02, the main hits are:

| Location | Note |
|----------|------|
| `src/api.rs` ~line 1564 | `fit_inner()` primary literal — receives real values |
| `src/api.rs` ~line 3639 | `synthetic_fit()` test helper — use `None` / empty |
| `src/api.rs` ~line 3642 | second test helper literal |

When `OuterResult` gains new fields (Step 6), also grep:

```bash
grep -rn "OuterResult {" src/
```

The confirmed constructors as of 2026-06-02 are listed in the Step 6 table
above.
