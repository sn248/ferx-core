use crate::types::{DoseEvent, Population, Subject};
use std::collections::HashMap;
use std::path::Path;

/// Read a NONMEM-format CSV file into a Population.
///
/// Expected columns (case-insensitive):
///   ID, TIME, DV, EVID, AMT, CMT, RATE, MDV, II, SS, CENS, [covariates...]
///
/// EVID: 0=observation, 1=dose, 4=reset+dose
/// MDV: 1=missing dependent variable
/// CENS: 1=observation is below LLOQ (DV carries the LLOQ value); 0 otherwise
///
/// `iov_column`: when `Some(name)`, that column is read as the occasion index
/// (integer) and stored in `Subject::occasions` / `Subject::dose_occasions`.
/// The column is excluded from the covariate auto-detection list.
pub fn read_nonmem_csv(
    path: &Path,
    covariate_columns: Option<&[&str]>,
    iov_column: Option<&str>,
) -> Result<Population, String> {
    let mut rdr = csv::ReaderBuilder::new()
        .flexible(true)
        .has_headers(true)
        .from_path(path)
        .map_err(|e| format!("Failed to open CSV: {}", e))?;

    // Preserve original header casing for covariate names. Standard NONMEM
    // columns are matched case-insensitively so that legacy CSVs (e.g. `Id`,
    // `TIME`) keep working; covariate lookups remain case-sensitive.
    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| format!("Failed to read headers: {}", e))?
        .iter()
        .map(|h| h.trim().to_string())
        .collect();

    let col_idx_ci =
        |name: &str| -> Option<usize> { headers.iter().position(|h| h.eq_ignore_ascii_case(name)) };
    let col_idx_cs = |name: &str| -> Option<usize> { headers.iter().position(|h| h == name) };

    let id_col = col_idx_ci("id").ok_or("Missing ID column")?;
    let time_col = col_idx_ci("time").ok_or("Missing TIME column")?;
    let dv_col = col_idx_ci("dv").ok_or("Missing DV column")?;
    let evid_col = col_idx_ci("evid");
    let amt_col = col_idx_ci("amt");
    let cmt_col = col_idx_ci("cmt");
    let rate_col = col_idx_ci("rate");
    let mdv_col = col_idx_ci("mdv");
    let ii_col = col_idx_ci("ii");
    let ss_col = col_idx_ci("ss");
    let cens_col = col_idx_ci("cens");

    // IOV occasion column (case-insensitive lookup of user-specified name)
    let occ_col: Option<usize> = iov_column.and_then(|name| col_idx_ci(name));
    if iov_column.is_some() && occ_col.is_none() {
        return Err(format!(
            "iov_column '{}' not found in dataset headers",
            iov_column.unwrap()
        ));
    }

    const STANDARD_COLS: &[&str] = &[
        "id", "time", "dv", "evid", "amt", "cmt", "rate", "mdv", "ii", "ss", "cens",
    ];
    let is_standard = |h: &str| {
        STANDARD_COLS.iter().any(|s| h.eq_ignore_ascii_case(s))
            || iov_column.map_or(false, |iov| h.eq_ignore_ascii_case(iov))
    };

    // Identify covariate columns (names preserved in their original case).
    let cov_names: Vec<String> = match covariate_columns {
        Some(cols) => cols.iter().map(|c| c.to_string()).collect(),
        None => headers
            .iter()
            .filter(|h| !is_standard(h))
            .cloned()
            .collect(),
    };
    let cov_indices: Vec<(String, usize)> = cov_names
        .iter()
        .filter_map(|name| col_idx_cs(name).map(|idx| (name.clone(), idx)))
        .collect();

    // Parse rows grouped by ID
    let mut rows_by_id: Vec<(String, Vec<Vec<String>>)> = Vec::new();
    let mut current_id = String::new();

    for result in rdr.records() {
        let record = result.map_err(|e| format!("CSV parse error: {}", e))?;
        let fields: Vec<String> = record.iter().map(|f| f.trim().to_string()).collect();

        let id = fields.get(id_col).cloned().unwrap_or_default();
        if id != current_id {
            current_id = id.clone();
            rows_by_id.push((id, Vec::new()));
        }
        rows_by_id.last_mut().unwrap().1.push(fields);
    }

    // Build subjects
    let mut subjects = Vec::new();
    let mut total_occ_failures: usize = 0;
    for (id, rows) in &rows_by_id {
        let (subject, occ_failures) = parse_subject(
            id,
            rows,
            time_col,
            dv_col,
            evid_col,
            amt_col,
            cmt_col,
            rate_col,
            mdv_col,
            ii_col,
            ss_col,
            cens_col,
            occ_col,
            &cov_indices,
        )?;
        subjects.push(subject);
        total_occ_failures += occ_failures;
    }

    // Surface a single warning if any OCC values were missing/unparseable.
    // Such rows are silently mapped to occ=0, mixing with valid occ=0 rows —
    // user should clean the dataset.
    if let Some(name) = iov_column {
        if total_occ_failures > 0 {
            eprintln!(
                "[ferx] warning: {} row(s) had missing or unparseable values in iov_column '{}'; \
                 these rows were assigned occasion=0 and may be grouped with valid occ=0 rows. \
                 Consider cleaning the dataset.",
                total_occ_failures, name
            );
        }
    }

    Ok(Population {
        subjects,
        covariate_names: cov_names,
        dv_column: "dv".to_string(),
    })
}

fn parse_f64(s: &str) -> f64 {
    s.parse::<f64>().unwrap_or(0.0)
}

fn parse_usize(s: &str) -> usize {
    s.parse::<usize>().unwrap_or(1)
}

/// Parse an occasion-column cell. Returns `None` for blank / `.` / NA / non-integer
/// values so the caller can warn about silently dropped rows. NONMEM convention
/// uses `.` for missing.
fn parse_occ(s: &str) -> Option<u32> {
    let t = s.trim();
    if t.is_empty() || t == "." || t.eq_ignore_ascii_case("na") || t.eq_ignore_ascii_case("nan") {
        return None;
    }
    t.parse::<u32>().ok()
}

#[allow(clippy::too_many_arguments)]
fn parse_subject(
    id: &str,
    rows: &[Vec<String>],
    time_col: usize,
    dv_col: usize,
    evid_col: Option<usize>,
    amt_col: Option<usize>,
    cmt_col: Option<usize>,
    rate_col: Option<usize>,
    mdv_col: Option<usize>,
    ii_col: Option<usize>,
    ss_col: Option<usize>,
    cens_col: Option<usize>,
    occ_col: Option<usize>,
    cov_indices: &[(String, usize)],
) -> Result<(Subject, usize), String> {
    let mut doses = Vec::new();
    let mut obs_times = Vec::new();
    let mut observations = Vec::new();
    let mut obs_cmts = Vec::new();
    let mut cens = Vec::new();
    let mut occasions: Vec<u32> = Vec::new();
    let mut dose_occasions: Vec<u32> = Vec::new();
    let mut occ_parse_failures: usize = 0;

    // Time-constant covariates: first non-missing value across all rows.
    // Used as the subject-static fallback (and for the AD fast path, which
    // does not yet read per-event snapshots).
    let mut covariates: HashMap<String, f64> = HashMap::new();
    for (name, idx) in cov_indices {
        for row in rows {
            if let Some(val_str) = row.get(*idx) {
                if let Ok(val) = val_str.parse::<f64>() {
                    if val.is_finite() {
                        covariates.insert(name.clone(), val);
                        break;
                    }
                }
            }
        }
    }

    // Detect which covariates are time-varying within this subject. Per-event
    // snapshots are only built when at least one is — keeps memory flat for
    // models with no TV covariates.
    let mut tv_names: Vec<&str> = Vec::new();
    let mut tv_indices: Vec<usize> = Vec::new();
    for (name, idx) in cov_indices {
        let mut first_val: Option<f64> = None;
        let mut is_tv = false;
        for row in rows {
            let v_opt = row
                .get(*idx)
                .and_then(|s| s.parse::<f64>().ok())
                .filter(|v| v.is_finite());
            if let Some(v) = v_opt {
                match first_val {
                    None => first_val = Some(v),
                    Some(fv) if (v - fv).abs() > 1e-12 => {
                        is_tv = true;
                        break;
                    }
                    _ => {}
                }
            }
        }
        if is_tv {
            tv_names.push(name.as_str());
            tv_indices.push(*idx);
        }
    }
    let any_tv = !tv_names.is_empty();

    // LOCF state for the per-event snapshot path. Initialized from the
    // subject-static `covariates` map so the first event sees something
    // sensible even if the row's own value is missing.
    let mut locf_state: HashMap<String, f64> = covariates.clone();

    // Per-event covariate snapshots (only populated when any_tv is true).
    let mut dose_covariates: Vec<HashMap<String, f64>> = Vec::new();
    let mut obs_covariates: Vec<HashMap<String, f64>> = Vec::new();

    for row in rows {
        // Update LOCF state from this row's TV-covariate values *before*
        // classifying the event, matching NONMEM's "$PK runs at the record
        // with this record's covariate values" semantics.
        if any_tv {
            for (name, idx) in tv_names.iter().zip(tv_indices.iter()) {
                if let Some(s) = row.get(*idx) {
                    if let Ok(v) = s.parse::<f64>() {
                        if v.is_finite() {
                            locf_state.insert((*name).to_string(), v);
                        }
                    }
                }
            }
        }

        let time = parse_f64(row.get(time_col).map(|s| s.as_str()).unwrap_or("0"));
        let evid = evid_col
            .and_then(|c| row.get(c))
            .map(|s| parse_usize(s))
            .unwrap_or(0);
        let mdv = mdv_col
            .and_then(|c| row.get(c))
            .map(|s| parse_usize(s))
            .unwrap_or(0);
        // Parse OCC. When iov_column is set but a row's value is missing or
        // unparseable, count it (caller emits a single summary warning) and
        // fall back to 0 — matching pre-warning behavior so existing fits
        // don't change. With no iov_column, parse failures are not tracked.
        let occ = if let Some(c) = occ_col {
            match row.get(c).and_then(|s| parse_occ(s)) {
                Some(n) => n,
                None => {
                    occ_parse_failures += 1;
                    0
                }
            }
        } else {
            0
        };

        if evid == 1 || evid == 4 {
            // Dose record
            let amt = amt_col
                .and_then(|c| row.get(c))
                .map(|s| parse_f64(s))
                .unwrap_or(0.0);
            let cmt = cmt_col
                .and_then(|c| row.get(c))
                .map(|s| parse_usize(s))
                .unwrap_or(1);
            let rate = rate_col
                .and_then(|c| row.get(c))
                .map(|s| parse_f64(s))
                .unwrap_or(0.0);
            let ii = ii_col
                .and_then(|c| row.get(c))
                .map(|s| parse_f64(s))
                .unwrap_or(0.0);
            let ss = ss_col
                .and_then(|c| row.get(c))
                .map(|s| parse_usize(s) > 0)
                .unwrap_or(false);

            doses.push(DoseEvent::new(time, amt, cmt, rate, ss, ii));
            if occ_col.is_some() {
                dose_occasions.push(occ);
            }
            if any_tv {
                dose_covariates.push(locf_state.clone());
            }
        } else if evid == 0 && mdv == 0 {
            // Observation record
            let dv = parse_f64(row.get(dv_col).map(|s| s.as_str()).unwrap_or("0"));
            let cmt = cmt_col
                .and_then(|c| row.get(c))
                .map(|s| parse_usize(s))
                .unwrap_or(1);
            let cens_flag = cens_col
                .and_then(|c| row.get(c))
                .map(|s| parse_usize(s))
                .unwrap_or(0);
            obs_times.push(time);
            observations.push(dv);
            obs_cmts.push(cmt);
            cens.push(if cens_flag > 0 { 1u8 } else { 0u8 });
            if occ_col.is_some() {
                occasions.push(occ);
            }
            if any_tv {
                obs_covariates.push(locf_state.clone());
            }
        }
    }

    // Sort doses by time (keeping dose_occasions and dose_covariates in sync).
    // Stable sort would be safer when two events share a time, but PartialOrd
    // sort_by gives us a stable answer for f64 ordered times, and matches
    // pre-existing behavior.
    let n_doses = doses.len();
    let mut perm: Vec<usize> = (0..n_doses).collect();
    perm.sort_by(|&a, &b| doses[a].time.partial_cmp(&doses[b].time).unwrap());
    let sorted_doses: Vec<DoseEvent> = perm.iter().map(|&i| doses[i].clone()).collect();
    let sorted_dose_occ: Vec<u32> = if occ_col.is_some() {
        perm.iter().map(|&i| dose_occasions[i]).collect()
    } else {
        Vec::new()
    };
    let sorted_dose_cov: Vec<HashMap<String, f64>> = if any_tv {
        perm.iter().map(|&i| dose_covariates[i].clone()).collect()
    } else {
        Vec::new()
    };

    Ok((
        Subject {
            id: id.to_string(),
            doses: sorted_doses,
            obs_times,
            observations,
            obs_cmts,
            covariates,
            dose_covariates: sorted_dose_cov,
            obs_covariates,
            cens,
            occasions,
            dose_occasions: sorted_dose_occ,
        },
        occ_parse_failures,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_csv(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn test_occ_absent_gives_empty_occasions() {
        let csv = "ID,TIME,DV,EVID,AMT\n1,0,.,1,100\n1,1,5.0,0,.\n1,2,3.0,0,.\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        assert!(pop.subjects[0].occasions.is_empty());
        assert!(pop.subjects[0].dose_occasions.is_empty());
    }

    #[test]
    fn test_parse_subject_reads_occ_column() {
        let csv = "ID,TIME,DV,EVID,AMT,OCC\n\
                   1,0,.,1,100,1\n\
                   1,1,5.0,0,.,1\n\
                   1,2,3.0,0,.,1\n\
                   1,7,.,1,100,2\n\
                   1,8,4.0,0,.,2\n\
                   1,9,2.5,0,.,2\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, Some("OCC")).unwrap();
        let subj = &pop.subjects[0];
        // Two obs in occ 1, two in occ 2 (dose rows are stripped from occasions)
        assert_eq!(subj.occasions, vec![1, 1, 2, 2]);
        assert_eq!(subj.dose_occasions, vec![1, 2]);
    }

    #[test]
    fn test_occ_column_excluded_from_covariates() {
        let csv = "ID,TIME,DV,EVID,AMT,OCC,WT\n\
                   1,0,.,1,100,1,70\n\
                   1,1,5.0,0,.,1,70\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, Some("OCC")).unwrap();
        // OCC should NOT appear as a covariate; WT should
        assert!(!pop.covariate_names.contains(&"OCC".to_string()));
        assert!(pop.covariate_names.contains(&"WT".to_string()));
    }

    #[test]
    fn test_missing_iov_column_errors() {
        let csv = "ID,TIME,DV,EVID,AMT\n1,0,.,1,100\n";
        let f = write_csv(csv);
        let result = read_nonmem_csv(f.path(), None, Some("OCC"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("iov_column"));
    }

    #[test]
    fn test_parse_occ_recognizes_missing_sentinels() {
        // NONMEM-style "." plus blanks/NAs should parse as None.
        assert_eq!(parse_occ(""), None);
        assert_eq!(parse_occ("."), None);
        assert_eq!(parse_occ("  "), None);
        assert_eq!(parse_occ("NA"), None);
        assert_eq!(parse_occ("nan"), None);
        // Non-integer or signed values that u32 can't parse
        assert_eq!(parse_occ("1.5"), None);
        assert_eq!(parse_occ("-1"), None);
        // Valid u32 round-trips
        assert_eq!(parse_occ("1"), Some(1));
        assert_eq!(parse_occ("42"), Some(42));
    }

    #[test]
    fn test_missing_occ_value_does_not_break_load_but_falls_back_to_zero() {
        // Row with OCC = "." gets occ=0; load still succeeds (warning is
        // emitted to stderr, not asserted here).
        let csv = "ID,TIME,DV,EVID,AMT,OCC\n\
                   1,0,.,1,100,1\n\
                   1,1,5.0,0,.,1\n\
                   1,2,3.0,0,.,.\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, Some("OCC")).unwrap();
        let subj = &pop.subjects[0];
        // Two obs: first has OCC=1, second had "." → 0
        assert_eq!(subj.occasions, vec![1, 0]);
    }

    #[test]
    fn test_no_tv_covariates_leaves_per_event_snapshots_empty() {
        // WT is constant — no per-event snapshots should be allocated.
        let csv = "ID,TIME,DV,EVID,AMT,WT\n\
                   1,0,.,1,100,70\n\
                   1,1,5.0,0,.,70\n\
                   1,2,3.0,0,.,70\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let subj = &pop.subjects[0];
        assert!(!subj.has_tv_covariates());
        assert!(subj.dose_covariates.is_empty());
        assert!(subj.obs_covariates.is_empty());
        // Static covariate map still populated.
        assert_eq!(subj.covariates["WT"], 70.0);
        // Fallback helpers return the static map.
        assert_eq!(subj.dose_cov(0)["WT"], 70.0);
        assert_eq!(subj.obs_cov(0)["WT"], 70.0);
    }

    #[test]
    fn test_tv_covariate_locf_per_event_snapshot() {
        // CR changes mid-subject. Each event must see the LOCF value at its
        // own row's time (NONMEM $PK semantics).
        let csv = "ID,TIME,DV,EVID,AMT,WT,CR\n\
                   1,0,.,1,100,70,1.0\n\
                   1,1,5.0,0,.,70,1.0\n\
                   1,2,3.0,0,.,70,1.5\n\
                   1,3,.,1,100,70,1.5\n\
                   1,4,2.5,0,.,70,2.0\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let subj = &pop.subjects[0];
        assert!(subj.has_tv_covariates());
        // 2 doses, 3 obs.
        assert_eq!(subj.dose_covariates.len(), 2);
        assert_eq!(subj.obs_covariates.len(), 3);
        // Static covariate map is the *first* CR value (1.0), kept for
        // the AD fast-path / fallback.
        assert_eq!(subj.covariates["CR"], 1.0);
        // Dose 1 (t=0): CR=1.0; Dose 2 (t=3): CR=1.5
        assert_eq!(subj.dose_covariates[0]["CR"], 1.0);
        assert_eq!(subj.dose_covariates[1]["CR"], 1.5);
        // Obs at t=1: CR=1.0; t=2: CR=1.5; t=4: CR=2.0
        assert_eq!(subj.obs_covariates[0]["CR"], 1.0);
        assert_eq!(subj.obs_covariates[1]["CR"], 1.5);
        assert_eq!(subj.obs_covariates[2]["CR"], 2.0);
        // WT is constant — but appears in every snapshot at its constant value.
        assert_eq!(subj.dose_covariates[0]["WT"], 70.0);
        assert_eq!(subj.obs_covariates[2]["WT"], 70.0);
    }

    #[test]
    fn test_tv_covariate_snapshot_keeps_dose_sort_alignment() {
        // Doses arrive in non-time order in the CSV. After sorting doses by
        // time, dose_covariates must follow the same permutation so each
        // dose still pairs with its own snapshot.
        let csv = "ID,TIME,DV,EVID,AMT,CR\n\
                   1,5,.,1,100,2.0\n\
                   1,0,.,1,100,1.0\n\
                   1,6,5.0,0,.,2.0\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let subj = &pop.subjects[0];
        assert!(subj.has_tv_covariates());
        // After sorting: dose 0 = t=0 (CR=1.0), dose 1 = t=5 (CR=2.0).
        assert_eq!(subj.doses[0].time, 0.0);
        assert_eq!(subj.doses[1].time, 5.0);
        assert_eq!(subj.dose_covariates[0]["CR"], 1.0);
        assert_eq!(subj.dose_covariates[1]["CR"], 2.0);
    }

    #[test]
    fn test_tv_covariate_locf_handles_missing_intermediate() {
        // Missing CR values are filled forward (LOCF), matching NONMEM.
        let csv = "ID,TIME,DV,EVID,AMT,CR\n\
                   1,0,.,1,100,1.0\n\
                   1,1,5.0,0,.,.\n\
                   1,2,3.0,0,.,2.0\n\
                   1,3,2.0,0,.,.\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        let subj = &pop.subjects[0];
        // Obs 0 (t=1, CR missing) → LOCF → 1.0.
        assert_eq!(subj.obs_covariates[0]["CR"], 1.0);
        // Obs 1 (t=2, CR=2.0) → 2.0.
        assert_eq!(subj.obs_covariates[1]["CR"], 2.0);
        // Obs 2 (t=3, CR missing) → LOCF → 2.0.
        assert_eq!(subj.obs_covariates[2]["CR"], 2.0);
    }
}
