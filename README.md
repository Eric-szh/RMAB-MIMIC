# RMAB-MIMIC

This repository now includes a reproducible RMAB workflow for ICU allocation experiments:

1. **Dataset preparation + transition estimation**
   - Builds daily patient states from MIMIC-IV hosp/icu tables.
   - Supports 4-variable or 8-variable discretized state spaces.
   - Learns action-conditioned transition matrices (`ICU`, `non-ICU`).
   - Enforces absorbing death state.
   - Replaces zero-row uniform fallback with support-threshold smoothing.
   - Produces bootstrap confidence intervals for self-loop uncertainty.
   - Freezes baseline `P_icu.npy` / `P_nonicu.npy` with checksums.

2. **RMAB simulation + offline policy evaluation**
   - Simulates fixed ICU occupancy windows (default 7 days).
   - Capacity-constrained daily ICU allocation (top-K admissions).
   - Includes policy baselines: random, severity, myopic, index-style.
   - Reports cumulative reward, mortality events, ICU utilization, fairness gap.
   - Runs sensitivity analysis for ICU capacity and ICU cost.

## Requirements

Python 3.10+ with:

- `numpy`
- `pandas`

## Prepare RMAB artifacts

```bash
python scripts/prepare_rmab_dataset.py \
  --repo-root . \
  --demo-zip data/mimic-iv-demo.zip \
  --out-dir artifacts/rmab \
  --state-spec 4var \
  --alpha 2.0 \
  --min-support 10 \
  --bootstraps 200
```

If you have full MIMIC-IV extracted locally, pass `--mimic-root /absolute/path/to/mimic-iv-clinical-database-...`.

### Key outputs

- `artifacts/rmab/baseline/manifest.json`
- `artifacts/rmab/diagnostics.json`
- `artifacts/rmab/row_support.csv`
- `artifacts/rmab/bootstrap_self_loop_ci.csv`
- `artifacts/rmab/matrices/P_icu_smoothed.npy`
- `artifacts/rmab/matrices/P_nonicu_smoothed.npy`
- `artifacts/rmab/processed/transitions.csv.gz`
- `artifacts/rmab/processed/initial_cohort.csv`

## Run RMAB policy evaluation

```bash
python scripts/run_rmab_experiment.py \
  --artifact-dir artifacts/rmab \
  --episodes 100 \
  --horizon-days 30 \
  --cohort-size 200 \
  --k-capacity 20 \
  --icu-stay-days 7
```

### Evaluation outputs

- `artifacts/rmab/evaluation/policy_comparison.csv`
- `artifacts/rmab/evaluation/sensitivity.csv`
- `artifacts/rmab/evaluation/summary.json`

## Notes / assumptions

- Decision and transition granularity are daily.
- Patients already occupying ICU are automatically continued until stay timer ends.
- Death state is absorbing under both actions.
- Fairness metric is inter-group reward gap across age groups.
