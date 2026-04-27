# RMAB-MIMIC

RMAB-MIMIC is a notebook-driven research project for modeling ICU allocation as a **Restless Multi-Armed Bandit (RMAB)** problem on the **eICU-CRD Demo v2.0.1**. It builds empirical patient-state transition matrices from EHR data, evaluates allocation heuristics, and compares a clinical-only model against a fairness-focused variant that adds insurance status to the state space.

## What this project does

The repository is organized around five notebooks that form an end-to-end workflow:

- [transition_matrix_pipeline.ipynb](transition_matrix_pipeline.ipynb) builds the clinical-only transition matrices `P_icu.npy` and `P_nonicu.npy` from the MIMIC-IV Demo tables.
- [model_A_fairness_pipeline.ipynb](model_A_fairness_pipeline.ipynb) extends the state space with insurance status and produces `P_icu_A.npy` and `P_nonicu_A.npy`.
- [rmab_allocation_heuristic.ipynb](rmab_allocation_heuristic.ipynb) simulates a first-pass ICU allocation policy using the learned transition matrices.
- [rmab_ml_policy_training.ipynb](rmab_ml_policy_training.ipynb) trains and evaluates a Q-learning policy on the RMAB dynamics.
- [rmab_compare_modelA_modelB.ipynb](rmab_compare_modelA_modelB.ipynb) compares the clinical-only and insurance-augmented models under the same simulation design.

In practice, the project turns raw gzipped CSV tables into daily patient states, discretizes those states into Markov transitions, and then uses the resulting dynamics to study ICU allocation policies.

## Why this project has relevance to social impact

- reproduce an RMAB pipeline on a real clinical dataset;
- compare clinical-only decision making with a fairness-aware alternative;
- reuse the learned transition matrices in downstream policy simulation or optimization;
- inspect how non-clinical attributes such as insurance status can affect allocation outcomes;
- keep the full workflow in notebooks so each stage remains easy to inspect and rerun.

The main outputs are reusable NumPy arrays and figures saved at the repository root or under `artifacts/rmab/`.

## Repository layout

```text
RMAB-MIMIC/
├── data/
│   └── mimic-iv-demo/
│       └── mimic-iv-clinical-database-demo-2.2/
├── artifacts/
│   └── rmab/
│       ├── baseline/
│       ├── evaluation/
│       ├── matrices/
│       └── processed/
├── transition_matrix_pipeline.ipynb
├── model_A_fairness_pipeline.ipynb
├── rmab_allocation_heuristic.ipynb
├── rmab_ml_policy_training.ipynb
├── rmab_compare_modelA_modelB.ipynb
├── P_icu.npy
├── P_nonicu.npy
├── P_icu_A.npy
└── P_nonicu_A.npy
```

The `data/mimic-iv-demo/mimic-iv-clinical-database-demo-2.2/` folder is expected to contain the extracted MIMIC-IV Demo release, including the `hosp/` and `icu/` tables used by the notebooks. See the bundled dataset documentation in [data/mimic-iv-demo/mimic-iv-clinical-database-demo-2.2/README.txt](data/mimic-iv-demo/mimic-iv-clinical-database-demo-2.2/README.txt).

## Getting started

### Prerequisites

- A recent Python 3 environment
- Jupyter Notebook or JupyterLab
- The MIMIC-IV Clinical Database Demo v2.2 extracted into `data/mimic-iv-demo/mimic-iv-clinical-database-demo-2.2/`

### Install dependencies

The notebooks use the standard scientific Python stack:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `jupyter`

On Windows PowerShell, you can set up a local virtual environment like this:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas matplotlib seaborn jupyter
```

### Run the workflow

1. Open the repository in JupyterLab or VS Code.
2. Run [transition_matrix_pipeline.ipynb](transition_matrix_pipeline.ipynb) first to regenerate `P_icu.npy` and `P_nonicu.npy`.
3. Run [model_A_fairness_pipeline.ipynb](model_A_fairness_pipeline.ipynb) to regenerate `P_icu_A.npy` and `P_nonicu_A.npy`.
4. Use [rmab_allocation_heuristic.ipynb](rmab_allocation_heuristic.ipynb) to inspect heuristic allocation behavior.
5. Use [rmab_ml_policy_training.ipynb](rmab_ml_policy_training.ipynb) to train and evaluate the Q-learning policy.
6. Use [rmab_compare_modelA_modelB.ipynb](rmab_compare_modelA_modelB.ipynb) to compare the two transition models.

If you only want to explore the saved outputs, the committed `.npy` files in the repository root are ready to load directly with NumPy.

### Example usage

```python
import numpy as np

P_icu = np.load("P_icu.npy")
P_nonicu = np.load("P_nonicu.npy")

print(P_icu.shape)
print(P_nonicu.shape)
```


If you plan to extend the project, the most useful additions are new notebooks, clearer data preprocessing steps, and reproducible evaluation outputs under `artifacts/rmab/`.
