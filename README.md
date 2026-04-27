# RMAB-MIMIC

RMAB-MIMIC is a notebook-driven research project for modeling ICU allocation as a **Restless Multi-Armed Bandit (RMAB)** problem using real-world clinical datasets. The project builds empirical patient-state transition matrices from EHR data, evaluates allocation heuristics, and studies fairness in resource allocation decisions.

---

## Repository Overview

This repository contains **two branches**, each corresponding to a different dataset and fairness setting:

### 🔹 `main` branch
- Dataset: **MIMIC-IV Clinical Database Demo v2**
- Fairness variable: **Insurance status**
- Focus: Compare clinical-only vs insurance-aware allocation

### 🔹 `eicu-demo-pipeline` branch
- Dataset: **eICU-CRD Demo v2.0.1**
- Fairness variable: **Hospital region (South / Midwest / Northeast+West)**
- Focus: Study fairness across geographic variation instead of insurance

---

## What this project does

Across both branches, the workflow follows a consistent pipeline:

1. **Data preprocessing**
   - Convert raw EHR tables into patient-day level observations
   - Discretize clinical features into a finite state space

2. **Transition matrix learning**
   - Learn:
     - `P_icu`: transition under ICU care
     - `P_nonicu`: transition without ICU care

3. **Model variants**
   - **Model B (baseline)**: clinical-only state space  
   - **Model A (fairness-aware)**: extends state space with non-clinical attributes

4. **Policy evaluation**
   - Simulate ICU allocation under capacity constraints
   - Compare heuristic and learning-based policies

---

## Notebook Structure (by branch)

### 📍 `main` (MIMIC-IV)

- `transition_matrix_pipeline.ipynb`  
  Builds clinical-only transition matrices (Model B)

- `model_A_fairness_pipeline.ipynb`  
  Adds **insurance status** → fairness-aware Model A

- `rmab_allocation_heuristic.ipynb`  
  Simulates heuristic ICU allocation

- `rmab_ml_policy_training.ipynb`  
  Trains Q-learning policy

- `rmab_compare_modelA_modelB.ipynb`  
  Compares fairness vs clinical-only models

---

### 📍 `eicu-demo-pipeline`

- `eicu_transition_matrix_pipeline.ipynb`  
  Model B (clinical-only, 82-state space)

- `eicu_model_A_fairness_pipeline.ipynb`  
  Model A using **hospital region** as fairness variable

---

## Key Idea

We model ICU allocation as an RMAB problem:

- Each patient = an arm  
- Action = assign ICU care or not  
- Patients evolve over time regardless of action (**restless**)  
- ICU capacity constraint limits number of treated patients  

---

## Social Impact

This project explores how AI-driven allocation decisions:

- Improve **patient outcomes**
- Optimize **limited ICU resources**
- Reveal potential **bias from non-clinical factors**
  - insurance (MIMIC-IV)
  - region (eICU)

---

## Repository Layout (example: `main` branch)

```text
RMAB-MIMIC/
├── data/
│   └── mimic-iv-demo/
├── artifacts/
│   └── rmab/
│       ├── matrices/
│       ├── processed/
│       └── evaluation/
├── transition_matrix_pipeline.ipynb
├── model_A_fairness_pipeline.ipynb
├── rmab_allocation_heuristic.ipynb
├── rmab_ml_policy_training.ipynb
├── rmab_compare_modelA_modelB.ipynb
├── P_icu.npy
├── P_nonicu.npy
├── P_icu_A.npy
└── P_nonicu_A.npy

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
