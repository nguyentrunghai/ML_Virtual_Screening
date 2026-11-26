# ML Virtual Screening

Machine-learning virtual screening pipeline for predicting ligand binding free energy ($\Delta G$) against five drug targets using an ensemble of Random Forest (`RF`), `XGBoost`, and Directed Message Passing Neural Network (`DMPNN`) models.

## Supported Drug Targets & Models

| Target | Models |
| :--- | :--- |
| `EGFR` | `RF`, `XGBoost`, `DMPNN` |
| `AChE` | `RF`, `XGBoost`, `DMPNN` |
| `BACE1` | `RF`, `XGBoost`, `DMPNN` |
| `SARS-CoV-2-MPro` | `RF`, `XGBoost`, `DMPNN` |
| `Neuraminidase-1` | `RF`, `XGBoost`, `DMPNN` |

- **RF & XGBoost**: Use 217 RDKit molecular descriptors extracted via `deepchem.feat.RDKitDescriptors` with saved imputation and standardization parameters.
- **DMPNN**: Uses `deepchem.feat.DMPNNFeaturizer` and a restored `deepchem.models.DMPNNModel` with its saved target transformer (`y_transformer.joblib`).
- **Ensemble prediction**: Averages predictions across all models configured for the selected target (`skipna=True`).

---

## Project Structure

```text
ML_Virtual_Screening/
├── README.md
├── data/
│   └── test/
│       └── <target>/<model>/test_data.csv   # Reference test sets for verification
├── models/
│   └── <target>/
│       ├── RF/rf_py310deepchem_01.pkl
│       ├── XGBoost/xbg_py310deepchem_01.pkl
│       └── DMPNN/                           # Checkpoint + y_transformer.joblib
└── src/
    ├── config.yaml                          # Root directory and model path definitions
    ├── ml_inference.py                      # Core RF, XGBoost, DMPNN, and ensemble inference
    ├── utils.py                             # YAML config loader and path verification
    ├── run_ml_inference_test.py             # CLI to verify all models against reference test sets
    ├── run_dG_prediction.py                 # CLI to predict ensemble dG for new compounds
    ├── environment.yml                      # Conda environment specification (deepchem)
    └── requirements.txt                     # Frozen pip package versions
```

---

## Requirements & Setup

- **Python**: 3.10+
- **DeepChem**: **`2.8.0` is strictly required** (RDKit descriptor extraction asserts 217 features matching DeepChem 2.8.0).
- Key packages: `deepchem==2.8.0`, `rdkit`, `scikit-learn`, `xgboost`, `torch`, `joblib`, `pandas`, `numpy`, `PyYAML`. See `src/environment.yml` and `src/requirements.txt` for the full dependency list.

### Create the Conda Environment

Create and activate the `deepchem` environment from `src/environment.yml`:

```bash
conda env create -f src/environment.yml
conda activate deepchem
```

### Configuration (`src/config.yaml`)

Before running the scripts on a new machine or directory, update `root_dir` in `src/config.yaml` to point to your local `ML_Virtual_Screening` directory:

```yaml
root_dir: /path/to/ML_Virtual_Screening

model_paths:
  EGFR:
    RF: models/EGFR/RF/rf_py310deepchem_01.pkl
    XGBoost: models/EGFR/XGBoost/xbg_py310deepchem_01.pkl
    DMPNN: models/EGFR/DMPNN
  ...
```

All `model_paths` entries are resolved relative to `root_dir` and verified on disk at startup.

---

## Usage

### 1. Verify Installation & Models (`run_ml_inference_test.py`)

`run_ml_inference_test.py` runs inference for every target and model defined in `config.yaml` against the reference test sets located at `<root_dir>/data/test/<target>/<model>/test_data.csv`. For each model, it compares the newly computed predictions against the stored `predicted_dG` column and marks the test `PASS` if the mean squared error (MSE) is strictly below `--mse-cutoff`.

**Run from `src/`:**

```bash
cd src
python run_ml_inference_test.py
```

**Or run from the repository root:**

```bash
python src/run_ml_inference_test.py --config src/config.yaml
```

**Options:**

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-c`, `--config` | `config.yaml` | Path to the YAML config file containing `root_dir` and `model_paths`. |
| `--mse-cutoff` | `1e-8` | Maximum allowed MSE between computed predictions and reference `predicted_dG`. |

**Exit codes:**
- `0`: All target/model checks passed.
- `1`: One or more checks failed.

---

### 2. Predict Binding Free Energy ($\Delta G$) (`run_dG_prediction.py`)

`run_dG_prediction.py` predicts ensemble $\Delta G$ values for a CSV file of compounds against a chosen drug target. It runs all models configured for `--target` on unique SMILES, averages their predictions, merges the result back onto the input table, and writes the output CSV.

**Input CSV format:**
- Must contain a column of SMILES strings (default column name: `smiles`).
- Any additional columns in the input CSV are preserved in the output CSV.

Example input (`compounds.csv`):

```csv
compound_id,smiles
CMPD_001,Cc1ccc(cc1)-c1nn(c2ncnc(N)c12)C(C)(C)C
CMPD_002,COc1cc(C=C(C#N)C#N)cc(O)c1O
```

**Run from `src/`:**

```bash
cd src
python run_dG_prediction.py \
  --target EGFR \
  --input_csv ../data/test/EGFR/RF/test_data.csv \
  --output_csv egfr_predictions.csv
```

**Or run from the repository root:**

```bash
python src/run_dG_prediction.py \
  --config src/config.yaml \
  --target EGFR \
  --input_csv compounds.csv \
  --smiles_col smiles \
  --output_csv egfr_predictions.csv \
  --pred_col ensemble_predicted_dG
```

**Options:**

| Flag | Required | Default | Description |
| :--- | :---: | :--- | :--- |
| `-c`, `--config` | No | `config.yaml` | Path to the YAML config file containing `root_dir` and `model_paths`. |
| `--target` | Yes | — | Drug target name (must match a key in `model_paths`, e.g., `EGFR`, `AChE`, `BACE1`, `SARS-CoV-2-MPro`, `Neuraminidase-1`). |
| `--input_csv` | Yes | — | Path to the input CSV file containing compound SMILES. |
| `--smiles_col` | No | `smiles` | Name of the SMILES column in the input CSV. |
| `--output_csv` | No | `output.csv` | Path where the output CSV with predictions will be saved. |
| `--pred_col` | No | `ensemble_predicted_dG` | Name of the ensemble prediction column added to the output CSV. |

**Example output (`egfr_predictions.csv`):**

```csv
compound_id,smiles,ensemble_predicted_dG
CMPD_001,Cc1ccc(cc1)-c1nn(c2ncnc(N)c12)C(C)(C)C,-8.570490578363147
CMPD_002,COc1cc(C=C(C#N)C#N)cc(O)c1O,-6.462504075584761
```
