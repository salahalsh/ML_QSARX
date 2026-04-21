# ML_QSARX — Systematic Multi-Target QSAR Benchmarking

Machine learning algorithms, molecular descriptors, and validation strategies benchmarked across four therapeutically diverse ChEMBL targets.

**Paper:** *Systematic Multi-Target QSAR Benchmarking: Machine Learning Algorithms, Molecular Descriptors, and Validation*
Salah A. Alshehade, Ghazi Al Jabal, Iqbal H. Jebril

**Interactive tool (web UI):** https://insilicosigma.com/qsar-x/

## Overview

Reproducible source code and curated datasets for the nine experiments reported in the paper:

| # | Experiment | Script |
|---|-----------|--------|
| 1 | Algorithm comparison (10 ML methods) | `run_multi_target.py` |
| 2 | Descriptor evaluation (5 representations) | `run_multi_target.py` |
| 3 | Feature selection impact | `run_multi_target.py` |
| 4 | Random vs. scaffold splitting | `run_multi_target.py` |
| 5 | Y-scrambling validation | `run_multi_target.py` |
| 6 | Cross-target consistency | `run_multi_target.py` |
| 7 | Bayesian hyperparameter optimization (Optuna) | `run_hpo.py` |
| 8 | Graph convolutional network baseline | `run_gnn.py` |
| 9 | Applicability domain analysis | `run_ad_analysis.py` |

## Datasets

Four ChEMBL (v34) targets spanning four protein families, totaling **33,751 compounds**:

| Target | ChEMBL ID | Family | Activity | Compounds |
|--------|-----------|--------|----------|-----------|
| EGFR   | CHEMBL203  | Kinase      | IC50 | 10,036 |
| DRD2   | CHEMBL217  | GPCR        | Ki   | 7,558  |
| BACE-1 | CHEMBL4822 | Protease    | IC50 | 8,080  |
| hERG   | CHEMBL240  | Ion Channel | IC50 | 8,077  |

Pre-curated datasets are provided in `data/`. To re-download and re-curate from ChEMBL:

```bash
cd data
python download_all_targets.py
```

## Installation

```bash
python -m venv qsar_env
source qsar_env/bin/activate    # Linux/Mac
qsar_env\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Reproduction

```bash
# Experiments 1-6 (core benchmarking)
python run_multi_target.py

# Experiment 7 (Bayesian HPO, ~hours)
python run_hpo.py

# Experiment 8 (GCN baseline, GPU recommended)
python run_gnn.py

# Experiment 9 (applicability domain)
python run_ad_analysis.py
```

All scripts fix `random_state=42` for full reproducibility. Results are written to a `results/` directory created at runtime.

## Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── run_multi_target.py        # Experiments 1-6
├── run_hpo.py                 # Experiment 7 (Optuna HPO)
├── run_gnn.py                 # Experiment 8 (GCN baseline)
├── run_ad_analysis.py         # Experiment 9 (applicability domain)
├── data/
│   ├── download_all_targets.py   # ChEMBL download + curation pipeline
│   ├── egfr_clean.csv
│   ├── drd2_clean.csv
│   ├── bace1_clean.csv
│   └── herg_clean.csv
└── supplementary/             # Tables S1-S7 (CSV)
```

## Key Results

| Finding | Value |
|---------|-------|
| Best algorithm + descriptor | RF + ECFP4 (mean R2 = 0.657 +/- 0.051) |
| Algorithm ranking consistency | Spearman rho = 0.83 - 0.99 |
| Scaffold gap range | 0.045 (BACE-1) to 0.253 (hERG) |
| GCN vs RF+ECFP4 deficit | mean -0.218 R2 (random split) |
| AD R2 drop (outside domain) | 0.31 - 0.51 units |

## Software Versions

Python 3.13, RDKit 2025.03.6, scikit-learn 1.7.2, XGBoost 2.1, LightGBM 4.5, Optuna 4.2, PyTorch 2.10, PyTorch Geometric 2.7.0, NumPy 2.1.3, Pandas 2.2, Matplotlib 3.10.

## Citation

```bibtex
@article{Alshehade2026qsar,
  title={Systematic Multi-Target QSAR Benchmarking: Machine Learning
         Algorithms, Molecular Descriptors, and Validation},
  author={Alshehade, Salah A. and Al Jabal, Ghazi and Jebril, Iqbal H.},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).
