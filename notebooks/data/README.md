# Data

Datasets used in the notebook series. Small files (subsets, splits, indices) are committed; large raw files are gitignored and re-downloaded on demand by helpers in `notebooks/utils/data_loading.py`.

## Layout (created on demand)

- `chembl/` — small ChEMBL SMILES subset for MLM pre-training (notebook 09)
- `zinc/` — small ZINC SMILES subset, alternate pre-training corpus
- `moleculenet/` — ESOL, BACE, BBBP, FreeSolv splits for fine-tuning
- `qm9/` — QM9 SMILES + targets, also used by the cross-repo GNN comparison (notebook 09.1)
