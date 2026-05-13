# Data

Datasets used in the notebook series. Small files (subsets, splits, indices) are committed; large raw files are gitignored and re-downloaded on demand by helpers in `notebooks/utils/data_loading.py`.

## Layout (created on demand)

- `chembl/` — small ChEMBL SMILES subset for MLM pre-training (notebook 09)
- `zinc/` — small ZINC SMILES subset, alternate pre-training corpus
- `moleculenet/` — raw CSVs (ESOL, BACE, BBBP, FreeSolv) used for the empirical token-length demo in notebook 01 and for fine-tuning in notebook 07. Downloaded on demand from the DeepChem S3 mirror.
- `qm9/` — QM9 SMILES + targets, also used by the cross-repo GNN comparison (notebook 09.1)
