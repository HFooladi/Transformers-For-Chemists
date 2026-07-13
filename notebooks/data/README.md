# Data

Datasets used in the notebook series. Small files (subsets, splits, indices) are committed; large raw files are gitignored and re-downloaded on demand by helpers in `notebooks/utils/data_loading.py`.

## Committed files

- `chembl/chembl_subset.csv` — small ChEMBL SMILES subset for MLM pre-training (notebook 09)
- `BBBP.csv` — MoleculeNet BBBP task, cached directly in `data/`. Used for fine-tuning (notebook 07) and the cross-repo GNN-vs-transformer comparison (notebook 09.1)

## Fetched on demand

The helpers in `notebooks/utils/data_loading.py` download and cache these on first use; they are **not** committed to the repo:

- Other MoleculeNet tasks — ESOL, BACE, FreeSolv (via `load_moleculenet`, cached alongside `BBBP.csv` in `data/`)
- A ZINC SMILES subset — alternate pre-training corpus (`load_zinc_subset`)
- QM9 SMILES + quantum-mechanical targets (`load_qm9`)
