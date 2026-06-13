"""Dataset loaders for the course.

The notebooks operate on small subsets of well-known chemistry datasets:

* **ChEMBL**: ~100k drug-like SMILES, used for MLM pre-training (notebook 09)
* **ZINC**: ~100k commercially-available SMILES, alternative pre-training set
* **MoleculeNet** (ESOL, BACE, BBBP, FreeSolv): standard small benchmark tasks
  for fine-tuning (notebooks 07, 09)
* **QM9**: small organic molecules with quantum-mechanical properties — the
  shared dataset between this repo and `GNNs-For-Chemists`

Each loader returns a list of SMILES (and a list of targets, for supervised
sets). Loaders are deliberately tiny — production users should use the HF
``datasets`` library directly (covered in notebook 10).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import List, Tuple


DATA_DIR = Path(__file__).parent.parent / "data"

_S3_BASE = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"

# MoleculeNet task → (cached filename, remote filename, smiles column,
# target column, task type). Column names are used (not positions) because the
# raw CSVs disagree on layout.
_MOLECULENET = {
    "bbbp": ("BBBP.csv", "BBBP.csv", "smiles", "p_np", "classification"),
    "bace": ("bace.csv", "bace.csv", "mol", "Class", "classification"),
    "esol": ("delaney-processed.csv", "delaney-processed.csv", "smiles",
             "measured log solubility in mols per litre", "regression"),
    "freesolv": ("SAMPL.csv", "SAMPL.csv", "smiles", "expt", "regression"),
}


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised only without pandas
        raise ImportError(
            "load_moleculenet / load_chembl_subset need pandas — "
            "`pip install pandas`."
        ) from exc
    return pd


def load_moleculenet(name: str) -> Tuple[List[str], List[float]]:
    """Load a MoleculeNet task by name.

    Downloads the CSV on first call and caches it under ``notebooks/data/``.
    SMILES are returned raw (not canonicalized) — apply
    :func:`utils.preprocessing.clean_smiles` in the notebook if you want the
    largest-fragment / canonical form.

    Parameters
    ----------
    name
        One of ``"esol"``, ``"bace"``, ``"bbbp"``, ``"freesolv"``.

    Returns
    -------
    smiles, targets
        Parallel lists. ``targets`` are ``0/1`` ints for classification tasks
        (``bbbp``, ``bace``) and floats for regression tasks (``esol``,
        ``freesolv``).
    """
    pd = _require_pandas()
    key = name.lower()
    if key not in _MOLECULENET:
        raise ValueError(
            f"unknown MoleculeNet task {name!r}; choose from {sorted(_MOLECULENET)}"
        )
    cached, remote, smiles_col, target_col, _ = _MOLECULENET[key]

    path = DATA_DIR / cached
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{_S3_BASE}/{remote}", path)

    df = pd.read_csv(path)
    df = df[[smiles_col, target_col]].dropna()
    df = df[df[smiles_col].str.len() > 0]
    smiles = df[smiles_col].astype(str).tolist()
    targets = df[target_col].tolist()
    return smiles, targets


def load_chembl_subset(n: int = 100_000) -> List[str]:
    """Return up to ``n`` SMILES from a small curated ChEMBL subset.

    Reads ``notebooks/data/chembl/chembl_subset.csv`` (a small file committed
    with the repo so the MLM notebook runs offline and reproducibly). If that
    file is missing it is downloaded and cached on first call.

    Parameters
    ----------
    n
        Maximum number of SMILES to return (the committed file holds a few
        thousand).
    """
    pd = _require_pandas()
    path = DATA_DIR / "chembl" / "chembl_subset.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/HFooladi/"
            "Transformers-For-Chemists/main/notebooks/data/chembl/chembl_subset.csv",
            path,
        )
    df = pd.read_csv(path)
    smiles = df["smiles"].dropna().astype(str).tolist()
    return smiles[:n]


def load_zinc_subset(n: int = 100_000) -> List[str]:
    """Return ``n`` SMILES from a curated ZINC subset."""
    raise NotImplementedError("Phase 3: implement in notebook 08 or 09.")


def load_qm9(target: str = "homo") -> Tuple[List[str], List[float]]:
    """Load QM9 SMILES and a single quantum-mechanical target."""
    raise NotImplementedError("Phase 3: implement in notebook 07 or 09.1.")
