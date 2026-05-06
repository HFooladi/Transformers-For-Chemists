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

from pathlib import Path
from typing import List, Tuple


DATA_DIR = Path(__file__).parent.parent / "data"


def load_chembl_subset(n: int = 100_000) -> List[str]:
    """Return ``n`` SMILES from a curated ChEMBL subset.

    Downloads the subset on first call and caches it under
    ``notebooks/data/chembl/``.
    """
    raise NotImplementedError("Phase 3: implement in notebook 08 or 09.")


def load_zinc_subset(n: int = 100_000) -> List[str]:
    """Return ``n`` SMILES from a curated ZINC subset."""
    raise NotImplementedError("Phase 3: implement in notebook 08 or 09.")


def load_moleculenet(name: str) -> Tuple[List[str], List[float]]:
    """Load a MoleculeNet task by name.

    Parameters
    ----------
    name
        One of ``"esol"``, ``"bace"``, ``"bbbp"``, ``"freesolv"``.

    Returns
    -------
    smiles, targets
        Parallel lists.
    """
    raise NotImplementedError("Phase 3: implement in notebook 07.")


def load_qm9(target: str = "homo") -> Tuple[List[str], List[float]]:
    """Load QM9 SMILES and a single quantum-mechanical target."""
    raise NotImplementedError("Phase 3: implement in notebook 07 or 09.1.")
