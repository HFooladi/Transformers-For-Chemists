"""Minimal SMILES standardization used by the empirical demos.

When the notebooks tokenize *real* chemistry datasets (MoleculeNet, ChEMBL,
ZINC, ...) the rows are not always clean: some entries are malformed and
fail to parse, others are co-crystals or salts written as multi-fragment
SMILES joined by ``.`` (e.g. ``[drug].[Na+].[Cl-]``). Reporting token-count
statistics on those raw strings is misleading — the long tail is counting
counterions rather than chemistry.

This module provides the smallest preprocessing recipe that makes such
statistics defensible:

1. Parse the SMILES with RDKit; drop rows that fail.
2. Keep only the largest connected fragment (by heavy-atom count).
3. Re-emit a canonical SMILES so equivalent inputs collapse to the same
   string.

That is **not** a substitute for a real preprocessing pipeline (tautomer
canonicalization, charge normalization, stereo cleanup, de-duplication) —
see RDKit's ``rdMolStandardize`` module for that. The recipe here is
deliberately tiny so the educational notebooks stay readable.
"""

from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING

from rdkit import Chem

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    import pandas as pd


def clean_smiles(smiles: str) -> Optional[str]:
    """Parse, take the largest fragment, return canonical SMILES.

    Parameters
    ----------
    smiles
        Raw SMILES string from a downloaded CSV. May be malformed or a
        multi-fragment co-crystal/salt joined by ``.``.

    Returns
    -------
    str or None
        The canonical SMILES of the largest organic fragment, or ``None``
        if RDKit cannot parse the input.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fragments = Chem.GetMolFrags(mol, asMols=True)
    if len(fragments) > 1:
        mol = max(fragments, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol)


def preprocess_smiles_series(
    smiles: "pd.Series",
) -> Tuple["pd.Series", int, int]:
    """Apply :func:`clean_smiles` to a pandas Series of raw SMILES.

    Parameters
    ----------
    smiles
        A pandas Series of raw SMILES strings (e.g. ``df["smiles"]``).

    Returns
    -------
    cleaned : pd.Series
        Canonical SMILES of the largest fragment for each row that parsed
        successfully. Unparseable rows are dropped and the index is reset.
    n_unparseable : int
        Number of input rows that RDKit could not parse.
    n_multifragment : int
        Number of input rows containing a ``.`` (i.e. salts, co-crystals,
        multi-component SMILES). For those rows the *largest* fragment is
        kept; this count is reported so the caller can report it.
    """
    cleaned = smiles.apply(clean_smiles)
    n_unparseable = int(cleaned.isna().sum())
    n_multifragment = int(smiles.str.contains(".", regex=False).sum())
    cleaned = cleaned.dropna().reset_index(drop=True)
    return cleaned, n_unparseable, n_multifragment
