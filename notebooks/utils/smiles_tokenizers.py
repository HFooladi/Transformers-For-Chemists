"""Educational SMILES tokenizers.

This module collects the four tokenization schemes covered in the course:

* **Character-level** (notebook 01): each SMILES character is its own token.
  Trivial to implement, large vocabulary on long molecules.
* **Atom-level** (notebook 02): multi-character atoms (``Br``, ``Cl``,
  ``[nH]``) are kept as single tokens via a regex match. Common baseline in
  cheminformatics.
* **Byte-Pair Encoding (BPE)** (notebook 02): standard learned subword
  tokenization, trained on a SMILES corpus.
* **SMILES-pair encoding** (notebook 02): Schwaller-style chemistry-aware
  variant that biases merges toward chemically meaningful substructures.

Each tokenizer exposes the same minimal API (``encode``, ``decode``, ``vocab``)
so notebooks downstream can swap them with a one-line change. Implementations
prioritize readability — production users should reach for the HuggingFace
``tokenizers`` library (covered in notebook 10).
"""

from __future__ import annotations

from typing import List, Sequence


# Standard atom-regex used in MolFormer and friends. Matches multi-character
# atoms (``Br``, ``Cl``), bracketed atoms (``[nH]``, ``[O-]``), bonds, and ring
# closures.
SMILES_ATOM_REGEX = (
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\\\|/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)


class CharTokenizer:
    """Character-level SMILES tokenizer (notebook 01)."""

    def __init__(self, vocab: Sequence[str] | None = None) -> None:
        raise NotImplementedError("Phase 2: implement in notebook 01.")

    def encode(self, smiles: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError


class AtomTokenizer:
    """Atom/regex-level SMILES tokenizer (notebook 02)."""

    def __init__(self, vocab: Sequence[str] | None = None) -> None:
        raise NotImplementedError("Phase 3: implement in notebook 02.")

    def encode(self, smiles: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError


class BPETokenizer:
    """Byte-Pair Encoding tokenizer trained on a SMILES corpus (notebook 02).

    Wraps the HuggingFace ``tokenizers`` library; the from-scratch BPE merge
    loop is illustrated separately in notebook 02 for pedagogical purposes.
    """

    def __init__(self) -> None:
        raise NotImplementedError("Phase 3: implement in notebook 02.")

    def train(self, smiles_corpus: Sequence[str], vocab_size: int = 1000) -> None:
        raise NotImplementedError

    def encode(self, smiles: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError


class SmilesPairTokenizer:
    """Schwaller-style SMILES-pair tokenizer (notebook 02).

    Same merge algorithm as BPE but operates on the atom-regex pre-tokens
    instead of raw characters, biasing merges toward chemically meaningful
    fragments.
    """

    def __init__(self) -> None:
        raise NotImplementedError("Phase 3: implement in notebook 02.")

    def train(self, smiles_corpus: Sequence[str], vocab_size: int = 1000) -> None:
        raise NotImplementedError

    def encode(self, smiles: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError
