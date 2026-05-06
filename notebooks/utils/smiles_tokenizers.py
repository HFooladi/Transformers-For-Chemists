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


# Special tokens used by every tokenizer in the course. The IDs are fixed so
# that a model trained with one tokenizer can be loaded with another (as long
# as the vocabularies otherwise agree). PAD must be 0 — many losses and
# attention-mask routines assume this implicitly.
SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]")
PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID, BOS_ID, EOS_ID = range(len(SPECIAL_TOKENS))


class CharTokenizer:
    """Character-level SMILES tokenizer (notebook 01).

    Treats every character of a SMILES string as a separate token. Stupidly
    simple, surprisingly effective on small datasets, and the right starting
    point for understanding what tokenization actually does.

    Parameters
    ----------
    vocab
        Optional pre-built character vocabulary (without special tokens — those
        are prepended automatically). If ``None``, the vocabulary must be
        built later via :meth:`build_vocab`.

    Examples
    --------
    >>> tk = CharTokenizer.from_smiles(["CCO", "c1ccccc1"])
    >>> ids = tk.encode("CCO", add_special_tokens=True)
    >>> tk.decode(ids, skip_special_tokens=True)
    'CCO'
    """

    def __init__(self, vocab: Sequence[str] | None = None) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        # Special tokens always come first so PAD == 0, etc.
        for tok in SPECIAL_TOKENS:
            self._add_token(tok)
        if vocab is not None:
            for tok in vocab:
                self._add_token(tok)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _add_token(self, token: str) -> None:
        if token in self.token_to_id:
            return
        idx = len(self.token_to_id)
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token

    def build_vocab(self, smiles_corpus: Sequence[str]) -> None:
        """Add every distinct character in ``smiles_corpus`` to the vocabulary."""
        seen: set[str] = set()
        for smi in smiles_corpus:
            for ch in smi:
                if ch not in seen:
                    seen.add(ch)
                    self._add_token(ch)

    @classmethod
    def from_smiles(cls, smiles_corpus: Sequence[str]) -> "CharTokenizer":
        """Convenience constructor: build vocab from a SMILES corpus."""
        tk = cls()
        tk.build_vocab(smiles_corpus)
        return tk

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def tokenize(self, smiles: str) -> List[str]:
        """Return the list of tokens (characters) for ``smiles``."""
        return list(smiles)

    def encode(
        self,
        smiles: str,
        add_special_tokens: bool = False,
    ) -> List[int]:
        """Convert a SMILES string to a list of token IDs.

        Unknown characters map to ``[UNK]``. With ``add_special_tokens=True``
        the result is wrapped in ``[CLS] ... [SEP]`` (BERT-style), which is
        what the encoder model expects in notebooks 07+.
        """
        ids = [self.token_to_id.get(ch, UNK_ID) for ch in smiles]
        if add_special_tokens:
            ids = [CLS_ID, *ids, SEP_ID]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        """Inverse of :meth:`encode`."""
        tokens = []
        for idx in ids:
            tok = self.id_to_token.get(int(idx), "[UNK]")
            if skip_special_tokens and tok in SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return "".join(tokens)

    # ------------------------------------------------------------------
    # Batching helper
    # ------------------------------------------------------------------

    def encode_batch(
        self,
        smiles_list: Sequence[str],
        add_special_tokens: bool = True,
        max_length: int | None = None,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Encode a batch of SMILES, padding to a common length.

        Returns
        -------
        input_ids, attention_mask
            Parallel lists of equal-length lists. ``attention_mask`` is 1 for
            real tokens and 0 for ``[PAD]`` positions.
        """
        encoded = [self.encode(s, add_special_tokens=add_special_tokens) for s in smiles_list]
        if max_length is not None:
            encoded = [seq[:max_length] for seq in encoded]
        target_len = max(len(seq) for seq in encoded)
        input_ids = [seq + [PAD_ID] * (target_len - len(seq)) for seq in encoded]
        attention_mask = [[1] * len(seq) + [0] * (target_len - len(seq)) for seq in encoded]
        return input_ids, attention_mask


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
