"""Educational SMILES tokenizers.

This module collects the four tokenization schemes covered in the course:

* **Character-level** (notebook 01): each SMILES character is its own token.
  Trivial to implement, large vocabulary on long molecules.
* **Atom-level** (notebook 02): multi-character atoms (``Br``, ``Cl``,
  ``[nH]``) are kept as single tokens via a regex match. Common baseline in
  cheminformatics.
* **Byte-Pair Encoding (BPE)** (notebook 02): standard learned subword
  tokenization, trained on a SMILES corpus.
* **SMILES-pair encoding** (notebook 02): chemistry-aware BPE variant
  (Li & Fourches 2021) that runs BPE merges over atom sequences, so every
  learned token is a sequence of *whole atoms*.

Each tokenizer exposes the same minimal API (``encode``, ``decode``, ``vocab``)
so notebooks downstream can swap them with a one-line change. Implementations
prioritize readability — production users should reach for the HuggingFace
``tokenizers`` library (covered in notebook 10).
"""

from __future__ import annotations

import re
from typing import List, Sequence


# Standard atom-regex used in MolFormer and friends. Matches multi-character
# atoms (``Br``, ``Cl``), bracketed atoms (``[nH]``, ``[O-]``), bonds, and ring
# closures.
SMILES_ATOM_REGEX = (
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)
_ATOM_PATTERN = re.compile(SMILES_ATOM_REGEX)


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


def atom_tokenize(smiles: str) -> List[str]:
    """Split a SMILES string into atom-level tokens via :data:`SMILES_ATOM_REGEX`.

    Multi-character atoms (``Br``, ``Cl``), bracketed atoms (``[nH]``,
    ``[O-]``), bonds, and ring closures each become single tokens. Anything
    the regex doesn't match (e.g. whitespace) is dropped silently — pass
    canonical SMILES.
    """
    return _ATOM_PATTERN.findall(smiles)


class AtomTokenizer:
    """Atom-level SMILES tokenizer (notebook 02).

    Uses :func:`atom_tokenize` as the pre-tokenizer so multi-character atoms
    stay together. Same minimal API as :class:`CharTokenizer`.
    """

    def __init__(self, vocab: Sequence[str] | None = None) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        for tok in SPECIAL_TOKENS:
            self._add_token(tok)
        if vocab is not None:
            for tok in vocab:
                self._add_token(tok)

    def _add_token(self, token: str) -> None:
        if token in self.token_to_id:
            return
        idx = len(self.token_to_id)
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token

    def build_vocab(self, smiles_corpus: Sequence[str]) -> None:
        for smi in smiles_corpus:
            for tok in atom_tokenize(smi):
                self._add_token(tok)

    @classmethod
    def from_smiles(cls, smiles_corpus: Sequence[str]) -> "AtomTokenizer":
        tk = cls()
        tk.build_vocab(smiles_corpus)
        return tk

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def tokenize(self, smiles: str) -> List[str]:
        return atom_tokenize(smiles)

    def encode(self, smiles: str, add_special_tokens: bool = False) -> List[int]:
        ids = [self.token_to_id.get(tok, UNK_ID) for tok in atom_tokenize(smiles)]
        if add_special_tokens:
            ids = [CLS_ID, *ids, SEP_ID]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        out: list[str] = []
        for idx in ids:
            tok = self.id_to_token.get(int(idx), "[UNK]")
            if skip_special_tokens and tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        return "".join(out)

    def encode_batch(
        self,
        smiles_list: Sequence[str],
        add_special_tokens: bool = True,
        max_length: int | None = None,
    ) -> tuple[list[list[int]], list[list[int]]]:
        encoded = [self.encode(s, add_special_tokens=add_special_tokens) for s in smiles_list]
        if max_length is not None:
            encoded = [seq[:max_length] for seq in encoded]
        target_len = max(len(seq) for seq in encoded)
        input_ids = [seq + [PAD_ID] * (target_len - len(seq)) for seq in encoded]
        attention_mask = [[1] * len(seq) + [0] * (target_len - len(seq)) for seq in encoded]
        return input_ids, attention_mask


class _HFTokenizerBackedTokenizer:
    """Shared base for the two BPE-flavoured tokenizers.

    Both :class:`BPETokenizer` and :class:`SmilesPairTokenizer` train a
    HuggingFace ``tokenizers.Tokenizer`` under the hood — they only differ in
    which pre-tokenizer they use. The encode/decode/batch methods are the same
    for both, so they live here.

    Calling :meth:`train` is required before :meth:`encode`.
    """

    def __init__(self) -> None:
        try:
            from tokenizers import Tokenizer  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The `tokenizers` package is required for BPE-style tokenizers. "
                "Install with: pip install tokenizers"
            ) from e
        self._hf = None  # populated by .train()

    # Subclasses override _build_tokenizer() to choose the pre-tokenizer.
    def _build_tokenizer(self):  # pragma: no cover - subclass responsibility
        raise NotImplementedError

    def train(self, smiles_corpus: Sequence[str], vocab_size: int = 1000) -> None:
        from tokenizers.trainers import BpeTrainer

        tk = self._build_tokenizer()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=[],
            show_progress=False,
        )
        tk.train_from_iterator(list(smiles_corpus), trainer=trainer)
        self._hf = tk

    @property
    def vocab_size(self) -> int:
        if self._hf is None:
            return 0
        return self._hf.get_vocab_size()

    def tokenize(self, smiles: str) -> List[str]:
        if self._hf is None:
            raise RuntimeError("Tokenizer not trained yet — call .train(corpus) first.")
        return self._hf.encode(smiles, add_special_tokens=False).tokens

    def encode(self, smiles: str, add_special_tokens: bool = False) -> List[int]:
        if self._hf is None:
            raise RuntimeError("Tokenizer not trained yet — call .train(corpus) first.")
        ids = self._hf.encode(smiles, add_special_tokens=False).ids
        if add_special_tokens:
            ids = [CLS_ID, *ids, SEP_ID]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        if self._hf is None:
            raise RuntimeError("Tokenizer not trained yet — call .train(corpus) first.")
        # The HF decoder joins tokens with spaces and replaces continuation
        # markers; for our SMILES use we want a faithful concatenation.
        out: list[str] = []
        for idx in ids:
            tok = self._hf.id_to_token(int(idx))
            if tok is None:
                tok = "[UNK]"
            if skip_special_tokens and tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        return "".join(out)

    def encode_batch(
        self,
        smiles_list: Sequence[str],
        add_special_tokens: bool = True,
        max_length: int | None = None,
    ) -> tuple[list[list[int]], list[list[int]]]:
        encoded = [self.encode(s, add_special_tokens=add_special_tokens) for s in smiles_list]
        if max_length is not None:
            encoded = [seq[:max_length] for seq in encoded]
        target_len = max(len(seq) for seq in encoded)
        input_ids = [seq + [PAD_ID] * (target_len - len(seq)) for seq in encoded]
        attention_mask = [[1] * len(seq) + [0] * (target_len - len(seq)) for seq in encoded]
        return input_ids, attention_mask


class BPETokenizer(_HFTokenizerBackedTokenizer):
    """Byte-Pair Encoding tokenizer trained on a SMILES corpus (notebook 02).

    Wraps HuggingFace ``tokenizers`` with the simplest possible setup:
    BPE model, no normalizer, no pre-tokenizer (BPE operates on the raw
    character stream). The from-scratch BPE merge loop is illustrated
    separately in notebook 02 for pedagogical purposes.
    """

    def _build_tokenizer(self):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE

        return Tokenizer(BPE(unk_token="[UNK]"))


class SmilesPairTokenizer(_HFTokenizerBackedTokenizer):
    """SMILES-pair encoding tokenizer (Li & Fourches 2021, notebook 02).

    Reference: Li, X. & Fourches, D. (2021). *SMILES Pair Encoding: A
    Data-Driven Substructure Tokenization Algorithm for Deep Learning.*
    J. Chem. Inf. Model. https://doi.org/10.1021/acs.jcim.0c01127

    Algorithm:

    1. **Atom-tokenize** each SMILES into a sequence of atom tokens
       (``Br``, ``[nH]``, ``c``, ``1``, …) using :data:`SMILES_ATOM_REGEX`.
    2. **Train BPE over those atom sequences**, treating each atom as one
       indivisible unit. Merge candidates are pairs of *atoms*, so any
       learned token is automatically a sequence of whole atoms — a
       bracketed atom like ``[nH]`` can never be split across a token
       boundary.

    **Implementation note.** HuggingFace ``tokenizers`` operates on Unicode
    character streams, not on lists of arbitrary tokens. To make BPE see
    each atom as one indivisible "character", this class maps every unique
    atom token in the training corpus to a single codepoint in the Unicode
    Private Use Area (U+E000 onwards), trains BPE on the encoded strings
    *without* a pre-tokenizer, and decodes each output token's placeholder
    characters back to atom strings at tokenize/decode time. This is
    equivalent to the original sentencepiece-based SPE training.

    A naive HF implementation that just sets a ``Split`` pre-tokenizer to
    the atom regex does **not** work: HF's BPE only merges within
    pre-tokens, so atom-pre-tokenized BPE never finds cross-atom merges
    and degenerates into atom-level tokenization.
    """

    # Unicode Basic Multilingual Plane Private Use Area: U+E000–U+F8FF,
    # 6 400 codepoints — plenty for any chemistry vocabulary.
    _PUA_BASE = 0xE000
    _PUA_CAPACITY = 0xF8FF - 0xE000 + 1

    def __init__(self) -> None:
        super().__init__()
        self._atom_to_char: dict[str, str] = {}
        self._char_to_atom: dict[str, str] = {}

    def _build_tokenizer(self):
        # No pre-tokenizer: BPE consumes the placeholder string directly,
        # which is what lets it learn merges across atom boundaries.
        from tokenizers import Tokenizer
        from tokenizers.models import BPE

        return Tokenizer(BPE(unk_token="[UNK]"))

    def _encode_to_placeholders(self, smiles: str) -> str:
        """Map atom tokens to private-use codepoints. Unknown atoms → ``\\x00`` → ``[UNK]``."""
        return "".join(self._atom_to_char.get(a, "\x00") for a in atom_tokenize(smiles))

    def _decode_from_placeholders(self, token_string: str) -> str:
        """Map a BPE token's placeholder characters back to the atom string they encode."""
        return "".join(self._char_to_atom.get(c, c) for c in token_string)

    def train(self, smiles_corpus: Sequence[str], vocab_size: int = 1000) -> None:
        from tokenizers.trainers import BpeTrainer

        # Pass 1: discover the atom-token alphabet across the corpus, then
        # assign each atom a unique placeholder codepoint. Sorted so the
        # mapping is deterministic across runs.
        all_atoms: set[str] = set()
        for smi in smiles_corpus:
            all_atoms.update(atom_tokenize(smi))
        if len(all_atoms) > self._PUA_CAPACITY:
            raise ValueError(
                f"Corpus has {len(all_atoms)} unique atom tokens but only "
                f"{self._PUA_CAPACITY} placeholder codepoints are available."
            )
        self._atom_to_char = {
            atom: chr(self._PUA_BASE + i) for i, atom in enumerate(sorted(all_atoms))
        }
        self._char_to_atom = {c: a for a, c in self._atom_to_char.items()}

        # Pass 2: encode the corpus to placeholder strings and train BPE.
        encoded_corpus = [self._encode_to_placeholders(s) for s in smiles_corpus]
        tk = self._build_tokenizer()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=[],
            show_progress=False,
        )
        tk.train_from_iterator(encoded_corpus, trainer=trainer)
        self._hf = tk

    def tokenize(self, smiles: str) -> List[str]:
        if self._hf is None:
            raise RuntimeError("Tokenizer not trained yet — call .train(corpus) first.")
        raw = self._hf.encode(self._encode_to_placeholders(smiles), add_special_tokens=False).tokens
        return [self._decode_from_placeholders(t) for t in raw]

    def encode(self, smiles: str, add_special_tokens: bool = False) -> List[int]:
        if self._hf is None:
            raise RuntimeError("Tokenizer not trained yet — call .train(corpus) first.")
        ids = self._hf.encode(self._encode_to_placeholders(smiles), add_special_tokens=False).ids
        if add_special_tokens:
            ids = [CLS_ID, *ids, SEP_ID]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        if self._hf is None:
            raise RuntimeError("Tokenizer not trained yet — call .train(corpus) first.")
        out: list[str] = []
        for idx in ids:
            tok = self._hf.id_to_token(int(idx))
            if tok is None:
                tok = "[UNK]"
            if skip_special_tokens and tok in SPECIAL_TOKENS:
                continue
            out.append(self._decode_from_placeholders(tok))
        return "".join(out)
