"""Training-loop helpers shared across notebooks.

Notebooks build their first training loops from scratch (notebook 07) and then
re-use the canonical version from here. The MLM masking collator (notebook 08)
and a basic ``train_one_epoch`` / ``evaluate`` pair are the main exports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class MLMMaskingConfig:
    """Configuration for the BERT-style 80/10/10 masking scheme.

    Attributes
    ----------
    mask_probability
        Fraction of tokens selected for masking (default 0.15).
    mask_token_fraction
        Of the selected tokens, the fraction replaced by ``[MASK]`` (default 0.8).
    random_token_fraction
        Of the selected tokens, the fraction replaced by a random vocab token
        (default 0.1).
    keep_token_fraction
        Of the selected tokens, the fraction left unchanged (default 0.1).
    """

    mask_probability: float = 0.15
    mask_token_fraction: float = 0.8
    random_token_fraction: float = 0.1
    keep_token_fraction: float = 0.1


class MLMCollator:
    """Batch collator that applies the BERT-style masking scheme.

    Used in notebook 08 (MLM) and notebook 09 (Tiny MolFormer pre-training).
    """

    def __init__(
        self,
        vocab_size: int,
        mask_token_id: int,
        pad_token_id: int,
        config: MLMMaskingConfig | None = None,
    ) -> None:
        raise NotImplementedError("Phase 3: implement in notebook 08.")

    def __call__(self, batch: Sequence[Sequence[int]]):
        raise NotImplementedError


def train_one_epoch(model, loader, optimizer, device: str = "cuda") -> float:
    """Standard one-epoch training loop. Returns average loss."""
    raise NotImplementedError("Phase 3: implement in notebook 07.")


def evaluate(model, loader, device: str = "cuda") -> dict:
    """Evaluate ``model`` on ``loader``; returns a dict of metric values."""
    raise NotImplementedError("Phase 3: implement in notebook 07.")


def count_parameters(model) -> int:
    """Total number of trainable parameters in ``model`` — used in notebook 09
    to check the tiny-MolFormer fits the size budget."""
    raise NotImplementedError("Phase 3: implement in notebook 09.")
