"""Training-loop helpers shared across notebooks.

Notebooks build their first training loops from scratch (notebook 07) and then
re-use the canonical version from here. The MLM masking collator (notebook 08)
and a basic ``train_one_epoch`` / ``evaluate`` pair are the main exports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    import torch
    from torch.nn import functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    F = None  # type: ignore

from .smiles_tokenizers import MASK_ID, PAD_ID, SPECIAL_TOKENS

# Default set of special-token ids that must never be masked or used as a
# random replacement. Matches the pinned ids in ``smiles_tokenizers``.
_DEFAULT_SPECIAL_IDS = tuple(range(len(SPECIAL_TOKENS)))


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
    """Batch collator that applies the BERT-style 80/10/10 masking scheme.

    Turn a list of token-ID sequences (variable length) into a padded,
    *corrupted* batch ready for masked-language-modelling. For every batch it:

    1. pads each sequence to the batch maximum with ``pad_token_id``;
    2. picks ~``mask_probability`` of the **non-special** positions to predict;
    3. splits those selected positions 80/10/10 into ``[MASK]`` / a random
       (non-special) token / unchanged — the trick from Devlin et al. (2019)
       that stops the model from only ever reacting to the literal ``[MASK]``
       symbol (which never appears at fine-tune time);
    4. builds a ``labels`` tensor that is ``-100`` everywhere except the
       selected positions, where it holds the *original* token id.

    ``-100`` is PyTorch's default ``ignore_index`` for ``cross_entropy``, so the
    loss is computed only where a token was hidden — including the 10%-random and
    10%-kept positions, per the BERT convention.

    Used in notebook 08 (MLM) and notebook 09 (Tiny MolFormer pre-training).

    Parameters
    ----------
    vocab_size
        Tokenizer vocabulary size; random replacements are drawn from
        ``[len(special_ids), vocab_size)`` so they are never special tokens.
    mask_token_id
        Id of the ``[MASK]`` token.
    pad_token_id
        Id used to pad short sequences in the batch.
    config
        An :class:`MLMMaskingConfig`; defaults to the standard 15% / 80-10-10.
    special_ids
        Token ids that are never selected for masking and never used as a random
        replacement. Defaults to all special tokens.
    seed
        Optional integer seed for a private :class:`torch.Generator`, so the same
        batch produces the same masking — handy for the equivalence check in
        notebook 08.

    Examples
    --------
    >>> collate = MLMCollator(vocab_size=40, mask_token_id=4, pad_token_id=0, seed=0)
    >>> batch = collate([[2, 7, 8, 9, 3], [2, 7, 3]])
    >>> batch["input_ids"].shape, batch["labels"].shape
    (torch.Size([2, 5]), torch.Size([2, 5]))
    """

    def __init__(
        self,
        vocab_size: int,
        mask_token_id: int = MASK_ID,
        pad_token_id: int = PAD_ID,
        config: MLMMaskingConfig | None = None,
        special_ids: Sequence[int] = _DEFAULT_SPECIAL_IDS,
        seed: int | None = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("MLMCollator requires PyTorch.")
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.config = config or MLMMaskingConfig()
        self.special_ids = tuple(special_ids)
        # Random replacements are drawn from the non-special part of the vocab.
        self.random_low = len(self.special_ids)
        self.generator = (
            torch.Generator().manual_seed(seed) if seed is not None else None
        )

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.generator)

    def __call__(self, batch: Sequence[Sequence[int]]):
        """Collate and mask a batch.

        ``batch``: a list of token-id sequences (each a ``list[int]``), or a list
        of ``(input_ids, attention_mask)`` pairs as returned per-example by a
        ``Dataset``. Returns a dict with ``input_ids`` (corrupted), the padded
        ``attention_mask`` (1 real / 0 pad), and ``labels`` (``-100`` except at
        masked positions) — all ``(batch, seq_len)`` long tensors.
        """
        # Accept either bare id-lists or (ids, mask) pairs; we recompute the
        # mask from padding either way, so the second element is optional.
        sequences = [b[0] if isinstance(b, tuple) else b for b in batch]
        target_len = max(len(seq) for seq in sequences)

        input_ids = torch.full((len(sequences), target_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), target_len), dtype=torch.long)
        for i, seq in enumerate(sequences):
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, : len(seq)] = 1

        original = input_ids.clone()

        # 1. Eligible positions: real, non-special tokens.
        eligible = attention_mask.bool()
        for sid in self.special_ids:
            eligible &= input_ids != sid

        # 2. Select ~mask_probability of the eligible positions.
        selected = (self._rand(input_ids.shape) < self.config.mask_probability) & eligible

        # 3. Split selected into 80% [MASK] / 10% random / 10% keep.
        c = self.config
        roll = self._rand(input_ids.shape)
        mask_to_mask = selected & (roll < c.mask_token_fraction)
        mask_to_random = (
            selected
            & (roll >= c.mask_token_fraction)
            & (roll < c.mask_token_fraction + c.random_token_fraction)
        )
        # the remaining selected positions are kept unchanged.

        input_ids[mask_to_mask] = self.mask_token_id
        n_random = int(mask_to_random.sum().item())
        if n_random:
            input_ids[mask_to_random] = torch.randint(
                self.random_low, self.vocab_size, (n_random,), generator=self.generator
            )

        # 4. labels = -100 except at every selected position (original id).
        labels = torch.full_like(input_ids, -100)
        labels[selected] = original[selected]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _move(batch, device):
    """Move a dict / tuple of tensors to ``device``."""
    if isinstance(batch, dict):
        return {k: v.to(device) for k, v in batch.items()}
    return tuple(t.to(device) for t in batch)


def _classification_loss_and_logits(model, batch, device):
    """Forward a classification batch ``(input_ids, attention_mask, labels)``.

    Expects ``model`` to return a ``(batch,)`` logit tensor. Returns
    ``(loss, logits, labels)``.
    """
    input_ids, attention_mask, labels = _move(batch, device)
    logits = model(input_ids, attention_mask)
    loss = F.binary_cross_entropy_with_logits(logits, labels.float())
    return loss, logits, labels


def _mlm_loss_and_logits(model, batch, device):
    """Forward an MLM batch dict ``{input_ids, attention_mask, labels}``.

    Expects ``model`` to return ``(batch, seq_len, vocab_size)`` logits. Returns
    ``(loss, logits, labels)``.
    """
    batch = _move(batch, device)
    logits = model(batch["input_ids"], batch["attention_mask"])
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), batch["labels"].view(-1), ignore_index=-100
    )
    return loss, logits, batch["labels"]


def _is_mlm_batch(batch) -> bool:
    return isinstance(batch, dict) and "labels" in batch


def train_one_epoch(model, loader, optimizer, device: str = "cuda", grad_clip=None) -> float:
    """Run one training epoch over ``loader``; return the mean batch loss.

    The loss is chosen from the batch shape, so the same loop drives both
    notebooks:

    * a dict ``{input_ids, attention_mask, labels}`` → masked cross-entropy
      (``ignore_index=-100``), the MLM objective of notebook 08;
    * a tuple ``(input_ids, attention_mask, labels)`` → binary cross-entropy on
      a single logit, the classification objective of notebook 07.

    ``grad_clip``, if given, is the max-norm passed to
    ``torch.nn.utils.clip_grad_norm_``.
    """
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        optimizer.zero_grad()
        if _is_mlm_batch(batch):
            loss, _, _ = _mlm_loss_and_logits(model, batch, device)
        else:
            loss, _, _ = _classification_loss_and_logits(model, batch, device)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def evaluate(model, loader, device: str = "cuda", task: str = "classification") -> dict:
    """Evaluate ``model`` on ``loader``; return a dict of metrics.

    ``task="classification"`` → ``{"loss", "accuracy", "roc_auc"}`` (ROC-AUC via
    scikit-learn, ``nan`` if a split has a single class). ``task="mlm"`` →
    ``{"loss", "masked_accuracy"}``, the fraction of masked positions predicted
    correctly.
    """
    model.eval()
    total_loss, n_batches = 0.0, 0

    if task == "mlm":
        n_correct, n_masked = 0, 0
        with torch.no_grad():
            for batch in loader:
                loss, logits, labels = _mlm_loss_and_logits(model, batch, device)
                total_loss += loss.item()
                n_batches += 1
                masked = labels != -100
                preds = logits.argmax(-1)
                n_correct += int((preds[masked] == labels[masked]).sum().item())
                n_masked += int(masked.sum().item())
        return {
            "loss": total_loss / max(n_batches, 1),
            "masked_accuracy": n_correct / max(n_masked, 1),
        }

    # classification
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            loss, logits, labels = _classification_loss_and_logits(model, batch, device)
            total_loss += loss.item()
            n_batches += 1
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    accuracy = float((preds == labels.long()).float().mean().item())

    roc_auc = float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        if len(set(labels.tolist())) > 1:
            roc_auc = float(roc_auc_score(labels.numpy(), probs.numpy()))
    except ImportError:
        pass

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": accuracy,
        "roc_auc": roc_auc,
    }


def count_parameters(model) -> int:
    """Total number of trainable parameters in ``model`` — used in notebook 09
    to check the tiny-MolFormer fits the size budget."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
