"""Attention-bias position encodings — the deep-dive variants for notebook 04.3.

Notebooks 03 and 04.2 covered two ways to give a transformer a sense of
position: *adding* a sinusoidal vector to the embeddings (absolute), and
*rotating* Q and K (RoPE, relative). This module implements a third family that
injects position as a **bias added directly to the attention scores**, just
before the softmax::

    attention = softmax( Q K^T / sqrt(d)  +  bias[i, j] )

Two members of that family:

* **ALiBi** (Press et al., 2021, *"Attention with Linear Biases"*) — a *fixed*,
  parameter-free bias that penalizes attention linearly with token distance,
  ``bias = -slope * |i - j|``. Each head gets a different slope, so some heads
  stay local while others see far. Because the bias is purely a function of
  distance, ALiBi extrapolates to sequences longer than any seen in training.

* **Relative-position bias** (Shaw et al., 2018; Raffel et al., 2020 / T5) — a
  *learned* bias looked up per relative distance. Distances are grouped into
  buckets (nearby distances get their own bucket; far ones share log-spaced
  buckets), and each bucket/head pair learns a scalar bias.

Both produce a ``(n_heads, seq_len, seq_len)`` tensor ready to add onto the
pre-softmax scores. The bidirectional (encoder) form is used throughout, since
the course targets an encoder-only, MolFormer-style model.
"""

from __future__ import annotations

import math

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # type: ignore


def alibi_slopes(n_heads: int):
    """Return the per-head ALiBi slopes as a 1-D tensor of length ``n_heads``.

    Press et al. choose slopes as a geometric sequence. For a power-of-two head
    count the sequence starts at ``2^(-8/n_heads)`` and multiplies by that same
    ratio each step, giving slopes that span from "very local" to "almost
    global". Non-power-of-two head counts interpolate the next power of two
    (matching the reference implementation).
    """
    def power_of_two_slopes(n: int):
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = power_of_two_slopes(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        slopes = power_of_two_slopes(closest)
        extra = power_of_two_slopes(2 * closest)[0::2][: n_heads - closest]
        slopes = slopes + extra
    return torch.tensor(slopes, dtype=torch.float)


def alibi_bias(seq_len: int, n_heads: int):
    """Fixed ALiBi bias of shape ``(n_heads, seq_len, seq_len)``.

    Entry ``[h, i, j] = -slope_h * |i - j|`` — a symmetric (bidirectional)
    penalty that grows with the distance between query ``i`` and key ``j``.
    Adding this to the attention scores nudges every query toward nearby keys,
    by an amount set by the head's slope.
    """
    positions = torch.arange(seq_len)
    distance = (positions[None, :] - positions[:, None]).abs().float()  # (L, L)
    slopes = alibi_slopes(n_heads)                                      # (H,)
    return -slopes[:, None, None] * distance[None, :, :]                # (H, L, L)


class RelativePositionBias(nn.Module if TORCH_AVAILABLE else object):
    """Learned T5-style relative-position bias (Raffel et al., 2020; notebook 04.3).

    Maps the relative distance ``j - i`` between every query/key pair to a
    bucket, then looks up a learned scalar bias per (bucket, head). Returns a
    ``(n_heads, seq_len, seq_len)`` tensor to add onto the attention scores.

    Parameters
    ----------
    n_heads
        Number of attention heads (one learned bias curve per head).
    num_buckets
        Total number of distance buckets (split half for negative, half for
        positive offsets in the bidirectional case).
    max_distance
        Distances beyond this are all folded into the final bucket.
    """

    def __init__(self, n_heads: int, num_buckets: int = 32, max_distance: int = 128) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bias = nn.Embedding(num_buckets, n_heads)

    @staticmethod
    def relative_bucket(relative_position, num_buckets: int = 32, max_distance: int = 128):
        """Map signed relative positions to bucket indices (bidirectional T5).

        Small distances get one bucket each (exact); larger distances are
        grouped logarithmically so a handful of buckets covers a wide range.
        The first half of the buckets is reserved for one sign, the second half
        for the other.
        """
        ret = torch.zeros_like(relative_position)
        half = num_buckets // 2
        # One sign gets the upper half of the bucket range.
        ret = ret + (relative_position > 0).long() * half
        n = relative_position.abs()

        max_exact = half // 2
        is_small = n < max_exact
        # Logarithmic spacing for the large distances.
        large = max_exact + (
            torch.log(n.float().clamp(min=1) / max_exact)
            / math.log(max_distance / max_exact)
            * (half - max_exact)
        ).long()
        large = torch.minimum(large, torch.full_like(large, half - 1))
        return ret + torch.where(is_small, n, large)

    def forward(self, seq_len: int):
        """Return the ``(n_heads, seq_len, seq_len)`` learned bias tensor."""
        positions = torch.arange(seq_len, device=self.bias.weight.device)
        relative = positions[None, :] - positions[:, None]      # (L, L) = j - i
        buckets = self.relative_bucket(relative, self.num_buckets, self.max_distance)
        values = self.bias(buckets)                              # (L, L, n_heads)
        return values.permute(2, 0, 1)                          # (n_heads, L, L)
