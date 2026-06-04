"""Rotary position embeddings (RoPE) — the deep-dive variant used in notebook 04.2.

Notebook 03 gave each token an *absolute* position by **adding** a sinusoidal
vector to its embedding. RoPE (Su et al., 2021, *RoFormer*) takes a different
route: it **rotates** each query and key vector by an angle proportional to the
token's position, *inside* the attention dot product.

Why rotation? Split a ``d``-dimensional vector into ``d/2`` coordinate pairs.
Rotating pair ``i`` of a query at position ``m`` and the same pair of a key at
position ``n``, each by an angle ``= position * theta_i``, makes their 2-D dot
product depend only on the **difference** ``m - n``::

    R(m*theta) q  .  R(n*theta) k   =   q . R((n - m)*theta) k

So the attention score between two tokens becomes a function of their *relative*
offset — the property notebook 03 had to verify after the fact for additive
sinusoidal PE, here baked in by construction. Rotations also preserve vector
norm, so RoPE never changes the magnitude of Q or K, only their direction.

The per-pair angular frequencies ``theta_i`` reuse the Vaswani inverse-frequency
schedule, so RoPE shares its "many frequencies at once" structure with the
sinusoidal encoding of notebook 03.

This module exposes:

* ``RotaryPositionalEncoding`` — precomputes the ``cos``/``sin`` tables and
  rotates a Q or K tensor of shape ``(batch, seq_len, dim)`` (``dim`` even).
* ``apply_rotary`` — a stateless helper that rotates Q and K given precomputed
  ``cos``/``sin`` tables, for notebooks that want to see the raw arithmetic.
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


def rotate_half(x):
    """Map ``(x1, x2)`` pairs to ``(-x2, x1)`` — a 90-degree turn per pair.

    RoPE rotates each 2-D coordinate pair by ``angle``. Written over the whole
    vector at once, a rotation is
    ``x * cos(angle) + rotate_half(x) * sin(angle)``, where ``rotate_half``
    supplies the "other axis" of every pair. We adopt the common interleaving
    where the first half of the dimensions holds the ``x1`` coordinates and the
    second half holds the ``x2`` coordinates.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x, cos, sin):
    """Rotate ``x`` by the position-dependent angles encoded in ``cos``/``sin``.

    ``x``: ``(..., seq_len, dim)``. ``cos``/``sin``: ``(seq_len, dim)`` tables
    where row ``p`` holds the cosines/sines of every pair's angle at position
    ``p``. Returns a tensor the same shape as ``x``.
    """
    return x * cos + rotate_half(x) * sin


class RotaryPositionalEncoding(nn.Module if TORCH_AVAILABLE else object):
    """Rotary position embedding (Su et al., 2021; notebook 04.2).

    Precomputes ``cos`` and ``sin`` lookup tables of shape ``(max_len, dim)``
    and applies them to a query or key tensor. Unlike
    ``SinusoidalPositionalEncoding`` (which *adds* to the token embeddings once,
    up front), RoPE is applied to **Q and K** just before the attention dot
    product, and it *rotates* rather than adds.

    Parameters
    ----------
    dim
        Per-head feature dimension to rotate. Must be even (it is split into
        ``dim / 2`` coordinate pairs).
    max_len
        Largest sequence length the tables support.
    base
        The frequency base (``10000`` in the original paper, matching Vaswani).

    Examples
    --------
    >>> rope = RotaryPositionalEncoding(dim=32, max_len=128)
    >>> q = torch.randn(2, 16, 32)
    >>> q_rot = rope(q)
    >>> q_rot.shape
    torch.Size([2, 16, 32])
    """

    def __init__(self, dim: int, max_len: int = 512, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE needs an even dim, got {dim}.")
        self.dim = dim
        self.max_len = max_len

        # One angular frequency per coordinate pair: theta_i = base^(-2i/dim).
        inv_freq = base ** (-torch.arange(0, dim, 2).float() / dim)  # (dim/2,)
        positions = torch.arange(max_len).float()                   # (max_len,)
        # Outer product: angle of pair i at position p is  p * theta_i.
        angles = torch.outer(positions, inv_freq)                   # (max_len, dim/2)
        # Duplicate across the two halves so `cos`/`sin` line up with the
        # first-half / second-half split that `rotate_half` uses.
        emb = torch.cat((angles, angles), dim=-1)                   # (max_len, dim)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, x, offset: int = 0):
        """Rotate ``x`` by its positions.

        ``x``: ``(batch, seq_len, dim)`` (or any ``(..., seq_len, dim)``).
        ``offset``: starting position index, for caching scenarios.
        Returns a tensor of the same shape as ``x``.
        """
        seq_len = x.size(-2)
        cos = self.cos[offset : offset + seq_len]   # (seq_len, dim)
        sin = self.sin[offset : offset + seq_len]
        return apply_rotary(x, cos, sin)
