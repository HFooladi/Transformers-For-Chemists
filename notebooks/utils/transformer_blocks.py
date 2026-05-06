"""Reference transformer-component implementations re-used across notebooks.

Each notebook builds the next layer of the stack from scratch, but later
notebooks re-import the canonical, well-commented version from here so they
don't have to copy-paste old code. The idea is the same as the GNN repo's
``enhanced_3d_visualizations.py``: a single, didactic implementation that
becomes the project's lingua franca.

Components:

* ``TokenEmbedding`` — learned lookup, scaled by ``sqrt(d_model)`` (notebook 03)
* ``SinusoidalPositionalEncoding`` — fixed sinusoidal PE (notebook 03)
* ``ScaledDotProductAttention`` — vanilla single-head attention (notebook 04)
* ``MultiHeadAttention`` — multi-head wrapper (notebook 05)
* ``FeedForward`` — two-layer MLP with GELU (notebook 06)
* ``EncoderBlock`` — pre-norm encoder block (notebook 06)
* ``TransformerEncoder`` — stack of encoder blocks (notebook 07)

Variants used in deep-dives (linear attention, RoPE, ALiBi) live in their own
modules under ``utils/`` to keep this one focused on the core stack.
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


class TokenEmbedding(nn.Module if TORCH_AVAILABLE else object):
    """Learned token embedding scaled by ``sqrt(d_model)`` (notebook 03).

    A thin wrapper around ``nn.Embedding`` that applies the
    ``sqrt(d_model)`` scaling from Vaswani et al. (2017). The scaling keeps
    the magnitude of the token embeddings on a comparable footing with the
    sinusoidal positional encodings that get added to them downstream.

    Padding tokens (id 0 by convention) are zeroed out and excluded from
    gradient updates via ``padding_idx=0``.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

    def forward(self, input_ids):
        """``input_ids``: ``(batch, seq_len)`` long tensor of token IDs.

        Returns a ``(batch, seq_len, d_model)`` float tensor.
        """
        return self.embedding(input_ids) * math.sqrt(self.d_model)


class SinusoidalPositionalEncoding(nn.Module if TORCH_AVAILABLE else object):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017; notebook 03).

    For each position ``pos`` and each pair of embedding dimensions ``(2i,
    2i+1)``, the encoding is::

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    The matrix is computed once at construction time and stored as a
    non-trainable buffer. ``forward`` adds it to the input embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()  # (max_len, 1)
        # Inverse frequencies for each pair of dimensions.
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as a buffer so it's saved with the model state but not
        # treated as a learnable parameter.
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Add positional encoding to ``x``.

        ``x``: ``(batch, seq_len, d_model)`` float tensor.
        Returns a tensor of the same shape.
        """
        return x + self.pe[: x.size(1)].unsqueeze(0)


class ScaledDotProductAttention(nn.Module if TORCH_AVAILABLE else object):
    """Single-head scaled dot-product attention (notebook 04)."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        raise NotImplementedError("Phase 3: implement in notebook 04.")


class MultiHeadAttention(nn.Module if TORCH_AVAILABLE else object):
    """Multi-head attention (notebook 05)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        raise NotImplementedError("Phase 3: implement in notebook 05.")


class FeedForward(nn.Module if TORCH_AVAILABLE else object):
    """Position-wise feed-forward network (notebook 06)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        raise NotImplementedError("Phase 3: implement in notebook 06.")


class EncoderBlock(nn.Module if TORCH_AVAILABLE else object):
    """Pre-norm transformer encoder block (notebook 06)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        raise NotImplementedError("Phase 3: implement in notebook 06.")


class TransformerEncoder(nn.Module if TORCH_AVAILABLE else object):
    """Stack of ``EncoderBlock`` layers (notebook 07)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        raise NotImplementedError("Phase 3: implement in notebook 07.")
