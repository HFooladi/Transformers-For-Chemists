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

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # type: ignore


class TokenEmbedding(nn.Module if TORCH_AVAILABLE else object):
    """Learned token embedding scaled by ``sqrt(d_model)`` (notebook 03)."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        raise NotImplementedError("Phase 3: implement in notebook 03.")


class SinusoidalPositionalEncoding(nn.Module if TORCH_AVAILABLE else object):
    """Fixed sinusoidal positional encoding from Vaswani et al. (notebook 03)."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        raise NotImplementedError("Phase 3: implement in notebook 03.")


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
