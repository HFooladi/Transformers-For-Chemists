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
    from torch.nn import functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # type: ignore
    F = None  # type: ignore


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
    """Single-head scaled dot-product attention (Vaswani et al., 2017; notebook 04).

    Computes ``softmax(QK^T / sqrt(d_k)) V``. The query, key, and value
    projections all read from the same input ``x`` (i.e., self-attention) and
    each maps ``d_model -> d_model``. Multi-head attention, which splits
    these projections across ``n_heads`` parallel sub-spaces, is built on
    top of this module in notebook 05.

    The forward pass returns both the attended output and the attention
    weights so downstream code (notebook 04 visualizations, later sanity
    checks) can inspect what the model is doing.

    Parameters
    ----------
    d_model
        Feature dimension of the input. Q, K, V each live in ``R^{d_model}``.
    dropout
        Dropout probability applied to the *attention weights* (not the
        output). 0.1 matches the original Transformer.

    Examples
    --------
    >>> attn = ScaledDotProductAttention(d_model=64)
    >>> x = torch.randn(2, 16, 64)
    >>> out, weights = attn(x)
    >>> out.shape, weights.shape
    (torch.Size([2, 16, 64]), torch.Size([2, 16, 16]))
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(
        self,
        x,
        mask=None,
        return_attention: bool = True,
    ):
        """Run self-attention on ``x``.

        ``x``: ``(batch, seq_len, d_model)`` float tensor.
        ``mask``: optional ``(batch, seq_len)`` tensor with 1 for real tokens
        and 0 for padding. The mask is applied on the *key* axis: padded
        positions cannot be attended to. ``None`` means no masking.

        Returns ``(output, attention_weights)`` where ``output`` has the same
        shape as ``x`` and ``attention_weights`` has shape
        ``(batch, seq_len, seq_len)``. If ``return_attention`` is ``False``,
        the second element is ``None``.
        """
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # (B, L, d) @ (B, d, L) -> (B, L, L). The 1/sqrt(d) scale stops the
        # softmax from saturating once d_model gets large; see notebook 04 §5.
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            # mask: (B, L). Broadcast onto the key axis so an attended-to
            # padding position gets -inf for *every* query.
            keep = mask.bool().unsqueeze(1)  # (B, 1, L)
            scores = scores.masked_fill(~keep, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        return (out, attn) if return_attention else (out, None)


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
