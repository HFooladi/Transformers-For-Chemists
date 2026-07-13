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
    """Multi-head scaled dot-product attention (Vaswani et al., 2017; notebook 05).

    Instead of one attention pattern over the full ``d_model`` space, this runs
    ``n_heads`` patterns *in parallel*, each in a cheaper ``d_k = d_model /
    n_heads`` sub-space. A single ``d_model -> d_model`` projection for each of
    Q/K/V is *reshaped* into ``n_heads`` heads (the heads cost no extra
    parameters), attention is computed per head, the heads are concatenated, and
    a final output projection ``W_O`` mixes them back into one vector.

    The forward pass returns the **per-head** attention weights with shape
    ``(batch, n_heads, seq_len, seq_len)`` — note the extra head axis compared
    with :class:`ScaledDotProductAttention`, which returns ``(batch, seq_len,
    seq_len)``. That axis is what ``plot_per_head_grid`` (notebook 05)
    visualizes.

    Parameters
    ----------
    d_model
        Feature dimension of the input. Must be divisible by ``n_heads``.
    n_heads
        Number of parallel attention heads. Each head works in ``d_model /
        n_heads`` dimensions.
    dropout
        Dropout probability applied to the attention weights.

    Examples
    --------
    >>> mha = MultiHeadAttention(d_model=64, n_heads=8)
    >>> x = torch.randn(2, 16, 64)
    >>> out, weights = mha(x)
    >>> out.shape, weights.shape
    (torch.Size([2, 16, 64]), torch.Size([2, 8, 16, 16]))
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        # One projection each for Q, K, V, plus the output projection W_O that
        # mixes the concatenated heads back together.
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # NB: scale by 1/sqrt(d_k), the *per-head* dimension — not d_model.
        self.scale = 1.0 / math.sqrt(self.d_k)

    def _split_heads(self, t):
        """``(B, L, d_model)`` -> ``(B, n_heads, L, d_k)``."""
        B, L, _ = t.shape
        return t.view(B, L, self.n_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        x,
        mask=None,
        return_attention: bool = True,
    ):
        """Run multi-head self-attention on ``x``.

        ``x``: ``(batch, seq_len, d_model)`` float tensor.
        ``mask``: optional ``(batch, seq_len)`` tensor with 1 for real tokens
        and 0 for padding, applied on the *key* axis (same convention as
        :class:`ScaledDotProductAttention`). ``None`` means no masking.

        Returns ``(output, attention_weights)`` where ``output`` has the same
        shape as ``x`` and ``attention_weights`` has shape
        ``(batch, n_heads, seq_len, seq_len)``. If ``return_attention`` is
        ``False``, the second element is ``None``.
        """
        B, L, _ = x.shape

        # Project then reshape into heads: (B, n_heads, L, d_k).
        Q = self._split_heads(self.w_q(x))
        K = self._split_heads(self.w_k(x))
        V = self._split_heads(self.w_v(x))

        # (B, n_heads, L, d_k) @ (B, n_heads, d_k, L) -> (B, n_heads, L, L).
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            # mask: (B, L). Broadcast onto (head, query) axes so a padded key
            # position gets -inf for every head and every query.
            keep = mask.bool().unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)
            scores = scores.masked_fill(~keep, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # (B, n_heads, L, L) @ (B, n_heads, L, d_k) -> (B, n_heads, L, d_k).
        ctx = torch.matmul(attn, V)

        # Concatenate heads back to (B, L, d_model) — the inverse of the split,
        # transpose-then-reshape so memory is laid out contiguously.
        ctx = ctx.transpose(1, 2).reshape(B, L, self.d_model)
        out = self.w_o(ctx)
        return (out, attn) if return_attention else (out, None)


class FeedForward(nn.Module if TORCH_AVAILABLE else object):
    """Position-wise feed-forward network (Vaswani et al., 2017; notebook 06).

    A two-layer MLP applied *independently* to every position::

        FFN(x) = W_2 . GELU(W_1 . x)

    The hidden layer expands the representation to ``d_ff`` (typically
    ``4 * d_model``) and the second layer projects it back, so the output shape
    matches the input. Where attention lets tokens *talk* to each other, the
    feed-forward network is where each token *thinks* — a non-linear
    transformation of the context it just gathered, with no mixing across
    positions. GELU is used (as in BERT and MolFormer) rather than ReLU.

    Parameters
    ----------
    d_model
        Input/output feature dimension.
    d_ff
        Hidden (expanded) dimension, usually ``4 * d_model``.
    dropout
        Dropout probability applied after the activation.

    Examples
    --------
    >>> ff = FeedForward(d_model=64, d_ff=256)
    >>> ff(torch.randn(2, 16, 64)).shape
    torch.Size([2, 16, 64])
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """``x``: ``(batch, seq_len, d_model)`` -> same shape."""
        return self.lin2(self.dropout(self.act(self.lin1(x))))


class EncoderBlock(nn.Module if TORCH_AVAILABLE else object):
    """Pre-norm transformer encoder block (notebook 06).

    Combines multi-head attention and a position-wise feed-forward network,
    each wrapped in a **pre-norm residual**::

        x = x + MultiHeadAttention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))

    The "pre-norm" placement (LayerNorm *inside* the residual branch, used by
    GPT-2+, ViT, and MolFormer) keeps the residual stream un-normalized, which
    gives a clean gradient highway and trains stably at depth — see the
    deep-dive in notebook 06.1. The block is **shape-preserving**
    (``(B, L, d_model)`` in and out), which is exactly the property that lets
    ``TransformerEncoder`` (notebook 07) stack it ``n_layers`` times.

    Parameters
    ----------
    d_model
        Feature dimension carried along the residual stream.
    n_heads
        Number of attention heads.
    d_ff
        Hidden dimension of the feed-forward network (usually ``4 * d_model``).
    dropout
        Dropout probability for attention weights, the feed-forward network, and
        the residual sub-layer outputs.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attention: bool = True):
        """``x``: ``(batch, seq_len, d_model)``; ``mask``: ``(batch, seq_len)``.

        Returns ``(output, attention_weights)`` with ``output`` the same shape
        as ``x`` and ``attention_weights`` of shape ``(batch, n_heads, seq_len,
        seq_len)`` (or ``None`` if ``return_attention`` is ``False``), so the
        attention pattern can be collected layer-by-layer when the block is
        stacked.
        """
        attended, attn = self.attn(
            self.norm1(x), mask=mask, return_attention=return_attention
        )
        x = x + self.dropout(attended)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, attn


class TransformerEncoder(nn.Module if TORCH_AVAILABLE else object):
    """Token embedding + positional encoding + a stack of ``EncoderBlock``\\ s (notebook 07).

    This is the complete, **task-agnostic** encoder body — the thing every
    downstream notebook reuses. It glues together the three pieces built in
    notebooks 03–06::

        input_ids ── TokenEmbedding ──┐
                                      ├─(+)─→ EncoderBlock × n_layers ─→ sequence_output
                  SinusoidalPositionalEncoding ─┘

    Crucially it returns the **full sequence** ``(batch, seq_len, d_model)`` and
    does *no* pooling: how to read the sequence is the head's job, not the
    body's. That separation is what lets the same encoder drive a ``[CLS]``
    classification head (notebook 07), a per-position masked-language-modelling
    head (notebook 08), and a fine-tuning head (notebook 09) without change.

    Parameters
    ----------
    vocab_size
        Size of the tokenizer vocabulary (including special tokens).
    d_model
        Feature dimension carried along the residual stream.
    n_heads
        Number of attention heads in each block (must divide ``d_model``).
    n_layers
        Number of stacked :class:`EncoderBlock` layers.
    d_ff
        Hidden dimension of each block's feed-forward network (usually
        ``4 * d_model``).
    max_len
        Maximum sequence length the positional encoding supports.
    dropout
        Dropout probability shared by the embedding, the blocks, and the
        residual sub-layers.

    Examples
    --------
    >>> enc = TransformerEncoder(vocab_size=40, d_model=64, n_heads=4, n_layers=2)
    >>> ids = torch.randint(0, 40, (2, 16))
    >>> seq, attns = enc(ids, return_attention=True)
    >>> seq.shape, len(attns), attns[0].shape
    (torch.Size([2, 16, 64]), 2, torch.Size([2, 4, 16, 16]))
    """

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
        super().__init__()
        self.d_model = d_model
        self.embed = TokenEmbedding(vocab_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None, return_attention: bool = False):
        """Encode a batch of token-ID sequences.

        ``input_ids``: ``(batch, seq_len)`` long tensor.
        ``attention_mask``: optional ``(batch, seq_len)`` tensor with 1 for real
        tokens and 0 for padding. It is threaded unchanged into every block (the
        blocks broadcast it onto the key axis), so padded positions never get
        attended to anywhere in the stack.

        Returns ``(sequence_output, per_layer_attention)`` where
        ``sequence_output`` is ``(batch, seq_len, d_model)`` and
        ``per_layer_attention`` is a list of ``n_layers`` tensors each of shape
        ``(batch, n_heads, seq_len, seq_len)`` — or ``None`` if
        ``return_attention`` is ``False``.
        """
        h = self.dropout(self.pos(self.embed(input_ids)))   # (B, L, d_model)
        attns = []
        for blk in self.blocks:
            h, w = blk(h, mask=attention_mask, return_attention=return_attention)
            if return_attention:
                attns.append(w)                              # (B, n_heads, L, L)
        return (h, attns) if return_attention else (h, None)
