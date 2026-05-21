"""Linear-attention variant of self-attention (notebook 04.1).

The companion module to :mod:`utils.transformer_blocks`. Notebook 04 built
``ScaledDotProductAttention`` — Vaswani et al.'s ``softmax(QKᵀ/√d) V`` — and
noted in its closing remarks that the deep-dive sub-series would explore
MolFormer's choice: a **linear-attention** variant that is ``O(N · d²)`` in
sequence length instead of ``O(N² · d)``.

This module holds the canonical, well-commented version of that variant that
notebook 04.1 builds up from scratch.

The implementation uses the **ELU+1 feature map** from Katharopoulos et al.
(2020), *"Transformers are RNNs: Fast Autoregressive Transformers with Linear
Attention"* — the simplest correct linear-attention kernel:

    φ(x) = elu(x) + 1   (always strictly positive)

Attention then becomes

    Attention(x)_i = ( φ(Q_i)ᵀ Σ_j φ(K_j) V_jᵀ ) / ( φ(Q_i)ᵀ Σ_j φ(K_j) )

which costs ``O(L · d²)`` per molecule instead of ``O(L² · d)`` because the
two sums over keys/values can be precomputed *once* and reused for every
query. MolFormer in production uses a Performer-style FAVOR+ feature map
instead of ELU+1; that variant gets its own supplementary notebook
(``04_1_FAVOR_Performer.ipynb``).
"""

from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # type: ignore
    F = None  # type: ignore


def elu_feature_map(x):
    """``φ(x) = elu(x) + 1`` — strictly positive, smooth, no learned parameters.

    Element-wise. Used as the default feature map in :class:`LinearAttention`.
    The ``+1`` shift guarantees ``φ(x) > 0`` everywhere (since
    ``elu(x) > -1``), which keeps the denominator of linear attention
    non-zero for any non-padded row.
    """
    return F.elu(x) + 1.0


class LinearAttention(nn.Module if TORCH_AVAILABLE else object):
    """Single-head linear self-attention (Katharopoulos et al., 2020; notebook 04.1).

    Replaces softmax-attention's ``softmax(QKᵀ/√d) V`` with a kernel-trick
    reformulation. With a feature map ``φ: R^d → R^d`` applied element-wise to
    Q and K, the output for query ``i`` is

        out_i = φ(Q_i)ᵀ · ( Σ_j φ(K_j) V_jᵀ )  /  φ(Q_i)ᵀ · ( Σ_j φ(K_j) )

    Both inner sums are independent of the query index ``i`` and can be
    precomputed once per molecule. The total cost is ``O(L · d²)`` instead of
    ``O(L² · d)`` — linear in sequence length, which is why MolFormer (Ross et
    al., 2022) uses a variant of this trick to scale pre-training to ~1
    billion SMILES.

    The interface mirrors :class:`ScaledDotProductAttention` exactly so the
    two are drop-in interchangeable in any encoder block.

    Parameters
    ----------
    d_model
        Feature dimension of the input. Q, K, V each live in ``R^{d_model}``.
    dropout
        Dropout probability applied to the attended output (not the implicit
        attention matrix — there is no normalized attention matrix to drop
        from in the fast path).
    eps
        Numerical floor for the per-query denominator. A row of all padding
        would otherwise divide by zero; this clamps that denominator to
        ``eps`` so the forward pass stays finite. ``1e-6`` is well below any
        non-degenerate value of ``φ(Q_i)ᵀ Σ_j φ(K_j)``.

    Examples
    --------
    >>> attn = LinearAttention(d_model=64)
    >>> x = torch.randn(2, 16, 64)
    >>> out, weights = attn(x)
    >>> out.shape, weights.shape
    (torch.Size([2, 16, 64]), torch.Size([2, 16, 16]))
    """

    def __init__(self, d_model: int, dropout: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.eps = eps

    def forward(
        self,
        x,
        mask=None,
        return_attention: bool = True,
    ):
        """Run linear self-attention on ``x``.

        ``x``: ``(batch, seq_len, d_model)`` float tensor.
        ``mask``: optional ``(batch, seq_len)`` tensor with 1 for real tokens
        and 0 for padding. Padded **keys** are zeroed out in ``φ(K)`` so they
        contribute nothing to any query's numerator or denominator. ``None``
        means no masking.

        Returns ``(output, attention_weights)``.

        ``output`` has the same shape as ``x``. ``attention_weights`` is the
        ``(batch, seq_len, seq_len)`` *implicit* attention matrix
        ``φ(Q) φ(K)ᵀ / normalizer``, reconstructed only when
        ``return_attention=True``. Reconstructing it defeats the ``O(L)``
        speed-up — the matrix exists purely for visualization and pedagogy.
        Set ``return_attention=False`` in any real training loop.
        """
        Q = self.w_q(x)                              # (B, L, d_model)
        K = self.w_k(x)
        V = self.w_v(x)

        # Apply the feature map element-wise to Q and K. Note that V is *not*
        # passed through φ — it is summed unchanged.
        Q_phi = elu_feature_map(Q)                   # (B, L, d_model)
        K_phi = elu_feature_map(K)                   # (B, L, d_model)

        if mask is not None:
            # Zero out padded *keys* so they contribute nothing to any query.
            # Shape: (B, L, 1) broadcasts over the feature axis.
            keep = mask.bool().unsqueeze(-1)         # (B, L, 1)
            K_phi = K_phi.masked_fill(~keep, 0.0)

        # Sum-of-outer-products: Σ_j φ(K_j) V_jᵀ  — shape (B, d_model, d_model).
        # This is the trick that makes linear attention O(L).
        KV = torch.einsum("bld,blm->bdm", K_phi, V)  # (B, d_model, d_model)

        # Σ_j φ(K_j) — shape (B, d_model). Used as the normalizer.
        K_sum = K_phi.sum(dim=1)                     # (B, d_model)

        # Numerator: φ(Q_i)ᵀ Σ_j φ(K_j) V_jᵀ — shape (B, L, d_model).
        numerator = torch.einsum("bld,bdm->blm", Q_phi, KV)

        # Denominator: φ(Q_i)ᵀ Σ_j φ(K_j) — shape (B, L). Clamp away from zero.
        denominator = torch.einsum("bld,bd->bl", Q_phi, K_sum)
        denominator = denominator.clamp(min=self.eps).unsqueeze(-1)  # (B, L, 1)

        out = numerator / denominator                # (B, L, d_model)
        out = self.dropout(out)

        attn = None
        if return_attention:
            attn = self._reconstruct_attention(Q_phi, K_phi)

        return out, attn

    def _reconstruct_attention(self, Q_phi, K_phi):
        """Build the implicit (B, L, L) attention matrix for visualization only.

        This explicitly materializes the matrix that the fast path *avoids* —
        useful for plotting heatmaps and comparing against softmax attention,
        but it brings the cost back to ``O(L²)`` and should never be done in
        a real training loop.
        """
        # Unnormalized: φ(Q) · φ(K)ᵀ — same shape as softmax(QKᵀ).
        unnorm = torch.matmul(Q_phi, K_phi.transpose(-2, -1))            # (B, L, L)
        # Row-wise renormalize so each row sums to 1 (where possible) — this
        # matches the per-query denominator used in the fast path.
        row_sum = unnorm.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        return unnorm / row_sum
