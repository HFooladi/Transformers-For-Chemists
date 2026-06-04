"""Performer / FAVOR+ attention — the deep-dive variant for notebook 04.2.

Notebook 04.1 built **linear attention** with the simplest correct feature map,
``φ(x) = elu(x) + 1`` (Katharopoulos et al., 2020). It carries the full ``O(L)``
complexity story but makes no attempt to *match softmax*: its attention pattern
is just "smoother", not a faithful approximation of ``softmax(QKᵀ/√d)``.

**Performer** (Choromanski et al., 2021) — the attention **MolFormer** actually
uses — closes that gap. Its FAVOR+ feature map (*Fast Attention Via positive
Orthogonal Random features*) **provably approximates the softmax kernel** while
keeping ``O(L)`` cost. The two ideas behind the name:

* **Positive random features** (the ``+``). The softmax similarity
  ``exp(qᵀk)`` can be written as an expectation over random projections
  ``w ~ N(0, I)``::

      exp(qᵀk) = E_w[ exp(wᵀq − ‖q‖²/2) · exp(wᵀk − ‖k‖²/2) ]

  Estimating it with ``m`` sampled ``w``'s gives a feature map
  ``φ(x) = exp(−‖x‖²/2)/√m · [exp(w₁ᵀx), …, exp(w_mᵀx)]`` whose entries are
  **always positive** — so the approximate attention weights stay non-negative
  and the denominator never collapses. (Earlier trigonometric features can go
  negative; see notebook 04.2 §C.)

* **Orthogonal random features** (the ``OR``). Making the rows of the
  projection matrix mutually orthogonal (rather than i.i.d. Gaussian) reduces
  the estimator variance with no added bias — a better approximation for the
  same ``m``.

``PerformerAttention`` mirrors the interface of ``ScaledDotProductAttention``
and ``LinearAttention`` (``forward(x, mask=None, return_attention=...)``), so it
is drop-in interchangeable in any encoder block.
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


def gaussian_orthogonal_random_matrix(n_features: int, dim: int, generator=None):
    """Build an ``(n_features, dim)`` random projection with orthogonal rows.

    Rows are generated in square ``dim × dim`` orthogonal blocks (via QR of a
    Gaussian matrix), then each row is rescaled so its norm matches that of a
    fresh ``N(0, I_dim)`` draw (a chi-distributed length). The result is
    unbiased for the softmax-kernel estimator but lower-variance than i.i.d.
    Gaussian rows — the "OR" in FAVOR+.
    """
    blocks = []
    n_full = n_features // dim
    for _ in range(n_full):
        g = torch.randn(dim, dim, generator=generator)
        q, _ = torch.linalg.qr(g)
        blocks.append(q)
    remaining = n_features - n_full * dim
    if remaining > 0:
        g = torch.randn(dim, dim, generator=generator)
        q, _ = torch.linalg.qr(g)
        blocks.append(q[:remaining])
    w = torch.cat(blocks, dim=0)                       # (n_features, dim), orthonormal rows

    # Rescale each (unit-norm) row to a chi-distributed length, so the row
    # norms match i.i.d. Gaussian rows — required for the estimator to be
    # unbiased.
    lengths = torch.randn(n_features, dim, generator=generator).norm(dim=1)
    return w * lengths.unsqueeze(-1)


def softmax_kernel_features(x, projection, is_query: bool, eps: float = 1e-12):
    """Positive random features ``φ(x)`` that estimate the softmax kernel.

    ``x``: ``(..., L, d)``. ``projection``: ``(m, d)`` random matrix.
    Returns ``(..., L, m)`` strictly-positive features such that
    ``φ(q) · φ(k) ≈ exp(qᵀk / √d)`` in expectation — i.e. the *scaled* softmax
    similarity from notebook 04.

    The ``d^(−1/4)`` pre-scaling of ``x`` folds the ``1/√d`` attention scaling
    into the kernel. A max-subtraction stabilizes the ``exp`` (per-query for
    queries, global for keys) exactly as in the reference Performer code.
    """
    d = x.shape[-1]
    m = projection.shape[0]
    x_scaled = x * (d ** -0.25)
    # w·x̃ for every random direction w (rows of `projection`).
    proj = torch.matmul(x_scaled, projection.transpose(-2, -1))   # (..., L, m)
    half_sq_norm = (x_scaled ** 2).sum(dim=-1, keepdim=True) / 2.0  # (..., L, 1)

    if is_query:
        stabilizer = proj.max(dim=-1, keepdim=True).values
    else:
        stabilizer = proj.max()
    feats = torch.exp(proj - half_sq_norm - stabilizer) + eps
    return feats * (m ** -0.5)


class PerformerAttention(nn.Module if TORCH_AVAILABLE else object):
    """Single-head Performer / FAVOR+ attention (Choromanski et al., 2021; notebook 04.2).

    Approximates ``softmax(QKᵀ/√d) V`` in ``O(L · m · d)`` time using ``m``
    positive orthogonal random features, instead of the exact ``O(L² · d)``.
    Interface matches ``ScaledDotProductAttention`` / ``LinearAttention`` so the
    three are interchangeable.

    Parameters
    ----------
    d_model
        Feature dimension. Q, K, V each live in ``R^{d_model}``.
    n_features
        Number of random features ``m``. More features → closer to true
        softmax, at higher cost. ``None`` defaults to ``d_model * ceil(log d)``,
        a common heuristic.
    dropout
        Dropout on the attended output.
    redraw
        If ``True``, resample the random projection on every forward pass
        (Performer's "feature redraw", which de-biases training). Default
        ``False`` for reproducible, inspectable behaviour in the notebook.
    """

    def __init__(self, d_model: int, n_features: int | None = None,
                 dropout: float = 0.1, redraw: bool = False, eps: float = 1e-12) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features or int(d_model * math.ceil(math.log(max(d_model, 2))))
        self.redraw = redraw
        self.eps = eps
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # Fixed projection stored as a buffer (saved with the model, not trained).
        self.register_buffer(
            "projection",
            gaussian_orthogonal_random_matrix(self.n_features, d_model),
        )

    def _projection_for(self, device):
        if self.redraw and self.training:
            return gaussian_orthogonal_random_matrix(self.n_features, self.d_model).to(device)
        return self.projection

    def forward(self, x, mask=None, return_attention: bool = False):
        """Run Performer self-attention on ``x``.

        ``x``: ``(batch, seq_len, d_model)``. ``mask``: optional
        ``(batch, seq_len)`` with 1 = real, 0 = pad; padded keys are zeroed in
        feature space. Returns ``(output, attention_weights)``; the weights are
        ``None`` unless ``return_attention=True``, in which case the implicit
        ``(batch, seq_len, seq_len)`` approximate-softmax matrix is rebuilt for
        visualization (this costs ``O(L²)`` and is for inspection only).
        """
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        proj = self._projection_for(x.device)
        Qf = softmax_kernel_features(Q, proj, is_query=True)    # (B, L, m)
        Kf = softmax_kernel_features(K, proj, is_query=False)   # (B, L, m)

        if mask is not None:
            keep = mask.bool().unsqueeze(-1)                    # (B, L, 1)
            Kf = Kf.masked_fill(~keep, 0.0)
            V = V.masked_fill(~keep, 0.0)

        # Linear-attention contraction in feature space — never forms (L, L).
        kv = torch.einsum("blm,bld->bmd", Kf, V)               # (B, m, d)
        k_sum = Kf.sum(dim=1)                                   # (B, m)
        numerator = torch.einsum("blm,bmd->bld", Qf, kv)        # (B, L, d)
        denominator = torch.einsum("blm,bm->bl", Qf, k_sum)     # (B, L)
        out = numerator / denominator.clamp(min=self.eps).unsqueeze(-1)
        out = self.dropout(out)

        attn = None
        if return_attention:
            scores = torch.einsum("blm,bnm->bln", Qf, Kf)       # (B, L, L) approx exp(QKᵀ/√d)
            attn = scores / scores.sum(dim=-1, keepdim=True).clamp(min=self.eps)

        return out, attn
