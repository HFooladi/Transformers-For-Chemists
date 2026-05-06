"""Attention visualizations.

Helpers for the visualizations used in notebooks 04+:

* ``plot_attention_heatmap`` — square ``(seq, seq)`` heatmap with token labels
  on both axes.
* ``plot_attention_on_smiles`` — overlay attention weight from a chosen token
  back onto the SMILES string itself, colouring each character by its attention
  weight.
* ``plot_per_head_grid`` — small-multiples grid showing the same attention
  pattern across multiple heads, useful for showing head specialization.
* ``animate_attention_over_layers`` — frame-by-frame animation showing how the
  attention pattern evolves across encoder layers.

All functions return matplotlib figures so they can be saved to ``assets/`` for
the README.
"""

from __future__ import annotations

from typing import Sequence

try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


def plot_attention_heatmap(
    attention: "np.ndarray",
    tokens: Sequence[str],
    title: str | None = None,
):
    """Plot a single ``(seq, seq)`` attention matrix as a heatmap."""
    raise NotImplementedError("Phase 2: implement in notebook 04.")


def plot_attention_on_smiles(
    attention_row: "np.ndarray",
    tokens: Sequence[str],
    query_index: int,
):
    """Show the attention weights *from* one query token *to* every other
    token, by colouring the SMILES tokens with their weights."""
    raise NotImplementedError("Phase 2: implement in notebook 04.")


def plot_per_head_grid(
    attention: "np.ndarray",
    tokens: Sequence[str],
    n_cols: int = 4,
):
    """Small-multiples grid: one heatmap per attention head."""
    raise NotImplementedError("Phase 3: implement in notebook 05.")


def animate_attention_over_layers(
    attention_per_layer: Sequence["np.ndarray"],
    tokens: Sequence[str],
):
    """Animate attention patterns across successive transformer layers."""
    raise NotImplementedError("Phase 3: implement in notebook 06.")
