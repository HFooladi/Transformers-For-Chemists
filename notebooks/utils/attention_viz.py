"""Attention visualizations.

Helpers for the visualizations used in notebooks 04+:

* ``plot_attention_heatmap`` — square ``(seq, seq)`` heatmap with token labels
  on both axes.
* ``plot_attention_on_smiles`` — overlay attention weight from a chosen token
  back onto the SMILES string itself, colouring each tokenized cell by its
  attention weight.
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
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

# Re-use the row-packing logic from notebook 02's tokenization viz so the
# attention overlay shares the same cell aesthetic as the token grid.
from .tokenization_viz import (
    _DEFAULT_CHAR_WIDTH,
    _DEFAULT_MAX_ROW_WIDTH,
    _DEFAULT_MIN_CELL_WIDTH,
    _DEFAULT_PADDING,
    _pack_tokens,
)


def _to_numpy(x):
    """Best-effort tensor → ndarray conversion (handles torch + numpy)."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def plot_attention_heatmap(
    attention,
    tokens: Sequence[str],
    title: str | None = None,
    cmap: str | None = None,
    figsize: tuple[float, float] = (7.5, 6.0),
    fontsize: int = 8,
):
    """Plot a single ``(seq, seq)`` attention matrix as a heatmap.

    The colour map is chosen automatically from the data:

    * If every entry is non-negative (e.g. the output of a softmax), use
      ``"Blues"`` with ``vmin=0``.
    * Otherwise (e.g. raw ``QK^T`` scores), use the diverging ``"RdBu_r"``
      with a symmetric colour range so 0 maps to white.

    Parameters
    ----------
    attention
        ``(L, L)`` attention matrix as numpy ndarray or torch tensor. Rows
        index the **query** token, columns the **key** token.
    tokens
        Length-``L`` sequence of token strings used to label both axes.
    title
        Optional axes title.
    cmap
        Override the auto-chosen colour map.
    figsize
        Matplotlib figure size in inches.
    fontsize
        Font size for the tick labels.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    if not MPL_AVAILABLE:
        raise RuntimeError("matplotlib is not installed; run `pip install matplotlib`.")
    if not NUMPY_AVAILABLE:
        raise RuntimeError("numpy is not installed; run `pip install numpy`.")

    A = _to_numpy(attention)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"expected a square (L, L) matrix, got shape {A.shape}")
    L = A.shape[0]
    if len(tokens) != L:
        raise ValueError(f"len(tokens) = {len(tokens)} does not match attention shape {A.shape}")

    is_nonneg = np.all(A >= 0)
    if cmap is None:
        cmap = "Blues" if is_nonneg else "RdBu_r"
    if is_nonneg:
        vmin, vmax = 0.0, float(A.max()) if A.size else 1.0
        cbar_label = "attention weight"
    else:
        v = float(np.abs(A).max()) if A.size else 1.0
        vmin, vmax = -v, v
        cbar_label = "score"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(A, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(L))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=fontsize)
    ax.set_yticks(range(L))
    ax.set_yticklabels(tokens, fontsize=fontsize)
    ax.set_xlabel("Key  (attended to)")
    ax.set_ylabel("Query  (attending from)")

    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.85)

    if title:
        ax.set_title(title, fontsize=11, pad=10)
    fig.tight_layout()
    return fig


def plot_attention_on_smiles(
    attention_row,
    tokens: Sequence[str],
    query_index: int,
    title: str | None = None,
    cmap: str = "Blues",
    char_width: float = _DEFAULT_CHAR_WIDTH,
    cell_height: float = 0.8,
    padding: float = _DEFAULT_PADDING,
    min_cell_width: float = _DEFAULT_MIN_CELL_WIDTH,
    max_row_width: float = _DEFAULT_MAX_ROW_WIDTH,
    fontsize: int = 11,
):
    """Show the attention weights *from* one query token *to* every other
    token, by colouring the tokenized SMILES cells with their weights.

    The cells are laid out exactly like ``plot_token_grid`` (notebook 02), so
    the visual language is consistent across the course. The query token is
    outlined in red so it's obvious which row of the attention matrix is
    being shown.

    Parameters
    ----------
    attention_row
        Length-``L`` vector of attention weights from a single query token.
        Should already be softmaxed (non-negative, sums to ~1).
    tokens
        Length-``L`` sequence of token strings.
    query_index
        Position of the query token in ``tokens`` (the one whose attention
        row is being visualised).
    title
        Optional figure title. A sensible default is generated otherwise.
    cmap
        Matplotlib colour map name.
    char_width, cell_height, padding, min_cell_width, max_row_width, fontsize
        Layout knobs, identical to ``plot_token_grid``.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    if not MPL_AVAILABLE:
        raise RuntimeError("matplotlib is not installed; run `pip install matplotlib`.")
    if not NUMPY_AVAILABLE:
        raise RuntimeError("numpy is not installed; run `pip install numpy`.")

    weights = _to_numpy(attention_row).astype(float).ravel()
    if weights.shape[0] != len(tokens):
        raise ValueError(
            f"len(attention_row) = {weights.shape[0]} does not match len(tokens) = {len(tokens)}"
        )
    if not (0 <= query_index < len(tokens)):
        raise ValueError(f"query_index {query_index} out of range for {len(tokens)} tokens")

    # Map weights -> [0, 1]. Use a fixed vmin=0 so a uniform 1/L attention
    # row reads as "uniformly faint", not "uniformly saturated".
    vmin, vmax = 0.0, max(float(weights.max()), 1e-12)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmapper = plt.cm.get_cmap(cmap)

    widths, rows = _pack_tokens(tokens, char_width, padding, min_cell_width, max_row_width)
    n_rows = max(1, len(rows))
    max_w = max((r[2] for r in rows), default=min_cell_width)

    # Reserve a bit of width for the colorbar on the right.
    fig_height = cell_height * n_rows + 0.9
    fig, (ax, cax) = plt.subplots(
        1,
        2,
        figsize=(max_w + 1.6, fig_height),
        gridspec_kw={"width_ratios": [max_w, 0.25]},
    )

    for row_idx, (start, end, _) in enumerate(rows):
        y = n_rows - 1 - row_idx
        x = 0.0
        for i in range(start, end):
            tok = tokens[i]
            w = widths[i]
            facecolor = cmapper(norm(weights[i]))
            is_query = i == query_index
            ax.add_patch(
                Rectangle(
                    (x, y),
                    w,
                    1,
                    facecolor=facecolor,
                    edgecolor="#cc0000" if is_query else "black",
                    linewidth=2.2 if is_query else 0.7,
                )
            )
            # Pick text colour by the *luminance* of the cell — bright cells
            # (high attention) need dark text, faint ones can use black.
            r, g, b, _ = facecolor
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_colour = "white" if luminance < 0.5 else "black"
            ax.text(
                x + w / 2,
                y + 0.5,
                tok,
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                color=text_colour,
            )
            x += w

    ax.set_xlim(0, max_w)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is None:
        title = (
            f"Attention from token '{tokens[query_index]}'  "
            f"(query position {query_index})"
        )
    ax.set_title(title, fontsize=11, pad=8)

    # Colorbar.
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmapper)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label="attention weight")

    fig.tight_layout()
    return fig


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
