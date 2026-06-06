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
    attention,
    tokens: Sequence[str],
    n_cols: int = 4,
    title: str | None = None,
    cmap: str = "Blues",
    panel_size: float = 3.0,
    fontsize: int = 6,
):
    """Small-multiples grid: one heatmap per attention head.

    Multi-head attention (notebook 05) produces one ``(L, L)`` pattern per head.
    Laying them out side by side makes head *specialization* visible at a
    glance — each head attends differently. All panels share a single colour
    scale (``vmin=0``, ``vmax`` = global max over heads) so the comparison is
    honest: a faint panel really is attending more weakly than a bright one.

    Parameters
    ----------
    attention
        ``(n_heads, L, L)`` array (numpy ndarray or torch tensor). Typically
        ``attn_weights[batch_index]`` from
        :class:`~utils.transformer_blocks.MultiHeadAttention`.
    tokens
        Length-``L`` sequence of token strings used to label both axes.
    n_cols
        Number of columns in the grid. Rows are inferred from ``n_heads``.
    title
        Optional overall figure title (``suptitle``).
    cmap
        Matplotlib colour map name (default ``"Blues"`` for softmaxed weights).
    panel_size
        Size in inches of each per-head panel.
    fontsize
        Font size for the per-axis token tick labels.

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
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(f"expected an (n_heads, L, L) tensor, got shape {A.shape}")
    n_heads, L, _ = A.shape
    if len(tokens) != L:
        raise ValueError(
            f"len(tokens) = {len(tokens)} does not match attention shape {A.shape}"
        )

    n_cols = min(n_cols, n_heads)
    n_rows = (n_heads + n_cols - 1) // n_cols
    # Shared colour scale across all heads for an honest comparison.
    vmax = float(A.max()) if A.size else 1.0

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_size * n_cols, panel_size * n_rows),
        squeeze=False,
    )
    im = None
    for h in range(n_rows * n_cols):
        ax = axes[h // n_cols][h % n_cols]
        if h >= n_heads:
            ax.axis("off")
            continue
        im = ax.imshow(A[h], cmap=cmap, vmin=0.0, vmax=vmax, aspect="equal")
        ax.set_title(f"head {h}", fontsize=10)
        ax.set_xticks(range(L))
        ax.set_xticklabels(tokens, rotation=90, fontsize=fontsize)
        ax.set_yticks(range(L))
        ax.set_yticklabels(tokens, fontsize=fontsize)

    if im is not None:
        fig.colorbar(
            im, ax=axes, label="attention weight", shrink=0.8, location="right"
        )
    if title:
        fig.suptitle(title, fontsize=13)
    return fig


def animate_attention_over_layers(
    attention_per_layer: Sequence["np.ndarray"],
    tokens: Sequence[str],
    static: bool = True,
    cmap: str = "Blues",
    interval: int = 700,
):
    """Show how an attention pattern evolves across stacked transformer layers.

    Given one ``(L, L)`` matrix per layer (e.g. the head-averaged attention from
    each :class:`~utils.transformer_blocks.EncoderBlock` in a stack), this
    renders the layer-by-layer progression two ways:

    * ``static=True`` (default) — a small-multiples grid, one panel per layer.
      Always renders in a static notebook/GitHub preview and can be saved to
      ``assets/`` for the README.
    * ``static=False`` — a true :class:`matplotlib.animation.FuncAnimation`
      that steps through the layers. Display it in a notebook with
      ``from IPython.display import HTML; HTML(anim.to_jshtml())``.

    All frames share one colour scale (``vmin=0``, global ``vmax``) so changes
    in attention mass between layers are comparable.

    Parameters
    ----------
    attention_per_layer
        Sequence of ``(L, L)`` arrays (numpy or torch), one per layer.
    tokens
        Length-``L`` token strings used to label the axes.
    static
        If ``True`` return a grid ``Figure``; if ``False`` return a
        ``FuncAnimation``.
    cmap
        Matplotlib colour map name.
    interval
        Milliseconds between frames (animation only).

    Returns
    -------
    matplotlib.figure.Figure or matplotlib.animation.FuncAnimation
        A static grid figure when ``static`` is ``True``, otherwise an
        animation object.
    """
    if not MPL_AVAILABLE:
        raise RuntimeError("matplotlib is not installed; run `pip install matplotlib`.")
    if not NUMPY_AVAILABLE:
        raise RuntimeError("numpy is not installed; run `pip install numpy`.")

    layers = [_to_numpy(a) for a in attention_per_layer]
    if not layers:
        raise ValueError("attention_per_layer is empty")
    L = layers[0].shape[0]
    for k, a in enumerate(layers):
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError(f"layer {k}: expected a square (L, L) matrix, got {a.shape}")
        if a.shape[0] != L:
            raise ValueError(f"layer {k}: shape {a.shape} disagrees with layer 0 ({L})")
    if len(tokens) != L:
        raise ValueError(f"len(tokens) = {len(tokens)} does not match (L = {L})")

    n_layers = len(layers)
    vmax = max((float(a.max()) for a in layers), default=1.0)

    def _label(ax):
        ax.set_xticks(range(L))
        ax.set_xticklabels(tokens, rotation=90, fontsize=6)
        ax.set_yticks(range(L))
        ax.set_yticklabels(tokens, fontsize=6)

    if static:
        n_cols = min(4, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows), squeeze=False
        )
        im = None
        for k in range(n_rows * n_cols):
            ax = axes[k // n_cols][k % n_cols]
            if k >= n_layers:
                ax.axis("off")
                continue
            im = ax.imshow(layers[k], cmap=cmap, vmin=0.0, vmax=vmax, aspect="equal")
            ax.set_title(f"layer {k + 1}", fontsize=10)
            _label(ax)
        if im is not None:
            fig.colorbar(im, ax=axes, label="attention weight", shrink=0.8, location="right")
        return fig

    # Animation: one axes, step through the layers.
    from matplotlib import animation

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    im = ax.imshow(layers[0], cmap=cmap, vmin=0.0, vmax=vmax, aspect="equal")
    _label(ax)
    fig.colorbar(im, ax=ax, label="attention weight", shrink=0.85)

    def _update(k):
        im.set_data(layers[k])
        ax.set_title(f"layer {k + 1} / {n_layers}", fontsize=12)
        return (im,)

    anim = animation.FuncAnimation(
        fig, _update, frames=n_layers, interval=interval, blit=False
    )
    # Avoid a duplicate static frame showing in notebooks.
    plt.close(fig)
    return anim
