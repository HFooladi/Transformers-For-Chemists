"""Tokenization visualizations.

Helpers for the visualizations used in notebooks 01, 02, and 02.1:

* ``plot_token_grid`` — horizontal grid of cells, one per token, each labelled
  with the token text and coloured by its category (atom, bond, bracket, ring,
  special token). Cells are sized to fit their token text and rows wrap when
  the line gets too wide, so long subword tokens like ``c1ccccc1C(=O)O`` stay
  inside their own cell instead of overflowing onto the next.
* ``plot_molecule_with_tokens`` — side-by-side: the RDKit 2D depiction of a
  molecule on the left, the (possibly-wrapped) token grid on the right.
* ``compare_tokenizations`` — vertically stacked token grids for the same
  molecule across different tokenizers (notebook 02), with each subplot's
  height scaled to the number of wrapped rows it needs.

All functions return matplotlib figures.
"""

from __future__ import annotations

from typing import Sequence

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# Coarse-grained colour palette, chosen to be colour-blind friendly and to
# echo the Jmol / CPK conventions where it makes sense (red-ish for O,
# yellow for S, etc.).
_TOKEN_COLOURS = {
    "atom_C": "#cccccc",
    "atom_aromatic": "#aab4ff",
    "atom_N": "#3050f8",
    "atom_O": "#ff5050",
    "atom_S": "#dddd00",
    "atom_halogen": "#90e050",
    "atom_other": "#9966cc",
    "bond": "#666666",
    "bracket": "#444444",
    "ring": "#ff9900",
    "special": "#000000",
    "default": "#888888",
}


_SPECIAL_TOKEN_SET = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"}


def _classify_token(tok: str) -> str:
    """Return a coarse category string for ``tok`` used to look up its colour."""
    if tok in _SPECIAL_TOKEN_SET:
        return "special"
    if tok.startswith("[") and tok.endswith("]"):
        return "bracket"
    if tok in {"=", "#", "-", "/", "\\", ":", "~", "."}:
        return "bond"
    if tok in {"(", ")"}:
        return "bracket"
    if tok.isdigit():
        return "ring"
    if tok in {"C"}:
        return "atom_C"
    if tok in {"c", "n", "o", "s", "p", "b"}:
        return "atom_aromatic"
    if tok == "N":
        return "atom_N"
    if tok == "O":
        return "atom_O"
    if tok == "S":
        return "atom_S"
    if tok in {"F", "Cl", "Br", "I"}:
        return "atom_halogen"
    if tok and tok[0].isalpha():
        return "atom_other"
    return "default"


def _token_colour(tok: str) -> str:
    return _TOKEN_COLOURS[_classify_token(tok)]


# Layout constants used as defaults across the three plotting functions. The
# values are in figure-units (≈ inches) because we use aspect="equal" and a
# cell of width 1 unit corresponds to one inch of figure width.
_DEFAULT_CHAR_WIDTH = 0.18      # estimated horizontal space per character (in figure units)
_DEFAULT_PADDING = 0.30         # extra padding inside each cell (split left/right)
_DEFAULT_MIN_CELL_WIDTH = 0.55  # never shrink a cell below this (so 1-char tokens stay readable)
_DEFAULT_MAX_ROW_WIDTH = 12.0   # wrap to a new row once a row exceeds this many units


def _pack_tokens(
    tokens: Sequence[str],
    char_width: float,
    padding: float,
    min_cell_width: float,
    max_row_width: float,
) -> tuple[list[float], list[tuple[int, int, float]]]:
    """Pack ``tokens`` left-to-right into rows that don't exceed ``max_row_width``.

    Each token is given a cell width of ``max(min_cell_width,
    len(token) * char_width + padding)``. Tokens are then greedily placed onto
    the current row; when adding the next token would push the row past
    ``max_row_width``, the row is closed and a new row started. A single token
    that is itself wider than ``max_row_width`` is given its own row.

    Returns
    -------
    widths
        Per-token cell widths.
    rows
        List of ``(start_index, end_index, total_width)`` tuples — one per row.
    """
    widths = [
        max(min_cell_width, len(tok) * char_width + padding) for tok in tokens
    ]
    rows: list[tuple[int, int, float]] = []
    cur_start = 0
    cur_w = 0.0
    for i, w in enumerate(widths):
        # Wrap if adding this token would exceed the row budget AND the
        # current row already has at least one token (so we never start a row
        # with an immediate overflow check that produces an empty row).
        if cur_w + w > max_row_width and i > cur_start:
            rows.append((cur_start, i, cur_w))
            cur_start = i
            cur_w = 0.0
        cur_w += w
    if tokens:
        rows.append((cur_start, len(tokens), cur_w))
    return widths, rows


def plot_token_grid(
    tokens: Sequence[str],
    title: str | None = None,
    ax=None,
    char_width: float = _DEFAULT_CHAR_WIDTH,
    cell_height: float = 0.8,
    padding: float = _DEFAULT_PADDING,
    min_cell_width: float = _DEFAULT_MIN_CELL_WIDTH,
    max_row_width: float = _DEFAULT_MAX_ROW_WIDTH,
    fontsize: int = 11,
):
    """Draw a horizontal grid of coloured cells, one per token.

    Each cell is sized to fit its token text. When a row would overflow
    ``max_row_width`` (in figure units, ≈ inches) the layout wraps to a new
    row underneath. Useful for subword tokenizers whose tokens can be much
    longer than a single character (``c1ccccc1C(=O)O``, ``CC(=O)O``, ...).

    Parameters
    ----------
    tokens
        The tokens to display, in order.
    title
        Optional axes title.
    ax
        Optional pre-existing matplotlib axes to draw into. If omitted a new
        figure is created and sized to fit the wrapped layout.
    char_width
        Horizontal space allocated per character (figure units).
    cell_height
        Per-cell vertical size (figure units).
    padding
        Extra horizontal padding added to each cell beyond the text width.
    min_cell_width
        Minimum cell width — keeps single-character tokens from looking like
        a sliver.
    max_row_width
        Wrap to a new row once the current row exceeds this width.
    fontsize
        Font size for the token labels.
    """
    if not MPL_AVAILABLE:
        raise RuntimeError("matplotlib is not installed; run `pip install matplotlib`.")

    widths, rows = _pack_tokens(tokens, char_width, padding, min_cell_width, max_row_width)
    n_rows = max(1, len(rows))
    max_w = max((r[2] for r in rows), default=min_cell_width)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max_w + 0.5, cell_height * n_rows + 0.6))
    else:
        fig = ax.figure

    for row_idx, (start, end, _) in enumerate(rows):
        # Top row first: highest y value.
        y = n_rows - 1 - row_idx
        x = 0.0
        for i in range(start, end):
            tok = tokens[i]
            w = widths[i]
            colour = _token_colour(tok)
            ax.add_patch(
                Rectangle(
                    (x, y),
                    w,
                    1,
                    facecolor=colour,
                    edgecolor="black",
                    linewidth=0.7,
                    alpha=0.7,
                )
            )
            text_colour = (
                "white"
                if colour in {"#000000", "#3050f8", "#666666", "#444444"}
                else "black"
            )
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
    if title:
        ax.set_title(title, fontsize=11, pad=8)

    return fig


def plot_molecule_with_tokens(
    smiles: str,
    tokens: Sequence[str],
    title: str | None = None,
    mol_size: int = 300,
    max_row_width: float = 9.0,
):
    """Side-by-side RDKit 2D depiction + token grid for ``smiles``.

    The token grid wraps automatically if it has more tokens than will fit
    in ``max_row_width`` units of horizontal space.

    Falls back to a token-only display if RDKit is not available.
    """
    if not MPL_AVAILABLE:
        raise RuntimeError("matplotlib is not installed; run `pip install matplotlib`.")

    # Compute the wrapped layout up front so we can size the figure to it.
    _, rows = _pack_tokens(
        tokens,
        _DEFAULT_CHAR_WIDTH,
        _DEFAULT_PADDING,
        _DEFAULT_MIN_CELL_WIDTH,
        max_row_width,
    )
    grid_w = max(2.0, max((r[2] for r in rows), default=1.0))
    grid_h = max(1.0, len(rows) * 0.8 + 0.4)

    fig_height = max(3.2, grid_h + 0.4)
    fig = plt.figure(figsize=(grid_w + 4.0, fig_height))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.0, grid_w])

    ax_mol = fig.add_subplot(gs[0])
    if RDKIT_AVAILABLE:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            img = Draw.MolToImage(mol, size=(mol_size, mol_size))
            ax_mol.imshow(img)
    ax_mol.set_xticks([])
    ax_mol.set_yticks([])
    for spine in ax_mol.spines.values():
        spine.set_visible(False)
    ax_mol.set_title(f"SMILES: {smiles}", fontsize=10)

    ax_grid = fig.add_subplot(gs[1])
    plot_token_grid(tokens, ax=ax_grid, max_row_width=max_row_width)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def compare_tokenizations(
    smiles: str,
    tokenizations: dict[str, Sequence[str]],
    title: str | None = None,
    max_row_width: float = 12.0,
):
    """Stack token grids vertically — one row per tokenizer (notebook 02).

    Each subplot's height is scaled to the number of wrapped rows it needs,
    so a tokenizer that produces 1 long row (e.g. character-level on a big
    molecule) and one that produces 3 wrapped rows render side by side
    without one squashing the other.
    """
    if not MPL_AVAILABLE:
        raise RuntimeError("matplotlib is not installed; run `pip install matplotlib`.")

    # First pass: compute layout for each tokenizer so we know per-subplot
    # row counts and the widest row across them all.
    layouts = []  # list of (name, tokens, n_rows, max_row_extent)
    overall_max_w = 1.0
    for name, toks in tokenizations.items():
        _, rows = _pack_tokens(
            toks,
            _DEFAULT_CHAR_WIDTH,
            _DEFAULT_PADDING,
            _DEFAULT_MIN_CELL_WIDTH,
            max_row_width,
        )
        n_r = max(1, len(rows))
        max_r = max((r[2] for r in rows), default=1.0)
        overall_max_w = max(overall_max_w, max_r)
        layouts.append((name, toks, n_r, max_r))

    n_subplots = len(layouts)
    height_ratios = [n_r for _, _, n_r, _ in layouts]
    fig_width = overall_max_w + 0.6
    # 0.85 inch per token-row + 0.5 inch per subplot title
    fig_height = sum(height_ratios) * 0.85 + n_subplots * 0.5

    fig, axes = plt.subplots(
        n_subplots,
        1,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": height_ratios},
    )
    if n_subplots == 1:
        axes = [axes]

    for ax, (name, toks, _, _) in zip(axes, layouts):
        plot_token_grid(
            toks,
            title=f"{name}  ({len(toks)} tokens)",
            ax=ax,
            max_row_width=max_row_width,
        )

    if title is None:
        title = f"Tokenizations of {smiles}"
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig
