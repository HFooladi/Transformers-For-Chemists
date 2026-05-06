"""Tokenization visualizations.

Helpers for the visualizations used in notebooks 01, 02, and 02.1:

* ``plot_token_grid`` — horizontal grid of cells, one per token, each labelled
  with the token text and coloured by its category (atom, bond, bracket, ring,
  special token).
* ``plot_molecule_with_tokens`` — side-by-side: the RDKit 2D depiction of a
  molecule on the left, the token grid on the right, useful for showing
  exactly *which* parts of the SMILES become which tokens.
* ``compare_tokenizations`` — vertically stacked token grids for the same
  molecule across different tokenizers (notebook 02).

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


def plot_token_grid(
    tokens: Sequence[str],
    title: str | None = None,
    cell_width: float = 0.55,
    cell_height: float = 0.7,
    fontsize: int = 11,
    ax=None,
):
    """Draw a horizontal grid of coloured cells, one per token.

    Parameters
    ----------
    tokens
        The tokens to display, in order.
    title
        Optional axes title.
    cell_width, cell_height
        Per-cell dimensions in inches; the figure is sized accordingly.
    fontsize
        Font size for the token labels.
    ax
        Optional pre-existing matplotlib axes to draw into. If omitted a new
        figure and axes are created and returned.
    """
    if not MPL_AVAILABLE:
        raise RuntimeError("matplotlib is not installed; run `pip install matplotlib`.")

    n = len(tokens)
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(2.0, cell_width * n + 0.5), cell_height + 0.6))
    else:
        fig = ax.figure

    for i, tok in enumerate(tokens):
        colour = _token_colour(tok)
        ax.add_patch(
            Rectangle(
                (i, 0),
                1,
                1,
                facecolor=colour,
                edgecolor="black",
                linewidth=0.7,
                alpha=0.7,
            )
        )
        text_colour = "white" if colour in {"#000000", "#3050f8", "#666666", "#444444"} else "black"
        ax.text(
            i + 0.5,
            0.5,
            tok,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=text_colour,
        )

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
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
):
    """Side-by-side RDKit 2D depiction + token grid for ``smiles``.

    Falls back to a token-only display if RDKit is not available.
    """
    if not MPL_AVAILABLE:
        raise RuntimeError("matplotlib is not installed; run `pip install matplotlib`.")

    n = len(tokens)
    grid_w = max(2.0, 0.55 * n + 0.5)
    fig = plt.figure(figsize=(grid_w + 4.0, 3.2))
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
    plot_token_grid(tokens, ax=ax_grid)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def compare_tokenizations(
    smiles: str,
    tokenizations: dict[str, Sequence[str]],
    title: str | None = None,
):
    """Stack token grids vertically — one row per tokenizer (notebook 02)."""
    if not MPL_AVAILABLE:
        raise RuntimeError("matplotlib is not installed; run `pip install matplotlib`.")

    n_rows = len(tokenizations)
    max_n = max(len(toks) for toks in tokenizations.values())
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(max(3.0, 0.55 * max_n + 0.5), n_rows * 1.0 + 0.5),
    )
    if n_rows == 1:
        axes = [axes]

    for ax, (name, toks) in zip(axes, tokenizations.items()):
        plot_token_grid(toks, title=f"{name}  ({len(toks)} tokens)", ax=ax)

    if title is None:
        title = f"Tokenizations of {smiles}"
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig
