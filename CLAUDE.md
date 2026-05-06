# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an educational repository focused on **transformer models for chemistry**. It is a sister course to [GNNs-For-Chemists](https://github.com/HFooladi/GNNs-For-Chemists) and follows the same teaching philosophy: from-scratch implementations, chemist-first intuition, paired `.ipynb`/`.py` notebooks, rich visualizations.

The course builds toward a small **MolFormer-style, encoder-only, MLM-pretrained** model that fits in a free Colab. Causal/decoder transformers are mentioned for context but are not the focus.

## Repository Structure

- **notebooks/**: Main educational materials
  - Numbered tutorial series (01–10) covering the core transformer stack for chemistry
  - Both `.ipynb` (Jupyter) and `.py` (Jupytext-paired) versions
  - Sub-series (02.1, 04.1, 04.2, 04.3, 08.1, 09.1) for deep-dives
  - `data/` subdirectory for cached datasets (small subsets of ChEMBL, ZINC, MoleculeNet, QM9)
  - `utils/` shared educational helper modules
- **assets/**: Repository banner and images
- **docs/**: Documentation and teaching notes

## Key Dependencies

The notebooks primarily use:
- **PyTorch**: Core deep learning framework
- **RDKit**: Chemical informatics toolkit for molecular processing and SMILES handling
- **HuggingFace `tokenizers`**: BPE / WordPiece training and inference (notebook 02 onward)
- **HuggingFace `transformers`**: Reference implementation and final-notebook bridge (notebook 10)
- **HuggingFace `datasets`**: Convenient loading of ChEMBL/ZINC subsets
- **NumPy / SciPy**: Numerical computing
- **Matplotlib / Seaborn**: Static plots and attention heatmaps
- **Plotly**: Interactive plots where helpful
- **scikit-learn**: Train/test splits, metrics, baselines

Installation is done via pip cells inside notebooks (Colab-friendly):
```bash
!pip install -q torch rdkit tokenizers transformers datasets matplotlib seaborn scikit-learn
```

There is intentionally **no top-level `requirements.txt`** — each notebook installs only what it needs.

## Notebook Architecture

### Progressive Learning Structure (Core)
1. **01_From_SMILES_to_Tokens** — SMILES recap, character-level tokenization
2. **02_Subword_Tokenization** — BPE and SMILES-pair encoding
3. **03_Embeddings_and_Positions** — Token embeddings and sinusoidal positional encoding
4. **04_Self_Attention_From_Scratch** — Q/K/V scaled dot-product
5. **05_Multi_Head_Attention** — Multiple heads and head specialization
6. **06_The_Transformer_Block** — Encoder block: attention + FFN + LayerNorm + residuals
7. **07_Training_a_Property_Predictor** — Single-block transformer trained supervised
8. **08_Masked_Language_Modeling** — MLM objective on SMILES
9. **09_Tiny_MolFormer** — End-to-end pre-train + fine-tune
10. **10_HuggingFace_Reimplementation** — Same model via the HF stack

### Sub-series (0X.Y notebooks)
- **02.1**: How tokenization choice affects downstream property prediction
- **04.1**: Linear attention (MolFormer's choice)
- **04.2**: Rotary position embeddings (RoPE)
- **04.3**: Other position encodings (ALiBi, relative position)
- **08.1**: Empirical study of MLM masking ratios
- **09.1**: GNN vs encoder-transformer head-to-head on MoleculeNet

### Code Patterns
- Each notebook is self-contained: starts with a `!git clone` cell that pulls this repo and adds `notebooks/` to `sys.path`, followed by pip-install for libs that notebook needs
- From-scratch implementations prioritize clarity over efficiency
- Common pattern: SMILES loading → tokenization → batching → model definition → training → evaluation → visualization
- Exercises use TODO comments with starter code + commented solution blocks
- Dependency availability checks (`RDKIT_AVAILABLE`, `HF_AVAILABLE`, `TORCH_AVAILABLE`) enable graceful fallbacks in `utils/`

## Development Workflow

### Notebook Syncing (jupytext)
- `.ipynb` and `.py` files are paired via `jupytext.toml` (light format)
- Pre-commit hooks auto-sync on commit: `nbstripout` strips outputs, then `jupytext --sync` keeps pairs aligned
- To manually sync after editing a `.py` file: `jupytext --sync notebooks/<name>.py`
- The `.py` files use jupytext light format with `# +` / `# -` cell markers and `# + [markdown]` for markdown cells

### Running Notebooks
- Notebooks designed for Google Colab (each starts with a Colab badge link)
- First cell: `!git clone https://github.com/HFooladi/Transformers-For-Chemists.git` + `sys.path.append('Transformers-For-Chemists/notebooks')`
- Can also be run locally with Jupyter

### Data Handling
- Pre-training datasets: small ChEMBL and ZINC SMILES subsets (~100k each), fetched on demand
- Fine-tuning datasets: MoleculeNet (ESOL, BACE, BBBP, FreeSolv) and QM9
- Small data files committed to `notebooks/data/`; large files gitignored and re-downloaded

### Code Style
- Educational focus: implementations prioritize clarity over efficiency
- Extensive comments and markdown explanations
- From-scratch implementations to illustrate concepts
- PyTorch object-oriented design for models
- Standard chemical color schemes (Jmol/CPK) for any structural visualizations

## Common Tasks

### Adding a New Notebook
1. Follow the numbering scheme (NN_Title.ipynb for core, NN_M_Title.ipynb for deep-dives)
2. Include learning objectives at the top, checkpoint exercises at the end
3. First cell: Colab badge + `!git clone` + `sys.path.append` setup
4. Second cell: pip install for libs this notebook needs
5. Use `🧪 Chemical Intuition`, `💡 Key Insight`, `⚠️ Note`, `🔬 Try This` callout boxes
6. Generate both `.ipynb` and `.py` via jupytext

### Adding to `utils/`
1. Add module docstring describing the educational role
2. Wrap optional imports in try/except, set `_AVAILABLE` flags
3. All visualizations should default to publication-ready styling
4. Functions should have multi-paragraph docstrings explaining intent

### Common Patterns Across Notebooks
- **Tokenize → embed → attend → predict** pipeline visualized at every stage
- Side-by-side molecule + tokenized-form comparison views
- Attention heatmaps overlayed on the SMILES string

## Educational Focus

This repository is designed for **learning, not production use**. Code prioritizes clarity over efficiency, with from-scratch implementations and extensive comments. The end goal is that a chemist with intermediate Python can read every line of a tiny MolFormer and understand exactly what it does.

## Gotchas

- **Pre-commit hooks require staging**: Run `git add` on both `.py` and `.ipynb` files before `pre-commit run` or commit; the jupytext hook checks the git index
- **nbstripout modifies files**: The first pre-commit run strips notebook outputs, modifying the `.ipynb` — you may need to re-stage
- **Colab `git clone` is slow on first run**: Each notebook re-clones the repo, which adds ~10s of setup. Acceptable for an educational notebook
- **Tokenizers Rust dependency**: `tokenizers` requires a Rust toolchain to build from source on some platforms; the pre-built Colab wheel works fine
- **MLM masking subtlety**: The standard 80/10/10 split (mask/random/keep) is non-obvious; the masking-ratio deep-dive (08.1) is the place to explain why
