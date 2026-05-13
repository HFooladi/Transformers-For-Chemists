# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,notebooks//py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# <a href="https://colab.research.google.com/github/HFooladi/Transformers-For-Chemists/blob/main/notebooks/01_From_SMILES_to_Tokens.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 01 · From SMILES to Tokens
#
# Welcome to the first notebook in **Transformers For Chemists**! Before any
# neural network can *read* a molecule, that molecule has to become a sequence
# of integers. This notebook introduces **tokenization**: the process of
# chopping a SMILES string into discrete pieces and assigning each piece a
# unique numeric ID.
#
# We'll start with the simplest possible scheme — one token per character —
# and write the whole thing from scratch so there's no magic. Notebook 02 will
# then show why that scheme isn't enough and graduate to subword tokenizers.

# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# 1. Explain why a transformer needs a **sequence of integers**, not raw text.
# 2. Build a **character-level SMILES tokenizer** from scratch.
# 3. Construct a **vocabulary** from a small chemical corpus.
# 4. Use **special tokens** like `[PAD]`, `[CLS]`, `[SEP]`, `[MASK]`, `[UNK]` and
#    explain what each one is for.
# 5. **Encode** a molecule into IDs and **decode** it back, round-trip.
# 6. **Pad** a batch of variable-length SMILES into a rectangular tensor and
#    keep track of which positions are real with an **attention mask**.
# 7. Spot the **limitations of character-level tokenization** — the motivation
#    for the subword schemes in notebook 02.

# ## Setup
#
# The cell below clones this repository (so the `utils/` helpers become
# importable), installs the small set of packages we need, and verifies the
# environment. It's safe to re-run in any environment.

# +
import os
import sys

# Clone the repo if we don't already have access to its utils.
REPO_URL = "https://github.com/HFooladi/Transformers-For-Chemists.git"
REPO_DIR = "Transformers-For-Chemists"
if not any(os.path.isdir(os.path.join(p, "utils")) for p in ("notebooks", f"{REPO_DIR}/notebooks", ".", "..")):
    # Quiet clone; harmless if it fails (e.g. when running from inside the repo).
    os.system(f"git clone -q {REPO_URL}")

# Make `utils` importable from any reasonable working directory.
for p in (f"{REPO_DIR}/notebooks", "notebooks", ".", ".."):
    if os.path.isdir(os.path.join(p, "utils")) and p not in sys.path:
        sys.path.insert(0, p)

from utils.colab_setup import ensure_environment

ensure_environment(["rdkit", "matplotlib"])

# +
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

from utils.smiles_tokenizers import (
    CharTokenizer,
    SPECIAL_TOKENS,
    PAD_ID,
    CLS_ID,
    SEP_ID,
    MASK_ID,
)
from utils.preprocessing import preprocess_smiles_series
from utils.tokenization_viz import plot_molecule_with_tokens, plot_token_grid
# -

# ---
# ## 1. SMILES: molecules as strings
#
# A **SMILES** string (Simplified Molecular Input Line Entry System) is a
# textual encoding of a molecular graph. The chemist writes the atoms in the
# order they would walk through the molecule, with a small number of
# punctuation conventions for branches, rings, and bond orders.
#
# Here are three molecules you probably know:

# +
EXAMPLES = {
    "ethanol":    "CCO",
    "aspirin":    "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine":   "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
}

mols = [Chem.MolFromSmiles(s) for s in EXAMPLES.values()]
img = Draw.MolsToGridImage(
    mols,
    molsPerRow=3,
    subImgSize=(260, 220),
    legends=[f"{name}\n{smi}" for name, smi in EXAMPLES.items()],
)
img
# -

# 🧪 **Chemical Intuition.** A SMILES is a *graph traversal* written down
# linearly. `CCO` is ethanol because we start at one carbon, hop to the next
# carbon, then end at oxygen. `c1ccccc1` is benzene — six lowercase `c`s
# (aromatic carbons) closed into a ring with the matching digits `1...1`.
# Brackets `(...)` mark side branches. The string is short and unambiguous,
# which is exactly why it's a great input for any text-based pipeline.
#
# 💡 **Key Insight.** Even though SMILES *looks* like text, every character
# carries chemical meaning. Our job for the rest of this notebook is to turn
# that text into a sequence of integers — without losing the chemistry.

# ## 2. Why tokenize?
#
# A neural network cannot operate on Python strings. It operates on tensors of
# numbers. So we need a deterministic recipe for turning a SMILES string into
# a list of integers (a **sequence of token IDs**) and back again.
#
# The recipe has three parts:
#
# 1. **Pre-tokenize**: split the string into smaller pieces (the "tokens").
# 2. **Build a vocabulary**: map every distinct token to a unique integer ID.
# 3. **Encode** = look up IDs for a string. **Decode** = reverse.
#
# In notebook 01 we'll use the simplest possible pre-tokenization: every
# character is its own token. In notebook 02 we'll see what we lose by being
# this naïve.

# ## 3. Character tokenization, from scratch
#
# Let's write the tokenizer in three lines before reaching for any helper.

# +
# A tiny corpus to learn the vocabulary from.
corpus = [
    "CCO",                 # ethanol
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
    "Cc1ccccc1",           # toluene
    "C(C(=O)O)N",          # glycine
]

# Step 1: pre-tokenize each SMILES. For character-level, that's just list().
print("ethanol  ->", list(corpus[0]))
print("aspirin  ->", list(corpus[1]))

# Step 2: collect the unique characters.
unique_chars = sorted({ch for smi in corpus for ch in smi})
print("\nUnique characters across the corpus:")
print(unique_chars)
print(f"\n{len(unique_chars)} distinct characters total.")

# +
# Step 3: build a {char: id} mapping. Encode and decode.
vocab = {ch: i for i, ch in enumerate(unique_chars)}

def encode_naive(smiles: str) -> list[int]:
    return [vocab[ch] for ch in smiles]

def decode_naive(ids: list[int]) -> str:
    inv_vocab = {i: ch for ch, i in vocab.items()}
    return "".join(inv_vocab[i] for i in ids)

ids = encode_naive("CCO")
print("CCO encoded ->", ids)
print("decoded     ->", decode_naive(ids))
# -

# That's it — that's the whole tokenizer in three lines. It works perfectly on
# anything we trained the vocabulary on. But it has two glaring problems:
#
# 1. **Out-of-vocabulary characters crash it.** Try `encode_naive("CCBr")` —
#    `B` and `r` are not in the corpus, so the lookup fails.
# 2. **It has no concept of sequence boundaries**, padding, or "this position
#    is masked" — all of which the transformer will need.
#
# Let's fix both with the idea of *special tokens*.

# ## 4. Special tokens
#
# Every transformer training pipeline reserves a handful of token IDs for
# meta-purposes. They are not characters in any SMILES — we *choose* them. The
# canonical set we'll use throughout the course:

for tok_id, tok in enumerate(SPECIAL_TOKENS):
    print(f"  id {tok_id}  ->  {tok}")

# Their roles:
#
# | Token | Purpose | First introduced in |
# |-------|---------|---------------------|
# | `[PAD]` | Filler so all sequences in a batch are the same length. Always ID 0. | this notebook |
# | `[UNK]` | Stand-in for any character missing from the vocabulary. | this notebook |
# | `[CLS]` | "Start of sequence" marker; its final embedding is used as a whole-molecule summary in classification tasks. | notebook 07 |
# | `[SEP]` | "End of sequence" marker (or boundary between two sequences). | notebook 07 |
# | `[MASK]` | Placeholder for tokens we hide during masked language modelling. | notebook 08 |
# | `[BOS]`, `[EOS]` | Beginning/end of sequence in causal-style models. Provided for completeness. | not used in this course |
#
# ⚠️ **Note.** We pin `[PAD]` to ID 0 because many PyTorch utilities assume
# pad-id-zero by default (`nn.CrossEntropyLoss(ignore_index=0)`,
# `nn.Embedding(padding_idx=0)`, etc.). Treat the special-token IDs as
# load-bearing constants for the rest of the course.

# ## 5. A proper `CharTokenizer`
#
# The `utils/smiles_tokenizers.py` module ships a `CharTokenizer` class that
# wraps everything we built above plus the special tokens, padding, and
# unknown-character handling. Let's use it.

tokenizer = CharTokenizer.from_smiles(corpus)
print(f"vocab_size = {tokenizer.vocab_size}")
print(f"first 10 entries:")
for tok, idx in list(tokenizer.token_to_id.items())[:10]:
    print(f"  {idx:>2}  {tok!r}")

# Round-trip a molecule through the tokenizer.
smi = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
ids = tokenizer.encode(smi, add_special_tokens=True)
print(f"SMILES:  {smi}")
print(f"IDs:     {ids}")
print(f"Length:  {len(ids)} tokens")
print(f"Decoded: {tokenizer.decode(ids, skip_special_tokens=True)}")
print(f"Decoded (with specials): {tokenizer.decode(ids)}")

# 💡 **Key Insight.** The `add_special_tokens=True` flag wraps the sequence in
# `[CLS] ... [SEP]`, the way BERT does. We won't *need* these tokens until
# notebook 07 (when we build a property predictor and pool the `[CLS]`
# embedding), but it costs us nothing to include them now.

# ## 6. Visualizing tokenization
#
# A picture is worth a thousand IDs. The helper `plot_molecule_with_tokens`
# draws the molecule alongside its token grid, with each token colored by what
# it represents (atoms, bonds, brackets, ring closures, special tokens).

for name, smi in [("aspirin", EXAMPLES["aspirin"]), ("caffeine", EXAMPLES["caffeine"])]:
    tokens = tokenizer.tokenize(smi)
    fig = plot_molecule_with_tokens(smi, tokens, title=f"{name}: character-level tokenization")
    plt.show()

# 🧪 **Chemical Intuition.** Notice how each ring-closure digit (`1`, `2`)
# becomes its own orange cell, and every `(` and `)` becomes a grey cell. A
# transformer reading this sequence sees brackets and digits the same way it
# sees atoms — as positions in a sequence. It will have to *learn* that `(`
# and `)` come in matched pairs and that two `1`s on either side of a span
# mean a ring. Those are not free; they have to be discovered from data.

# ## 7. Batches: padding and the attention mask
#
# Real training batches contain many SMILES of different lengths. To stack
# them into a single tensor, we pad the short ones with `[PAD]` until every
# row is the same length, and we record an **attention mask** so the model
# knows which positions are real and which are padding.

# +
batch_smiles = ["CCO", "CC(=O)O", "c1ccccc1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
input_ids, attention_mask = tokenizer.encode_batch(batch_smiles, add_special_tokens=True)

print("Padded input IDs (rows = molecules, columns = positions):")
for smi, row in zip(batch_smiles, input_ids):
    print(f"  {smi:35s}  {row}")

print("\nAttention mask (1 = real token, 0 = padding):")
for smi, row in zip(batch_smiles, attention_mask):
    print(f"  {smi:35s}  {row}")
# -

# ⚠️ **Note.** During training, every loss term and every attention score over
# a `[PAD]` position must be **masked out** so it doesn't contribute to the
# gradient. We'll come back to this when we build the model in notebook 06,
# but it starts here, with the binary mask we just produced.

# ## 8. Limitations of character-level tokenization
#
# Character-level is the simplest scheme that works. It has two weaknesses
# that motivate everything in notebook 02.

# ### Weakness 1: multi-character atoms get split
#
# `Br` (bromine) is one atom. `Cl` (chlorine) is one atom. `[nH]` (an aromatic
# nitrogen with an explicit hydrogen) is one atom. But the character tokenizer
# splits each of them into multiple tokens.

weird = "BrCCCl"
tokens = tokenizer.tokenize(weird)
print(f"SMILES:  {weird}")
print(f"Tokens:  {tokens}")
print(f"That's {len(tokens)} tokens for what a chemist sees as 4 atoms (Br, C, C, Cl).")

# Worse, capital `B` and lowercase `b` (aromatic boron) become *the same kind
# of token* as the `B` inside `Br` — context is lost.
#
# 🔬 **Try This.** What does the tokenizer do with a SMILES that contains a
# character we never trained on (say, `P` for phosphorus)? Predict the answer,
# then run the cell.

unseen = "P(=O)(O)(O)O"  # phosphoric acid
tokens = tokenizer.tokenize(unseen)
ids = tokenizer.encode(unseen)
print(f"SMILES:  {unseen}")
print(f"Tokens:  {tokens}")
print(f"IDs:     {ids}     (1 = [UNK])")

# `P` becomes `[UNK]`. The model now has *no way* to distinguish phosphoric
# acid from any other molecule containing an unknown character. In a tiny
# corpus this happens constantly.

# ### Weakness 2: long molecules become very long sequences
#
# Self-attention costs scale **quadratically** in the sequence length: a
# 200-token sequence is 4× more expensive than a 100-token sequence. A
# character-level tokenizer is the *least efficient* possible scheme — every
# atom of every molecule contributes (at least) one token.
#
# Notebook 02 introduces **subword tokenizers** (BPE and SMILES-pair) that
# compress common substructures (e.g. `c1ccccc1` for benzene) into single
# tokens, dramatically shortening the sequence and giving the model better
# inductive biases for chemistry.

# ### 🔬 Sense of scale: how long do real molecules actually get?
#
# We've claimed character-level tokenization produces "long" sequences — but
# how long is "long" on real data? Let's tokenize three small, widely-used
# chemistry datasets and look at the actual distribution of token counts:
#
# - **FreeSolv** (≈ 640 molecules) — small organic solutes with hydration
#   free energies.
# - **ESOL** (≈ 1,128) — drug-like organics labeled with water solubility.
# - **BBBP** (≈ 2,050) — drug candidates labeled for blood-brain-barrier
#   permeability.
#
# All three are part of [MoleculeNet](https://moleculenet.org/) and are tiny
# (< 1 MB each) to download. We will do one round of **minimal preprocessing**
# first (parse with RDKit, keep the largest organic fragment, canonicalize)
# so the numbers reflect real molecules rather than salts and counterions,
# then tokenize and look at the distribution of token counts. Since
# self-attention is **O(N²)**, the *distribution* of N matters as much as
# its mean — a single long molecule in a batch dominates the compute and
# memory for that batch.

# +
# Download the three CSVs (cached locally; only fetched on first run).
import urllib.request
from pathlib import Path

import pandas as pd

# Cache under notebooks/data/moleculenet, falling back gracefully if the
# notebook is run from an unusual working directory.
_candidates = [
    "notebooks/data",
    "Transformers-For-Chemists/notebooks/data",
    "../notebooks/data",
]
_data_root = next(
    (Path(p) for p in _candidates if Path(p).exists()),
    Path("notebooks/data"),
)
DATA_DIR = _data_root / "moleculenet"
DATA_DIR.mkdir(parents=True, exist_ok=True)

_BASE = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"
DATASETS = {
    "FreeSolv": f"{_BASE}/SAMPL.csv",
    "ESOL":     f"{_BASE}/delaney-processed.csv",
    "BBBP":     f"{_BASE}/BBBP.csv",
}

dataframes = {}
for name, url in DATASETS.items():
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        print(f"Downloading {name} -> {path}")
        urllib.request.urlretrieve(url, path)
    dataframes[name] = pd.read_csv(path)
    print(f"  {name}: {len(dataframes[name])} molecules")

# Before tokenizing, two cleanup steps that any serious downstream pipeline
# would do first:
#
# 1. **Parse each SMILES with RDKit** and drop the rare row that fails — a
#    handful of BBBP entries are malformed.
# 2. **Take the largest organic fragment.** Some rows are co-crystals or
#    salts written as multi-fragment SMILES joined by `.`
#    (e.g. `[drug].[Na+].[Cl-]`); a transformer trained on those would
#    waste capacity modeling counterions. We canonicalize the result so
#    equivalent SMILES collapse to the same string.
#
# What we are **not** doing here (but you should in a production pipeline):
# tautomer canonicalization, charge normalization, de-duplication, stereo
# cleanup. See RDKit's `rdMolStandardize` module for the full toolbox.

# +
from rdkit import RDLogger

# Silence RDKit's per-row "could not parse" warnings; we count them
# explicitly below instead.
RDLogger.DisableLog("rdApp.*")


# `preprocess_smiles_series` is a tiny helper in `utils/preprocessing.py`
# that wraps the parse → largest-fragment → canonicalize recipe (also used
# in notebook 02).
clean_smiles_by_dataset = {}
for name, df in dataframes.items():
    cleaned, n_unparseable, n_multifrag = preprocess_smiles_series(df["smiles"])
    clean_smiles_by_dataset[name] = cleaned
    print(
        f"  {name}: kept {len(cleaned)}/{len(df)}  "
        f"(dropped {n_unparseable} unparseable; "
        f"{n_multifrag} were multi-fragment → largest kept)"
    )

# +
# Tokenize each *cleaned* dataset with the SAME `tokenizer` we built in
# Section 5 — same tool, real data — and gather summary statistics.
def token_lengths(smiles_series):
    return smiles_series.apply(
        lambda s: len(tokenizer.encode(s, add_special_tokens=True))
    )

lengths_by_dataset = {
    name: token_lengths(smiles) for name, smiles in clean_smiles_by_dataset.items()
}

stats = pd.DataFrame(
    {
        "n":      [len(L) for L in lengths_by_dataset.values()],
        "mean":   [round(L.mean(), 1) for L in lengths_by_dataset.values()],
        "median": [int(L.median()) for L in lengths_by_dataset.values()],
        "p95":    [int(L.quantile(0.95)) for L in lengths_by_dataset.values()],
        "max":    [int(L.max()) for L in lengths_by_dataset.values()],
    },
    index=list(lengths_by_dataset.keys()),
)
stats.index.name = "dataset"
stats

# +
# Overlay the three distributions on a single axis so the rightward shift
# from small organics to real drug candidates is obvious at a glance.
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = {"FreeSolv": "#4C72B0", "ESOL": "#DD8452", "BBBP": "#55A868"}

max_len = max(L.max() for L in lengths_by_dataset.values())
bins = range(0, int(max_len) + 5, 5)  # 5-token bins

for name, lengths in lengths_by_dataset.items():
    ax.hist(
        lengths, bins=bins, alpha=0.55, color=colors[name],
        label=f"{name} (n={len(lengths)})",
    )
    ax.axvline(lengths.median(), color=colors[name], linestyle="--", linewidth=1.2)

ax.set_xlabel("Tokens per SMILES (including [CLS] / [SEP])")
ax.set_ylabel("Number of molecules")
ax.set_title("Character-level token counts on real molecules")
ax.legend()
fig.tight_layout()
plt.show()
# -

# 💡 **Key Insight — read the medians, not the means.** Outliers skew the
# mean, so the median is the better "typical molecule" number.
#
# - **FreeSolv** typical molecule: ~14 tokens — small organics, shorter than
#   aspirin.
# - **ESOL** typical molecule: ~22 tokens — roughly aspirin-sized.
# - **BBBP** typical molecule: ~42 tokens — *nearly twice as long as
#   aspirin*, and the top 5% exceed ~100 tokens.
#
# Aspirin (Section 5) was 23 tokens. Attention compute scales
# **quadratically** in sequence length, so:
#
# - A p95 BBBP drug candidate (~101 tokens) costs **(101/23)² ≈ 19×** the
#   attention work of aspirin *per layer, per molecule*.
# - The longest cleaned BBBP entry is ~340 tokens — about **220×** the
#   attention work of aspirin. That tail is *real* chemistry (large
#   natural-product-like scaffolds with many fused rings), not the salt
#   artifact we filtered out in the preprocessing step.
#
# Exact numbers will vary slightly between runs of the dataset, but the
# headline is robust: **realistic drug-discovery data already pushes
# character-level sequences into a range where O(N²) attention starts to
# hurt meaningfully.** Notebook 02 begins paying that cost down — a subword
# tokenizer trades each character for a *piece* of chemistry, shortening
# sequences several-fold and giving the model better inductive biases in
# the process.

# 🔬 **Try This.** Pull the longest molecule out of (cleaned) BBBP and look
# at its tokens — how many of them are bare aromatic carbons (`c`) inside a
# ring? A starting snippet:
#
# ```python
# bbbp_lengths = lengths_by_dataset["BBBP"]
# longest_smi  = clean_smiles_by_dataset["BBBP"].loc[bbbp_lengths.idxmax()]
# print(f"Longest cleaned BBBP SMILES ({bbbp_lengths.max()} tokens):")
# print(longest_smi)
# print("Tokens:", tokenizer.tokenize(longest_smi))
# ```
#
# Notebook 02's BPE will collapse common runs like `c1ccccc1` (benzene) and
# other recurring substructures into single tokens — often compressing such
# SMILES by 2–4× and meaningfully cutting the attention bill.

# ---
# ## Checkpoint exercises
#
# Spend a few minutes on these before moving on. Each exercise cell ends with
# a commented-out solution under a `--- Solution ---` divider — try it
# yourself first, then peek.

# +
# Exercise 1
# -----------
# Encode the SMILES for caffeine, then decode it. Verify the decoded string
# is exactly the original (excluding [CLS] and [SEP]).

# YOUR CODE HERE

# --- Solution (try the exercise first, then peek) ---
# smi = EXAMPLES["caffeine"]
# ids = tokenizer.encode(smi, add_special_tokens=True)
# back = tokenizer.decode(ids, skip_special_tokens=True)
# assert back == smi, (back, smi)
# print("Round-trip OK")

# +
# Exercise 2
# -----------
# Build a *new* CharTokenizer from a different corpus (e.g. the four amino
# acid SMILES below). Print: total vocabulary size, the number of special
# tokens, and the number of *content* (non-special) tokens.

amino_acids = [
    "C(C(=O)O)N",                    # glycine
    "CC(C(=O)O)N",                   # alanine
    "CC(C)C(C(=O)O)N",               # valine
    "C1=CC=C(C=C1)CC(C(=O)O)N",      # phenylalanine
]

# YOUR CODE HERE

# --- Solution (try the exercise first, then peek) ---
# aa_tk = CharTokenizer.from_smiles(amino_acids)
# n_special = len(SPECIAL_TOKENS)
# print(f"vocab_size: {aa_tk.vocab_size}")
# print(f"specials:   {n_special}")
# print(f"content:    {aa_tk.vocab_size - n_special}")

# +
# Exercise 3
# -----------
# Find a SMILES (real or made up) that produces at least one [UNK] token under
# the `tokenizer` we built from the original corpus. Print the tokens to
# confirm.

# YOUR CODE HERE

# --- Solution (try the exercise first, then peek) ---
# Anything containing an unseen character works: 'PCl5', 'CSi(C)C', 'CB(O)O', ...
# weird = "CB(O)O"  # boronic acid
# print(tokenizer.tokenize(weird))
# print(tokenizer.encode(weird))
# -

# ---
# ## What's next
#
# In **notebook 02** we'll move from character-level to **subword
# tokenization**: atom-level (regex), Byte-Pair Encoding (BPE), and the
# chemistry-aware **SMILES-pair encoding** of Li & Fourches. We'll see why
# subwords give us much shorter, more chemically meaningful token sequences —
# and then in **notebook 03** we'll start turning those tokens into vectors.
#
# 📚 **References.**
# - Weininger, D. (1988). *SMILES, a chemical language and information system.*
#   J. Chem. Inf. Model. 28(1), 31-36.
# - Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional
#   Transformers for Language Understanding.* — origin of `[CLS]`/`[SEP]`/`[MASK]`.
# - Ross, J. et al. (2022). *Large-Scale Chemical Language Representations
#   Capture Molecular Structure and Properties* (MolFormer paper).
