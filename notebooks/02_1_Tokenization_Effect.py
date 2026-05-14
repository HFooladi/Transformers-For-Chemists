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

# <a href="https://colab.research.google.com/github/HFooladi/Transformers-For-Chemists/blob/main/notebooks/02_1_Tokenization_Effect.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 02.1 · Tokenization Effect in Practice
#
# Notebook 02 introduced four ways to tokenize a SMILES — Char, Atom, BPE,
# SPE — and showed how they shrink the median sequence length on real
# MoleculeNet data. That's the *headline* number. This deep-dive notebook
# asks four follow-up questions a chemist actually faces when picking a
# tokenizer for a real project:
#
# 1. **Variance.** Is "median tokens per molecule" the whole story, or does
#    the spread of sequence lengths matter for batching and attention cost?
# 2. **Transfer.** What happens when a tokenizer trained on dataset A meets
#    a new dataset B — does it see anything it doesn't recognize?
# 3. **Canonicalization.** A molecule has many valid SMILES strings. Do
#    different strings of the *same* molecule produce different tokens? How
#    different?
# 4. **Interpretability.** Do BPE/SPE actually learn chemistry, or are their
#    merges just frequent strings that happen to look chemical?
#
# **Scope:** this notebook is analysis-only. No models are trained. The
# downstream-prediction story (does the tokenizer change the predicted
# log-solubility?) belongs to **notebook 07**. The job of 02.1 is to give
# you the empirical intuition for *why* the answer there will be "yes,
# noticeably."

# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# 1. Explain why **sequence-length variance** (not just the mean) drives
#    attention cost in a real batched training loop.
# 2. Measure **out-of-vocabulary (OOV) rates** when transferring a learned
#    BPE/SPE vocabulary from one chemistry dataset to another.
# 3. Demonstrate that **BPE/SPE are string-dependent** by feeding randomized
#    SMILES of the same molecule and watching the tokens shift.
# 4. Inspect the **top learned tokens** of a BPE/SPE tokenizer and verify,
#    with RDKit substructure matching, which of them correspond to real
#    chemical motifs.
# 5. Pick a tokenizer for your own project using a short **decision guide**.

# ## Setup

# +
import os
import subprocess
import sys

REPO_OWNER = "HFooladi"
REPO_NAME = "Transformers-For-Chemists"
REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"

if not any(os.path.isdir(os.path.join(p, "utils")) for p in (f"{REPO_NAME}/notebooks", "notebooks", ".", "..")):
    subprocess.run(["git", "clone", "--depth", "1", "-q", REPO_URL], check=False)

for p in (f"{REPO_NAME}/notebooks", "notebooks", ".", ".."):
    if os.path.isdir(os.path.join(p, "utils")) and p not in sys.path:
        sys.path.insert(0, p)

from utils.colab_setup import ensure_environment

ensure_environment(["rdkit", "matplotlib", "tokenizers", "mols2grid"])

# +
from collections import Counter
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")  # silence per-row parse warnings

from utils.smiles_tokenizers import (
    CharTokenizer,
    AtomTokenizer,
    BPETokenizer,
    SmilesPairTokenizer,
    SPECIAL_TOKENS,
)
from utils.preprocessing import preprocess_smiles_series
from utils.tokenization_viz import plot_token_grid
# -

# ## Load and clean three MoleculeNet datasets
#
# We reuse the same three CSVs that notebooks 01 and 02 used —
# **FreeSolv** (hydration free energies), **ESOL** (aqueous solubility),
# and **BBBP** (blood-brain barrier permeability). They're cached locally;
# if not, the cell below downloads them.

# +
_candidates = ["notebooks/data", f"{REPO_NAME}/notebooks/data", "../notebooks/data"]
_data_root = next((Path(p) for p in _candidates if Path(p).exists()), Path("notebooks/data"))
DATA_DIR = _data_root / "moleculenet"
DATA_DIR.mkdir(parents=True, exist_ok=True)

_BASE = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"
DATASETS = {
    "FreeSolv": f"{_BASE}/SAMPL.csv",
    "ESOL":     f"{_BASE}/delaney-processed.csv",
    "BBBP":     f"{_BASE}/BBBP.csv",
}

clean = {}
for name, url in DATASETS.items():
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        print(f"Downloading {name} -> {path}")
        urllib.request.urlretrieve(url, path)
    cleaned, n_unp, n_mf = preprocess_smiles_series(pd.read_csv(path)["smiles"])
    clean[name] = list(cleaned)
    print(f"  {name:8s} {len(cleaned):>4d} molecules (dropped {n_unp} unparseable, {n_mf} multi-fragment)")

UNION = clean["FreeSolv"] + clean["ESOL"] + clean["BBBP"]
print(f"\nUnion corpus: {len(UNION)} molecules")
# -

# ## Train four reference tokenizers
#
# We train one of each scheme on the union of all three datasets, with
# vocab=1000 for BPE/SPE (a realistic small-to-medium chemistry vocab).
# Every later section reuses these unless it explicitly re-trains.

# +
char_tk = CharTokenizer.from_smiles(UNION)
atom_tk = AtomTokenizer.from_smiles(UNION)

bpe_tk = BPETokenizer()
bpe_tk.train(UNION, vocab_size=1000)

spe_tk = SmilesPairTokenizer()
spe_tk.train(UNION, vocab_size=1000)

TOKENIZERS = {"Char": char_tk, "Atom": atom_tk, "BPE": bpe_tk, "SPE": spe_tk}
COLORS = {"Char": "#888888", "Atom": "#1f77b4", "BPE": "#d62728", "SPE": "#2ca02c"}

for name, tk in TOKENIZERS.items():
    print(f"  {name:5s} vocab_size = {tk.vocab_size}")
# -

# ---
# ## Section A · Sequence-length variance, not just the mean
#
# Notebook 02 reported median tokens per molecule. Median tells you the
# *typical* sequence; it tells you nothing about the *worst* sequence in a
# batch — and worst-case-per-batch is what actually costs you compute.
# Self-attention is `O(L²)` and every batch is padded to its longest
# sequence, so a few long molecules drag the whole batch's cost up.

# +
def token_count(tk, smi: str) -> int:
    return len(tk.encode(smi, add_special_tokens=True))


# Compute per-molecule token counts for every (dataset, tokenizer) cell.
lengths = {}
for ds_name, smis in clean.items():
    lengths[ds_name] = {tk_name: np.array([token_count(tk, s) for s in smis])
                        for tk_name, tk in TOKENIZERS.items()}

# Summary table: mean / median / p95 / max / std on BBBP (the largest set).
rows = []
for tk_name in TOKENIZERS:
    L = lengths["BBBP"][tk_name]
    rows.append({"tokenizer": tk_name,
                 "mean": L.mean(), "median": np.median(L),
                 "p95": np.percentile(L, 95), "max": L.max(),
                 "std": L.std()})
print("BBBP sequence-length statistics (tokens per molecule):")
print(pd.DataFrame(rows).round(1).to_string(index=False))
# -

# 💡 **Key Insight.** Look at the **std** column. Char has a standard
# deviation of ~30 tokens; BPE/SPE are ~6. That's a 5× tighter distribution.
# In a batched training loop where attention is `O(max_L²)`, a fat-tailed
# distribution like Char's makes the *worst* batches enormously expensive
# even when the median molecule is small.

# Violin plot: full distribution per (dataset, tokenizer).
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, (ds_name, by_tk) in zip(axes, lengths.items()):
    data = [by_tk[t] for t in TOKENIZERS]
    parts = ax.violinplot(data, showmedians=True, widths=0.85)
    for pc, name in zip(parts["bodies"], TOKENIZERS):
        pc.set_facecolor(COLORS[name]); pc.set_alpha(0.65)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(list(TOKENIZERS))
    ax.set_title(f"{ds_name}  (n={len(clean[ds_name])})")
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("Tokens per molecule (incl. [CLS]/[SEP])")
fig.suptitle("Full sequence-length distribution per tokenizer", y=1.02)
fig.tight_layout()
plt.show()

# The Char and Atom violins are *long* — the upper tail extends well past
# 100 tokens on BBBP, with maxima above 300. The BPE and SPE violins look
# almost like flat coins by comparison: every molecule lands within a
# narrow band, ~5–25 tokens. That tight band is what makes BPE/SPE batches
# *predictable*, not just short.

# +
# Padding-waste simulation: pretend we batch in chunks of 32 (in dataset
# order — no length-sort), pad each batch to its longest sequence, and
# measure how many of the padded cells are pad tokens.
def padding_waste(L: np.ndarray, batch_size: int = 32) -> float:
    n = len(L)
    waste = 0.0
    used = 0.0
    for i in range(0, n, batch_size):
        b = L[i:i + batch_size]
        cells = len(b) * b.max()
        waste += cells - b.sum()
        used += cells
    return waste / used


waste_rows = []
for ds_name, by_tk in lengths.items():
    waste_rows.append({"dataset": ds_name,
                       **{t: f"{padding_waste(by_tk[t]) * 100:.1f}%" for t in TOKENIZERS}})
print("Fraction of attention cells wasted on [PAD] (batch_size=32):")
print(pd.DataFrame(waste_rows).to_string(index=False))
# -

# 🧪 **Chemical Intuition.** A character tokenizer on BBBP wastes roughly
# 40 % of every batch's attention work on padding. BPE/SPE waste 20-25 %.
# In raw `FLOPs` per epoch you save *more* than the median compression
# suggests — because attention's `O(L²)` cost is dominated by the longest
# molecule in each batch, and BPE/SPE tighten the worst case dramatically.
# In production, length-sorted batching ("smart batching") cuts this
# further, but only after you've already paid for a tokenizer that doesn't
# emit 300-token outliers.

# ---
# ## Section B · OOV and cross-dataset transfer
#
# The reference tokenizers above were trained on the *union* of all three
# datasets — so they have already seen every molecule we evaluate on. In
# real production you would train your tokenizer on a large pre-training
# corpus (ChEMBL, ZINC, PubChem) and then fine-tune the model on a small
# downstream task. What happens when the downstream task has chemistry
# the tokenizer hasn't seen?
#
# We simulate this by training BPE and SPE only on **ESOL** and then
# tokenizing the *other* two datasets. Char and Atom are kept as
# references — their "vocab" is the character/atom alphabet, which
# generalizes essentially perfectly across drug-like chemistry.

# +
bpe_esol = BPETokenizer();        bpe_esol.train(clean["ESOL"], vocab_size=1000)
spe_esol = SmilesPairTokenizer(); spe_esol.train(clean["ESOL"], vocab_size=1000)
bpe_bbbp = BPETokenizer();        bpe_bbbp.train(clean["BBBP"], vocab_size=1000)
spe_bbbp = SmilesPairTokenizer(); spe_bbbp.train(clean["BBBP"], vocab_size=1000)

print("Trained four single-dataset tokenizers (vocab=1000 each).")


def unk_rates(tk, smis):
    n_tok = 0; n_unk = 0; mol_unk = 0
    for s in smis:
        toks = tk.tokenize(s)
        u = sum(1 for t in toks if t == "[UNK]")
        n_tok += len(toks); n_unk += u
        if u > 0:
            mol_unk += 1
    return n_unk / max(n_tok, 1), mol_unk / len(smis)


rows = []
for train_ds, (bpe_x, spe_x) in [("ESOL", (bpe_esol, spe_esol)),
                                  ("BBBP", (bpe_bbbp, spe_bbbp))]:
    for eval_ds in clean:
        if eval_ds == train_ds:
            continue
        for tk_name, tk in [("BPE", bpe_x), ("SPE", spe_x)]:
            tu, mu = unk_rates(tk, clean[eval_ds])
            rows.append({"trained_on": train_ds, "evaluated_on": eval_ds,
                         "tokenizer": tk_name,
                         "token_UNK_%": round(tu * 100, 2),
                         "molecule_UNK_%": round(mu * 100, 1)})
print("\nCross-dataset OOV (lower is better):")
print(pd.DataFrame(rows).to_string(index=False))
# -

# Same numbers as a bar chart for the ESOL-trained case.
fig, ax = plt.subplots(figsize=(8, 4))
sub = pd.DataFrame([r for r in rows if r["trained_on"] == "ESOL"])
x = np.arange(len(sub["evaluated_on"].unique()))
width = 0.35
for i, tk_name in enumerate(("BPE", "SPE")):
    vals = sub[sub["tokenizer"] == tk_name]["molecule_UNK_%"].values
    ax.bar(x + (i - 0.5) * width, vals, width,
           label=tk_name, color=COLORS[tk_name])
ax.set_xticks(x)
ax.set_xticklabels(sub["evaluated_on"].unique())
ax.set_ylabel("Fraction of molecules with ≥1 [UNK]  (%)")
ax.set_title("OOV when tokenizer is trained on ESOL only")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
plt.show()

# ⚠️ **Note: OOV scales with chemistry mismatch.** A BPE trained on ESOL
# (mostly small organics) hits BBBP (drug-like CNS molecules) and chokes —
# about a third of BBBP molecules contain at least one `[UNK]`. The same
# tokenizer on FreeSolv (also small organics) does fine, ~8 %. The lesson
# isn't "BPE is fragile" — it's "BPE-style vocabs encode the chemistry of
# their training set." This is exactly why production chemistry models
# train their tokenizer on the broadest, biggest corpus they can get
# (ChEMBL ~2 M molecules, ZINC ~1 B): so the vocab covers anything the
# downstream task is likely to throw at it.
#
# 💡 **Key Insight.** Char and Atom (not shown) have essentially zero OOV
# on drug-like chemistry because their "vocabulary" is the character or
# atom alphabet — finite, fully enumerable, and shared across organic
# molecules. **OOV is a BPE/SPE problem.** When you adopt a subword
# tokenizer you take on the responsibility of training it on a
# corpus broad enough that `[UNK]` is rare on your downstream task.

# ---
# ## Section C · Canonicalization sensitivity
#
# A molecule has many valid SMILES strings. RDKit returns one canonical
# form by default, but `MolToSmiles(mol, doRandom=True)` will hand back
# a fresh, random-but-valid SMILES every time. BPE and SPE operate on
# **strings**, so the same molecule fed in via different SMILES strings
# routes through different BPE merges and lands on different token
# sequences.

# +
# 5 representative molecules.
DEMO = {
    "aspirin":   "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine":  "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "indole":    "c1ccc2[nH]ccc2c1",
    "tyrosine":  "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
}


def random_smiles(canonical_smi: str, n: int = 10, seed: int = 0) -> list[str]:
    """Return up to ``n`` distinct random SMILES of the molecule, plus the canonical one."""
    mol = Chem.MolFromSmiles(canonical_smi)
    canonical = Chem.MolToSmiles(mol)
    s = {canonical}
    rng = np.random.RandomState(seed)
    for _ in range(n * 4):  # over-sample because randoms can collide
        s.add(Chem.MolToSmiles(mol, doRandom=True, canonical=False))
        if len(s) >= n + 1:
            break
    return sorted(s)


def tokenization_stability(tk, smiles_list):
    """Return (#distinct sequences, union size, intersection size) of token sets."""
    seqs = {tuple(tk.tokenize(s)) for s in smiles_list}
    sets = [set(tk.tokenize(s)) for s in smiles_list]
    union = set().union(*sets)
    inter = set.intersection(*sets) if sets else set()
    return len(seqs), len(union), len(inter)


stab_rows = []
for label, canon in DEMO.items():
    smis = random_smiles(canon, n=10)
    for tk_name, tk in TOKENIZERS.items():
        n_seq, n_union, n_inter = tokenization_stability(tk, smis)
        stab_rows.append({"molecule": label, "n_SMILES": len(smis),
                          "tokenizer": tk_name,
                          "distinct_seqs": n_seq,
                          "vocab_union": n_union,
                          "vocab_intersection": n_inter,
                          "intersection/union": f"{n_inter/n_union:.0%}"})

print("Tokenization stability over 10 randomized SMILES per molecule:")
print(pd.DataFrame(stab_rows).to_string(index=False))
# -

# Two columns matter here:
#
# - **`distinct_seqs`** — how many *different* token sequences the 11
#   SMILES produce. For all four schemes this is essentially the number
#   of unique input SMILES strings (each rearrangement gives different
#   tokens). This is bad news if your model treats sequences atomically.
# - **`intersection/union`** — of the tokens used across all 11 SMILES,
#   what fraction shows up in *every* representation? Tokens in the
#   intersection are "robust" — the model would see them regardless of
#   which SMILES it gets. A low ratio means most tokens are
#   string-specific.

# Visual: three different SMILES strings of the same aspirin molecule,
# BPE token grid for each. Each row is the same molecule.
aspirin_strings = random_smiles(DEMO["aspirin"], n=2, seed=42)[:3]
fig, axes = plt.subplots(len(aspirin_strings), 1,
                         figsize=(11, 1.2 * len(aspirin_strings) + 0.5))
for ax, smi in zip(axes, aspirin_strings):
    toks = bpe_tk.tokenize(smi)
    plot_token_grid(toks, ax=ax, max_row_width=11.0)
    ax.set_title(f"{smi}    ({len(toks)} BPE tokens)", fontsize=10, loc="left")
fig.suptitle("Three SMILES for the same aspirin molecule, three different token streams",
             y=1.02)
fig.tight_layout()
plt.show()

# 🧪 **Chemical Intuition.** A BPE/SPE tokenizer cannot see the molecule —
# it sees the *string*. Two valid SMILES of the same compound, identical
# under RDKit, can produce nearly disjoint token sequences. The practical
# consequences:
#
# - **Always canonicalize before tokenizing.** Pipe every SMILES through
#   `Chem.MolToSmiles(Chem.MolFromSmiles(s))` so each molecule has exactly
#   one representation.
# - **Or commit to augmentation.** Some pretraining recipes deliberately
#   feed random SMILES of each molecule as a form of data augmentation
#   ("SMILES randomization"). This works precisely *because* the tokens
#   change — the model is forced to learn invariances. If you go this
#   route, do it intentionally and consistently.
#
# Atom and Char tokenizers are equally string-dependent in
# `distinct_seqs`, but their tokens are smaller units (single atoms,
# single chars), so the "information" each token carries is independent
# of the larger string context. A BPE token like `c1ccc(C(=O)N` carries
# heavy contextual baggage that a `[nH]` atom token simply doesn't.

# ---
# ## Section D · Do BPE/SPE actually learn chemistry?
#
# BPE has no notion of valence, bond, or atom. It merges adjacent symbols
# by *frequency*. If a SMILES corpus is large enough, frequent strings
# happen to be chemically meaningful — so frequency-driven merges
# *rediscover* substructures. Let's verify this on our union corpus.

# +
# Token frequencies across the union corpus.
def token_frequencies(tk, smiles_list):
    c = Counter()
    for s in smiles_list:
        c.update(tk.tokenize(s))
    return c


bpe_freq = token_frequencies(bpe_tk, UNION)
spe_freq = token_frequencies(spe_tk, UNION)

top_bpe = bpe_freq.most_common(20)
top_spe = spe_freq.most_common(20)
print(f"{'rank':>4}  {'BPE token':<14} {'count':>6}    {'SPE token':<14} {'count':>6}")
for i, ((bt, bc), (st, sc)) in enumerate(zip(top_bpe, top_spe), 1):
    print(f"{i:>4}  {bt!r:<14} {bc:>6}    {st!r:<14} {sc:>6}")
# -

# Skim the top-20 of each. You should see:
#
# * **`c1ccccc1`** — benzene, present in both lists.
# * **`C(=O)`** — carbonyl.
# * **`C(=O)N`** (SPE) — amide, the single most common bond in drug
#   chemistry.
# * **`CC(C)`** — isopropyl.
# * **`Cl`** — chloride (kept as a single atom token by both).
# * **`CCCC`, `CCC`, `CC1`** — alkyl chains and ring carbons.
#
# These are exactly the substructures a medicinal chemist would
# enumerate as "things you see everywhere in drug molecules."

# +
# Hand-picked watchlist of chemically meaningful SMILES fragments.
# For each, check whether the BPE/SPE tokenizer learned the exact string
# as a token, and its frequency.
WATCHLIST = [
    ("benzene",            "c1ccccc1"),
    ("carboxylic acid",    "C(=O)O"),
    ("amide",              "C(=O)N"),
    ("ester (acetyl)",     "CC(=O)O"),
    ("pyrrolic N-H",       "[nH]"),
    ("pyridine",           "c1ccncc1"),
    ("isopropyl",          "CC(C)"),
    ("ethoxy",             "OCC"),
    ("chloride",           "Cl"),
    ("bromide",            "Br"),
    ("fluoride",           "F"),
    ("nitro",              "N(=O)=O"),
]

cov_rows = []
for label, frag in WATCHLIST:
    cov_rows.append({"motif": label, "smiles": frag,
                     "BPE_present": frag in bpe_freq, "BPE_count": bpe_freq.get(frag, 0),
                     "SPE_present": frag in spe_freq, "SPE_count": spe_freq.get(frag, 0)})
print("Did the tokenizer learn this exact chemical motif as a single token?")
print(pd.DataFrame(cov_rows).to_string(index=False))

# +
# Substructure verification: take a few high-frequency tokens that LOOK
# chemical (`c1ccccc1`, `C(=O)O`, `[nH]`, `Cl`, `CC(C)`) and check whether
# they are real chemical substructures of the molecules they appear in.
#
# A token can be a substring of the SMILES *without* corresponding to a
# substructure of the molecule (e.g. a BPE token like `c1` spans a ring
# opening only because of the surrounding context). RDKit substructure
# matching gives us the chemistry-side answer.
TO_VERIFY = ["c1ccccc1", "C(=O)O", "C(=O)N", "[nH]", "Cl", "CC(C)"]

ver_rows = []
for tok in TO_VERIFY:
    smarts_mol = Chem.MolFromSmarts(tok)
    if smarts_mol is None:
        ver_rows.append({"token": tok, "n_evaluated": "—", "match_rate_%": "n/a SMARTS"})
        continue
    # Sample molecules whose tokenization contains this token, up to 50.
    hits = []
    for s in UNION:
        if tok in bpe_tk.tokenize(s):
            hits.append(s)
        if len(hits) >= 50:
            break
    matches = sum(1 for s in hits if Chem.MolFromSmiles(s).HasSubstructMatch(smarts_mol))
    rate = matches / max(len(hits), 1)
    ver_rows.append({"token": tok, "n_evaluated": len(hits),
                     "match_rate_%": f"{rate * 100:.0f}"})

print("If a molecule's BPE tokenization contains this token, does the molecule")
print("actually contain the corresponding chemical substructure?")
print(pd.DataFrame(ver_rows).to_string(index=False))
# -

# **See it, don't just count it.** The table above says "100 % of
# molecules tokenized with `c1ccccc1` actually contain a benzene ring."
# That number is more convincing when you can scroll through the
# molecules and watch the highlighted substructure light up in each
# one. The `mols2grid` package renders RDKit molecules as a paginated
# image grid; if we pre-mark each molecule's atom indices that match
# the SMARTS, mols2grid will highlight them automatically.

# +
import mols2grid
from IPython.display import display as ip_display


def mols_with_highlight(token: str, n: int = 8) -> pd.DataFrame:
    """Find up to ``n`` corpus molecules containing the SMARTS ``token``,
    with the matching atom indices stored on each mol so that mols2grid
    will draw them highlighted.

    Returns a DataFrame with columns ``mol`` and ``SMILES`` ready to be
    passed to ``mols2grid.display(df, mol_col="mol", ...)``.
    """
    pattern = Chem.MolFromSmarts(token)
    if pattern is None:
        return pd.DataFrame()
    rows = []
    for s in UNION:
        if len(rows) >= n:
            break
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        match = mol.GetSubstructMatch(pattern)
        if match:
            # mols2grid reads this private attribute and passes the indices
            # straight to RDKit's DrawMolecule(highlightAtoms=...). See
            # mols2grid/molgrid.py:266-271.
            mol.__sssAtoms = list(match)
            rows.append({"mol": mol, "SMILES": s})
    return pd.DataFrame(rows)


# Four interpretable tokens, one mini-grid each.
for tok in ["c1ccccc1", "C(=O)O", "C(=O)N", "[nH]"]:
    df_mols = mols_with_highlight(tok, n=8)
    if df_mols.empty:
        print(f"  no matches for token {tok!r}")
        continue
    print(f"BPE token {tok!r} highlighted in {len(df_mols)} example molecules:")
    ip_display(mols2grid.display(
        df_mols,
        mol_col="mol",
        smiles_col="SMILES",
        template="static",
        n_cols=4,
        size=(180, 140),
        prerender=True,
        subset=["img", "SMILES"],
        border="1px solid #cccccc",
    ))
# -

# 🧪 **Chemical Intuition.** Each highlighted region in the grid above
# is the substructure that the BPE token was *named after*. For
# `c1ccccc1` you should see a six-membered aromatic ring lit up in every
# molecule — even when the ring is fused to another ring, BPE still
# matches it. For `C(=O)N`, the amide carbonyl-plus-nitrogen lights up.
# This is the payoff: **a single BPE token ID, when handed to the model,
# corresponds to a coherent piece of chemistry that a chemist would
# recognize on sight.** That correspondence is what makes attention
# heads downstream interpretable — when a head attends to one BPE token
# from another, it is attending across a chemical relationship, not
# just across two raw characters.

# 💡 **Key Insight.** A token like `c1ccccc1` shows up in roughly 200
# molecules in our corpus, and **every one of those molecules contains an
# actual benzene ring** (RDKit confirms). The frequency-driven BPE
# algorithm has rediscovered a piece of chemistry it was never told
# about. Compare this to a generic NLP BPE trained on English text — the
# top tokens there are `the`, `of`, `to`, which are syntactically common
# but semantically empty. On SMILES, frequency *is* chemistry, because
# SMILES is a chemistry-driven language.
#
# ⚠️ **Note.** Not every top BPE token is a clean substructure. Tokens
# like `c1` or `CC1` are *fragments* of substructures — they span a ring
# opening or part of a chain, and depend on neighboring tokens to make
# chemical sense. SPE constrains its merges to atom boundaries, so every
# SPE token is at least a valid sequence of whole atoms (even if it
# isn't a closed substructure). This is the chemistry-safety guarantee
# we paid for in notebook 02.

# ---
# ## Section E · Practical guide: which tokenizer, when?
#
# Synthesizing everything above plus notebook 02.
#
# | If your project is...                                           | Use         | Why |
# |----------------------------------------------------------------- |-------------|-----|
# | Teaching / sanity-checking / under 1 k molecules                 | **Atom**    | Trivial setup, no training, 100 % chemistry-aware, no OOV. |
# | Pretraining on millions of molecules                             | **SPE**     | Strong compression *and* atom-locked tokens. MolFormer's choice. |
# | You'll fine-tune from pretrained MolFormer / ChemBERTa / etc.    | **theirs**  | Always reuse the upstream tokenizer; retraining breaks every learned embedding. |
# | Reaction prediction (Schwaller-style)                            | **Atom-augmented BPE** | You need a special token for `>>`; see Molecular Transformer (2019). |
# | Quick prototyping where vocab size doesn't matter                | **Char**    | One-liner, no `[UNK]` risk, easy to debug. |
#
# ### Vocab-size heuristic
#
# Use a **vocab-size sweep on your own corpus** (notebook 02 has the
# template), and pick a vocab size near the inflection of the
# avg-tokens-per-molecule curve. As a rough order of magnitude:
#
# | Corpus size    | Sensible BPE/SPE vocab |
# |----------------|------------------------|
# | < 1 k          | 100-300                |
# | 1 k – 10 k     | 500-2 000              |
# | 10 k – 1 M     | 2 000-8 000            |
# | > 1 M          | 8 000-32 000           |
#
# Going much larger than this on a given corpus pushes you back into
# memorization territory (notebook 02 demonstrated this on a 61-molecule
# toy corpus, but the same pattern scales up).
#
# ### Mandatory hygiene
#
# 1. **Canonicalize** SMILES before tokenizing — or commit explicitly to
#    SMILES-augmentation training.
# 2. **Monitor the `[UNK]` rate** on every new dataset you tokenize. If
#    it's >1 % of molecules you should re-think your vocabulary.
# 3. **Train the tokenizer on a corpus at least 10× larger** than the
#    downstream data you'll fine-tune on. A tokenizer trained on the same
#    1 k molecules you'll evaluate on is overfit.

# ---
# ## Checkpoint exercises

# +
# Exercise 1
# -----------
# Train a BPE tokenizer on ESOL with vocab_size=300. Find every token in
# its vocabulary that begins with "[" but does not contain the matching
# "]" — i.e. a token that splits a bracketed atom. Print each such broken
# token along with one molecule from CORPUS that uses it.

# YOUR CODE HERE

# --- Solution (try it first, then peek) ---
# bpe_ex = BPETokenizer(); bpe_ex.train(clean["ESOL"], vocab_size=300)
# vocab = list(bpe_ex._hf.get_vocab().keys())
# broken = [t for t in vocab if "[" in t and t.count("[") != t.count("]")]
# print(f"{len(broken)} bracket-broken tokens: {broken}")
# for tok in broken:
#     for s in clean["ESOL"]:
#         if tok in bpe_ex.tokenize(s):
#             print(f"  {tok!r:8s} appears in: {s}")
#             break

# +
# Exercise 2
# -----------
# For caffeine, generate 20 randomized SMILES. Tokenize each with the
# SPE tokenizer (`spe_tk`) and count how many DISTINCT token sequences
# you get. Then count the size of the intersection of token sets. What
# does this tell you about using SMILES randomization as a data
# augmentation technique even with a chemistry-safe tokenizer?

# YOUR CODE HERE

# --- Solution ---
# caf_smiles = random_smiles(DEMO["caffeine"], n=20, seed=1)
# seqs = {tuple(spe_tk.tokenize(s)) for s in caf_smiles}
# sets = [set(spe_tk.tokenize(s)) for s in caf_smiles]
# inter = set.intersection(*sets); union = set().union(*sets)
# print(f"distinct sequences: {len(seqs)}")
# print(f"tokens in EVERY representation: {len(inter)}/{len(union)} ({100*len(inter)/len(union):.0f}%)")
# # Takeaway: a small intersection means a model trained on canonical
# # SPE tokens would see mostly unfamiliar tokens when handed a
# # randomized SMILES at inference. Conversely, training *with*
# # randomization forces the model to learn molecule-level invariance.

# +
# Exercise 3
# -----------
# Compute the standard deviation of token count per scheme on BBBP.
# Rank tokenizers by predictability (lower std = tighter distribution =
# less padding waste). Does the ranking match what you'd guess from
# notebook 02's *median* compression?

# YOUR CODE HERE

# --- Solution ---
# rank = sorted(
#     ((tk_name, lengths["BBBP"][tk_name].std()) for tk_name in TOKENIZERS),
#     key=lambda x: x[1],
# )
# for tk_name, s in rank:
#     print(f"  {tk_name:5s} std = {s:5.2f}")
# # Char tops the chart by std (~30 tokens), Atom ~20, BPE/SPE ~6.
# # The std ranking matches the median ranking, but the *gap* between
# # Char and BPE/SPE is much bigger in std (~5×) than in median (~5×).
# # Both effects compound: shorter mean AND tighter spread => much
# # cheaper attention per batch.
# -

# ---
# ## What's next
#
# Notebook 02 told you *what* the four tokenizers look like; this notebook
# told you *how* they behave when you turn them loose on real data.
# **Notebook 07** will plug a tokenizer into an actual one-block
# transformer and train it to predict a molecular property — at which
# point everything in this notebook will show up as concrete RMSE / AUC
# differences. **Notebook 09** scales tokenization to a real
# 100 k-molecule pretraining corpus and shows the regime where BPE/SPE
# really shine.
#
# 📚 **References.**
# - Li, X. & Fourches, D. (2021). *SMILES Pair Encoding.* J. Chem. Inf.
#   Model. https://doi.org/10.1021/acs.jcim.0c01127 — SPE paper.
# - Schwaller, P. et al. (2019). *Molecular Transformer.* — popularized
#   the atom-regex tokenization and the reaction-arrow handling.
# - Ross, J. et al. (2022). *MolFormer.* — production-scale SPE on 1.1 B
#   molecules; the reference point for what large-corpus tokenization
#   looks like.
# - Sennrich, R. et al. (2016). *Neural Machine Translation of Rare Words
#   with Subword Units.* — original BPE paper.
