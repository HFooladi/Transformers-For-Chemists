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

# <a href="https://colab.research.google.com/github/HFooladi/Transformers-For-Chemists/blob/main/notebooks/02_Subword_Tokenization.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 02 · Subword Tokenization
#
# Notebook 01 ended on a cliffhanger: character-level tokenization splits
# multi-character atoms (`Br`, `Cl`, `[nH]`), produces very long sequences,
# and gives the model no compositional vocabulary. This notebook fixes all
# three problems by moving from **characters** to **subwords**.
#
# We'll cover three subword schemes, in order of sophistication:
#
# 1. **Atom-level** tokenization — a single regex that respects chemistry.
# 2. **Byte-Pair Encoding (BPE)** — a learned compression algorithm; the
#    workhorse of GPT, BERT, and most modern NLP.
# 3. **SMILES-pair encoding (SPE)** — Li & Fourches' chemistry-aware
#    variant that combines BPE's compression with the atom regex's chemical
#    guard.

# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# 1. Explain the **three problems** with character-level tokenization and how
#    each subword scheme addresses them.
# 2. Use the **SMILES atom regex** to pre-tokenize a string into chemically
#    meaningful pieces.
# 3. Walk through the **BPE merge algorithm** by hand on a toy corpus.
# 4. Train a **production BPE tokenizer** on a SMILES corpus using the
#    HuggingFace `tokenizers` library.
# 5. Train a **SMILES-pair tokenizer** and explain why its merges are
#    chemically interpretable while BPE's are not.
# 6. **Compare four tokenization schemes** on the same molecule and read
#    their tradeoffs (vocabulary size vs. average sequence length).

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

ensure_environment(["rdkit", "matplotlib", "tokenizers"])

# +
from collections import Counter

import matplotlib.pyplot as plt
from rdkit import Chem

from utils.smiles_tokenizers import (
    SMILES_ATOM_REGEX,
    atom_tokenize,
    CharTokenizer,
    AtomTokenizer,
    BPETokenizer,
    SmilesPairTokenizer,
)
from utils.preprocessing import preprocess_smiles_series
from utils.tokenization_viz import (
    plot_token_grid,
    plot_molecule_with_tokens,
    compare_tokenizations,
)
# -

# ## A working corpus
#
# Subword schemes only show their power on a corpus with some variety. We'll
# use a hand-picked set of ~60 small drug-like molecules — varied scaffolds,
# halogens, heteroatoms, rings of different sizes. This is small enough to
# train on instantly, big enough that the algorithms have something to chew
# on. Notebook 09 will swap this for a real ChEMBL/ZINC subset.

CORPUS = [
    # Common drugs
    "CC(=O)Oc1ccccc1C(=O)O",                                      # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",                               # caffeine
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",                                 # ibuprofen
    "CN(C)CCC=C1c2ccccc2CCc2ccccc21",                             # amitriptyline
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",                                 # caffeine (alt SMILES)
    "CC(=O)Nc1ccc(O)cc1",                                         # paracetamol
    "OC(=O)Cc1ccc(N)cc1",                                         # 4-aminophenylacetic acid
    "Clc1ccc(C(=O)NC2CC2)cc1",                                    # cyclopropyl benzamide w/ Cl
    "CCN(CC)CCNC(=O)c1cc(Cl)c(N)cc1OC",                           # metoclopramide
    "Brc1ccc(N)cc1",                                              # 4-bromoaniline
    "Fc1ccc(cc1)C(=O)NCCc1c[nH]c2ccccc12",                        # tryptamine derivative w/ F
    "ClC(=C)Cl",                                                  # 1,1-dichloroethylene
    "OC(=O)CC(N)Cc1ccc(O)cc1",                                    # tyrosine
    "OC(=O)C(N)Cc1c[nH]c2ccccc12",                                # tryptophan
    "OC(=O)C(N)Cc1ccccc1",                                        # phenylalanine
    "C(C(=O)O)N",                                                 # glycine
    "CC(C(=O)O)N",                                                # alanine
    "CC(C)C(C(=O)O)N",                                            # valine
    "OC(=O)C(N)CCSC",                                             # methionine
    "OC(=O)C(N)CO",                                               # serine
    # Aromatic & heterocyclic scaffolds
    "c1ccc2[nH]ccc2c1",                                           # indole
    "c1ccc2c(c1)oc1ccccc12",                                      # dibenzofuran
    "c1ccc2c(c1)sc1ccccc12",                                      # dibenzothiophene
    "c1ccc2nc3ccccc3nc2c1",                                       # phenazine
    "c1ccc2[nH]c3ccccc3c2c1",                                     # carbazole
    "n1ccccc1",                                                   # pyridine
    "c1cnc2[nH]ccc2c1",                                           # pyrrolopyridine
    "Cc1ccc(O)cc1",                                               # cresol
    "Cc1ccc(N)cc1",                                               # toluidine
    "Cc1ccccc1C",                                                 # ortho-xylene
    "CCc1ccccc1",                                                 # ethylbenzene
    "C1CCCCC1",                                                   # cyclohexane
    "C1CCNCC1",                                                   # piperidine
    "C1CCOCC1",                                                   # tetrahydropyran
    "C1COCCN1",                                                   # morpholine
    "C1=CC=CC=C1",                                                # Kekulé benzene
    "c1ccncc1",                                                   # pyridine (alt)
    "c1ccsc1",                                                    # thiophene
    "c1ccoc1",                                                    # furan
    "c1cc[nH]c1",                                                 # pyrrole
    # Functional groups & small acids/amines
    "OCC(O)CO",                                                   # glycerol
    "CC(O)C(=O)O",                                                # lactic acid
    "OC(=O)C(=O)O",                                               # oxalic acid
    "OC(=O)CCC(=O)O",                                             # succinic acid
    "NCCN",                                                       # ethylenediamine
    "NCCO",                                                       # ethanolamine
    "NCCCN",                                                      # 1,3-diaminopropane
    "C(=O)O",                                                     # formic acid
    "CC(=O)O",                                                    # acetic acid
    "CCC(=O)O",                                                   # propionic acid
    "CCCC(=O)O",                                                  # butyric acid
    "CCCCC(=O)O",                                                 # pentanoic acid
    "CCCCCC(=O)O",                                                # hexanoic acid
    "CCO",                                                        # ethanol
    "CCCO",                                                       # propanol
    "CCCCO",                                                      # butanol
    "CC(C)O",                                                     # isopropanol
    "CC(C)(C)O",                                                  # tert-butanol
    "CCN",                                                        # ethylamine
    "CCNC",                                                       # N-methylethylamine
    "CCN(C)C",                                                    # N,N-dimethylethylamine
]
print(f"Corpus size: {len(CORPUS)} SMILES")

# ## 1. Atom-level tokenization
#
# The simplest fix to character-level is to keep multi-character atoms
# together. We do this with a regex — `SMILES_ATOM_REGEX` in
# `utils/smiles_tokenizers.py` — that matches one of: a bracketed atom
# (`[nH]`, `[O-]`), a two-letter atom (`Br`, `Cl`), a one-letter atom (`C`,
# `N`, `O`, ...), a bond symbol (`=`, `#`), a bracket (`(`, `)`), or a ring
# closure digit.

# +
print("The atom regex (truncated for display):")
print(SMILES_ATOM_REGEX[:80] + " ...")
print()

# Try it on the molecules that broke character tokenization in notebook 01.
for smi in ["BrCCCl", "c1ccc2[nH]ccc2c1", "Fc1ccc(C(=O)O)cc1", "CC(=O)O"]:
    print(f"  {smi:30s} -> {atom_tokenize(smi)}")
# -

# 🧪 **Chemical Intuition.** The regex encodes one piece of chemistry: **an
# atom is the smallest unit a chemist would name aloud.** `Br` is one atom,
# not "B then r". `[nH]` is an aromatic nitrogen with an explicit hydrogen,
# not five separate characters. The model receives one ID per atom, which is
# both shorter and chemically correct.

# +
# Wrap it in the AtomTokenizer class (same minimal API as CharTokenizer).
atom_tk = AtomTokenizer.from_smiles(CORPUS)
print(f"AtomTokenizer vocab size: {atom_tk.vocab_size}")

aspirin = "CC(=O)Oc1ccccc1C(=O)O"
ids = atom_tk.encode(aspirin, add_special_tokens=True)
print(f"\naspirin atom-tokens: {atom_tk.tokenize(aspirin)}")
print(f"aspirin IDs:        {ids}")
print(f"round-trip:         {atom_tk.decode(ids, skip_special_tokens=True)}")


# -

# ## 2. Byte-Pair Encoding (BPE), from scratch
#
# Atom-level fixes the multi-character atom problem but doesn't shorten
# sequences. **BPE** does both. The idea is dead simple:
#
# > Start with characters. Find the most common adjacent pair. Merge it into
# > a new symbol. Repeat until the vocabulary is the size you want.
#
# After many merges, common substructures (`c1ccccc1` for benzene,
# `C(=O)O` for carboxylic acid, `CC(C)` for isopropyl) become single tokens.

# Let's run the algorithm by hand on a tiny corpus so we can see exactly what
# it does. Treat each word as a sequence of characters separated by `▁` so
# we can see the merges happen.

# +
def get_pair_counts(words: dict[tuple[str, ...], int]) -> Counter:
    """Count adjacent symbol pairs across all words."""
    pairs = Counter()
    for word, freq in words.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def merge_pair(words: dict[tuple[str, ...], int], pair: tuple[str, str]) -> dict:
    """Replace every occurrence of ``pair`` in every word with the merged symbol."""
    merged_symbol = pair[0] + pair[1]
    new_words = {}
    for word, freq in words.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(merged_symbol)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_words[tuple(new_word)] = freq
    return new_words


# Toy corpus: three "words" with character-level breakdowns.
toy = {
    tuple("CCO"): 5,                    # ethanol, seen 5 times
    tuple("CCC"): 3,                    # propane, seen 3 times
    tuple("CC(=O)O"): 4,                # acetic acid, seen 4 times
}

print("Initial words (character-level):")
for w, f in toy.items():
    print(f"  freq={f:>2}  {' '.join(w)}")

# Run 4 merges and watch the most common pair get absorbed each round.
for step in range(1, 5):
    pairs = get_pair_counts(toy)
    best_pair, best_count = pairs.most_common(1)[0]
    toy = merge_pair(toy, best_pair)
    print(f"\nMerge {step}: combine {best_pair[0]!r} + {best_pair[1]!r} (count={best_count}) -> {best_pair[0] + best_pair[1]!r}")
    for w, f in toy.items():
        print(f"  freq={f:>2}  {' '.join(w)}")
# -

# 💡 **Key Insight.** BPE doesn't know any chemistry. It only knows
# *frequency*. The first merge always grabs the single most common adjacent
# pair, and it just happens that in SMILES corpora those tend to be things
# like `CC`, `(=`, `=O`, `)O`, `c1` — which read as *partial* chemistry
# (carbon chains, carbonyls, ring openings). The algorithm rediscovers
# chemistry from frequency alone.
#
# ⚠️ **Note.** BPE is allowed to merge across atom boundaries. It might
# happily merge `[` + `n` into `[n` even though that's chemically nonsense —
# `[n` is a fragment of `[nH]`. Notebook section 4 (SMILES-pair) fixes that.

# ## 3. BPE in production: the `tokenizers` library
#
# We don't actually run our hand-coded BPE in real training. The
# HuggingFace `tokenizers` library implements the same algorithm in Rust,
# trains in milliseconds, and handles all the edge cases. The
# `BPETokenizer` class in `utils/smiles_tokenizers.py` is a thin wrapper.

# +
bpe = BPETokenizer()
bpe.train(CORPUS, vocab_size=100)
print(f"BPE vocab size: {bpe.vocab_size}")

print("\nA few aspirin/caffeine/ibuprofen tokenizations:")
for smi in [
    "CC(=O)Oc1ccccc1C(=O)O",                  # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",           # caffeine
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",             # ibuprofen
]:
    toks = bpe.tokenize(smi)
    print(f"  {smi:35s} -> {len(toks):2d} tokens: {toks}")
# -

# Round-trip a molecule to make sure BPE is lossless.
smi = "CC(=O)Oc1ccccc1C(=O)O"
ids = bpe.encode(smi, add_special_tokens=True)
print(f"SMILES:  {smi}")
print(f"IDs:     {ids}  ({len(ids)} tokens including [CLS]/[SEP])")
print(f"Decoded: {bpe.decode(ids, skip_special_tokens=True)}")

# 🔬 **Try This.** Look at the BPE tokens for aspirin. Do they include
# `c1ccccc1` (benzene) as a single token? Do they include `C(=O)O`
# (carboxylic acid)? With our small 61-molecule corpus and vocab of 100, BPE
# can already find a handful of meaningful chunks — and it gets dramatically
# better with real corpora (notebook 09 uses 100k+ SMILES).

# ## 4. SMILES-pair encoding (SPE)
#
# BPE's only flaw: it doesn't know that `[` and `n` belong together inside
# `[nH]`. **SMILES-pair encoding** (Li & Fourches, 2021) fixes this by
# pre-tokenizing the input with the atom regex *before* running BPE merges.
# The merge candidates are atom tokens, not characters — so any merged
# fragment is automatically a sequence of *whole atoms*.

# +
spe = SmilesPairTokenizer()
spe.train(CORPUS, vocab_size=100)
print(f"SPE vocab size: {spe.vocab_size}")

# The same molecules, now tokenized chemically.
print("\nSPE tokenizations:")
for smi in [
    "CC(=O)Oc1ccccc1C(=O)O",                  # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",           # caffeine
    "c1ccc2[nH]ccc2c1",                       # indole
]:
    toks = spe.tokenize(smi)
    print(f"  {smi:35s} -> {len(toks):2d} tokens: {toks}")
# -

# Notice what happens for indole (`c1ccc2[nH]ccc2c1`): SPE can never split
# `[nH]` into `[`, `n`, `H`, `]` — those are inside an atomic unit, locked
# together by the regex. BPE made no such promise.

# ## 5. Side-by-side comparison
#
# A single-molecule view to drive the differences home.

# +
char_tk = CharTokenizer.from_smiles(CORPUS)

target_smiles = "CC(=O)Oc1ccccc1C(=O)O"   # aspirin
tokenizations = {
    f"Char  ({char_tk.vocab_size}-vocab)": char_tk.tokenize(target_smiles),
    f"Atom  ({atom_tk.vocab_size}-vocab)": atom_tk.tokenize(target_smiles),
    f"BPE   ({bpe.vocab_size}-vocab)":     bpe.tokenize(target_smiles),
    f"SPE   ({spe.vocab_size}-vocab)":     spe.tokenize(target_smiles),
}
fig = compare_tokenizations(target_smiles, tokenizations)
plt.show()
# -

# 💡 **Key Insight.** The four schemes form a clean spectrum:
#
# | Scheme | Vocab size | Sequence length | Chemistry-aware? |
# |--------|------------|-----------------|------------------|
# | Character | smallest | longest         | no |
# | Atom      | small    | shorter         | yes |
# | BPE       | medium-large | shortest    | partial (frequency only) |
# | SPE       | medium-large | short        | yes (atom-locked) |

# ## 6. The vocabulary-size tradeoff
#
# Bigger vocabulary = shorter sequences = cheaper attention. But bigger
# vocabulary also means more embedding parameters and rarer tokens (each one
# seen less often during training, harder to learn). Let's plot the curve.

# +
SAMPLE_FRACTION = 1.0   # use all of CORPUS
vocab_sizes = [50, 100, 200, 400, 800]

bpe_lens = []
spe_lens = []
for v in vocab_sizes:
    b = BPETokenizer(); b.train(CORPUS, vocab_size=v)
    s = SmilesPairTokenizer(); s.train(CORPUS, vocab_size=v)
    bpe_lens.append(sum(len(b.tokenize(s_)) for s_ in CORPUS) / len(CORPUS))
    spe_lens.append(sum(len(s.tokenize(s_)) for s_ in CORPUS) / len(CORPUS))

# +
char_avg = sum(len(char_tk.tokenize(s)) for s in CORPUS) / len(CORPUS)
atom_avg = sum(len(atom_tk.tokenize(s)) for s in CORPUS) / len(CORPUS)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.axhline(char_avg, ls="--", color="gray", label=f"Char (vocab={char_tk.vocab_size}): {char_avg:.1f}")
ax.axhline(atom_avg, ls=":", color="black", label=f"Atom (vocab={atom_tk.vocab_size}): {atom_avg:.1f}")
ax.plot(vocab_sizes, bpe_lens, "o-", label="BPE",  color="#cc6699")
ax.plot(vocab_sizes, spe_lens, "s-", label="SPE",  color="#669966")
ax.set_xscale("log")
ax.set_xlabel("Vocabulary size")
ax.set_ylabel("Average tokens per molecule")
ax.set_title("Bigger vocabulary → shorter sequences (on the toy corpus)")
ax.legend()
ax.grid(alpha=0.3)
plt.show()
# -

# ⚠️ **Limitation: tokenizer overfitting on tiny corpora.** The shape of
# this curve is what matters; the absolute numbers depend heavily on
# corpus size. With our 61-molecule toy corpus, BPE runs out of
# cross-molecule patterns to merge once the vocabulary grows past ~100
# and starts memorizing whole training SMILES as private tokens. At
# `vocab_size = 300` the trainer hits a hard ceiling at 246 tokens and
# **each of the 61 training molecules collapses to exactly one token** —
# that's not learning, it's a lookup table. This is why we kept
# `vocab_size = 100` for both BPE and SPE earlier: small enough to stay
# in the honest subword regime, where learned tokens are still
# recognizable substructures (`c1ccccc1`, `C(=O)O`, `[nH]`) rather than
# memorized molecules. Real chemistry tokenizers (ChemBERTa: 77 M SMILES;
# MolFormer: 1.1 B) need enormous corpora precisely so the merges
# represent genuinely reusable substructures. On a real 100 k-molecule
# corpus (notebook 09), a 1000-token vocabulary cuts average sequence
# length by **5–10×** *and* the merges remain meaningful.

# ## 7. Why this matters for the transformer
#
# Self-attention costs $O(L^2)$ where $L$ is the sequence length. Halving
# the average sequence length (BPE/SPE vs character) makes attention 4×
# cheaper. For the same compute budget, you can:
#
# - process 4× longer molecules, or
# - train 4× more steps, or
# - use 4× wider hidden dimensions.
#
# This is why every modern chemical language model — MolFormer, ChemBERTa,
# MolBERT, MolGPT — uses subword tokenization. Notebook 09 will pre-train a
# tiny MolFormer using SPE for exactly this reason.

# ## 8. Sense of scale revisited: how much does subword tokenization save?
#
# Notebook 01 made a promise: subword tokenizers trade each individual
# character for a *piece* of chemistry, shortening sequences several-fold.
# Let's check that empirically by running the same three MoleculeNet
# datasets — **FreeSolv**, **ESOL**, **BBBP** — through all four
# tokenization schemes and comparing the distributions of token counts.
#
# **One important caveat first.** The `bpe` and `spe` tokenizers trained
# earlier in this notebook only saw the 61-molecule toy corpus — their
# vocabularies are far too small to fairly handle ~3 800 real drug
# candidates. We will therefore train a fresh pair of BPE and SPE
# tokenizers on the *cleaned* MoleculeNet data itself (under the names
# `bpe_big` and `spe_big`). The existing `bpe`/`spe` stay in scope under
# their original names. Notebook 09 will scale the training corpus up to
# 100k+ molecules; this is the educational halfway point.

# +
# Same MoleculeNet datasets as notebook 01 — small drug-discovery CSVs,
# cached locally so this cell is fast on re-run.
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # silence per-row "could not parse" warnings

_candidates = [
    "notebooks/data",
    f"{REPO_NAME}/notebooks/data",
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

clean_smiles_by_dataset = {}
for name, url in DATASETS.items():
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        print(f"Downloading {name} -> {path}")
        urllib.request.urlretrieve(url, path)
    df = pd.read_csv(path)
    cleaned, n_unp, n_mf = preprocess_smiles_series(df["smiles"])
    clean_smiles_by_dataset[name] = cleaned
    print(
        f"  {name}: kept {len(cleaned)}/{len(df)} "
        f"(dropped {n_unp} unparseable; {n_mf} multi-fragment → largest kept)"
    )

# +
# Train new BPE and SPE on the union of the three cleaned datasets so the
# tokenizers' vocabularies are appropriate to the chemistry we are
# evaluating on. Also build a Char and Atom tokenizer over the same corpus
# so all four schemes see the same character set.
training_corpus = (
    list(clean_smiles_by_dataset["FreeSolv"])
    + list(clean_smiles_by_dataset["ESOL"])
    + list(clean_smiles_by_dataset["BBBP"])
)
print(f"Training corpus: {len(training_corpus)} molecules")

char_big = CharTokenizer.from_smiles(training_corpus)
atom_big = AtomTokenizer.from_smiles(training_corpus)

bpe_big = BPETokenizer()
bpe_big.train(training_corpus, vocab_size=1000)

spe_big = SmilesPairTokenizer()
spe_big.train(training_corpus, vocab_size=1000)

print(f"  char_big: vocab_size = {char_big.vocab_size}")
print(f"  atom_big: vocab_size = {atom_big.vocab_size}")
print(f"  bpe_big : vocab_size = {bpe_big.vocab_size}")
print(f"  spe_big : vocab_size = {spe_big.vocab_size}")

# +
# Tokenize every cleaned molecule with every scheme. ~15k tokenizations
# total — still sub-second.
tokenizers_big = {
    "Char": char_big,
    "Atom": atom_big,
    "BPE":  bpe_big,
    "SPE":  spe_big,
}


def token_lengths(tk, smiles):
    return smiles.apply(lambda s: len(tk.encode(s, add_special_tokens=True)))


# Per (dataset, tokenizer): median token count.
medians = pd.DataFrame(
    {
        tk_name: [int(token_lengths(tk, clean_smiles_by_dataset[ds]).median())
                  for ds in clean_smiles_by_dataset]
        for tk_name, tk in tokenizers_big.items()
    },
    index=list(clean_smiles_by_dataset.keys()),
)
medians.index.name = "dataset"
print("Median tokens per molecule (with [CLS] / [SEP]):")
print(medians)

# Compression ratio = Char median / scheme median. Bigger is better. The
# attention-cost saving per layer is this ratio *squared*.
compression = pd.DataFrame(
    {
        "Char/Atom": (medians["Char"] / medians["Atom"]).round(2),
        "Char/BPE":  (medians["Char"] / medians["BPE"]).round(2),
        "Char/SPE":  (medians["Char"] / medians["SPE"]).round(2),
    },
    index=medians.index,
)
print("\nCompression ratio vs character tokenizer (median):")
print(compression)

# +
# Clustered bar chart: x = dataset, y = median tokens, color = tokenizer.
fig, ax = plt.subplots(figsize=(9, 4.5))
colors = {"Char": "#888888", "Atom": "#1f77b4", "BPE": "#d62728", "SPE": "#2ca02c"}
x = np.arange(len(medians))
width = 0.2

for i, tk_name in enumerate(("Char", "Atom", "BPE", "SPE")):
    ax.bar(
        x + (i - 1.5) * width,
        medians[tk_name],
        width,
        label=f"{tk_name} (vocab {tokenizers_big[tk_name].vocab_size})",
        color=colors[tk_name],
    )

ax.set_xticks(x)
ax.set_xticklabels(medians.index)
ax.set_ylabel("Median tokens per molecule")
ax.set_title("Median sequence length by tokenizer (cleaned MoleculeNet)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
plt.show()
# -

# 💡 **Key Insight — subword tokenization delivers on notebook 01's
# promise.** On BBBP (most realistic of the three):
#
# | Scheme | Median tokens | Compression vs Char | Attention saving (≈²) |
# |--------|--------------:|--------------------:|----------------------:|
# | Char   | 42            | 1.0×                | 1×                    |
# | Atom   | 39            | 1.08×               | 1.2×                  |
# | BPE    | 9             | 4.7×                | **~22×**              |
# | SPE    | 9             | 4.7×                | **~22×**              |
#
# **BPE and SPE both win big on raw compression** — a 1000-token vocabulary
# lets them fuse common drug-like substrings (rings, amide bonds, aliphatic
# chains) into single tokens, shrinking the median BBBP sequence from 42
# down to 9. Since attention cost is O(N²), that's roughly **22× less
# attention work per molecule per layer**.
#
# **BPE vs SPE: identical compression, different safety profile.** Both
# reach the same median sequence length on real chemistry, but where BPE
# can in principle learn a merge that splits a bracketed atom like `[nH]`
# (the merge `[n` + `H]` would be legal if frequent enough — see notebook
# section 4), **SPE guarantees by construction that no token ever crosses
# an atom boundary in a chemistry-meaningless way**. Try it: every SPE
# token in your output above is a sequence of *whole atoms*.
#
# **Atom is barely better than Char** — most SMILES characters are already
# single atoms; the regex only helps for `Br`, `Cl`, bracketed atoms, and
# the like.
#
# ⚠️ **A note on fair comparison.** BPE and SPE were trained on the same
# ~3 800 molecules we evaluated on, so the compression numbers above are
# an upper bound — the *shape* of the comparison (BPE ≈ SPE ≫ Atom > Char)
# is the durable takeaway. In a production setting you would train on a
# much larger, *separate* corpus (ChEMBL/ZINC/PubChem) and then evaluate
# on held-out task data. Notebook 09 will revisit this with a 100k+
# pre-training corpus.

# ---
# ## Checkpoint exercises

# +
# Exercise 1
# -----------
# Train a BPE tokenizer with vocab_size=100 on CORPUS. Find any token that
# crosses an atom boundary in a chemically meaningless way (e.g. starts with
# "[" but doesn't include the matching "]"). Print it.

# YOUR CODE HERE

# --- Solution (try the exercise first, then peek) ---
# bpe_small = BPETokenizer(); bpe_small.train(CORPUS, vocab_size=100)
# vocab = list(bpe_small._hf.get_vocab().keys())
# bad = [t for t in vocab if t.count("[") != t.count("]")]
# print(f"{len(bad)} tokens with mismatched brackets:")
# for t in bad[:5]:
#     print(f"  {t!r}")

# +
# Exercise 2
# -----------
# Take the SPE tokenizer trained earlier. For each molecule in CORPUS,
# compute the compression ratio = char_tokens / spe_tokens. Print the 5
# molecules with the highest compression (i.e. SPE saves the most over
# character).

# YOUR CODE HERE

# --- Solution (try the exercise first, then peek) ---
# ratios = []
# for smi in CORPUS:
#     n_char = len(char_tk.tokenize(smi))
#     n_spe  = len(spe.tokenize(smi))
#     ratios.append((n_char / n_spe, smi))
# ratios.sort(reverse=True)
# for r, smi in ratios[:5]:
#     print(f"  {r:.2f}x   {smi}")

# +
# Exercise 3
# -----------
# A single SMILES can be written in many equivalent ways (different atom
# orderings). Take aspirin written two ways:
#    "CC(=O)Oc1ccccc1C(=O)O"
#    "O=C(C)Oc1ccccc1C(=O)O"
# Pass each through your trained SPE tokenizer. Are the token sequences the
# same? Why or why not?

# YOUR CODE HERE

# --- Solution ---
# Different SMILES, even if equivalent molecules, give different token
# sequences. SPE (and BPE) operate on the string, not the molecule. To get
# canonical tokenization, canonicalize the SMILES first via RDKit:
#    smi_canon = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
# Then tokenize. We'll do this routinely in notebook 07+.
# -

# ---
# ## What's next
#
# Now that we can chop a SMILES into a sequence of meaningful integer IDs,
# **notebook 03** turns those IDs into vectors — **token embeddings** — and
# adds **positional information** so the transformer knows which token came
# first. Then in **notebook 04** we build the centerpiece: **self-attention**.
#
# 📚 **References.**
# - Sennrich, R. et al. (2016). *Neural Machine Translation of Rare Words
#   with Subword Units.* — original BPE paper.
# - Schwaller, P. et al. (2019). *Molecular Transformer: A Model for
#   Uncertainty-Calibrated Chemical Reaction Prediction.* — popularized
#   the atom-regex tokenization used here for `AtomTokenizer`.
# - Li, X. & Fourches, D. (2021). *SMILES Pair Encoding: A Data-Driven
#   Substructure Tokenization Algorithm for Deep Learning.* J. Chem. Inf.
#   Model. https://doi.org/10.1021/acs.jcim.0c01127 — original SPE paper.
# - Ross, J. et al. (2022). *MolFormer: Large-Scale Chemical Language
#   Representations Capture Molecular Structure and Properties.*
