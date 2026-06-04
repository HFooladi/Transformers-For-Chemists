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

# <a href="https://colab.research.google.com/github/HFooladi/Transformers-For-Chemists/blob/main/notebooks/04_4_Other_Position_Encodings.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 04.4 · Other Position Encodings — ALiBi & relative-position bias
#
# We now have two ways to tell a transformer about position: **add** a
# sinusoidal vector to the embeddings (notebook 03, *absolute*) and **rotate**
# Q and K (notebook 04.3, RoPE, *relative*). This notebook covers a third
# family that is, in some ways, the simplest of all: don't touch the embeddings
# or the projections at all — just **add a bias number directly to the
# attention scores**, before the softmax.
#
# Two members of that family, both still in active use:
#
# - **ALiBi** (*Attention with Linear Biases*): a *fixed*, parameter-free
#   penalty that grows with token distance. Cheap, and famous for
#   extrapolating to sequences longer than anything seen in training.
# - **Relative-position bias** (T5-style): a *learned* bias looked up per
#   relative distance.
#
# **Scope.** This closes the position-encoding thread of the 04 sub-series. We
# build both biases from scratch, visualize them, and finish with a single
# capstone figure comparing *all four* position schemes on one molecule.

# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# 1. Place every position method on one map by *where* it injects position:
#    at the embedding, inside Q/K, or onto the scores.
# 2. Build the **ALiBi** bias `−slope · |i − j|` and explain why each head gets
#    a different slope.
# 3. Show that ALiBi pulls attention toward nearby tokens, and that it is
#    defined for *any* sequence length (the extrapolation property).
# 4. Explain **relative-position bucketing** (T5) and build a learned
#    relative-position bias.
# 5. Compare absolute PE, RoPE, ALiBi, and relative bias side by side on a real
#    molecule.

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

ensure_environment(["torch", "rdkit", "matplotlib", "tokenizers"])

# +
import math

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import torch
from torch import nn

from utils.smiles_tokenizers import AtomTokenizer
from utils.transformer_blocks import TokenEmbedding
from utils.rope import RotaryPositionalEncoding
from utils.position_biases import alibi_slopes, alibi_bias, RelativePositionBias

torch.manual_seed(0)  # reproducible random projections
# -

# ---
# ## 1. A map of position encodings: *where* does position enter?
#
# Every method we've seen answers the same question — "how does attention know
# where each token sits?" — but injects the answer at a different point in the
# attention pipeline. Seeing them on one diagram makes the whole landscape
# click.

# +
fig, ax = plt.subplots(figsize=(12, 4.2))
ax.axis("off")

stages = [
    (0.5,  "token\nembedding"),
    (2.4,  "Q, K, V\nprojections"),
    (4.3,  "Q·Kᵀ\nscores"),
    (6.2,  "+ bias"),
    (7.7,  "softmax"),
    (9.2,  "× V"),
]
for x, label in stages:
    ax.add_patch(FancyBboxPatch((x, 1.3), 1.2, 0.9, boxstyle="round,pad=0.05",
                                facecolor="#eef2f7", edgecolor="black", lw=1.2))
    ax.text(x + 0.6, 1.75, label, ha="center", va="center", fontsize=8.5)
for x in [1.7, 3.6, 5.5, 7.0, 8.5]:
    ax.add_patch(FancyArrowPatch((x, 1.75), (x + 0.7, 1.75),
                                 arrowstyle="-|>", mutation_scale=12, color="grey"))

# Annotate where each method injects position.
def annotate(x, text, color):
    ax.add_patch(FancyArrowPatch((x, 0.55), (x, 1.25), arrowstyle="-|>",
                                 mutation_scale=12, color=color, lw=2))
    ax.text(x, 0.35, text, ha="center", va="top", fontsize=8.5, color=color,
            fontweight="bold")

annotate(1.1, "absolute PE\n(nb 03): ADD", "#1f77b4")
annotate(3.0, "RoPE\n(nb 04.3): ROTATE", "#2ca02c")
annotate(6.8, "ALiBi / relative\n(this nb): BIAS", "#d62728")

ax.set_xlim(0, 10.6)
ax.set_ylim(-0.6, 2.5)
ax.set_title("Three places to inject position into attention")
plt.tight_layout()
plt.show()
# -

# 💡 **Key Insight.** Absolute PE acts the earliest (on the embedding), RoPE
# acts in the middle (on Q and K), and ALiBi / relative bias act the latest —
# right on the score matrix, one number per token pair. The later you inject,
# the more directly you can express "tokens this far apart should interact this
# much," which is exactly what ALiBi and relative bias do.

# ---
# ## 2. ALiBi: penalize attention by distance
#
# ALiBi adds nothing to the embeddings and learns nothing. For a query at
# position `i` and a key at position `j`, it simply adds
#
# $$ \text{bias}(i, j) = -\,m \cdot |i - j| $$
#
# to the score, where `m` is a fixed per-head **slope**. Nearby tokens get a
# bias near 0; distant tokens get a large negative bias that softmax then
# squashes toward zero weight. Here is the bias matrix for one head:

# +
SEQ = 24
slope = 0.2
positions = torch.arange(SEQ)
distance = (positions[None, :] - positions[:, None]).abs().float()
bias_one_head = -slope * distance

fig, ax = plt.subplots(figsize=(6.8, 5.6))
im = ax.imshow(bias_one_head.numpy(), cmap="magma")
ax.set_xlabel("key position  j")
ax.set_ylabel("query position  i")
ax.set_title(f"ALiBi bias  −{slope}·|i − j|  (one head)\n0 on the diagonal, more negative farther away")
fig.colorbar(im, ax=ax, label="bias added to score")
plt.tight_layout()
plt.show()
# -

# 🧪 **Chemical Intuition.** This is a *locality prior* written directly into
# attention: "atoms close together in the SMILES string are more likely to
# matter to each other." For many molecular properties that's a sensible
# default — neighbouring atoms share bonds — and ALiBi gives the model that bias
# for free, before it has learned anything.

# ---
# ## 3. One slope per head: from local to global
#
# A single slope would force *every* head to be equally local. ALiBi instead
# gives each head its **own** slope, spaced geometrically: steep-slope heads
# see only their immediate neighbourhood, shallow-slope heads see almost the
# whole molecule. (This is why ALiBi pairs naturally with the multi-head
# attention of notebook 05.)

# +
n_heads = 8
slopes = alibi_slopes(n_heads)
dists = np.arange(0, 40)

fig, ax = plt.subplots(figsize=(7.8, 4.4))
colors = plt.cm.viridis(np.linspace(0, 1, n_heads))
for h in range(n_heads):
    ax.plot(dists, -slopes[h].item() * dists, color=colors[h],
            label=f"head {h}  (slope {slopes[h].item():.3f})")
ax.set_xlabel("token distance  |i − j|")
ax.set_ylabel("ALiBi bias added to score")
ax.set_title("Each head gets its own slope: some stay local, some see far")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# -

# ⚠️ **Note.** The slopes are a fixed geometric sequence (ratio `2^(−8/H)` for
# `H` heads) — not learned. ALiBi's whole appeal is that it adds **zero
# parameters** and **zero runtime cost** beyond an addition, yet gives the model
# a structured, multi-scale sense of distance.

# ---
# ## 4. ALiBi inside attention, on caffeine
#
# Let's add the bias to real attention scores. We build Q/K for caffeine
# (random weights, as in notebook 04), compute `Q·Kᵀ`, and compare the softmax
# **with** and **without** the ALiBi bias for one head.

# +
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
CORPUS = ["CCO", "CC(=O)Oc1ccccc1C(=O)O", CAFFEINE, "BrCCCl", "c1ccc2[nH]ccc2c1"]
tokenizer = AtomTokenizer.from_smiles(CORPUS)

caffeine_ids, _ = tokenizer.encode_batch([CAFFEINE], add_special_tokens=True)
caffeine_ids = torch.tensor(caffeine_ids)
caffeine_tokens = ["[CLS]"] + tokenizer.tokenize(CAFFEINE) + ["[SEP]"]

D_MODEL = 32
token_embedding = TokenEmbedding(tokenizer.vocab_size, D_MODEL)
emb = token_embedding(caffeine_ids)                  # (1, L, d) — no positional info yet
L = emb.size(1)

W_q = nn.Linear(D_MODEL, D_MODEL, bias=False)
W_k = nn.Linear(D_MODEL, D_MODEL, bias=False)
Q, K = W_q(emb)[0], W_k(emb)[0]
scale = 1.0 / math.sqrt(D_MODEL)
raw_scores = (Q @ K.T) * scale

alibi = alibi_bias(L, n_heads=4)                     # (4, L, L)
attn_no_pos = torch.softmax(raw_scores, dim=-1)
attn_alibi  = torch.softmax(raw_scores + alibi[1], dim=-1)   # head 1's slope

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, A, ttl in [
    (axes[0], attn_no_pos.detach(), "no position info  (Q·Kᵀ only)"),
    (axes[1], attn_alibi.detach(),  "with ALiBi bias  (pulled toward the diagonal)"),
]:
    im = ax.imshow(A.numpy(), cmap="Blues", vmin=0, vmax=float(A.max()), aspect="equal")
    ax.set_xticks(range(len(caffeine_tokens)))
    ax.set_xticklabels(caffeine_tokens, rotation=90, fontsize=6)
    ax.set_yticks(range(len(caffeine_tokens)))
    ax.set_yticklabels(caffeine_tokens, fontsize=6)
    ax.set_title(ttl)
    fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()
# -

# 💡 **Key Insight.** The ALiBi panel concentrates weight near the diagonal:
# every token now attends most strongly to its sequence neighbours, tapering off
# with distance. Notice we never gave the model a position vector — the entire
# effect comes from one subtracted ramp on the scores.

# ⚠️ **Note (extrapolation).** Because the bias is *defined by a formula*, it
# exists for any distance, including ones longer than the model ever trained on.
# A length-1000 molecule? `−m·|i−j|` is still perfectly well-defined. This is the
# property that made ALiBi famous: train short, evaluate long.

# The same slope, evaluated at a length far beyond a "training" window — no
# lookup table to run off the end of, unlike a learned position embedding.
long_bias = alibi_bias(512, n_heads=4)
print(f"ALiBi bias is defined for length 512 with no extra parameters: "
      f"{tuple(long_bias.shape)}")
print(f"bias at distance 400 (head 1): {(-alibi_slopes(4)[1] * 400).item():.2f}")

# ---
# ## 5. Relative-position bias: *learn* the distance curve (T5)
#
# ALiBi fixes the shape of the distance penalty in advance (a straight line).
# T5's **relative-position bias** instead *learns* it: a separate scalar for
# each relative distance, added to the scores. To keep the parameter count
# small, distances are grouped into **buckets** — nearby distances get their own
# bucket, far ones share log-spaced buckets (you rarely need to distinguish
# "300 apart" from "320 apart").

# +
rel_pos = torch.arange(-60, 61)
buckets = RelativePositionBias.relative_bucket(rel_pos, num_buckets=32, max_distance=128)

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(rel_pos.numpy(), buckets.numpy(), drawstyle="steps-mid", color="purple")
ax.set_xlabel("relative position  (j − i)")
ax.set_ylabel("bucket index")
ax.set_title("T5 bucketing: nearby distances get unique buckets,\nfar ones are grouped logarithmically")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# -

# Each (bucket, head) pair owns a learnable scalar. Before training those
# scalars are random — but we can still *look at* the resulting bias matrix to
# see its banded, distance-only structure (every diagonal is one bucket, so one
# value):

# +
rel_bias_module = RelativePositionBias(n_heads=4, num_buckets=32, max_distance=128)
rel_bias = rel_bias_module(L)                       # (4, L, L), random until trained

fig, ax = plt.subplots(figsize=(6.8, 5.6))
im = ax.imshow(rel_bias[0].detach().numpy(), cmap="RdBu_r", aspect="equal")
ax.set_xlabel("key position  j")
ax.set_ylabel("query position  i")
ax.set_title("Learned relative bias (head 0, random init)\nconstant along diagonals — it depends only on j − i")
fig.colorbar(im, ax=ax, shrink=0.85, label="bias")
plt.tight_layout()
plt.show()
# -

# ⚠️ **Note.** Random-initialized here, so the *values* are meaningless — but the
# *structure* is the point: like ALiBi and RoPE, the bias is constant along each
# diagonal, i.e. a pure function of relative distance. Training shapes the curve
# (T5 learns a gentle local-favouring profile, not unlike ALiBi's straight
# line).

# ---
# ## 6. Capstone: all four position schemes on one molecule
#
# Time to put the entire position-encoding story in a single figure. Same
# caffeine, same random projections — only the position mechanism changes:
#
# 1. **No position** — `Q·Kᵀ` with nothing added.
# 2. **RoPE** (04.3) — rotate Q and K.
# 3. **ALiBi** (this notebook) — subtract a distance ramp from the scores.
# 4. **Relative bias** (this notebook) — add a learned per-distance bias.

# +
# (1) no position — reuse attn_no_pos from §4.
# (2) RoPE
rope = RotaryPositionalEncoding(dim=D_MODEL, max_len=128)
Qr, Kr = rope(Q.unsqueeze(0))[0], rope(K.unsqueeze(0))[0]
attn_rope = torch.softmax((Qr @ Kr.T) * scale, dim=-1)
# (3) ALiBi — reuse attn_alibi from §4.
# (4) relative bias
attn_rel = torch.softmax(raw_scores + rel_bias[0], dim=-1)

panels = [
    ("1 · no position",      attn_no_pos),
    ("2 · RoPE (rotate Q,K)", attn_rope),
    ("3 · ALiBi (score bias)", attn_alibi),
    ("4 · relative bias (learned)", attn_rel),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 11))
for ax, (ttl, A) in zip(axes.flat, panels):
    im = ax.imshow(A.detach().numpy(), cmap="Blues", vmin=0,
                   vmax=float(A.max()), aspect="equal")
    ax.set_xticks(range(len(caffeine_tokens)))
    ax.set_xticklabels(caffeine_tokens, rotation=90, fontsize=5)
    ax.set_yticks(range(len(caffeine_tokens)))
    ax.set_yticklabels(caffeine_tokens, fontsize=5)
    ax.set_title(ttl, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle("Four ways to give attention a sense of position (caffeine, random weights)",
             fontsize=13, y=0.995)
plt.tight_layout()
plt.show()
# -

# 🔬 **Try This.** With random weights the differences are subtle, but two
# patterns already stand out: the position-aware panels (ALiBi especially) tilt
# weight toward the diagonal, while "no position" has no notion of *near* vs
# *far*. After training, each scheme expresses locality a little differently —
# RoPE and relative bias can learn non-monotonic distance profiles, while ALiBi
# is locked to a straight line. Try raising the ALiBi slope and watch the
# diagonal band tighten.

# 💡 **Key Insight (which to use?).** There's no universal winner. **MolFormer**
# chose **RoPE** (04.3) because it composes cleanly with linear attention (04.1)
# and needs no parameters or score-matrix surgery. **ALiBi** shines when you
# need length extrapolation for almost free. **Relative bias** is the most
# expressive but adds parameters and assumes you'll form the score matrix (so it
# doesn't pair with linear attention). The right choice depends on your
# molecules, your sequence lengths, and your attention implementation.

# ---
# ## Checkpoint exercises

# +
# Exercise 1 — build an ALiBi bias and check its properties
# ----------------------------------------------------------
# Using `alibi_bias(seq_len=8, n_heads=4)`, verify (i) the diagonal is exactly 0
# (a token has zero distance to itself), (ii) the matrix is symmetric for the
# bidirectional encoder form, and (iii) entries get more negative with distance.

# YOUR CODE HERE

# --- Solution (try the exercise first, then peek) ---
# b = alibi_bias(8, 4)
# print(f"diagonal all zero? {torch.allclose(b.diagonal(dim1=-2, dim2=-1), torch.zeros(4, 8))}")
# print(f"symmetric?        {torch.allclose(b, b.transpose(-1, -2))}")
# print(f"farther = more negative? {(b[0, 0, 7] < b[0, 0, 1]).item()}")

# +
# Exercise 2 — extrapolation
# ---------------------------
# A learned absolute position embedding of size (max_len, d) breaks for any
# position >= max_len. Show that ALiBi does not: build alibi_bias for a length
# LONGER than the molecules in CORPUS, and confirm it returns a finite,
# fully-defined tensor with no NaNs.

# YOUR CODE HERE

# --- Solution ---
# big = alibi_bias(600, n_heads=4)
# print(f"shape: {tuple(big.shape)}  finite? {torch.isfinite(big).all().item()}  "
#       f"any nan? {torch.isnan(big).any().item()}")

# +
# Exercise 3 — read the bucketing
# --------------------------------
# Using `RelativePositionBias.relative_bucket`, print the bucket index for the
# relative distances [0, 1, 2, 4, 8, 16, 64]. Which distances land in their own
# bucket, and where does the logarithmic grouping start to kick in?

# YOUR CODE HERE

# --- Solution ---
# d = torch.tensor([0, 1, 2, 4, 8, 16, 64])
# print(f"distances: {d.tolist()}")
# print(f"buckets:   {RelativePositionBias.relative_bucket(d, 32, 128).tolist()}")
# # Small distances map 1:1 (exact buckets); beyond max_exact they compress
# # logarithmically, so 16 and 64 are only a couple of buckets apart.

# ---
# ## What's next
#
# That completes the **04 sub-series** — the four deep-dives into how a
# MolFormer-style encoder differs from a textbook transformer:
#
# - **04.1** linear attention (`O(N)` cost),
# - **04.2** the FAVOR+/Performer feature map,
# - **04.3** rotary position embeddings,
# - **04.4** ALiBi and relative-position bias *(this notebook)*.
#
# Back on the main track, **notebook 05** runs several attention heads in
# parallel — the natural home for ALiBi's per-head slopes — and **notebook 09**
# assembles attention, RoPE, and MLM into a tiny end-to-end MolFormer.
#
# 📚 **References.**
# - Press, O. et al. (2022). *Train Short, Test Long: Attention with Linear
#   Biases Enables Input Length Extrapolation* (ALiBi).
# - Shaw, P. et al. (2018). *Self-Attention with Relative Position
#   Representations.* — the first relative-position attention.
# - Raffel, C. et al. (2020). *Exploring the Limits of Transfer Learning with a
#   Unified Text-to-Text Transformer* (T5) — the bucketed relative-position bias
#   built here.
