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

# <a href="https://colab.research.google.com/github/HFooladi/Transformers-For-Chemists/blob/main/notebooks/04_Self_Attention_From_Scratch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 04 · Self-Attention From Scratch
#
# Notebook 03 left us with a `(batch, seq_len, d_model)` tensor: every token
# carries its content (which atom or bond it is) and its position (where in
# the SMILES string it sits). What it *cannot* do — yet — is **look at the
# other tokens**. The carbonyl carbon in caffeine has no idea there's a
# nitrogen two tokens away; the ring-closure digit `1` doesn't yet know
# which atom on the other side of the ring it pairs with.
#
# This notebook builds **self-attention** — the operation that finally lets
# every token look at every other token and decide who matters. We start
# from a single dot product, build up to scaled dot-product attention with
# masking, and end with a clean `nn.Module` we'll reuse for the rest of the
# course.

# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# 1. Explain why per-token vectors with content + position aren't enough —
#    the tokens still need a way to **exchange information**.
# 2. Build the **query / key / value** projections from scratch and explain
#    what each one does.
# 3. Compute the unnormalized attention scores `QKᵀ` and visualize them as
#    a heatmap on a real molecule (caffeine).
# 4. Justify the `1/√d_k` scaling factor by measuring how the variance of
#    raw scores grows with `d_k`.
# 5. Apply softmax and the value projection to produce the final attention
#    output.
# 6. Build a **padding mask** and verify it stops attention from leaking
#    onto `[PAD]` tokens.
# 7. Wrap everything into a reusable `ScaledDotProductAttention` module and
#    sanity-check it against the hand-rolled version.

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
import numpy as np
import torch
from torch import nn

from utils.smiles_tokenizers import AtomTokenizer
from utils.transformer_blocks import (
    ScaledDotProductAttention,
    SinusoidalPositionalEncoding,
    TokenEmbedding,
)
from utils.tokenization_viz import plot_molecule_with_tokens
from utils.attention_viz import plot_attention_heatmap, plot_attention_on_smiles

torch.manual_seed(0)  # so all the random projections below are reproducible
# -

# ---
# ## 1. From "stamped vectors" to talking tokens
#
# We pick up exactly where notebook 03 left off: tokenize a molecule, embed
# it, add positional encoding. **Caffeine** is our hero molecule for this
# notebook (it has three nitrogens, two carbonyls, and two ring closures —
# plenty of structure for an attention pattern to chew on).

# +
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
CORPUS = [
    "CCO",                                  # ethanol
    "CC(=O)Oc1ccccc1C(=O)O",                # aspirin
    CAFFEINE,                               # caffeine
    "BrCCCl",
    "c1ccc2[nH]ccc2c1",                     # indole
]
tokenizer = AtomTokenizer.from_smiles(CORPUS)

caffeine_ids, _ = tokenizer.encode_batch([CAFFEINE], add_special_tokens=True)
caffeine_ids = torch.tensor(caffeine_ids)                            # (1, L)
caffeine_tokens = ["[CLS]"] + tokenizer.tokenize(CAFFEINE) + ["[SEP]"]

D_MODEL = 32
token_embedding = TokenEmbedding(tokenizer.vocab_size, D_MODEL)
positional      = SinusoidalPositionalEncoding(d_model=D_MODEL, max_len=128)

x = positional(token_embedding(caffeine_ids))                        # (1, L, d_model)
print(f"caffeine sequence length: {x.size(1)}")
print(f"x shape: {tuple(x.shape)}  # (batch, seq_len, d_model)")
# -

# Quick reminder of the molecule and its atom-level tokens:

plot_molecule_with_tokens(CAFFEINE, caffeine_tokens, max_row_width=12.0)
plt.show()

# 🧪 **Chemical Intuition.** A 2D depiction of caffeine shows you *bonds* —
# at a glance you see the carbonyl `C` sits next to a `=O` and an `N`. The
# SMILES string doesn't show bonds; it just shows tokens in a line.
# Self-attention is how a transformer rebuilds "which tokens relate to
# which" — a learned, soft, all-pairs version of the bond list.

# ---
# ## 2. The Q / K / V mental model
#
# Self-attention applies three learned linear projections to the input,
# producing a **query** (`Q = x W_Q`), a **key** (`K = x W_K`), and a
# **value** (`V = x W_V`) for every token. A library-catalog version of the
# story:
#
# > Every token has a **query** (a question it's asking), a **key** (a
# > topic-tag advertising what it has to offer), and a **value** (the
# > actual content). Token *i* compares its query against everyone's key.
# > The answer is a weighted sum of everyone's value, weighted by how well
# > their key matched the query.
#
# In matrix form, a single self-attention head is:
#
# $$
# \mathrm{Attention}(x) = \mathrm{softmax}\!\Big(\frac{(xW_Q)(xW_K)^\top}{\sqrt{d_k}}\Big)\,(xW_V)
# $$
#
# We'll build this expression piece by piece. First, just the projections:

# +
W_q = nn.Linear(D_MODEL, D_MODEL, bias=False)
W_k = nn.Linear(D_MODEL, D_MODEL, bias=False)
W_v = nn.Linear(D_MODEL, D_MODEL, bias=False)

Q = W_q(x)
K = W_k(x)
V = W_v(x)
print(f"Q shape: {tuple(Q.shape)}")   # (1, L, d_model)
print(f"K shape: {tuple(K.shape)}")
print(f"V shape: {tuple(V.shape)}")
# -

# 💡 **Key Insight.** Q, K, and V live in the same `d_model`-dimensional
# space here because we're doing **single-head** attention. In notebook 05
# we'll split each projection into `n_heads` parallel sub-spaces of
# dimension `d_k = d_model / n_heads`. The mechanics below are identical
# inside each head.

# ---
# ## 3. Step 1: just dot the tokens with themselves
#
# Before introducing the projections, let's see what happens if we compute
# attention scores using the embeddings *directly* — i.e., set
# `Q = K = V = x`. The resulting matrix is `x · xᵀ`: entry `(i, j)` is the
# dot product of token `i`'s embedding with token `j`'s embedding.

xx = x[0] @ x[0].T                          # (L, L)
print(f"x · xᵀ shape: {tuple(xx.shape)}")
print(f"is symmetric? {torch.allclose(xx, xx.T, atol=1e-4)}")
print(f"first 5 diagonal entries: {[round(v.item(), 1) for v in xx.diag()[:5]]}")

# Visualize this as a heatmap. Token labels on both axes, and we let the
# helper auto-pick a diverging colormap because the raw scores can be
# negative.

plot_attention_heatmap(xx.detach(), caffeine_tokens, title="x · xᵀ  (raw embedding similarity)")
plt.show()

# 💡 **Key Insight.** Two things stand out:
#
# 1. The matrix is **symmetric** — `dot(a, b) = dot(b, a)`.
# 2. The **diagonal dominates** — each token overlaps with itself most.
#
# Both are properties of the *embedding space*, not of attention. They are
# also bad: a useful attention pattern needs to be *asymmetric* (token A
# attending to B doesn't have to match B attending to A) and **off-diagonal**
# (a token usually wants information from *other* tokens, not from itself).
# The Q and K projections are how attention escapes both limits.

# ---
# ## 4. Step 2: separate Q and K projections
#
# Replace `x · xᵀ` with `Q · Kᵀ`. Now token `i`'s query and token `i`'s key
# are *different vectors* — there is no longer any reason that
# `score(i, j) = score(j, i)`. The model has the freedom to learn "tokens
# like me attend to tokens like that" without forcing the reverse to also
# be true.

QK = Q[0] @ K[0].T                          # (L, L)
print(f"Q · Kᵀ shape: {tuple(QK.shape)}")
print(f"is symmetric? {torch.allclose(QK, QK.T, atol=1e-4)}")

# Side-by-side comparison: the pre-projection matrix on the left, the
# post-projection matrix on the right. Same molecule, same input — different
# math.

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, mat, ttl in [
    (axes[0], xx.detach(), "x · xᵀ  (no projections, symmetric)"),
    (axes[1], QK.detach(), "Q · Kᵀ  (with W_Q, W_K, asymmetric)"),
]:
    v = float(mat.abs().max())
    im = ax.imshow(mat.numpy(), cmap="RdBu_r", vmin=-v, vmax=v, aspect="equal")
    ax.set_xticks(range(len(caffeine_tokens)))
    ax.set_xticklabels(caffeine_tokens, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(caffeine_tokens)))
    ax.set_yticklabels(caffeine_tokens, fontsize=7)
    ax.set_title(ttl)
    fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()

# ⚠️ **Note.** With *random* `W_Q` and `W_K`, the right-hand matrix isn't
# yet meaningful — it just shows that the asymmetry is now *possible*. After
# training (notebooks 07–09), structure emerges: rows attend to chemically
# related tokens, ring-closure digits attend to each other across the
# molecule, special tokens act as hubs.

# ---
# ## 5. Step 3: scale by √d_k, then softmax
#
# Two things still need to happen to turn raw scores into attention weights:
#
# 1. **Scale** the scores by `1/√d_k`. Without this, large `d_k` makes the
#    raw dot products blow up, the softmax saturates, and gradients vanish.
# 2. **Softmax** each row so the weights sum to 1 — every token now
#    distributes a budget of 1.0 across the others.
#
# Let's prove the variance claim numerically. For unit-Gaussian `q`, `k`
# vectors of dimension `d_k`, the dot product `qᵀk` has variance equal to
# `d_k` — so the typical magnitude grows as `√d_k`. Dividing by `√d_k`
# undoes the growth.

# +
def random_qk_score_var(d_k: int, n_samples: int = 4096) -> float:
    """Empirical variance of qᵀk for unit-Gaussian q, k of dimension d_k."""
    q = torch.randn(n_samples, d_k)
    k = torch.randn(n_samples, d_k)
    return (q * k).sum(dim=-1).var().item()


d_ks = [4, 8, 16, 32, 64, 128, 256]
raw    = [random_qk_score_var(d) for d in d_ks]
scaled = [v / d for v, d in zip(raw, d_ks)]   # variance of qᵀk / √d_k is var(qᵀk) / d_k

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(d_ks, raw,    marker="o", label="raw  qᵀk")
ax.plot(d_ks, scaled, marker="o", label="scaled  qᵀk / √d_k")
ax.set_xscale("log")
ax.set_xlabel("d_k")
ax.set_ylabel("variance of score")
ax.set_title("Why we scale: raw qᵀk variance grows linearly with d_k")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# -

# ⚠️ **Note.** `1/√d_k` is **not** a hyperparameter — it's a
# numerical-stability fix. Without it, large dot products push softmax into
# saturation (one token gets ~100% of the weight) and gradients to all the
# other tokens vanish. With it, attention starts soft and *learns* to
# sharpen during training.

# Same lesson, viewed on a single row: softmax with vs without the scale.

scores_one  = QK[0]                                   # row for the [CLS] token
sm_unscaled = torch.softmax(scores_one,                       dim=-1).detach()
sm_scaled   = torch.softmax(scores_one / math.sqrt(D_MODEL),  dim=-1).detach()
print(f"max prob without scaling: {sm_unscaled.max().item():.3f}  (peakier — worse gradients)")
print(f"max prob with    scaling: {sm_scaled.max().item():.3f}  (softer  — trainable)")

# Now apply the full softmax(QKᵀ / √d_k) to the whole molecule — this is
# the actual attention matrix.

# +
scale = 1.0 / math.sqrt(D_MODEL)
attn = torch.softmax(QK * scale, dim=-1)              # (L, L), each row sums to 1
print(f"row sums (should all be ≈ 1): {[round(v, 3) for v in attn.sum(dim=-1)[:5].tolist()]}")

plot_attention_heatmap(attn.detach(), caffeine_tokens,
                       title="softmax(Q·Kᵀ / √d_k)   (random weights — pattern is noisy)")
plt.show()
# -

# ---
# ## 6. Step 4: weighted sum with the value projection
#
# The attention matrix says *how much* each token should look at every
# other; the **value** projection says *what* it sees when it does. The
# attention output for token `i` is a weighted sum of all the value vectors:

output = attn @ V[0]                         # (L, L) @ (L, d_model) → (L, d_model)
print(f"attention output shape: {tuple(output.shape)}  # one new vector per token")

# Every token now has a representation **informed by every other token** —
# that's the entire point of self-attention. And it took just four lines of
# linear algebra: Q/K/V projections, scaled `Q·Kᵀ`, softmax, and a final
# `@ V`.

# ---
# ## 7. Padding masks: stop attending to `[PAD]`
#
# Real batches mix sequences of different lengths, so shorter ones are
# padded with `[PAD]` (id 0). Attention has no idea those positions are
# fake — without a mask, it will happily distribute weight onto them. Let's
# make this concrete by stacking **ethanol** and **caffeine** into a single
# batch.

batch_smiles = ["CCO", CAFFEINE]
batch_ids, batch_mask = tokenizer.encode_batch(batch_smiles, add_special_tokens=True)
batch_ids  = torch.tensor(batch_ids)            # (2, L_max)
batch_mask = torch.tensor(batch_mask)           # (2, L_max), 1 = real, 0 = pad
print(f"batch_ids  shape: {tuple(batch_ids.shape)}")
print(f"ethanol mask : {batch_mask[0].tolist()}")
print(f"caffeine mask: {batch_mask[1, :8].tolist()}...   (all 1s)")

# Run the full attention pipeline on the batch — first **without** masking:

# +
xb = positional(token_embedding(batch_ids))                 # (2, L, d_model)
Qb, Kb, Vb = W_q(xb), W_k(xb), W_v(xb)
scores_b = (Qb @ Kb.transpose(-2, -1)) * scale              # (2, L, L)

attn_unmasked = torch.softmax(scores_b, dim=-1)             # (2, L, L)
n_real_eth = int(batch_mask[0].sum().item())                # = 5  ([CLS] C C O [SEP])
leak = attn_unmasked[0, :n_real_eth, n_real_eth:].sum(dim=-1)
print(f"ethanol real-token rows attending to pad cols (should be > 0 here):")
print(f"  {[round(v, 3) for v in leak.tolist()]}")
# -

# And **with** masking applied on the key axis. Setting masked scores to
# `-inf` makes their softmax contribution exactly 0 and the remaining real
# tokens re-normalize to sum to 1.

# +
keep = batch_mask.bool().unsqueeze(1)                       # (2, 1, L) key-side
scores_masked = scores_b.masked_fill(~keep, float("-inf"))
attn_masked = torch.softmax(scores_masked, dim=-1)

leak_after = attn_masked[0, :n_real_eth, n_real_eth:].sum(dim=-1)
real_sums  = attn_masked[0, :n_real_eth, :n_real_eth].sum(dim=-1)
print(f"ethanol attention onto pad cols (should be 0):     "
      f"{[round(v, 6) for v in leak_after.tolist()]}")
print(f"ethanol real-token rows still sum to 1:            "
      f"{[round(v, 3) for v in real_sums.tolist()]}")
# -

# Side-by-side: ethanol's attention matrix before and after masking. Pad
# columns leak attention on the left; they're correctly zeroed on the right.

ethanol_tokens = (
    ["[CLS]"] + tokenizer.tokenize("CCO") + ["[SEP]"]
    + ["[PAD]"] * (batch_ids.size(1) - n_real_eth)
)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, A, ttl in [
    (axes[0], attn_unmasked[0].detach(), "before masking — leaks onto [PAD]"),
    (axes[1], attn_masked[0].detach(),   "after masking — pad columns are zero"),
]:
    im = ax.imshow(A.numpy(), cmap="Blues", vmin=0, vmax=float(A.max()), aspect="equal")
    ax.set_xticks(range(len(ethanol_tokens)))
    ax.set_xticklabels(ethanol_tokens, rotation=90, fontsize=6)
    ax.set_yticks(range(len(ethanol_tokens)))
    ax.set_yticklabels(ethanol_tokens, fontsize=6)
    ax.set_title(ttl)
    fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()

# ⚠️ **Note.** Padding masks aren't a stylistic choice. Without them,
# attention mass leaks onto `[PAD]`, the softmax denominator is wrong, and
# gradients propagate into the pad columns of `W_V`. Always mask. (The
# brief mention of *causal* masks — which restrict each token to looking
# leftward — is in Exercise 3.)

# ---
# ## 8. The same thing, packaged
#
# All of the above lives in one tiny `nn.Module` —
# `ScaledDotProductAttention` — under `utils/transformer_blocks.py`. Same
# math, batched, with the padding mask handled inside.

# +
attn_module = ScaledDotProductAttention(d_model=D_MODEL, dropout=0.0)
# Copy our hand-rolled projections into the module so its output should
# match `attn_masked` from §7 to numerical precision.
attn_module.w_q.weight.data = W_q.weight.data.clone()
attn_module.w_k.weight.data = W_k.weight.data.clone()
attn_module.w_v.weight.data = W_v.weight.data.clone()
attn_module.w_q.bias.data.zero_()
attn_module.w_k.bias.data.zero_()
attn_module.w_v.bias.data.zero_()

out_module, attn_module_w = attn_module(xb, mask=batch_mask)
print(f"output shape:    {tuple(out_module.shape)}")
print(f"attention shape: {tuple(attn_module_w.shape)}")
print(f"matches the hand-rolled version? "
      f"{torch.allclose(attn_module_w, attn_masked, atol=1e-5)}")
# -

# A quick look at what the `[CLS]` token of caffeine attends to, painted
# back onto the tokenized SMILES. The query token (`[CLS]`, position 0) is
# outlined in red.

plot_attention_on_smiles(
    attn_module_w[1, 0].detach(),    # batch index 1 = caffeine, query row 0 = [CLS]
    caffeine_tokens,
    query_index=0,
)
plt.show()

# 🔬 **Try This.** Re-run this notebook with a different `torch.manual_seed`.
# The attention pattern changes completely — because `W_Q`, `W_K`, `W_V`
# are all random. Real (trained) attention patterns carry structure:
# special tokens act as hubs, ring-closure digits attend to each other,
# carbonyl carbons attend to their neighbouring oxygens. With random
# weights you get noise — which is fine: we're learning the **mechanism**,
# not its trained behaviour. Notebooks 07 and 09 will train and revisit
# these pictures.

# ---
# ## Checkpoint exercises

# +
# Exercise 1
# -----------
# Given Q, K of shape (seq_len=4, d_k=8), compute the unnormalized scores
# Q · Kᵀ. What is the resulting shape? Verify by hand that `scores[1, 2]`
# equals the dot product of `Q[1]` with `K[2]`.

# YOUR CODE HERE

# --- Solution (try the exercise first, then peek) ---
# torch.manual_seed(1)
# Q_ex = torch.randn(4, 8)
# K_ex = torch.randn(4, 8)
# scores_ex = Q_ex @ K_ex.T
# print(f"shape: {tuple(scores_ex.shape)}")            # (4, 4)
# print(f"matches dot? {torch.allclose(scores_ex[1, 2], Q_ex[1] @ K_ex[2])}")

# +
# Exercise 2
# -----------
# Build a padding mask for the batch ["CCO", "CC(=O)Oc1ccccc1C(=O)O"] using
# `tokenizer.encode_batch`. Apply it to a random (B, L, L) score matrix and
# verify (i) every real-token row of the resulting softmaxed attention sums
# to 1, and (ii) attention onto pad columns is exactly zero.

# YOUR CODE HERE

# --- Solution ---
# ids_ex, mask_ex = tokenizer.encode_batch(
#     ["CCO", "CC(=O)Oc1ccccc1C(=O)O"], add_special_tokens=True
# )
# mask_ex = torch.tensor(mask_ex)                                # (2, L)
# B, L = mask_ex.shape
# scores_rand = torch.randn(B, L, L)
# keep_ex = mask_ex.bool().unsqueeze(1)                          # (2, 1, L)
# attn_ex = torch.softmax(
#     scores_rand.masked_fill(~keep_ex, float("-inf")), dim=-1
# )
# # (i) every real-token row sums to 1
# real_rows = mask_ex.bool()                                     # (B, L)
# row_sums  = attn_ex.sum(dim=-1)
# print(f"real rows all sum to 1: "
#       f"{torch.allclose(row_sums[real_rows], torch.ones(int(real_rows.sum())))}")
# # (ii) attention onto pad columns is exactly zero
# n_real_short = int(mask_ex[0].sum())
# print(f"attention onto pad cols (short seq): "
#       f"{attn_ex[0, :, n_real_short:].max().item()}  (= 0.0)")

# +
# Exercise 3 — causal masking
# ----------------------------
# Modify the attention call so token `i` can only attend to tokens 0..i (a
# lower-triangular mask). Run on caffeine and plot the result with
# `plot_attention_heatmap`. In one sentence: why is this *not* what we want
# for our encoder-only, MolFormer-style models?

# YOUR CODE HERE

# --- Solution ---
# L = x.size(1)
# causal = torch.tril(torch.ones(L, L, dtype=torch.bool))
# scores_caffeine = (Q[0] @ K[0].T) * scale
# scores_causal   = scores_caffeine.masked_fill(~causal, float("-inf"))
# attn_causal     = torch.softmax(scores_causal, dim=-1)
# plot_attention_heatmap(attn_causal.detach(), caffeine_tokens,
#                        title="causal attention — token i only sees 0..i")
# plt.show()
# # Causal masks are for left-to-right *generators* (GPT). MolFormer-style
# # encoders are trained with masked language modelling and need to see the
# # whole molecule at once — masking out the right half throws away
# # exactly the context that disambiguates ring closures and stereochemistry.

# ---
# ## What's next
#
# We now have a single attention head: `Q·Kᵀ` scaled, softmaxed, masked,
# multiplied by `V`. **Notebook 05** runs several heads in parallel — each
# one free to learn a different "way of looking" at the molecule (one head
# tracks ring closures, another tracks heteroatom neighbourhoods, another
# acts as a `[CLS]` aggregator, ...). The mechanics inside each head are
# exactly what we just built.
#
# 📚 **Deep-dive sub-series**
# - **04.1**: Linear attention — MolFormer's `O(N)` swap for `softmax(QKᵀ)V`.
# - **04.2**: Rotary position embeddings (RoPE) — folding position into Q/K.
# - **04.3**: Other position encodings (ALiBi, relative-position bias).
#
# 📚 **References.**
# - Bahdanau, D. et al. (2015). *Neural Machine Translation by Jointly
#   Learning to Align and Translate.* — the original attention paper.
# - Vaswani, A. et al. (2017). *Attention Is All You Need.* — modern
#   scaled dot-product self-attention used here.
