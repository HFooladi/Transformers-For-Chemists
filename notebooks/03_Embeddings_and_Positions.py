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

# <a href="https://colab.research.google.com/github/HFooladi/Transformers-For-Chemists/blob/main/notebooks/03_Embeddings_and_Positions.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 03 · Embeddings & Positions
#
# Notebook 02 left us with a sequence of integer **token IDs** for every
# molecule. A neural network can't multiply integers — it needs **vectors**.
# This notebook builds the bridge: we turn each ID into a ``d_model``-dim
# vector via a learned **token embedding**, then stamp **positional
# information** onto each vector so the transformer knows that "first token"
# is different from "third token".
#
# Once we've done both, the input to every transformer block in the rest of
# the course is one tensor of shape ``(batch, seq_len, d_model)``. That
# shape is the lingua franca for everything that follows.

# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# 1. Explain why a transformer needs a **vector** per token, not an integer.
# 2. Implement a **token embedding** as a lookup table and use it on a real
#    SMILES batch.
# 3. Demonstrate that a transformer-style layer is **permutation-equivariant**
#    without positional information — and explain why that's a problem for
#    sequences.
# 4. Derive and visualize the **sinusoidal positional encoding** matrix
#    from Vaswani et al. (2017).
# 5. Show that the dot product of two PE rows depends only on their
#    **relative distance** — the property that makes the encoding useful.
# 6. Combine token embeddings + PE into the full input tensor and visualize
#    it as a heatmap.
# 7. Contrast **fixed sinusoidal** vs **learned** positional encodings and
#    say when each is preferable.

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

from utils.smiles_tokenizers import (
    AtomTokenizer,
    PAD_ID,
)
from utils.transformer_blocks import TokenEmbedding, SinusoidalPositionalEncoding

torch.manual_seed(0)  # so the random embeddings are reproducible
# -

# ---
# ## 1. From IDs to vectors: the token embedding
#
# A **token embedding** is the simplest possible way to turn a discrete ID
# into a dense vector: a giant lookup table. Row ``i`` of the table is the
# vector you get when the input token has ID ``i``.
#
# Concretely, if our vocabulary has ``V`` tokens and we want vectors of
# dimension ``d_model``, the embedding is a ``(V, d_model)`` matrix.

# Tokenize a small batch with the atom-level tokenizer from notebook 02.
CORPUS = [
    "CCO",                                  # ethanol
    "CC(=O)Oc1ccccc1C(=O)O",                # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",         # caffeine
    "BrCCCl",
    "c1ccc2[nH]ccc2c1",                     # indole
]
tokenizer = AtomTokenizer.from_smiles(CORPUS)
input_ids, attention_mask = tokenizer.encode_batch(
    ["CCO", "BrCCCl"], add_special_tokens=True
)
input_ids = torch.tensor(input_ids)         # shape (batch=2, seq_len=padded)
attention_mask = torch.tensor(attention_mask)
print(f"vocab size: {tokenizer.vocab_size}")
print(f"input_ids   shape: {tuple(input_ids.shape)}")
print(f"input_ids:\n{input_ids}")
print(f"attention_mask:\n{attention_mask}")

# +
D_MODEL = 16  # tiny embedding dim, just for visualization

# Build the embedding from scratch first, before reaching for the helper class.
embedding_table = torch.nn.Embedding(tokenizer.vocab_size, D_MODEL, padding_idx=PAD_ID)
print(f"embedding table shape: {tuple(embedding_table.weight.shape)}")
print(f"row 0 (pad): {embedding_table.weight[0]}")  # exactly zero
print(f"row 7 (one of the carbons in our vocab):")
print(f"  {embedding_table.weight[7]}")
# -

# Look up the batch through the table.
embedded = embedding_table(input_ids)
print(f"embedded shape: {tuple(embedded.shape)}  # (batch, seq_len, d_model)")
print(f"\nfirst molecule's embedded sequence (one row per token):")
print(embedded[0])

# 💡 **Key Insight.** "Embedding" sounds fancy. It is just a lookup table
# from `int → vector`. The whole table is a learnable parameter — every
# gradient step nudges the rows so that, eventually, tokens that play
# similar roles in similar molecules end up with similar vectors. We won't
# *train* until notebook 07; for now, the rows are random.

# ### A small but important detail: scaling by `sqrt(d_model)`
#
# Vaswani et al. multiply the raw embedding output by `sqrt(d_model)`
# before adding the positional encoding. The reason is mundane:
# `nn.Embedding` initializes rows with a unit-Gaussian, so the per-element
# magnitude is ~1. The sinusoidal PE we'll add in a moment also has
# per-element magnitude ~1. Without scaling, the embedding would *vanish*
# next to the PE. Scaling by `sqrt(d_model)` boosts the embedding's
# magnitude back into the same range as the PE.
#
# The course `TokenEmbedding` class wraps this for you.

token_embedding = TokenEmbedding(vocab_size=tokenizer.vocab_size, d_model=D_MODEL)
out = token_embedding(input_ids)
print(f"shape: {tuple(out.shape)}")
print(f"per-element std without scaling: {embedded.std().item():.3f}")
print(f"per-element std with    scaling: {out.std().item():.3f}  ≈ sqrt({D_MODEL}) × {embedded.std().item():.3f}")

# ---
# ## 2. Why position matters
#
# Without positional information, a transformer treats the input as a **set
# of tokens**, not a *sequence*. To see why that's broken, let's run a tiny
# experiment: take the same set of tokens in two different orders and check
# whether their *bag of embeddings* is identical (it is) and whether a real
# sequence encoder would care (it should).

# +
shuffled_ids = input_ids[1].clone()                 # BrCCCl with [CLS]/[SEP] (6 tokens)
permutation = torch.tensor([0, 4, 2, 1, 3, 5])      # keep [CLS]/[SEP] in place; shuffle the middle
print(f"original ids   : {shuffled_ids.tolist()}")
permuted = shuffled_ids[permutation]
print(f"permuted ids   : {permuted.tolist()}")

emb_orig = token_embedding(shuffled_ids.unsqueeze(0))[0]
emb_perm = token_embedding(permuted.unsqueeze(0))[0]
print(f"\nbag-of-embeddings (sum across positions):")
print(f"  original: {emb_orig.sum(dim=0)[:5]}...")
print(f"  permuted: {emb_perm.sum(dim=0)[:5]}...")
print(f"  identical? {torch.allclose(emb_orig.sum(0), emb_perm.sum(0), atol=1e-5)}")


# -

# 🧪 **Chemical Intuition.** The point isn't that the *embeddings* are
# identical — it's that any operation that **sums across positions**
# (like the bag-of-words baseline above, or — as we'll see in notebook 04 —
# self-attention without positional info) cannot distinguish
# `BrCCCl` from `ClCCBr` from `CCBrCl`. Those are different chemical
# strings: `BrCCCl` is 1-bromo-2-chloroethane, `CCBrCl` is just garbage.
# The model must know what came first, second, third, ...
#
# 💡 **Key Insight.** This problem is the mirror image of the *desirable*
# permutation invariance in graph neural networks (your GNN course covers
# why nodes shouldn't depend on their indexing). Sequences are different:
# **order is part of the signal**. Positional encoding restores that order.

# ---
# ## 3. Sinusoidal positional encoding from scratch
#
# The Vaswani et al. recipe is a closed-form formula — no learned parameters,
# generalizes to any sequence length:
#
# $$
# \mathrm{PE}(pos, 2i)   = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \\
# \mathrm{PE}(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
# $$
#
# Reading the formula: every position gets a unique vector, made of sines
# and cosines at a range of frequencies. The first few dimensions oscillate
# fast (with position), the last few oscillate so slowly they're nearly
# constant. The model can read off "absolute position" from the fast
# dimensions, "rough region" from the slow ones, and **relative position**
# from a clever interaction between the two (we'll prove this in section 5).

# +
def compute_pe_matrix(max_len: int, d_model: int) -> torch.Tensor:
    """Build the (max_len, d_model) sinusoidal PE matrix from scratch."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(max_len).unsqueeze(1).float()           # (max_len, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


pe_matrix = compute_pe_matrix(max_len=64, d_model=64)
print(f"PE matrix shape: {tuple(pe_matrix.shape)}")
print(f"PE[0]: {pe_matrix[0, :8].tolist()}")
print(f"PE[1]: {pe_matrix[1, :8].tolist()}")
print(f"PE[5]: {pe_matrix[5, :8].tolist()}")
# -

# ### The famous sinusoidal heatmap
#
# Plot the matrix with positions on the y-axis and embedding dimensions on
# the x-axis. The diagonal-stripe pattern is the visual signature of the
# encoding.

fig, ax = plt.subplots(figsize=(7.5, 5))
im = ax.imshow(pe_matrix.numpy(), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xlabel("Embedding dimension")
ax.set_ylabel("Position in sequence")
ax.set_title("Sinusoidal positional encoding  (max_len=64, d_model=64)")
fig.colorbar(im, ax=ax, label="PE value")
plt.tight_layout()
plt.show()

# 💡 **Key Insight.** Look at the columns. The leftmost dimensions
# oscillate quickly with position (many stripes); the rightmost dimensions
# oscillate so slowly they look almost flat. Together, every row is a
# unique fingerprint that encodes "I am at position 5 of the sequence".

# ### A second view: a few PE dimensions vs position
#
# Pick four embedding dimensions and plot their value as a function of
# position. This makes the multi-frequency structure obvious.

fig, ax = plt.subplots(figsize=(7.5, 4))
positions = np.arange(64)
for dim in [0, 4, 8, 16]:
    ax.plot(positions, pe_matrix[:, dim].numpy(), label=f"dim {dim}", alpha=0.85)
ax.set_xlabel("Position")
ax.set_ylabel("PE value")
ax.set_title("Each embedding dimension is a sinusoid at a different frequency")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ---
# ## 4. Putting it together: input = token_embedding + PE
#
# The course `SinusoidalPositionalEncoding` class precomputes the matrix
# and adds it to the input embeddings. The full input pipeline is::
#
#     ids        ─►  TokenEmbedding(ids)        # (B, L, d_model)
#                ─►  + PE(positions)            # (B, L, d_model)
#                ─►  feed into transformer block
#
# In code:

positional = SinusoidalPositionalEncoding(d_model=D_MODEL, max_len=128)
final_input = positional(token_embedding(input_ids))
print(f"final_input shape: {tuple(final_input.shape)}  # (batch, seq_len, d_model)")

# ### Visualize a single molecule's input tensor
#
# We'll switch to a slightly bigger `d_model` (32) and a larger molecule so
# the heatmap is interesting.

# +
D_MODEL_VIS = 32
big_token_emb = TokenEmbedding(tokenizer.vocab_size, D_MODEL_VIS)
big_pos_enc = SinusoidalPositionalEncoding(d_model=D_MODEL_VIS, max_len=128)

caffeine_ids, caffeine_mask = tokenizer.encode_batch(
    ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C"], add_special_tokens=True
)
caffeine_ids = torch.tensor(caffeine_ids)
caffeine_input = big_pos_enc(big_token_emb(caffeine_ids))[0]    # (seq_len, d_model)

caffeine_tokens = ["[CLS]"] + tokenizer.tokenize("CN1C=NC2=C1C(=O)N(C(=O)N2C)C") + ["[SEP]"]
print(f"caffeine input tensor shape: {tuple(caffeine_input.shape)}")

fig, ax = plt.subplots(figsize=(11, 4.5))
im = ax.imshow(caffeine_input.detach().numpy(), aspect="auto", cmap="RdBu_r")
ax.set_xticks(range(D_MODEL_VIS))
ax.set_xlabel("Embedding dimension")
ax.set_yticks(range(len(caffeine_tokens)))
ax.set_yticklabels(caffeine_tokens, fontsize=8)
ax.set_ylabel("Position (token)")
ax.set_title("Caffeine input to the transformer = token_embedding + sinusoidal PE")
fig.colorbar(im, ax=ax, label="value")
plt.tight_layout()
plt.show()


# -

# 🔬 **Try This.** Look at row 1 (the first `C`) and row 12 (one of the
# later `C`s). Both are the same token. They look *similar* in some
# columns (left side, where token info dominates) and *different* in
# others (right side, where positional info dominates). That's the
# embedding doing both jobs at once.

# ---
# ## 5. The relative-position property
#
# Sinusoidal PE has a beautiful property that learned PE doesn't:
# the **dot product** between any two PE rows depends only on their
# **distance**, not on their absolute positions. Concretely:
#
# $$
# \langle \mathrm{PE}(i),\ \mathrm{PE}(i+k)\rangle \approx f(k)
# $$
#
# regardless of `i`. We'll verify it numerically.

# +
def pe_dot(i: int, k: int, pe: torch.Tensor) -> float:
    return torch.dot(pe[i], pe[i + k]).item()


big_pe = compute_pe_matrix(max_len=256, d_model=64)
print("dot(PE(i), PE(i+k)) sampled at five values of i:")
print(f"{'k':>4s}   " + "  ".join(f"i={i:>3d}" for i in [0, 20, 50, 100, 150]))
for k in [1, 5, 10, 25, 50]:
    vals = [pe_dot(i, k, big_pe) for i in [0, 20, 50, 100, 150]]
    print(f"{k:>4d}   " + "  ".join(f"{v:7.3f}" for v in vals))
# -

# Every row is the same — within rounding error. Now plot the dot product
# as a function of distance. Notice it decays smoothly with `k`: nearby
# positions have high similarity, far positions have low similarity. This
# is *exactly* the prior the model needs for a sequence.

distances = list(range(0, 100))
dots = [pe_dot(0, k, big_pe) for k in distances]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(distances, dots, marker=".")
ax.set_xlabel("Distance k = |i − j|")
ax.set_ylabel("dot(PE(i), PE(j))")
ax.set_title("Sinusoidal PE encodes relative position via dot product")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 💡 **Key Insight.** This relative-position property is what makes the
# sinusoidal scheme work so well in attention: when the model computes
# `query · key`, the contribution from positions automatically encodes
# *how far apart* the tokens are. In notebook 04.2 (deep-dive) we'll meet
# **rotary positional embeddings (RoPE)** — MolFormer's choice — which
# rebuild this exact property by *rotating* the query and key vectors
# instead of adding to them.

# ---
# ## 6. Sinusoidal vs learned positional encodings
#
# The most common alternative to fixed sinusoidal PE is **learned PE**: a
# trainable `(max_len, d_model)` matrix. BERT uses learned PE; GPT-2 uses
# learned PE; ALBERT, RoBERTa, ChemBERTa — all use learned PE.
#
# | Property | Sinusoidal | Learned |
# |----------|-----------|---------|
# | Number of parameters | 0 | `max_len × d_model` |
# | Generalizes to longer sequences than `max_len`? | Yes (formula extrapolates) | No |
# | Encodes relative distance via dot product? | Yes (provably) | Only if learned to |
# | Performance in practice | ≈ | ≈ |
# | Pedagogy | Visualizable, parameter-free | Just a matrix |
#
# In this course we use sinusoidal PE for the core sequence (it's
# parameter-free and the math is transparent), and revisit position
# encoding as a deep-dive in notebooks 04.2 (RoPE) and 04.3 (ALiBi,
# relative-position attention).
#
# ⚠️ **Note.** A common misconception: "sinusoidal PE *gives* the model a
# notion of position." It really *enables* the model to discover position;
# the model still has to learn to use the encoding via its weights. With a
# tiny untrained model the embeddings are just random — useful position
# behaviour emerges during training.

# ---
# ## Checkpoint exercises

# +
# Exercise 1
# -----------
# Build a TokenEmbedding for a vocabulary of size 100 and d_model=64.
# How many learnable parameters does it have? What is the per-element
# standard deviation after the sqrt(d_model) scaling? (Hint: feed in a
# random batch and use .std()).

# YOUR CODE HERE

# --- Solution (try the exercise first, then peek) ---
# te = TokenEmbedding(vocab_size=100, d_model=64)
# n_params = sum(p.numel() for p in te.parameters())
# print(f"Parameters: {n_params}  (= 100 × 64)")
# x = torch.randint(0, 100, (4, 32))
# print(f"Output std: {te(x).std().item():.3f}  (≈ sqrt(64) × 1 = 8)")

# +
# Exercise 2
# -----------
# Construct a sinusoidal PE matrix with max_len=200, d_model=64. Find the
# pair of positions (i, j) with i ≠ j whose PE rows are *most* similar
# (largest cosine similarity). Are they close in position or far?

# YOUR CODE HERE

# --- Solution (try the exercise first, then peek) ---
# pe = compute_pe_matrix(max_len=200, d_model=64)
# pe_norm = pe / pe.norm(dim=-1, keepdim=True)
# sim = pe_norm @ pe_norm.T
# sim.fill_diagonal_(-1)  # exclude self-similarity
# best = sim.argmax()
# i, j = best // 200, best % 200
# print(f"Most similar non-self pair: positions {i.item()} and {j.item()}, "
#       f"cosine sim = {sim[i, j].item():.3f}")
# # In sinusoidal PE the most similar rows are always *adjacent* —
# # |i - j| = 1 — because dot(PE(i), PE(i+k)) decays monotonically in k.

# +
# Exercise 3
# -----------
# The course uses sinusoidal PE because it's parameter-free. But suppose
# you wanted to swap in a *learned* PE. Write a tiny `LearnedPositionalEncoding`
# nn.Module: a single nn.Embedding of shape (max_len, d_model) indexed by
# the position 0..L-1, added to the input. Verify it produces the right
# output shape on the caffeine batch above.

# YOUR CODE HERE

# --- Solution ---
# class LearnedPositionalEncoding(torch.nn.Module):
#     def __init__(self, max_len, d_model):
#         super().__init__()
#         self.embed = torch.nn.Embedding(max_len, d_model)
#     def forward(self, x):
#         positions = torch.arange(x.size(1), device=x.device)
#         return x + self.embed(positions).unsqueeze(0)
#
# learned_pe = LearnedPositionalEncoding(max_len=128, d_model=D_MODEL_VIS)
# out = learned_pe(big_token_emb(caffeine_ids))
# print(out.shape)  # same shape as the sinusoidal version
# -

# ---
# ## What's next
#
# Now we have a tensor of shape `(batch, seq_len, d_model)` for every
# molecule, with both *content* (which token) and *position* (which slot
# in the sequence) information packed into every vector. **Notebook 04**
# introduces the operation that finally lets tokens **talk to each other**:
# **self-attention**. We'll build it from scratch from the same Q/K/V
# intuition that powers every modern language model — and we'll use the
# embeddings we just built as its input.
#
# 📚 **References.**
# - Vaswani, A. et al. (2017). *Attention Is All You Need.* — original
#   sinusoidal PE.
# - Devlin, J. et al. (2019). *BERT.* — popularized learned PE.
# - Su, J. et al. (2021). *RoFormer: Enhanced Transformer with Rotary
#   Position Embedding* — RoPE, covered in notebook 04.2.
