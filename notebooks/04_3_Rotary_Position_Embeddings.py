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

# <a href="https://colab.research.google.com/github/HFooladi/Transformers-For-Chemists/blob/main/notebooks/04_3_Rotary_Position_Embeddings.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 04.3 · Rotary Position Embeddings (RoPE)
#
# Notebook 03 gave each token its position by **adding** a sinusoidal vector to
# its embedding — an *absolute* address ("you are token #7"). Notebook 04 then
# projected everything to Q/K/V and let attention compare tokens.
#
# RoPE takes a strikingly different route. Instead of *adding* position, it
# **rotates** each query and key vector by an angle proportional to its
# position. The beautiful consequence: when two rotated vectors are compared in
# the attention dot product, the result depends only on their **relative**
# offset `m − n` — not on where either token sits in absolute terms. This is
# the positional scheme **MolFormer** uses.
#
# **Scope.** A *mechanism* deep-dive: we build RoPE from scratch, visualize why
# rotation encodes relative position, and apply it to attention on a real
# molecule. Using it inside the full model is notebook 09.

# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# 1. Contrast **adding** position (notebook 03) with **rotating** by position
#    (RoPE).
# 2. Picture RoPE as literally spinning a 2-D vector by an angle that grows
#    with position.
# 3. Extend the 2-D picture to `d` dimensions: `d/2` pairs, each spun at its own
#    frequency (the same frequency schedule as sinusoidal PE).
# 4. Implement RoPE from scratch with the `rotate_half` trick and confirm it
#    **preserves vector norm**.
# 5. Show empirically that `rotate(q, m) · rotate(k, n)` depends only on
#    `m − n` — the relative-position property, baked in.
# 6. Apply RoPE inside attention on caffeine and compare with additive PE.

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
from utils.transformer_blocks import TokenEmbedding
from utils.rope import RotaryPositionalEncoding, apply_rotary, rotate_half
from utils.attention_viz import plot_attention_heatmap

torch.manual_seed(0)  # reproducible random projections
# -

# ---
# ## 1. The relative-distance idea, recapped
#
# Notebook 03 ended with a quiet but important observation: the dot product of
# two sinusoidal position vectors, `PE(i) · PE(j)`, depends mostly on the
# *distance* `|i − j|`, decaying smoothly as tokens get farther apart. That's
# the property a chemist wants — "how far apart along the SMILES are these two
# atoms?" — and additive PE only gets it *approximately*, as a side effect.
#
# RoPE asks: what if we build that relative-distance property in **exactly**,
# by construction, instead of hoping it emerges? The answer is rotation.

# 🧪 **Chemical Intuition.** When you read a SMILES like `CN1C=NC2=C1...`, what
# matters for the ring-closure digit `1` is not "I am the 3rd character" but
# "my partner `1` is 4 atoms further along." Relative position is the natural
# language of molecular structure. RoPE encodes exactly that.

# ---
# ## 2. RoPE in 2-D: position is how far you spin the arrow
#
# Start with the simplest case: a 2-D vector. RoPE rotates it by an angle
# `θ = position × ω`, where `ω` is a fixed frequency. Token 0 isn't rotated at
# all; token 1 is rotated by `ω`; token 2 by `2ω`; and so on. Same vector,
# spun a little more at each step down the molecule.

# +
v = torch.tensor([1.0, 0.3])          # a single 2-D "content" vector
omega = 0.45                           # one frequency
positions = range(8)

fig, ax = plt.subplots(figsize=(6.5, 6.5))
colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))
for p, c in zip(positions, colors):
    angle = p * omega
    rot = torch.tensor([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle),  math.cos(angle)],
    ])
    rv = rot @ v
    ax.annotate("", xy=(rv[0], rv[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
    ax.text(rv[0] * 1.12, rv[1] * 1.12, f"pos {p}", color=c, fontsize=9,
            ha="center", va="center")
lim = 1.3
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6)
ax.set_aspect("equal")
ax.set_title("Same vector, rotated a bit more at each position\n(RoPE in 2-D)")
plt.tight_layout()
plt.show()
# -

# 💡 **Key Insight.** Every arrow has the **same length** — rotation only
# changes *direction*, never *magnitude*. This is the first big difference from
# additive PE (notebook 03), which adds a vector and so changes the length of
# the embedding. RoPE leaves the "content" magnitude untouched and stores
# position purely in the *angle*.

# ---
# ## 3. From 2-D to d dimensions: many frequencies at once
#
# A real embedding has `d` dimensions, not 2. RoPE splits it into `d/2`
# coordinate **pairs** and rotates each pair by its own frequency. Low-index
# pairs spin fast (fine, local position); high-index pairs spin slowly (coarse,
# long-range position) — the *exact* inverse-frequency schedule from the
# sinusoidal encoding in notebook 03.
#
# We can visualize the whole schedule as a heatmap of the rotation **angle**
# for every (position, pair). Compare it to notebook 03's PE heatmap — same
# diagonal-stripe DNA, because it's the same set of frequencies.

# +
D_MODEL = 32
rope = RotaryPositionalEncoding(dim=D_MODEL, max_len=64)

# Recover the per-pair angles from the cos/sin tables: angle = atan2(sin, cos).
angles = torch.atan2(rope.sin, rope.cos)[:, : D_MODEL // 2]   # (max_len, d/2)

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(angles.numpy(), aspect="auto", cmap="twilight",
               vmin=-math.pi, vmax=math.pi)
ax.set_xlabel("coordinate pair  (fast spin ──► slow spin)")
ax.set_ylabel("position in sequence")
ax.set_title("RoPE rotation angle per (position, pair)\nfast pairs cycle quickly, slow pairs barely move")
fig.colorbar(im, ax=ax, label="rotation angle (radians)")
plt.tight_layout()
plt.show()
# -

# ⚠️ **Note.** The leftmost columns flip through the full colour wheel many
# times as you go down the positions (high frequency); the rightmost columns
# stay almost one colour (low frequency). A model reads fine-grained position
# from the fast pairs and long-range position from the slow ones — multiple
# rulers at multiple scales, all at once.

# ---
# ## 4. Build it from scratch
#
# Rotating every pair with an explicit 2×2 matrix would be fiddly. There's a
# slick vectorized form. Writing the embedding as `x` and using a helper
# `rotate_half(x)` that swaps each pair's two coordinates with a sign flip
# (`(x1, x2) → (−x2, x1)`), a full rotation by per-pair angles is just:
#
# $$ \text{RoPE}(x) = x \odot \cos(\theta) + \text{rotate\_half}(x) \odot \sin(\theta) $$
#
# where `θ` holds each pair's angle at this position. Let's confirm two things:
# the output is the genuine rotation, and — the signature property — it
# **preserves the norm** of every vector.

# +
q = torch.randn(1, 10, D_MODEL)        # 10 tokens, random "content"
q_rot = rope(q)                         # rotate by position

norms_before = q.norm(dim=-1)
norms_after  = q_rot.norm(dim=-1)
print(f"norms preserved by rotation? "
      f"{torch.allclose(norms_before, norms_after, atol=1e-5)}")
print(f"example token norm  before: {norms_before[0, 3]:.4f}   "
      f"after: {norms_after[0, 3]:.4f}")

# And position 0 is rotated by angle 0 — i.e. left unchanged:
print(f"position 0 unchanged? {torch.allclose(q[0, 0], q_rot[0, 0], atol=1e-6)}")
# -

# 💡 **Key Insight.** `rotate_half` is the entire trick that lets us rotate all
# `d/2` pairs in two element-wise multiplies instead of a Python loop over
# 2×2 matrices. It's exactly what `utils/rope.py` does, and what every
# production RoPE implementation does.

# ---
# ## 5. The payoff: attention sees only *relative* position
#
# Here is the property that makes RoPE worth the trouble. Take a **fixed**
# content vector for both query and key (so any difference in their dot product
# comes purely from position). Rotate the query to position `m` and the key to
# position `n`, then take their dot product, for *every* pair `(m, n)`.
#
# Under RoPE, the result is constant along each diagonal — it depends **only**
# on `m − n`. Under additive PE (notebook 03), cross-terms leak in and the
# clean diagonal structure breaks. Side by side:

# +
SEQ = 40
content = torch.randn(D_MODEL)                       # same content for q and k
rope_big = RotaryPositionalEncoding(dim=D_MODEL, max_len=SEQ)

# --- RoPE: rotate the shared content to every position, then all pairwise dots.
rot_all = rope_big(content.expand(1, SEQ, D_MODEL))[0]     # (SEQ, d)
rope_scores = rot_all @ rot_all.T                          # (SEQ, SEQ)

# --- Additive PE (notebook 03 style): add a sinusoidal vector, then dot.
pe = torch.zeros(SEQ, D_MODEL)
pos = torch.arange(SEQ).unsqueeze(1).float()
div = torch.exp(torch.arange(0, D_MODEL, 2).float() * (-math.log(10000.0) / D_MODEL))
pe[:, 0::2] = torch.sin(pos * div)
pe[:, 1::2] = torch.cos(pos * div)
add_all = content.unsqueeze(0) + pe                       # (SEQ, d)
add_scores = add_all @ add_all.T                          # (SEQ, SEQ)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
for ax, S, ttl in [
    (axes[0], rope_scores, "RoPE: q·k depends only on (m − n)\n→ clean constant diagonals"),
    (axes[1], add_scores,  "Additive PE: cross-terms leak in\n→ diagonals are not clean"),
]:
    im = ax.imshow(S.detach().numpy(), cmap="RdBu_r", aspect="equal")
    ax.set_xlabel("key position  n")
    ax.set_ylabel("query position  m")
    ax.set_title(ttl)
    fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()
# -

# Let's make the "depends only on `m − n`" claim quantitative. For RoPE, every
# entry on a given diagonal (fixed `m − n`) should be identical:

offset = 5
diag_vals = [rope_scores[m, m - offset].item()
             for m in range(offset, SEQ)]
print(f"RoPE scores along the diagonal m − n = {offset}:")
print(f"  min={min(diag_vals):.4f}  max={max(diag_vals):.4f}  "
      f"(all equal? {max(diag_vals) - min(diag_vals) < 1e-4})")

# 💡 **Key Insight.** The left heatmap is *striped*: slide along any diagonal
# and the value never changes, because the only thing that survives the
# rotation is the **gap** `m − n` between the two tokens. The model literally
# cannot tell the difference between "tokens 3 and 8" and "tokens 30 and 35" —
# only that both are 5 apart. That is exactly the relative-position bias we
# want for molecules, and RoPE gives it for free, with no extra parameters.

# ---
# ## 6. RoPE inside attention, on caffeine
#
# Finally, plug RoPE into the attention we built in notebook 04. The only
# change: rotate Q and K *after* projecting and *before* the dot product. **V is
# left alone** — position should steer *who attends to whom*, not corrupt the
# content being read out.

# +
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
CORPUS = ["CCO", "CC(=O)Oc1ccccc1C(=O)O", CAFFEINE, "BrCCCl", "c1ccc2[nH]ccc2c1"]
tokenizer = AtomTokenizer.from_smiles(CORPUS)

caffeine_ids, _ = tokenizer.encode_batch([CAFFEINE], add_special_tokens=True)
caffeine_ids = torch.tensor(caffeine_ids)
caffeine_tokens = ["[CLS]"] + tokenizer.tokenize(CAFFEINE) + ["[SEP]"]

token_embedding = TokenEmbedding(tokenizer.vocab_size, D_MODEL)
emb = token_embedding(caffeine_ids)                  # (1, L, d) — NO additive PE
L = emb.size(1)

W_q = nn.Linear(D_MODEL, D_MODEL, bias=False)
W_k = nn.Linear(D_MODEL, D_MODEL, bias=False)
Q, K = W_q(emb), W_k(emb)                             # (1, L, d)

rope_L = RotaryPositionalEncoding(dim=D_MODEL, max_len=128)
Q_rot, K_rot = rope_L(Q), rope_L(K)                  # rotate by position

scale = 1.0 / math.sqrt(D_MODEL)
attn_rope = torch.softmax((Q_rot[0] @ K_rot[0].T) * scale, dim=-1)
attn_norope = torch.softmax((Q[0] @ K[0].T) * scale, dim=-1)   # no position at all

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, A, ttl in [
    (axes[0], attn_norope.detach(), "no position info  (Q·Kᵀ only)"),
    (axes[1], attn_rope.detach(),   "with RoPE  (position folded into Q, K)"),
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

# ⚠️ **Note.** With random (untrained) weights the patterns are noisy — as in
# notebook 04, we're learning the *mechanism*, not a trained behaviour. But
# notice RoPE adds a gentle distance-dependent tilt: nearby tokens get a
# systematic nudge relative to far ones, because their rotation angles are
# closer. After training, this becomes a powerful, parameter-free locality
# prior.

# 🔬 **Try This.** RoPE has no learnable parameters and no `max_len` buffer to
# add to the embeddings — it's applied on the fly inside attention. Try
# rotating to positions *beyond* the molecule's length (e.g. evaluate a 60-token
# sequence with a `max_len=128` table). Unlike a learned position embedding,
# RoPE extrapolates smoothly — a big reason it's popular for variable-length
# inputs like SMILES.

# ---
# ## Checkpoint exercises

# +
# Exercise 1 — rotation preserves norm
# --------------------------------------
# Create a random tensor of shape (1, 12, 32), rotate it with a
# RotaryPositionalEncoding(dim=32), and verify every token's L2 norm is
# unchanged. (This is what distinguishes rotating from adding.)

# YOUR CODE HERE

# --- Solution (try the exercise first, then peek) ---
# r = RotaryPositionalEncoding(dim=32, max_len=64)
# z = torch.randn(1, 12, 32)
# z_rot = r(z)
# print(f"norms preserved? "
#       f"{torch.allclose(z.norm(dim=-1), z_rot.norm(dim=-1), atol=1e-5)}")

# +
# Exercise 2 — depends only on (m − n)
# -------------------------------------
# Take a single fixed content vector. Using `apply_rotary` (or the RoPE module),
# compute rotate(q, m) · rotate(k, n) for two pairs with the SAME gap, e.g.
# (m, n) = (9, 4) and (m, n) = (20, 15). Confirm the two dot products are equal.

# YOUR CODE HERE

# --- Solution ---
# r = RotaryPositionalEncoding(dim=32, max_len=64)
# c = torch.randn(32)
# def rot_to(vec, p):
#     return apply_rotary(vec, r.cos[p], r.sin[p])
# a = rot_to(c, 9)  @ rot_to(c, 4)
# b = rot_to(c, 20) @ rot_to(c, 15)
# print(f"gap-5 dot products equal? {torch.allclose(a, b, atol=1e-4)}  "
#       f"({a.item():.4f} vs {b.item():.4f})")

# +
# Exercise 3 — swap RoPE into notebook 04's attention
# ----------------------------------------------------
# Using the caffeine Q, K from §6, build attention TWO ways: (a) additive
# sinusoidal PE added to the embedding before projecting (notebook 03 style),
# and (b) RoPE applied to Q and K. Plot both with `plot_attention_heatmap`.
# In one sentence: what does RoPE change about WHERE position enters the model?

# YOUR CODE HERE

# --- Solution ---
# # (a) additive PE before projection
# from utils.transformer_blocks import SinusoidalPositionalEncoding
# pe_mod = SinusoidalPositionalEncoding(d_model=D_MODEL, max_len=128)
# emb_pe = pe_mod(token_embedding(caffeine_ids))
# Qa, Ka = W_q(emb_pe)[0], W_k(emb_pe)[0]
# attn_add = torch.softmax((Qa @ Ka.T) * scale, dim=-1)
# plot_attention_heatmap(attn_add.detach(), caffeine_tokens, title="additive PE")
# plt.show()
# # (b) RoPE (already computed above as attn_rope)
# plot_attention_heatmap(attn_rope.detach(), caffeine_tokens, title="RoPE")
# plt.show()
# # Additive PE enters ONCE at the input (absolute); RoPE enters INSIDE
# # attention, on Q and K only, and encodes relative position by construction.

# ---
# ## What's next
#
# RoPE and **linear attention** (notebook 04.1) are two of the architectural
# swaps that turn a textbook encoder into a **MolFormer**-style one: linear
# attention for `O(N)` cost, RoPE for parameter-free relative positions.
# **Notebook 04.4** surveys yet other position schemes (ALiBi, relative-position
# bias) that chase the same relative-position goal by different means, and
# **notebook 09** assembles attention, RoPE, and MLM into a tiny end-to-end
# MolFormer.
#
# 📚 **Deep-dive sub-series**
# - **04.1**: Linear attention — MolFormer's `O(N)` swap for `softmax(QKᵀ)V`.
# - **04.2**: The Performer-style FAVOR+ feature map MolFormer actually uses.
# - **04.3 (this notebook)**: Rotary position embeddings (RoPE).
# - **04.4**: Other position encodings (ALiBi, relative-position bias).
#
# 📚 **References.**
# - Su, J. et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position
#   Embedding.* — the paper that introduced RoPE.
# - Vaswani, A. et al. (2017). *Attention Is All You Need.* — the additive
#   sinusoidal encoding RoPE is an alternative to.
# - Ross, J. et al. (2022). *Large-Scale Chemical Language Representations
#   Capture Molecular Structure and Properties* (MolFormer) — RoPE + linear
#   attention on 1.1B molecules.
