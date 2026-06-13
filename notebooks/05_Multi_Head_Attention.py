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

# <a href="https://colab.research.google.com/github/HFooladi/Transformers-For-Chemists/blob/main/notebooks/05_Multi_Head_Attention.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 05 · Multi-Head Attention
#
# Notebook 04 built a single attention head: `softmax(Q·Kᵀ / √d_k) V`. It
# works — but it forces every token to summarize *all* of its relationships
# into **one** weighted average. Caffeine's carbonyl carbon would like to
# look left at its `=O`, look right at its ring nitrogen, *and* contribute to
# the `[CLS]` summary — but one softmax row is a single probability
# distribution. It can emphasize one of those at the expense of the others;
# it cannot do all three at once.
#
# **Multi-head attention** is the fix, and it is almost embarrassingly
# simple: run several attention heads **in parallel**, each in its own cheap
# `d_k = d_model / n_heads`-dimensional sub-space, then stitch their outputs
# back together. Each head is free to learn a different "way of looking" at
# the molecule. The mechanics *inside* each head are exactly what we built in
# notebook 04 — we're just doing the same thing `n_heads` times at once.

# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# 1. Explain the expressivity limit of a single head — one averaging pattern
#    per token — and why several patterns are useful for molecules.
# 2. Perform the `d_model → (n_heads, d_k)` **split-and-reshape** and track
#    every tensor shape along the way.
# 3. Run all heads at once with a single batched matmul — no Python loop.
# 4. Recombine the heads via **concatenation + an output projection `W_O`**,
#    and explain what `W_O` is for.
# 5. Visualize per-head attention on caffeine and **quantify** how diverse the
#    heads are (even with random weights).
# 6. Connect multi-head attention to ALiBi's per-head slopes (notebook 04.4).
# 7. Watch head specialization **emerge** in a tiny optional training run.
# 8. Package everything into a reusable `MultiHeadAttention` module and
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

from utils.smiles_tokenizers import AtomTokenizer, PAD_ID, CLS_ID, SEP_ID, MASK_ID
from utils.transformer_blocks import (
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    TokenEmbedding,
)
from utils.position_biases import alibi_bias, alibi_slopes
from utils.tokenization_viz import plot_molecule_with_tokens
from utils.attention_viz import plot_attention_on_smiles, plot_per_head_grid

torch.manual_seed(0)  # so the random projections below are reproducible
# -

# ---
# ## 1. One head, one story
#
# We reuse notebook 04's hero molecule, **caffeine**, with the same embed +
# positional-encoding pipeline.

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
N_HEADS = 4
D_K = D_MODEL // N_HEADS                                             # = 8
token_embedding = TokenEmbedding(tokenizer.vocab_size, D_MODEL)
positional      = SinusoidalPositionalEncoding(d_model=D_MODEL, max_len=128)

x = positional(token_embedding(caffeine_ids))                        # (1, L, d_model)
L = x.size(1)
print(f"caffeine length L = {L}")
print(f"x shape: {tuple(x.shape)}  # (batch, seq_len, d_model)")
print(f"splitting d_model={D_MODEL} into n_heads={N_HEADS} × d_k={D_K}")
# -

plot_molecule_with_tokens(CAFFEINE, caffeine_tokens, max_row_width=12.0)
plt.show()

# A single head produces **one** attention matrix — one probability
# distribution per query token. But caffeine has several *different* kinds of
# relationship a token might care about. To make this concrete, here are two
# hand-built "wish-list" patterns: a **local** one (each token leans on its
# immediate neighbours) and a **ring-closure** one (the two ring-closure
# digits `1` and `2` reach across the molecule to their partners).

# +
def local_pattern(seq_len: int, width: float = 1.2) -> torch.Tensor:
    """Each query attends to nearby positions (a soft band around the diagonal)."""
    pos = torch.arange(seq_len).float()
    dist = (pos[:, None] - pos[None, :]).abs()
    return torch.softmax(-(dist ** 2) / (2 * width ** 2), dim=-1)


def ringclosure_pattern(tokens, seq_len: int) -> torch.Tensor:
    """Ring-digit tokens attend to each other; everyone else stays on themselves."""
    scores = torch.eye(seq_len) * 3.0                       # default: look at self
    digit_pos = [i for i, t in enumerate(tokens) if t in {"1", "2"}]
    for i in digit_pos:                                     # link ring digits together
        for j in digit_pos:
            if i != j:
                scores[i, j] = 4.0
    return torch.softmax(scores, dim=-1)


target_local = local_pattern(L)
target_ring  = ringclosure_pattern(caffeine_tokens, L)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, mat, ttl in [
    (axes[0], target_local, "pattern A — local neighbours"),
    (axes[1], target_ring,  "pattern B — ring-closure pairs"),
]:
    im = ax.imshow(mat.numpy(), cmap="Blues", vmin=0, aspect="equal")
    ax.set_xticks(range(L)); ax.set_xticklabels(caffeine_tokens, rotation=90, fontsize=6)
    ax.set_yticks(range(L)); ax.set_yticklabels(caffeine_tokens, fontsize=6)
    ax.set_title(ttl)
    fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()
# -

# 💡 **Key Insight.** A single head has to pick *one* of these (or some blurry
# average). `n` heads give you `n` independent attention patterns computed at
# once — one head can chase pattern A while another chases pattern B. And the
# cost barely changes: each head works in a `d_k = d_model/n_heads`-dim
# subspace, so the total amount of arithmetic is roughly what a single full
# head would cost.

# ---
# ## 2. The split: `d_model → (n_heads, d_k)`
#
# Here's the move that makes "several heads" cost almost nothing extra. We do
# **not** create `n_heads` separate `d_model × d_model` projections. We keep
# *one* `W_Q` (and one `W_K`, one `W_V`), project as usual to get a
# `d_model`-vector per token, then **reshape** that vector into `n_heads`
# contiguous chunks of size `d_k`. Each chunk becomes one head.

# +
W_q = nn.Linear(D_MODEL, D_MODEL, bias=False)
W_k = nn.Linear(D_MODEL, D_MODEL, bias=False)
W_v = nn.Linear(D_MODEL, D_MODEL, bias=False)

B = x.size(0)
Q_full = W_q(x)                                          # (B, L, d_model)
print(f"after projection:        {tuple(Q_full.shape)}   # (B, L, d_model)")

# .view splits the last axis into (n_heads, d_k); .transpose moves the head
# axis next to the batch so each head is a clean (L, d_k) matrix.
Q = Q_full.view(B, L, N_HEADS, D_K).transpose(1, 2)      # (B, n_heads, L, d_k)
K = W_k(x).view(B, L, N_HEADS, D_K).transpose(1, 2)
V = W_v(x).view(B, L, N_HEADS, D_K).transpose(1, 2)
print(f"after split into heads:  {tuple(Q.shape)}   # (B, n_heads, L, d_k)")
# -

# A picture of the split: one `d_model`-wide projected vector is sliced into
# `n_heads` coloured segments, and each segment is handed to its own head.

fig, ax = plt.subplots(figsize=(9, 3.2))
colours = plt.cm.tab10(np.linspace(0, 1, N_HEADS))
seg = D_MODEL / N_HEADS
# Top bar: the full d_model projected vector, partitioned by colour.
for h in range(N_HEADS):
    ax.add_patch(plt.Rectangle((h * seg, 2.2), seg, 0.7, facecolor=colours[h], edgecolor="black"))
ax.text(D_MODEL / 2, 3.05, f"one projected vector  (d_model = {D_MODEL})", ha="center", fontsize=11)
# Bottom: the same segments pulled apart into n_heads small d_k vectors.
gap = 1.4
for h in range(N_HEADS):
    x0 = h * (seg + gap)
    ax.add_patch(plt.Rectangle((x0, 0.4), seg, 0.7, facecolor=colours[h], edgecolor="black"))
    ax.text(x0 + seg / 2, 0.1, f"head {h}\n(d_k={D_K})", ha="center", va="top", fontsize=8)
    ax.annotate("", xy=(x0 + seg / 2, 1.2), xytext=(h * seg + seg / 2, 2.1),
                arrowprops=dict(arrowstyle="->", color=colours[h], lw=1.6))
ax.set_xlim(-0.5, max(D_MODEL, N_HEADS * (seg + gap)) + 0.5)
ax.set_ylim(-0.7, 3.4)
ax.axis("off")
ax.set_title("The split is a reshape — no new parameters")
plt.tight_layout()
plt.show()

# ⚠️ **Note.** The split is a **view**, not new weights. `W_Q` is still a
# single `d_model × d_model` matrix. The heads only become *independent*
# because head `h`'s query slice is matmul'd against head `h`'s key slice —
# the slices never cross. That's why `n_heads` heads cost the same parameters
# and (almost) the same compute as one full-width head.

# ---
# ## 3. All heads at once
#
# Scaled dot-product attention is *identical* inside each head; we just keep
# the head axis around and let `torch.matmul` broadcast over it. The one
# subtlety: we scale by `1/√d_k`, **not** `1/√d_model` — each dot product now
# runs over `d_k` dimensions, so (recall notebook 04 §5) its variance is `d_k`.

# +
scale = 1.0 / math.sqrt(D_K)
scores = torch.matmul(Q, K.transpose(-2, -1)) * scale    # (B, n_heads, L, L)
attn = torch.softmax(scores, dim=-1)                     # (B, n_heads, L, L)
head_out = torch.matmul(attn, V)                         # (B, n_heads, L, d_k)

print(f"scores:   {tuple(scores.shape)}   # (B, n_heads, L, L)")
print(f"attn:     {tuple(attn.shape)}   # one (L, L) pattern per head")
print(f"head_out: {tuple(head_out.shape)}   # (B, n_heads, L, d_k)")
row_sums = attn[0].sum(dim=-1)
print(f"every head's rows sum to 1? {torch.allclose(row_sums, torch.ones_like(row_sums))}")
# -

# ⚠️ **Note.** Scaling by `√d_k` instead of `√d_model` is easy to get wrong
# and easy to miss. With four heads the two differ by a factor of 2 in the
# logits — enough to noticeably change how peaky the softmax is. The correct
# choice is always the dimension the dot product actually runs over: `d_k`.

print(f"√d_k = {math.sqrt(D_K):.3f}   vs   √d_model = {math.sqrt(D_MODEL):.3f}")

# ---
# ## 4. Per-head attention on caffeine
#
# Now the payoff visualization. Each head, with *random* weights, already
# produces a **different** `(L, L)` pattern, because each head's Q/K slices
# project caffeine into a different random subspace. `plot_per_head_grid`
# lays all the heads out side by side on a shared colour scale.

plot_per_head_grid(attn[0].detach(), caffeine_tokens,
                   title="Per-head attention on caffeine (random weights)")
plt.show()

# 🧪 **Chemical Intuition.** Right now these patterns are noise — the weights
# are random. But notice they are *different* noises: the heads are not
# redundant copies. After training (notebooks 07 & 09), this is exactly where
# chemistry shows up — one head tends to ring-closure partners, another to
# carbonyl `C=O` adjacency, another acts as a `[CLS]` aggregator. The
# *capacity* for that specialization is built into the architecture; §7 below
# gives a first glimpse of it emerging.

# ---
# ## 5. Concatenate the heads + the output projection `W_O`
#
# Each head returns a `d_k`-dim vector per token. We **concatenate** them back
# into a `d_model`-dim vector (the exact inverse of the §2 split), then apply
# one more learned projection `W_O`. Concatenation just stacks the heads;
# `W_O` is what lets them **talk to each other** and blend into a single
# representation.

# Inverse of the split: move the head axis back and merge it into d_model.
merged = head_out.transpose(1, 2).reshape(B, L, D_MODEL)   # (B, L, d_model)
W_o = nn.Linear(D_MODEL, D_MODEL)
output = W_o(merged)                                       # (B, L, d_model)
print(f"merged (concat heads): {tuple(merged.shape)}")
print(f"output (after W_O):    {tuple(output.shape)}   # back to one vector per token")

# 💡 **Key Insight.** Without `W_O` the heads would be concatenated but never
# *combined* — downstream layers would see `n_heads` separate opinions glued
# end to end. `W_O` mixes them into one consensus vector. It's also why the
# **order** of the heads doesn't matter: a permutation of the heads is just a
# permutation of `W_O`'s input columns, which the model can absorb.

# What does one head's `[CLS]` row look like, painted back onto the SMILES?

plot_attention_on_smiles(attn[0, 0, 0].detach(), caffeine_tokens, query_index=0,
                         title="head 0 · attention from [CLS]  (random weights)")
plt.show()

# ---
# ## 6. How different are the heads, really?
#
# We claimed the heads are diverse. Let's *measure* it. For every pair of
# heads we compute `1 − cosine-similarity` between their flattened attention
# matrices: 0 means identical patterns, larger means more different. We also
# compute each head's mean row **entropy** — a peaky head (low entropy)
# focuses on a few tokens; a diffuse head (high entropy) spreads its
# attention out.

# +
flat = attn[0].reshape(N_HEADS, -1)                       # (n_heads, L*L)
flat = flat / flat.norm(dim=-1, keepdim=True)
cos = flat @ flat.T                                       # (n_heads, n_heads)
dissim = (1.0 - cos).clamp(min=0).detach().numpy()

# Mean row entropy per head (in nats).
eps = 1e-9
row_entropy = -(attn[0] * (attn[0] + eps).log()).sum(-1)  # (n_heads, L)
mean_entropy = row_entropy.mean(-1).detach().numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
im = ax1.imshow(dissim, cmap="magma", vmin=0)
ax1.set_xticks(range(N_HEADS)); ax1.set_yticks(range(N_HEADS))
ax1.set_xlabel("head"); ax1.set_ylabel("head")
ax1.set_title("pairwise dissimilarity  (1 − cosine)")
for i in range(N_HEADS):
    for j in range(N_HEADS):
        ax1.text(j, i, f"{dissim[i, j]:.2f}", ha="center", va="center",
                 color="white" if dissim[i, j] > dissim.max() / 2 else "black", fontsize=9)
fig.colorbar(im, ax=ax1, shrink=0.8)

ax2.bar(range(N_HEADS), mean_entropy, color=plt.cm.tab10(np.linspace(0, 1, N_HEADS)))
ax2.set_xlabel("head"); ax2.set_ylabel("mean row entropy (nats)")
ax2.set_title("how diffuse is each head?")
ax2.set_xticks(range(N_HEADS))
plt.tight_layout()
plt.show()
# -

# 🔬 **Try This.** Re-run with `N_HEADS = 8` (so `d_k = 4`). More heads means
# each works in a smaller subspace — cheaper, and often *more* diverse
# pairwise. There's a catch, though: past some point heads start to
# **duplicate** each other and many can be pruned with no loss. That
# redundancy is the whole subject of the deep-dive in notebook 05.1.

# ---
# ## 7. ALiBi pairs naturally with multi-head
#
# Notebook 04.4 introduced **ALiBi** — a per-head bias `−slopeₕ · |i − j|`
# added to the scores before softmax, nudging each head toward nearby tokens.
# That "per-head" qualifier finally has a home: with `n_heads` heads we can
# give each a *different* slope, so some heads stay strictly local while
# others see the whole molecule.

# +
slopes = alibi_slopes(N_HEADS)
bias = alibi_bias(L, N_HEADS)                             # (n_heads, L, L)
print(f"ALiBi slopes (one per head): {[round(s, 3) for s in slopes.tolist()]}")

scores_alibi = scores + bias.unsqueeze(0)                 # broadcast over batch
attn_alibi = torch.softmax(scores_alibi, dim=-1)

plot_per_head_grid(attn_alibi[0].detach(), caffeine_tokens,
                   title="Per-head attention + ALiBi (small slope = global, large = local)")
plt.show()
# -

# 💡 **Key Insight.** Look across the heads: the small-slope head is almost
# unchanged (it still sees everywhere), while the large-slope head collapses
# onto the diagonal (it only sees its neighbours). **Multi-head is *why*
# ALiBi's per-head slopes exist** — with a single head you'd be forced to pick
# one locality scale for the entire model.

# ---
# ## 8. (Optional) Watch specialization emerge — a tiny MLM run
#
# Everything above used *random* weights, so the per-head patterns were
# structured-but-meaningless. This optional section trains a tiny one-layer
# model with a masked-language-modelling objective (predict hidden atoms) for
# a few hundred steps, then re-plots the heads. We follow the same self-
# contained recipe as notebook 04.1 — no special training utilities. It runs
# in well under a minute on CPU; skip it if you just want the mechanism.

# +
import urllib.request
from pathlib import Path

# Download a few hundred real molecules (MoleculeNet BBBP), cached locally.
DATA_DIR = Path("notebooks/data") if Path("notebooks/data").exists() else Path(f"{REPO_NAME}/notebooks/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
BBBP_CSV = DATA_DIR / "BBBP.csv"
if not BBBP_CSV.exists():
    urllib.request.urlretrieve(
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv", BBBP_CSV
    )

bbbp_smiles = []
with open(BBBP_CSV) as f:
    next(f)
    for line in f:
        parts = line.strip().split(",")
        if len(parts) >= 4:
            bbbp_smiles.append(parts[-1])
try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    bbbp_smiles = [s for s in bbbp_smiles if Chem.MolFromSmiles(s) is not None]
except ImportError:
    pass

# Keep only molecules short enough for a quick toy run, and train a tokenizer.
bbbp_smiles = [s for s in bbbp_smiles if len(s) <= 80][:600]
train_tok = AtomTokenizer.from_smiles(bbbp_smiles)
print(f"training corpus: {len(bbbp_smiles)} molecules, vocab {train_tok.vocab_size}")

# +
class TinyMLM(nn.Module):
    """Embedding + positional + one multi-head attention layer + MLM head."""

    def __init__(self, vocab_size, d_model=D_MODEL, n_heads=N_HEADS, max_len=128):
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=0.0)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask):
        h = self.pos(self.embed(input_ids))
        a, attn = self.attn(self.norm(h), mask=attention_mask)
        h = h + a
        return self.head(h), attn


def make_mlm_batch(ids_list, mask_list, vocab_size, mask_prob=0.15,
                   special_ids=(PAD_ID, CLS_ID, SEP_ID, MASK_ID)):
    """Replace 15% of non-special tokens with [MASK]; label is -100 elsewhere."""
    input_ids = torch.tensor(ids_list)
    attention_mask = torch.tensor(mask_list)
    eligible = torch.ones_like(input_ids, dtype=torch.bool)
    for sid in special_ids:
        eligible &= input_ids != sid
    mlm_mask = (torch.rand_like(input_ids, dtype=torch.float) < mask_prob) & eligible
    corrupted = input_ids.clone()
    corrupted[mlm_mask] = MASK_ID
    labels = torch.full_like(input_ids, -100)
    labels[mlm_mask] = input_ids[mlm_mask]
    return corrupted, labels, attention_mask


def train_tiny(model, smiles, tok, n_steps=400, batch_size=32, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for step in range(n_steps):
        idx = torch.randint(0, len(smiles), (batch_size,))
        batch = [smiles[i] for i in idx]
        ids, mask = tok.encode_batch(batch, add_special_tokens=True)
        corrupted, labels, attn_mask = make_mlm_batch(ids, mask, tok.vocab_size)
        logits, _ = model(corrupted, attn_mask)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
        )
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses


torch.manual_seed(0)
mlm = TinyMLM(train_tok.vocab_size)
losses = train_tiny(mlm, bbbp_smiles, train_tok)

fig, ax = plt.subplots(figsize=(7.5, 4))
smooth = np.convolve(losses, np.ones(15) / 15, mode="valid")
ax.plot(smooth, color="#4c72b0", lw=2)
ax.set_xlabel("optimizer step (15-step moving average)")
ax.set_ylabel("MLM cross-entropy loss")
ax.set_title("Tiny MLM training — the heads are now learning")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
print(f"loss: {losses[0]:.3f} → {np.mean(losses[-20:]):.3f}")
# -

# Now re-encode caffeine with the **trained** model's tokenizer and look at
# its heads. The patterns are no longer random — they have started to pick up
# structure (compare with the noise in §4).

# +
caf_ids2, caf_mask2 = train_tok.encode_batch([CAFFEINE], add_special_tokens=True)
caf_tokens2 = ["[CLS]"] + train_tok.tokenize(CAFFEINE) + ["[SEP]"]
mlm.eval()
with torch.no_grad():
    _, attn_trained = mlm(torch.tensor(caf_ids2), torch.tensor(caf_mask2))

plot_per_head_grid(attn_trained[0], caf_tokens2,
                   title="Per-head attention on caffeine — AFTER tiny training")
plt.show()
# -

# 🔬 **Try This.** Crank `n_steps` up to a few thousand and watch the heads
# sharpen further. Even this tiny single-layer model on 600 molecules starts
# to differentiate its heads — proof that the *capacity* we built in §1–5 is
# real, not decorative. Notebook 09 trains the full stack and the patterns
# become genuinely chemically interpretable.

# ---
# ## 9. The same thing, packaged
#
# All of §2–5 lives in one `nn.Module` — `MultiHeadAttention` — in
# `utils/transformer_blocks.py`. We copy our hand-rolled projections into it
# (zeroing the biases, since our hand-roll used `bias=False`) and confirm the
# module reproduces the §3 result to numerical precision. We test on a batch
# that mixes **ethanol** and **caffeine** so padding masking is exercised too.

# +
batch_smiles = ["CCO", CAFFEINE]
batch_ids, batch_mask = tokenizer.encode_batch(batch_smiles, add_special_tokens=True)
batch_ids  = torch.tensor(batch_ids)                       # (2, L_max)
batch_mask = torch.tensor(batch_mask)                      # (2, L_max)
xb = positional(token_embedding(batch_ids))                # (2, L, d_model)

mha = MultiHeadAttention(d_model=D_MODEL, n_heads=N_HEADS, dropout=0.0)
mha.w_q.weight.data = W_q.weight.data.clone(); mha.w_q.bias.data.zero_()
mha.w_k.weight.data = W_k.weight.data.clone(); mha.w_k.bias.data.zero_()
mha.w_v.weight.data = W_v.weight.data.clone(); mha.w_v.bias.data.zero_()
mha.w_o.weight.data = W_o.weight.data.clone(); mha.w_o.bias.data = W_o.bias.data.clone()

out_m, attn_m = mha(xb, mask=batch_mask)
print(f"output shape:    {tuple(out_m.shape)}   # (B, L, d_model)")
print(f"attention shape: {tuple(attn_m.shape)}   # (B, n_heads, L, L)")

# Re-derive the single-molecule (caffeine) result by hand to compare.
out_hand = W_o(torch.matmul(
    torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * scale, dim=-1), V
).transpose(1, 2).reshape(B, L, D_MODEL))
# caffeine is batch index 1; its first L positions are real (no padding).
print(f"matches hand-rolled caffeine output? "
      f"{torch.allclose(out_m[1, :L], out_hand[0], atol=1e-5)}")

n_real_eth = int(batch_mask[0].sum().item())
print(f"no head attends to ethanol's [PAD] columns? "
      f"{float(attn_m[0, :, :, n_real_eth:].max()) == 0.0}")
# -

# ⚠️ **Note.** `MultiHeadAttention` returns its weights with shape
# `(B, n_heads, L, L)` — the extra head axis compared with notebook 04's
# `ScaledDotProductAttention`, which returned `(B, L, L)`. That axis is what
# `plot_per_head_grid` consumes, and what notebook 06 will collect layer by
# layer.

plot_per_head_grid(attn_m[1].detach(), caffeine_tokens,
                   title="MultiHeadAttention module — caffeine, all heads")
plt.show()

# ---
# ## Checkpoint exercises

# +
# Exercise 1 — shape surgery
# ---------------------------
# Given Q_full of shape (2, 10, 32) and n_heads=4, (a) split it into
# (2, 4, 10, 8), then (b) merge it back to (2, 10, 32). Verify the round-trip
# returns the original tensor (torch.allclose).

# YOUR CODE HERE

# --- Solution (try the exercise first, then peek) ---
# qf = torch.randn(2, 10, 32)
# nh, dk = 4, 8
# split = qf.view(2, 10, nh, dk).transpose(1, 2)          # (2, 4, 10, 8)
# merged = split.transpose(1, 2).reshape(2, 10, 32)       # (2, 10, 32)
# print(f"split:  {tuple(split.shape)}")
# print(f"merged: {tuple(merged.shape)}")
# print(f"round-trip exact? {torch.allclose(qf, merged)}")   # True

# +
# Exercise 2 — the √d_k vs √d_model trap
# --------------------------------------
# Take one head's scores from §3. Softmax it twice: once scaled by 1/√d_k
# (correct) and once by 1/√d_model (wrong). Report the max probability of the
# [CLS] row each way. Which one saturates more, and why is that bad? (Recall
# notebook 04 §5.)

# YOUR CODE HERE

# --- Solution ---
# raw = torch.matmul(Q, K.transpose(-2, -1))[0, 0]        # head 0, (L, L)
# p_dk = torch.softmax(raw / math.sqrt(D_K),    dim=-1)[0]
# p_dm = torch.softmax(raw / math.sqrt(D_MODEL), dim=-1)[0]
# print(f"max prob with 1/√d_k    : {p_dk.max():.3f}")
# print(f"max prob with 1/√d_model: {p_dm.max():.3f}  (different scale → different sharpness)")
# # Wrong scaling shifts how peaky the softmax is; in a real (large d_k) model
# # the wrong factor pushes the softmax toward saturation, shrinking gradients.

# +
# Exercise 3 — per-head masking + ALiBi
# -------------------------------------
# Build the padding mask for ["CCO", "CC(=O)Oc1ccccc1C(=O)O"], run
# MultiHeadAttention, and verify (i) no head attends to any pad column, and
# (ii) after adding alibi_bias to the scores the rows STILL sum to 1.

# YOUR CODE HERE

# --- Solution ---
# ids_ex, mask_ex = tokenizer.encode_batch(
#     ["CCO", "CC(=O)Oc1ccccc1C(=O)O"], add_special_tokens=True)
# ids_ex, mask_ex = torch.tensor(ids_ex), torch.tensor(mask_ex)
# xe = positional(token_embedding(ids_ex))
# _, attn_ex = mha(xe, mask=mask_ex)
# n_real = int(mask_ex[0].sum())
# print(f"(i) attention onto pad cols (short seq): "
#       f"{attn_ex[0, :, :, n_real:].max().item()}  (= 0.0)")
# Le = xe.size(1)
# sc = (torch.randn(2, N_HEADS, Le, Le) + alibi_bias(Le, N_HEADS))
# keep = mask_ex.bool().unsqueeze(1).unsqueeze(1)
# at = torch.softmax(sc.masked_fill(~keep, float("-inf")), dim=-1)
# print(f"(ii) real rows still sum to 1? "
#       f"{torch.allclose(at[0, :, :n_real, :].sum(-1), torch.ones(N_HEADS, n_real))}")

# ---
# ## What's next
#
# Heads give us several parallel "ways of looking" at a molecule. But notice
# what attention still *cannot* do: it only ever **moves** information between
# tokens (a weighted average of other tokens' values). There's no per-token
# non-linear processing, and a single attention layer is shallow. **Notebook
# 06** wraps multi-head attention with a position-wise **feed-forward
# network**, **residual connections**, and **LayerNorm** to form a complete,
# composable **transformer block** — the unit we'll stack to build the real
# model.
#
# 📚 **Deep-dive sub-series**
# - **05.1**: Head specialization, redundancy, and pruning — do we really need
#   all these heads? ("Are Sixteen Heads Really Better than One?")
#
# 📚 **References.**
# - Vaswani, A. et al. (2017). *Attention Is All You Need.* — introduces
#   multi-head attention.
# - Michel, P., Levy, O. & Neubig, G. (2019). *Are Sixteen Heads Really Better
#   than One?* — head importance and pruning (notebook 05.1).
# - Press, O., Smith, N. & Lewis, M. (2022). *Train Short, Test Long:
#   Attention with Linear Biases Enables Input Length Extrapolation (ALiBi).*
# - Ross, J. et al. (2022). *Large-Scale Chemical Language Representations
#   Capture Molecular Structure and Properties (MolFormer).*
