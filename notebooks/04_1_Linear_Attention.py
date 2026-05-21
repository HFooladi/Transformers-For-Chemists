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

# <a href="https://colab.research.google.com/github/HFooladi/Transformers-For-Chemists/blob/main/notebooks/04_1_Linear_Attention.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 04.1 · Linear Attention — MolFormer's `O(N)` swap for softmax
#
# Notebook 04 built **scaled dot-product self-attention** from scratch:
#
# $$
# \mathrm{Attention}(x) = \mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)\,V .
# $$
#
# That expression hides a quiet cost. The matrix `QKᵀ` is `(L, L)` — one entry
# per *pair* of tokens. Compute it, softmax it, multiply by `V`: every step
# scales as `L²`. For a single caffeine string (`L ≈ 25`) that's fine. For
# **MolFormer pre-trained on ~1.1 billion SMILES** (Ross et al., 2022) it is
# not — even when most molecules are short, the `L²` constant factor stacks up
# across a billion forward passes per epoch.
#
# MolFormer's answer is **linear attention**: a kernel-trick reformulation
# that costs `O(L · d²)` instead of `O(L² · d)`, *linear* in sequence length.
# This deep-dive notebook explains the math, builds it from scratch in the
# same style as notebook 04, and shows side-by-side what softmax and linear
# attention produce on the same molecule — including a tiny MLM training run
# to confirm that both flavours actually learn.
#
# **Scope.** This is an analysis-and-tiny-training notebook. The `O(N)` win
# matters at scale; the toy training run here just shows that linear
# attention isn't a degenerate operation. Full pre-training lives in
# **notebook 09 (Tiny MolFormer)**.

# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# 1. Explain why softmax-attention's cost is `O(L²)` and which line of code
#    is the culprit.
# 2. Derive the **kernel-trick reformulation** that turns
#    `softmax(QKᵀ)V` into a per-query linear combination of two precomputed
#    sums.
# 3. Implement **`LinearAttention`** from scratch using the
#    `φ(x) = elu(x)+1` feature map (Katharopoulos et al., 2020).
# 4. Compare softmax and linear attention **side-by-side on caffeine** and
#    describe how the patterns differ.
# 5. Measure wall-clock cost vs. sequence length and locate the **crossover
#    point** where linear attention starts to win.
# 6. Run a **tiny MLM training comparison** and read the resulting loss
#    curves honestly — what the toy run does and does not tell you.
# 7. Name **MolFormer's actual feature map** (Performer / FAVOR+) and
#    where it sits relative to the simpler ELU+1 you implemented.

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

ensure_environment(["torch", "rdkit", "matplotlib"])

# +
import math
import time
from pathlib import Path
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils.smiles_tokenizers import (
    AtomTokenizer,
    SPECIAL_TOKENS,
    PAD_ID, CLS_ID, SEP_ID, MASK_ID,
)
from utils.transformer_blocks import (
    ScaledDotProductAttention,
    SinusoidalPositionalEncoding,
    TokenEmbedding,
)
from utils.linear_attention import LinearAttention, elu_feature_map
from utils.tokenization_viz import plot_molecule_with_tokens
from utils.attention_viz import plot_attention_heatmap, plot_attention_on_smiles

torch.manual_seed(0)
np.random.seed(0)
# -

# ---
# ## A. The cost of softmax — what makes attention quadratic
#
# Pulling the relevant line straight out of notebook 04:
#
# ```python
# scores = torch.matmul(Q, K.transpose(-2, -1)) * scale   # (B, L, L)
# attn   = F.softmax(scores, dim=-1)                      # (B, L, L)
# out    = torch.matmul(attn, V)                          # (B, L, d_model)
# ```
#
# That `(B, L, L)` tensor is the whole story. We have to materialize it
# (because softmax needs to see all `L` keys at once to normalize), and then
# multiply it by `V`. **Both of those operations cost `O(L² · d)`.**
#
# Let's quantify "how big does `L` actually get?" for chemistry. We'll
# tokenize a sample of real SMILES from MoleculeNet's BBBP set and look at
# how `L²` (the size of the attention matrix) scales.

# +
DATA_DIR = Path("notebooks/data") if Path("notebooks/data").exists() else Path(f"{REPO_NAME}/notebooks/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
BBBP_CSV = DATA_DIR / "BBBP.csv"
if not BBBP_CSV.exists():
    urllib.request.urlretrieve(
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        BBBP_CSV,
    )

bbbp_smiles = []
with open(BBBP_CSV) as f:
    next(f)  # header
    for line in f:
        parts = line.strip().split(",")
        if len(parts) >= 4:
            bbbp_smiles.append(parts[-1])

# Lightweight RDKit filter — drop anything that won't parse — so the length
# distribution is on real molecules, not on garbled rows.
try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    bbbp_smiles = [s for s in bbbp_smiles if Chem.MolFromSmiles(s) is not None]
except ImportError:
    pass

tokenizer = AtomTokenizer.from_smiles(bbbp_smiles)
lengths = np.array([len(tokenizer.tokenize(s)) for s in bbbp_smiles])
print(f"BBBP: {len(bbbp_smiles)} valid SMILES")
print(f"token-length: median {int(np.median(lengths))}, "
      f"p90 {int(np.percentile(lengths, 90))}, max {int(lengths.max())}")

# +
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.hist(lengths, bins=40, color="#4c72b0", edgecolor="white")
ax1.set_xlabel("sequence length L (atom tokens)")
ax1.set_ylabel("number of molecules")
ax1.set_title("BBBP — token-length distribution")

ax2.hist(lengths ** 2, bins=40, color="#dd8452", edgecolor="white")
ax2.set_xlabel("attention-matrix size  L²  (entries per molecule)")
ax2.set_ylabel("number of molecules")
ax2.set_title("BBBP — softmax attention work per molecule")

fig.tight_layout()
plt.show()
# -

# 💡 **Key Insight.** The right-hand histogram has a long tail that the
# left-hand one does not. A molecule twice as long doesn't take twice as much
# attention compute — it takes **four times** as much. Multiply by a billion
# molecules × dozens of pre-training epochs and the constant factor on `L²`
# becomes the rate-limiting cost of MolFormer-scale training.

# ---
# ## B. The kernel-trick view — turning `L²` into `L`
#
# Here's the trick (Katharopoulos et al., 2020, *Transformers are RNNs*).
# Softmax-attention can be written entry-wise as
#
# $$
# \mathrm{out}_i \;=\; \frac{\sum_j \mathrm{sim}(Q_i, K_j)\, V_j}{\sum_j \mathrm{sim}(Q_i, K_j)},
# \qquad \mathrm{sim}(q, k) = \exp(q^\top k / \sqrt{d}) .
# $$
#
# The `exp(qᵀk)` similarity is what makes the formula `O(L²)`: every pair
# `(i, j)` has its own value, and the matrix can't be factored.
#
# **The swap.** Replace `sim(q, k)` with a *factorizable* kernel of the form
# `φ(q)ᵀ φ(k)` for some feature map `φ: ℝᵈ → ℝᵈ`. Then
#
# $$
# \mathrm{out}_i
# \;=\; \frac{\sum_j \big(\phi(Q_i)^\top \phi(K_j)\big)\, V_j}
#            {\sum_j \big(\phi(Q_i)^\top \phi(K_j)\big)}
# \;=\; \frac{\phi(Q_i)^\top \big(\sum_j \phi(K_j)\, V_j^\top\big)}
#            {\phi(Q_i)^\top \big(\sum_j \phi(K_j)\big)} .
# $$
#
# Read the right-hand side carefully:
#
# - The two inner sums **don't depend on `i`**. We compute them **once per
#   molecule**, in `O(L · d²)` and `O(L · d)` time respectively.
# - For each query we do one dot product with each sum — `O(L · d²)` and
#   `O(L · d)`.
# - **Total cost: `O(L · d²)`** — linear in `L`. No `(L, L)` matrix ever gets
#   built (except by us, for visualization).
#
# That is the whole game. The only design choice left is which `φ` to use.

# 🧪 **Chemical Intuition.** Softmax says: "out of all keys, the one with
# the highest match wins by a big margin." Linear attention with a smooth
# `φ` says: "average everyone by their match." A token like the carbonyl
# `C` in caffeine that "really wants to look at" `=O` will still pull more
# value from `=O` than from a random ring atom — but the weighting is
# softer, less spiky. We will see this directly in Section D.

# **The feature map.** We use the simplest correct choice (Katharopoulos
# et al., 2020):
#
# $$
# \phi(x) = \mathrm{elu}(x) + 1
# $$
#
# applied **element-wise**. Two properties matter:
#
# 1. **Positive everywhere** — since `elu(x) > -1` for all `x`, `φ(x) > 0`.
#    That keeps the denominator sum positive and the row of weights
#    interpretable as something attention-like.
# 2. **No learned parameters** — `φ` itself is just a deterministic
#    activation. All the learning still happens in the same `W_Q, W_K, W_V`
#    projections as before.
#
# MolFormer actually uses a more sophisticated **Performer / FAVOR+** feature
# map (Choromanski et al., 2021) that approximates softmax more closely. We
# stick with ELU+1 for pedagogy — it is five lines of PyTorch and shares the
# same `O(L)` complexity. The Performer version gets its own supplementary
# notebook (`04_1_FAVOR_Performer.ipynb`).

xs = torch.linspace(-3, 3, 200)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(xs.numpy(), torch.softmax(xs, dim=0).numpy() * len(xs), label="softmax (rescaled)", lw=2)
ax.plot(xs.numpy(), elu_feature_map(xs).numpy(), label=r"$\varphi(x) = \mathrm{elu}(x) + 1$", lw=2)
ax.axhline(0, color="grey", lw=0.5)
ax.set_xlabel("x")
ax.set_ylabel("output")
ax.set_title("Softmax is sharp; ELU+1 is smooth and strictly positive")
ax.legend()
fig.tight_layout()
plt.show()

# ---
# ## C. Building `LinearAttention` from scratch
#
# We rebuild the operation step by step on caffeine, then wrap it in an
# `nn.Module` that mirrors `ScaledDotProductAttention`'s signature exactly so
# the two are drop-in interchangeable.
#
# We re-use exactly the **same input pipeline** as notebook 04 so the
# heatmaps in Section D are directly comparable: same hero molecule, same
# `D_MODEL`, same atom tokenizer, same `[CLS] ... [SEP]` framing.

# +
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
caffeine_tokens = ["[CLS]"] + tokenizer.tokenize(CAFFEINE) + ["[SEP]"]
caffeine_ids = torch.tensor(tokenizer.encode(CAFFEINE, add_special_tokens=True)).unsqueeze(0)

D_MODEL = 32
token_embedding = TokenEmbedding(tokenizer.vocab_size, D_MODEL)
positional      = SinusoidalPositionalEncoding(d_model=D_MODEL, max_len=128)

x = positional(token_embedding(caffeine_ids))   # (1, L, d_model)
print(f"L = {x.size(1)},  d_model = {D_MODEL}")
print(f"x shape: {tuple(x.shape)}")
# -

# Shared Q/K/V projections. We initialise them once and use the **same
# weights** for both attentions in Section D, so the only difference between
# the two heatmaps will be the kernel — softmax vs. ELU+1.

# +
torch.manual_seed(42)
W_q = nn.Linear(D_MODEL, D_MODEL)
W_k = nn.Linear(D_MODEL, D_MODEL)
W_v = nn.Linear(D_MODEL, D_MODEL)

Q = W_q(x)   # (1, L, d_model)
K = W_k(x)
V = W_v(x)
# -

# **Step 1 — apply the feature map.** Element-wise; no learned parameters.

phi_Q = elu_feature_map(Q)   # (1, L, d_model)
phi_K = elu_feature_map(K)   # (1, L, d_model)
print(f"φ(Q) min / max: {phi_Q.min().item():.3f} / {phi_Q.max().item():.3f}")
print(f"φ(K) min / max: {phi_K.min().item():.3f} / {phi_K.max().item():.3f}")
print(f"all strictly positive? {(phi_Q > 0).all().item() and (phi_K > 0).all().item()}")

# 💡 **Key Insight.** `φ(Q)` and `φ(K)` live in the same shape as `Q` and
# `K`. The feature map does not blow up the dimension. The "kernel trick"
# part is purely about *how we multiply them*, not about expanding to a
# higher-dimensional space.

# **Step 2 — precompute the two sums over keys.** This is the move that
# eliminates the `(L, L)` matrix.

KV    = torch.einsum("bld,blm->bdm", phi_K, V)    # (1, d_model, d_model)
K_sum = phi_K.sum(dim=1)                          # (1, d_model)
print(f"KV shape:    {tuple(KV.shape)}   # one (d, d) matrix per molecule")
print(f"K_sum shape: {tuple(K_sum.shape)}   # one (d,) vector per molecule")

# 💡 **Key Insight.** `KV` has shape `(d_model, d_model)` regardless of how
# long the molecule is. A 1 000-token molecule has the same-sized `KV` as a
# 10-token one. That's exactly why the cost stops being quadratic.

# **Step 3 — per-query division.** Each row's output is one dot product
# with `KV` divided by one dot product with `K_sum`.

numerator   = torch.einsum("bld,bdm->blm", phi_Q, KV)            # (1, L, d_model)
denominator = torch.einsum("bld,bd->bl",   phi_Q, K_sum).unsqueeze(-1)  # (1, L, 1)
linear_out_manual = numerator / denominator.clamp(min=1e-6)      # (1, L, d_model)
print(f"linear-attention output shape: {tuple(linear_out_manual.shape)}")

# **Step 4 — wrap it.** The canonical version lives in
# `utils/linear_attention.py`. We instantiate it, copy in the same Q/K/V
# projections, and check the module agrees with the hand-rolled forward
# pass to numerical tolerance.

# +
linear_attn = LinearAttention(d_model=D_MODEL, dropout=0.0)
with torch.no_grad():
    linear_attn.w_q.weight.copy_(W_q.weight); linear_attn.w_q.bias.copy_(W_q.bias)
    linear_attn.w_k.weight.copy_(W_k.weight); linear_attn.w_k.bias.copy_(W_k.bias)
    linear_attn.w_v.weight.copy_(W_v.weight); linear_attn.w_v.bias.copy_(W_v.bias)

linear_out_module, linear_attn_matrix = linear_attn(x)
print(f"hand-rolled vs module agree? "
      f"{torch.allclose(linear_out_manual, linear_out_module, atol=1e-6)}")
# -

# ⚠️ **Note — masking.** Linear attention has no softmax to saturate, so
# the `scores.masked_fill(~keep, -inf)` trick from notebook 04 doesn't
# apply. Instead, padded *keys* are zeroed in `φ(K)` itself — that
# subtracts their contribution from both the numerator and the
# denominator, exactly what we want. The `LinearAttention` module handles
# this internally; we will exercise it in Checkpoint Exercise 2.

# ---
# ## D. Side-by-side attention patterns on caffeine
#
# Now the payoff: the same `(Q, K, V)` projected the same way, attended to
# two different ways. The softmax path will be sharp; the linear path will
# be smoother. The two heatmaps share the same axis order as the one you
# saw at the end of notebook 04.

# +
softmax_attn = ScaledDotProductAttention(d_model=D_MODEL, dropout=0.0)
with torch.no_grad():
    softmax_attn.w_q.weight.copy_(W_q.weight); softmax_attn.w_q.bias.copy_(W_q.bias)
    softmax_attn.w_k.weight.copy_(W_k.weight); softmax_attn.w_k.bias.copy_(W_k.bias)
    softmax_attn.w_v.weight.copy_(W_v.weight); softmax_attn.w_v.bias.copy_(W_v.bias)

with torch.no_grad():
    softmax_out, softmax_matrix = softmax_attn(x)
    linear_out,  linear_matrix  = linear_attn(x)
# -

# Heatmaps. The helper auto-picks a `Blues` colormap because both matrices
# are non-negative softmax-like outputs.

plot_attention_heatmap(
    softmax_matrix[0].detach(), caffeine_tokens,
    title="softmax attention   softmax(QKᵀ/√d) — sharp peaks",
)
plt.show()

plot_attention_heatmap(
    linear_matrix[0].detach(), caffeine_tokens,
    title="linear attention   φ(Q)φ(K)ᵀ / Σφ(K) — softer mass",
)
plt.show()

# 🧪 **Chemical Intuition.** Pick any row (a "query token") and compare
# how its weight is distributed across the columns. In the softmax matrix
# you typically see one or two columns dominating — the network has
# already committed most of the query's attention to a handful of keys.
# In the linear matrix the mass is spread out: the same key still wins,
# but it wins by a smaller margin and more keys keep a non-negligible
# share. That softness is the cost MolFormer pays for `O(L)` compute — and
# at pre-training scale it's a cost that buys you a billion molecules in
# the budget you would otherwise spend on a tenth as many.

# Now overlay one query's attention back onto the SMILES string. We pick
# the `[CLS]` token because it is the global summary the model learns
# during MLM and is the most-watched row in any encoder-only transformer.

QUERY_IDX = 0  # [CLS]
plot_attention_on_smiles(
    softmax_matrix[0, QUERY_IDX].detach(), caffeine_tokens, QUERY_IDX,
    title=f"softmax — '{caffeine_tokens[QUERY_IDX]}' attending across caffeine",
)
plt.show()

plot_attention_on_smiles(
    linear_matrix[0, QUERY_IDX].detach(), caffeine_tokens, QUERY_IDX,
    title=f"linear  — '{caffeine_tokens[QUERY_IDX]}' attending across caffeine",
)
plt.show()

# 💡 **Key Insight.** If you take the `argmax` of each row, the two
# matrices often *agree* on which token wins — what differs is the
# *contrast*. Linear attention is the same conversation, in a quieter
# voice.

# ---
# ## E. Empirical scaling — when does linear actually win?
#
# Let's measure both forwards on synthetic random inputs at varying `L`.
# We use `d_model = 64` (a tiny MolFormer-sized model) and time the
# forward pass for `L` from 16 to 1024.

# +
def time_forward(module, x, n_warmup=3, n_iter=20):
    """Average wall-clock per forward pass, in milliseconds."""
    for _ in range(n_warmup):
        module(x, return_attention=False)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        module(x, return_attention=False)
    return (time.perf_counter() - t0) / n_iter * 1e3

D_BENCH = 64
softmax_bench = ScaledDotProductAttention(d_model=D_BENCH, dropout=0.0).eval()
linear_bench  = LinearAttention         (d_model=D_BENCH, dropout=0.0).eval()

Ls = [16, 32, 64, 128, 256, 512, 1024]
softmax_times, linear_times = [], []
for L in Ls:
    xb = torch.randn(1, L, D_BENCH)
    with torch.no_grad():
        softmax_times.append(time_forward(softmax_bench, xb))
        linear_times.append (time_forward(linear_bench,  xb))
    print(f"L = {L:>4}  |  softmax {softmax_times[-1]:7.3f} ms  "
          f"|  linear {linear_times[-1]:7.3f} ms")
# -

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(Ls, softmax_times, "o-", label="softmax  (O(L²))", lw=2, color="#dd8452")
ax.loglog(Ls, linear_times,  "s-", label="linear   (O(L))",  lw=2, color="#4c72b0")
ax.set_xlabel("sequence length L")
ax.set_ylabel("forward-pass wall-clock (ms)")
ax.set_title("Forward-pass cost vs. sequence length  (d_model = 64, CPU)")
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend()
fig.tight_layout()
plt.show()

# 💡 **Key Insight.** Three things to notice:
#
# 1. **Slope.** On a log–log plot, a slope-2 line is `L²` and slope-1 is
#    `L`. Softmax pulls away from linear once `L` gets large.
# 2. **Crossover.** For very small `L`, linear attention is **slower** in
#    absolute terms — its `O(d²)` constant per query is bigger than
#    softmax's `O(L · d)` constant. Crossover usually lands somewhere in
#    `L ≈ 64–256` on CPU; on GPU with cuDNN-tuned matmuls it can sit
#    higher.
# 3. **Why bother for chemistry, where `L` is mostly small?** Because
#    pre-training compute lives on the *tail* of the distribution
#    (Section A's right-hand histogram), and at a billion molecules even
#    a small constant on `L²` swallows the budget.

# ⚠️ **Note.** This benchmark is on CPU, single-threaded, and uses our
# educational implementations — not optimised CUDA kernels. The crossover
# and absolute timings on a real GPU MolFormer training run look
# different. The *shapes* of the two curves are what matter.

# ---
# ## F. Do they actually *learn* equivalently? A tiny MLM run
#
# Wall-clock complexity is only half the story. The other half is whether
# linear attention still **learns** — or whether the softness costs us so
# much expressiveness that the model can't fit anything.
#
# We borrow the **masked-language-modelling (MLM) objective** from notebook
# 08 (full theory lives there). For each input SMILES we mask 15% of the
# non-special tokens with `[MASK]`, run the model, and compute
# cross-entropy over just the masked positions. Two models — identical
# everywhere except the attention module — train side by side for 200
# steps on a few hundred BBBP SMILES. We plot the two loss curves on the
# same axes.
#
# **This is not a serious benchmark.** It is a sanity check: do both
# attentions get *off the floor* at this scale?

# +
class TinyMLM(nn.Module):
    """A minimal MLM head: embed → position → one attention block → vocab."""
    def __init__(self, vocab_size, d_model=64, attn_cls=ScaledDotProductAttention, max_len=128):
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, d_model)
        self.pos   = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
        self.attn  = attn_cls(d_model=d_model, dropout=0.1)
        self.norm  = nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask):
        h = self.pos(self.embed(input_ids))
        h_attn, _ = self.attn(h, mask=attention_mask, return_attention=False)
        h = self.norm(h + h_attn)         # one residual + post-norm
        return self.head(h)               # (B, L, vocab_size)


def make_mlm_batch(ids_list, mask_list, mask_prob=0.15, vocab_size=None,
                   special_ids=(PAD_ID, CLS_ID, SEP_ID, MASK_ID)):
    """Standard MLM masking: 15% of non-special tokens get replaced by [MASK].

    Returns (corrupted_ids, attention_mask, labels). Labels are -100 for
    positions that should be ignored by the loss (everything except the
    masked tokens), matching HuggingFace convention.
    """
    input_ids = torch.tensor(ids_list)
    attn_mask = torch.tensor(mask_list)
    labels = input_ids.clone()
    eligible = attn_mask.bool() & ~torch.isin(input_ids, torch.tensor(special_ids))
    mlm_mask = (torch.rand_like(input_ids, dtype=torch.float) < mask_prob) & eligible
    corrupted = input_ids.clone()
    corrupted[mlm_mask] = MASK_ID
    labels[~mlm_mask] = -100  # ignore index for CrossEntropyLoss
    return corrupted, attn_mask, labels


# Build a small training set from the already-loaded BBBP SMILES.
TRAIN_SMILES = bbbp_smiles[:512]
input_ids, attn_mask = tokenizer.encode_batch(TRAIN_SMILES, add_special_tokens=True, max_length=80)


def train_one(attn_cls, n_steps=200, batch_size=16, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = TinyMLM(vocab_size=tokenizer.vocab_size, d_model=64, attn_cls=attn_cls, max_len=128)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    n = len(input_ids)
    for step in range(n_steps):
        idx = np.random.choice(n, size=batch_size, replace=False)
        ids_b = [input_ids[i] for i in idx]
        mask_b = [attn_mask[i] for i in idx]
        corrupted, m, labels = make_mlm_batch(ids_b, mask_b)
        logits = model(corrupted, m)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


print("Training softmax-attention model ...")
softmax_losses = train_one(ScaledDotProductAttention)
print("Training linear-attention model ...")
linear_losses  = train_one(LinearAttention)

# +
def smooth(xs, k=10):
    xs = np.asarray(xs, dtype=float)
    if len(xs) < k:
        return xs
    return np.convolve(xs, np.ones(k) / k, mode="valid")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(smooth(softmax_losses), label="softmax attention", lw=2, color="#dd8452")
ax.plot(smooth(linear_losses),  label="linear attention",  lw=2, color="#4c72b0")
ax.set_xlabel("optimizer step (10-step moving average)")
ax.set_ylabel("masked-LM cross-entropy")
ax.set_title("Tiny MLM training — same model, two attentions")
ax.grid(True, ls=":", alpha=0.5)
ax.legend()
fig.tight_layout()
plt.show()

print(f"softmax  final loss: {np.mean(softmax_losses[-20:]):.3f}")
print(f"linear   final loss: {np.mean(linear_losses[-20:]):.3f}")
# -

# 💡 **Key Insight.** Both curves come down. Both attentions can *learn*
# the MLM objective from this tiny corpus. The two final losses are usually
# within a factor of 2× of each other at this scale — which is to say
# indistinguishable in any practical sense.
#
# What this run does **not** tell you:
#
# - Whether linear attention scales better than softmax on *real*
#   MolFormer-scale pre-training (it does, but you need millions of
#   molecules and many heads to see it).
# - Whether the **representations** the two models learn transfer to
#   downstream property prediction the same way (open question — varies
#   by task).
# - Anything about the choice of feature map. ELU+1 here is a teaching
#   tool; MolFormer uses FAVOR+, which approximates softmax more
#   faithfully and tends to track softmax-attention's metrics even more
#   closely.

# ---
# ## G. Tradeoffs and what MolFormer actually uses
#
# Putting it all together:
#
# | Property            | Softmax attention                            | Linear attention (ELU+1)                  |
# |---------------------|----------------------------------------------|-------------------------------------------|
# | Time complexity     | `O(L² · d)`                                  | `O(L · d²)`                               |
# | Memory complexity   | `O(L²)` attention matrix                     | `O(d²)` running sum                       |
# | Attention pattern   | Sharp, peaked, can hard-select a single key  | Smooth, distributed                       |
# | Numerical safety    | softmax handles ranges intrinsically         | denominator needs an `eps` floor          |
# | Sweet spot          | small `L`, expressive single-pair queries    | very large `L`, very large corpora        |
# | MolFormer uses it?  | No (pre-training)                            | Yes — but with a **different feature map** |
#
# ⚠️ **What MolFormer actually does.** Ross et al. (2022) use a
# **Performer-style FAVOR+** attention (Choromanski et al., 2021) rather
# than ELU+1. FAVOR+ uses random orthogonal feature maps that *provably
# approximate* the softmax kernel — so the linear-attention output is
# (up to noise) the same function softmax would compute, at `O(L)` cost.
# FAVOR+ has more moving parts than ELU+1, hence its own supplementary
# notebook **`04_1_FAVOR_Performer.ipynb`**. The `O(L)` *complexity story*
# you just learned is identical; only the choice of `φ` changes.

# ---
# ## Checkpoint exercises
#
# Each exercise has starter code and a commented-out solution block. Try
# the exercise first, then peek if you get stuck.

# +
# Exercise 1 — implement linear attention from scratch and check it
# -----------------------------------------------------------------
# Without using the `LinearAttention` module, write a function
# `my_linear_attention(Q, K, V)` that takes three `(B, L, d_model)` tensors
# (already projected) and returns the `(B, L, d_model)` linear-attention
# output. Use `φ(x) = elu(x) + 1`. Then assert it agrees with
# `LinearAttention(...).forward(x)` on the caffeine input above to a
# tolerance of `1e-5`.

# YOUR CODE HERE


# --- Solution ---
# def my_linear_attention(Q, K, V, eps=1e-6):
#     phi_q = F.elu(Q) + 1.0
#     phi_k = F.elu(K) + 1.0
#     KV    = torch.einsum("bld,blm->bdm", phi_k, V)
#     K_sum = phi_k.sum(dim=1)
#     num   = torch.einsum("bld,bdm->blm", phi_q, KV)
#     den   = torch.einsum("bld,bd->bl",   phi_q, K_sum).clamp(min=eps).unsqueeze(-1)
#     return num / den
#
# out_manual = my_linear_attention(Q, K, V)
# out_module, _ = linear_attn(x)
# print("agree?", torch.allclose(out_manual, out_module, atol=1e-5))

# +
# Exercise 2 — padding-aware linear attention
# -------------------------------------------
# The `LinearAttention` module zeroes `φ(K)` for padded keys. Build a
# two-molecule batch where the second molecule is shorter than the first;
# pass it through `LinearAttention(..., dropout=0.0)` with the correct
# `(B, L)` mask. Then verify the implicit attention matrix has zero in
# *every column* that corresponds to a padded position of the shorter
# molecule, and that the two columns for the longer molecule are
# untouched.

# YOUR CODE HERE


# --- Solution ---
# pair_ids, pair_mask = tokenizer.encode_batch(
#     [CAFFEINE, "CCO"], add_special_tokens=True
# )
# pair_ids  = torch.tensor(pair_ids)
# pair_mask = torch.tensor(pair_mask)
# x_pair = positional(token_embedding(pair_ids))
# _, attn_pair = linear_attn(x_pair, mask=pair_mask)
# # CCO has 5 real tokens ([CLS] C C O [SEP]); everything from index 5 on is padding.
# n_real_short = int(pair_mask[1].sum().item())
# print("padded-column max in row of short molecule:",
#       attn_pair[1, :, n_real_short:].max().item(), "  (should be 0.0)")
# print("real-column min in row of long  molecule:",
#       attn_pair[0, :, :pair_mask[0].sum()].sum(-1).min().item(),
#       "  (should be > 0)")

# +
# Exercise 3 — why ELU+1 and not ReLU?
# ------------------------------------
# Re-implement `my_linear_attention` from Exercise 1, but with
# `φ(x) = F.relu(x)` instead of `elu(x) + 1`. Run it on the same caffeine
# input. Are the outputs still finite? Re-plot the implicit attention
# matrix with `plot_attention_heatmap`. In one sentence: why is ReLU
# numerically riskier than ELU+1 here, and what would make the issue worse
# (deeper model, smaller `d_model`, ...)?

# YOUR CODE HERE


# --- Solution ---
# def relu_linear_attention(Q, K, V, eps=1e-6):
#     phi_q = F.relu(Q)
#     phi_k = F.relu(K)
#     KV    = torch.einsum("bld,blm->bdm", phi_k, V)
#     K_sum = phi_k.sum(dim=1)
#     num   = torch.einsum("bld,bdm->blm", phi_q, KV)
#     den   = torch.einsum("bld,bd->bl",   phi_q, K_sum).clamp(min=eps).unsqueeze(-1)
#     return num / den
#
# # ReLU zeroes any negative coordinate of Q or K. If a row of phi(Q) ends
# # up all-zero (negative everywhere), the *numerator* for that query is
# # zero — meaning the model contributes nothing useful for that token —
# # and the denominator is also zero, so the result is 0/eps ≈ 0 instead
# # of a sensible weighted average. With small d_model, this is alarmingly
# # easy: a single bias shift can collapse a query into the "all-zero"
# # regime. ELU+1 avoids this entirely because phi(x) > 0 for all x.
# -

# ---
# ## What's next
#
# You now have the second of MolFormer's three big architectural choices.
# **Notebook 04.2** covers the third — **rotary position embeddings (RoPE)**,
# which fold position information directly into `Q` and `K` and pair
# naturally with linear attention (the kernel-trick math still goes
# through). **Notebook 09** puts attention, RoPE, and MLM together into a
# tiny end-to-end MolFormer.
#
# 📚 **Deep-dive sub-series**
# - **04.1 (this notebook)** — Linear attention with ELU+1.
# - **04_1_FAVOR_Performer** — *coming soon.* The Performer-style feature
#   map MolFormer actually uses.
# - **04.2** — Rotary position embeddings (RoPE).
# - **04.3** — Other position encodings (ALiBi, relative-position bias).
#
# 📚 **References.**
# - Katharopoulos, A. et al. (2020). *Transformers are RNNs: Fast
#   Autoregressive Transformers with Linear Attention.* — the ELU+1
#   feature map and the kernel-trick derivation.
# - Choromanski, K. et al. (2021). *Rethinking Attention with Performers.*
#   — FAVOR+ random-feature attention, what MolFormer uses.
# - Ross, J. et al. (2022). *Large-Scale Chemical Language Representations
#   Capture Molecular Structure and Properties.* — MolFormer, 1.1B-SMILES
#   pre-training using linear attention.
