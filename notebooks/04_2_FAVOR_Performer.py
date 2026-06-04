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

# <a href="https://colab.research.google.com/github/HFooladi/Transformers-For-Chemists/blob/main/notebooks/04_2_FAVOR_Performer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 04.2 · FAVOR+ — the Performer feature map MolFormer actually uses
#
# Notebook 04.1 built **linear attention** with the simplest correct feature
# map, `φ(x) = elu(x) + 1`. It nailed the `O(L)` *cost* story — but it made no
# attempt to *match softmax*. Its attention pattern is just "softer"; it
# approximates **some** kernel, not the `exp(qᵀk)` of real attention.
#
# **MolFormer** (Ross et al., 2022) does better. It uses **Performer**
# attention (Choromanski et al., 2021) and its **FAVOR+** feature map — *Fast
# Attention Via positive Orthogonal Random features* — which **provably
# approximates the softmax kernel** while keeping the same `O(L)` cost. This
# notebook unpacks the three ideas hiding in that acronym:
#
# - **Random features** — `exp(qᵀk)` can be written as an *average* over random
#   projections, so we can estimate it with a few samples.
# - **Positive** (the `+`) — naive trigonometric features can produce *negative*
#   attention weights; positive features cannot.
# - **Orthogonal** (the `OR`) — making the random directions orthogonal samples
#   the space more evenly and lowers the approximation variance.
#
# **Scope.** A mechanism deep-dive: we build FAVOR+ from scratch, visualize why
# each trick matters, and show Performer attention converging to true softmax as
# we add features. It slots directly on top of 04.1's `O(L)` machinery. The full
# pre-trained model is **notebook 09**.

# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# 1. Explain why `elu(x)+1` carries the cost story but not the *softmax-fidelity*
#    story.
# 2. State the **random-feature identity** `exp(qᵀk) = E_w[φ_w(q)·φ_w(k)]` and
#    watch a Monte-Carlo estimate converge to it.
# 3. Explain why FAVOR+ uses **positive** features, by seeing trigonometric
#    features produce negative kernel estimates.
# 4. Explain how **orthogonal** random directions reduce variance for the same
#    feature budget.
# 5. Use a **`PerformerAttention`** module (drop-in with `LinearAttention`) and
#    measure its error to true softmax as a function of the feature count `m`.
# 6. Place Performer on the softmax↔linear spectrum and name what MolFormer
#    actually ships.

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

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils.smiles_tokenizers import AtomTokenizer
from utils.transformer_blocks import (
    ScaledDotProductAttention,
    SinusoidalPositionalEncoding,
    TokenEmbedding,
)
from utils.linear_attention import LinearAttention, elu_feature_map
from utils.performer import (
    PerformerAttention,
    softmax_kernel_features,
    gaussian_orthogonal_random_matrix,
)
from utils.attention_viz import plot_attention_heatmap

torch.manual_seed(0)
np.random.seed(0)
# -

# ---
# ## A. Where 04.1 stopped: approximating *a* kernel, not *the* softmax
#
# Both softmax and linear attention share one skeleton (notebook 04.1 §B):
#
# $$
# \mathrm{out}_i = \frac{\sum_j \mathrm{sim}(Q_i, K_j)\, V_j}{\sum_j \mathrm{sim}(Q_i, K_j)} .
# $$
#
# Softmax uses `sim(q, k) = exp(qᵀk / √d)`. Linear attention swaps in a
# *factorizable* `sim(q, k) = φ(q)ᵀφ(k)`. The `elu+1` choice factorizes
# beautifully — but the curve it implies is nothing like `exp`. Performer's goal
# is a `φ` whose `φ(q)ᵀφ(k)` actually **equals `exp(qᵀk)` on average**.

xs = torch.linspace(-2, 2, 200)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(xs.numpy(), torch.exp(xs).numpy(), lw=2.5, label="softmax similarity  exp(qᵀk)")
ax.plot(xs.numpy(), (elu_feature_map(xs) * elu_feature_map(torch.ones(1))).numpy(),
        lw=2, ls="--", label="elu+1 similarity  φ(q)·φ(1)")
ax.set_xlabel("qᵀk  (one coordinate, k fixed)")
ax.set_ylabel("similarity")
ax.set_title("elu+1 is smooth and positive — but it is not exp\nPerformer chases exp itself")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# 💡 **Key Insight.** `elu+1` gets the *shape* roughly right (monotone, positive)
# but it is a linear-ish ramp where softmax is an exponential. FAVOR+ asks: can
# we keep the `O(L)` factorization **and** make the similarity genuinely
# exponential? The trick is to stop trying to write `exp` as a fixed formula and
# instead write it as an **average**.

# ---
# ## B. Softmax as an average over random features
#
# Here is the identity that starts everything (a Gaussian integral). For any two
# vectors and a random direction `w` drawn from a standard Gaussian:
#
# $$
# \exp(q^\top k) \;=\; \mathbb{E}_{w \sim \mathcal{N}(0, I)}\!\Big[\,
#   \underbrace{e^{\,w^\top q - \|q\|^2/2}}_{\varphi_w(q)}\;\cdot\;
#   \underbrace{e^{\,w^\top k - \|k\|^2/2}}_{\varphi_w(k)} \Big] .
# $$
#
# Read it slowly: the *exponential-of-a-dot-product* on the left becomes an
# *average of a product of two separate functions* on the right — one depending
# only on `q`, one only on `k`. That separation is exactly the factorization
# linear attention needs. We can't take the expectation exactly, but we can
# **estimate** it by sampling `m` directions `w₁ … w_m` and averaging. More
# samples → better estimate.

# +
def phi_random(x, W):
    """Positive random features φ_w(x) = exp(wᵀx − ‖x‖²/2) for each row w of W."""
    return torch.exp(W @ x - (x @ x) / 2)


torch.manual_seed(1)
d = 8
q = torch.randn(d) * 0.5          # moderate norm keeps the estimator well-behaved
k = torch.randn(d) * 0.5
true_kernel = math.exp(q @ k)

ms = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
means, stds = [], []
for m in ms:
    estimates = []
    for _ in range(200):          # 200 independent feature draws per m
        W = torch.randn(m, d)
        estimates.append((phi_random(q, W) * phi_random(k, W)).mean().item())
    means.append(np.mean(estimates))
    stds.append(np.std(estimates))

means, stds = np.array(means), np.array(stds)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.axhline(true_kernel, color="black", ls="--", label=f"true  exp(qᵀk) = {true_kernel:.3f}")
ax.plot(ms, means, "o-", color="#4c72b0", label="random-feature estimate")
ax.fill_between(ms, means - stds, means + stds, color="#4c72b0", alpha=0.2,
                label="±1 std over 200 draws")
ax.set_xscale("log", base=2)
ax.set_xlabel("number of random features  m")
ax.set_ylabel("estimated similarity")
ax.set_title("More random features → the estimate homes in on exp(qᵀk)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
# -

# 💡 **Key Insight.** The estimate is **unbiased** at every `m` — it scatters
# *around* the true value, not off to one side — and the scatter (the shaded
# band) shrinks as `m` grows. This is the whole bet of Performer: trade a few
# hundred random features for an `exp`-faithful similarity that still
# factorizes into `O(L)` attention. `m` is the knob that buys accuracy.

# ---
# ## C. Why **positive** features (the `+` in FAVOR+)
#
# The features above, `φ_w(x) = exp(wᵀx − ‖x‖²/2)`, are **always positive** —
# they're exponentials. That is not a cosmetic detail. The *original* random
# features for the softmax/Gaussian kernel used **trigonometric** functions
# (`cos`, `sin`), and those can be **negative** — which lets the estimated
# "similarity" go negative, producing negative attention weights and a
# denominator that can collapse toward zero. Let's watch it happen.

# +
def kernel_estimate_positive(q, k, W):
    """Positive-feature estimate of exp(qᵀk) — always ≥ 0."""
    return (phi_random(q, W) * phi_random(k, W)).mean().item()


def kernel_estimate_trig(q, k, W):
    """Trigonometric-feature estimate of the same exp(qᵀk) — can be negative."""
    prefactor = math.exp((q @ q + k @ k) / 2)
    return (prefactor * torch.cos(W @ (q - k))).mean().item()


torch.manual_seed(2)
q = torch.randn(d) * 0.6
k = torch.randn(d) * 0.6
true_kernel = math.exp(q @ k)
m = 8
pos = [kernel_estimate_positive(q, k, torch.randn(m, d)) for _ in range(3000)]
trig = [kernel_estimate_trig(q, k, torch.randn(m, d)) for _ in range(3000)]

fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.linspace(-4, 6, 80)
ax.hist(trig, bins=bins, alpha=0.6, color="#dd8452", label="trigonometric features")
ax.hist(pos, bins=bins, alpha=0.7, color="#4c72b0", label="positive features (FAVOR+)")
ax.axvline(0, color="red", lw=1.5, ls=":", label="zero — weights must stay ≥ 0")
ax.axvline(true_kernel, color="black", lw=1.5, ls="--", label=f"true kernel {true_kernel:.2f}")
ax.set_xlabel("estimated softmax similarity  (m = 8 features)")
ax.set_ylabel("count over 3000 draws")
ax.set_title("Trig features spill across zero; positive features never do")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

frac_neg = np.mean(np.array(trig) < 0)
print(f"fraction of trig estimates that are NEGATIVE: {frac_neg:.1%}")
print(f"fraction of positive-feature estimates negative: {np.mean(np.array(pos) < 0):.1%}")
# -

# ⚠️ **Note.** A similarity is supposed to be a non-negative weight. The orange
# histogram pours a large fraction of its mass **left of the red line** — those
# are negative "weights". Plug them into attention and the softmax-replacement
# denominator `Σ φ(q)·φ(k)` can land near zero or negative, and the whole layer
# destabilizes. The blue histogram (positive features) lives entirely to the
# right of zero. **That is the `+` in FAVOR+** — and it is the single change
# that made random-feature attention trainable.

# ---
# ## D. Why **orthogonal** features (the `OR` in FAVOR+)
#
# The last trick. With `m` *independent* Gaussian directions, some happen to
# point almost the same way — wasted, redundant samples — while other parts of
# the space go unsampled. If instead we force the directions to be **mutually
# orthogonal**, they cover the space evenly, so each feature carries fresh
# information. Same `m`, lower variance. The picture in 2-D makes it obvious:

# +
torch.manual_seed(3)
n_dirs = 8

# i.i.d. Gaussian directions (normalized for display).
iid = F.normalize(torch.randn(n_dirs, 2), dim=-1)

# Orthogonal directions: blocks of 2 perpendicular vectors at random rotations.
orth_blocks = []
for _ in range(n_dirs // 2):
    a = torch.rand(1).item() * 2 * math.pi
    orth_blocks.append(torch.tensor([[math.cos(a), math.sin(a)],
                                     [-math.sin(a), math.cos(a)]]))
orth = torch.cat(orth_blocks, dim=0)

fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
for ax, dirs, ttl in [
    (axes[0], iid,  "i.i.d. random directions\n(clump together, leave gaps)"),
    (axes[1], orth, "orthogonal random directions\n(spread evenly, no redundancy)"),
]:
    circle = plt.Circle((0, 0), 1.0, fill=False, color="grey", ls=":")
    ax.add_patch(circle)
    for v in dirs:
        ax.annotate("", xy=(v[0], v[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#4c72b0", lw=2))
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(ttl)
fig.tight_layout()
plt.show()
# -

# 💡 **Key Insight.** Look at the left panel: a couple of arrows nearly overlap
# (redundant) while whole arcs of the circle have none (unsampled). The right
# panel spreads its arrows around the circle. Estimating an average is most
# efficient when your samples are *spread out* — so orthogonal features give a
# lower-variance estimate of the softmax kernel for the **same** feature count
# `m`. The effect is modest per layer but compounds across a deep,
# many-headed, billion-molecule model (Yu et al., 2016; Choromanski et al.,
# 2021). Our `gaussian_orthogonal_random_matrix` builds exactly these
# directions in `d` dimensions.

# ---
# ## E. PerformerAttention — and how close it gets to softmax
#
# Now we assemble FAVOR+ into attention. `utils/performer.py` provides
# `PerformerAttention` with the **same interface** as
# `ScaledDotProductAttention` and `LinearAttention`, so it is drop-in. We run it
# on caffeine and ask the key question: *how close is it to true softmax, and
# how does that depend on `m`?*
#
# We reuse notebook 04's input pipeline, with one realistic touch — a
# `LayerNorm` on `Q` and `K`. Real transformers normalize activations before
# attention, and that keeps `qᵀk` in the range where random-feature estimators
# behave well (very large `qᵀk` needs very many features; see §C's variance).

# +
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
CORPUS = ["CCO", "CC(=O)Oc1ccccc1C(=O)O", CAFFEINE, "BrCCCl", "c1ccc2[nH]ccc2c1"]
tokenizer = AtomTokenizer.from_smiles(CORPUS)

caffeine_ids, _ = tokenizer.encode_batch([CAFFEINE], add_special_tokens=True)
caffeine_ids = torch.tensor(caffeine_ids)
caffeine_tokens = ["[CLS]"] + tokenizer.tokenize(CAFFEINE) + ["[SEP]"]

D_MODEL = 32
token_embedding = TokenEmbedding(tokenizer.vocab_size, D_MODEL)
positional      = SinusoidalPositionalEncoding(d_model=D_MODEL, max_len=128)
x = positional(token_embedding(caffeine_ids))         # (1, L, d)
L = x.size(1)

torch.manual_seed(42)
W_q, W_k, W_v = (nn.Linear(D_MODEL, D_MODEL) for _ in range(3))
layer_norm = nn.LayerNorm(D_MODEL)
Q = layer_norm(W_q(x))[0].detach()                    # (L, d), well-scaled
K = layer_norm(W_k(x))[0].detach()

# Ground truth: exact softmax attention on these Q, K.
true_softmax = torch.softmax((Q @ K.T) / math.sqrt(D_MODEL), dim=-1)

# elu+1 linear attention (notebook 04.1) — a fixed approximation, for reference.
pq, pk = elu_feature_map(Q), elu_feature_map(K)
elu_scores = pq @ pk.T
elu_attn = elu_scores / elu_scores.sum(-1, keepdim=True)
elu_err = (elu_attn - true_softmax).norm().item()
print(f"elu+1 fixed error to softmax: {elu_err:.3f}")
# -

# Now sweep the feature count `m`. For each `m` we draw several orthogonal
# projections, build the Performer attention matrix, and measure its Frobenius
# distance to true softmax — averaged, with a spread band.

# +
def performer_attn_matrix(Q, K, W):
    qf = softmax_kernel_features(Q, W, is_query=True)
    kf = softmax_kernel_features(K, W, is_query=False)
    scores = qf @ kf.T
    return scores / scores.sum(-1, keepdim=True).clamp(min=1e-12)


torch.manual_seed(0)
m_values = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
perf_mean, perf_std = [], []
for m in m_values:
    errs = []
    for _ in range(20):
        W = gaussian_orthogonal_random_matrix(m, D_MODEL)
        errs.append((performer_attn_matrix(Q, K, W) - true_softmax).norm().item())
    perf_mean.append(np.mean(errs))
    perf_std.append(np.std(errs))

perf_mean, perf_std = np.array(perf_mean), np.array(perf_std)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.axhline(elu_err, color="#dd8452", ls="--", lw=2,
           label=f"elu+1 (04.1) — fixed bias {elu_err:.2f}")
ax.plot(m_values, perf_mean, "o-", color="#4c72b0", label="Performer (FAVOR+)")
ax.fill_between(m_values, perf_mean - perf_std, perf_mean + perf_std,
                color="#4c72b0", alpha=0.2)
ax.set_xscale("log", base=2)
ax.set_xlabel("number of random features  m")
ax.set_ylabel("Frobenius distance to true softmax")
ax.set_title("Performer converges toward exact softmax as m grows;\nelu+1 is stuck at a fixed bias")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
# -

# 💡 **Key Insight.** This is the punchline of the whole notebook. **Performer's
# error slides downward** as `m` grows — it is a *consistent* estimator of
# exact softmax, so with enough features it gets arbitrarily close. **elu+1**
# (the dashed line) has a *fixed bias*: it approximates a different kernel, so
# no amount of anything moves it. Below the crossover, the cheap `elu+1` is
# actually closer; past it, spending random features buys you genuinely
# more-softmax-like attention. MolFormer picks a point on this curve — enough
# features to track softmax, few enough to stay `O(L)`.

# A visual confirmation: the attention matrices themselves, from noisy
# (few features) to softmax-faithful (many).

# +
torch.manual_seed(0)
panels = [("true softmax", true_softmax)]
for m in [8, 64, 512]:
    W = gaussian_orthogonal_random_matrix(m, D_MODEL)
    panels.append((f"Performer  m={m}", performer_attn_matrix(Q, K, W)))
panels.append(("elu+1 (04.1)", elu_attn))

fig, axes = plt.subplots(1, 5, figsize=(18, 3.8))
vmax = float(true_softmax.max())
for ax, (ttl, A) in zip(axes, panels):
    im = ax.imshow(A.detach().numpy(), cmap="Blues", vmin=0, vmax=vmax, aspect="equal")
    ax.set_title(ttl, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Caffeine attention: Performer sharpens toward true softmax as m grows", y=1.04)
fig.tight_layout()
plt.show()
# -

# Finally, confirm the packaged `PerformerAttention` module runs as a drop-in
# replacement, mask and all. We feed it `LayerNorm`-scaled input — the same
# pre-norm regime every real encoder block uses, and the range where the
# random-feature estimator is well-behaved (§E intro).

# +
batch_ids, batch_mask = tokenizer.encode_batch(["CCO", CAFFEINE], add_special_tokens=True)
xb = F.layer_norm(positional(token_embedding(torch.tensor(batch_ids))), (D_MODEL,))
mask = torch.tensor(batch_mask)

performer = PerformerAttention(d_model=D_MODEL, n_features=256, dropout=0.0)
out, attn = performer(xb, mask=mask, return_attention=True)
real_rows = attn.sum(-1)[mask.bool()]
n_real_cco = int(mask[0].sum())
print(f"PerformerAttention output: {tuple(out.shape)}   (drop-in with LinearAttention)")
print(f"implicit attention matrix: {tuple(attn.shape)}")
print(f"real-token rows sum to 1? {torch.allclose(real_rows, torch.ones_like(real_rows), atol=1e-3)}")
print(f"attention onto pad columns (should be 0): {attn[0, :, n_real_cco:].max().item():.6f}")
# -

# ---
# ## F. Cost: linear in `L`, linear in `m`
#
# Performer keeps 04.1's `O(L)` scaling in sequence length — adding the random
# features makes the cost `O(L · m · d)`, linear in *both* `L` and `m`. So `m`
# is a dial trading accuracy (§E) against compute, with no return of the `L²`
# term.

# +
def time_forward(module, xb, n_warmup=2, n_iter=15):
    for _ in range(n_warmup):
        module(xb, return_attention=False)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        module(xb, return_attention=False)
    return (time.perf_counter() - t0) / n_iter * 1e3


# (i) cost vs sequence length L, at fixed m — compare to softmax's O(L²).
D_BENCH = 64
softmax_bench = ScaledDotProductAttention(d_model=D_BENCH, dropout=0.0).eval()
performer_bench = PerformerAttention(d_model=D_BENCH, n_features=128, dropout=0.0).eval()
Ls = [32, 64, 128, 256, 512, 1024]
t_softmax, t_performer = [], []
with torch.no_grad():
    for Lv in Ls:
        xb = torch.randn(1, Lv, D_BENCH)
        t_softmax.append(time_forward(softmax_bench, xb))
        t_performer.append(time_forward(performer_bench, xb))

# (ii) cost vs feature count m, at fixed L — should be ~linear in m.
ms_bench = [32, 64, 128, 256, 512]
t_vs_m = []
with torch.no_grad():
    xb = torch.randn(1, 256, D_BENCH)
    for m in ms_bench:
        mod = PerformerAttention(d_model=D_BENCH, n_features=m, dropout=0.0).eval()
        t_vs_m.append(time_forward(mod, xb))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
axes[0].loglog(Ls, t_softmax, "o-", color="#dd8452", label="softmax  O(L²)")
axes[0].loglog(Ls, t_performer, "s-", color="#4c72b0", label="Performer  O(L·m)")
axes[0].set_xlabel("sequence length L"); axes[0].set_ylabel("forward (ms)")
axes[0].set_title("Cost vs L  (m = 128)"); axes[0].legend(); axes[0].grid(True, which="both", ls=":", alpha=0.5)

axes[1].plot(ms_bench, t_vs_m, "o-", color="#4c72b0")
axes[1].set_xlabel("number of features m"); axes[1].set_ylabel("forward (ms)")
axes[1].set_title("Cost vs m  (L = 256) — linear"); axes[1].grid(ls=":", alpha=0.5)
fig.tight_layout()
plt.show()
# -

# ⚠️ **Note.** Educational CPU timings, not tuned CUDA kernels — read the
# *shapes*, not the absolute numbers. The left panel reproduces 04.1's story
# (softmax pulls away as `L²`); the right panel shows `m` is a smooth linear
# dial. MolFormer sits at large `L` and a moderate `m`, deep in the region where
# Performer wins.

# ---
# ## G. The spectrum, and what MolFormer ships
#
# Three attentions, one interface:
#
# | Attention                  | Similarity `sim(q,k)`             | Cost        | Tracks softmax? |
# |----------------------------|-----------------------------------|-------------|-----------------|
# | Softmax (nb 04)            | `exp(qᵀk/√d)` exactly             | `O(L²·d)`   | it *is* softmax |
# | Linear / elu+1 (nb 04.1)   | `φ(q)ᵀφ(k)`, `φ=elu+1`            | `O(L·d²)`   | no — fixed bias |
# | **Performer / FAVOR+** (here) | `φ(q)ᵀφ(k)`, positive orthogonal random features | `O(L·m·d)` | **yes — as `m→∞`** |
#
# ✅ **What MolFormer actually does.** Ross et al. (2022) use **Performer /
# FAVOR+** attention — the positive-orthogonal-random-feature map you just
# built — so their `O(L)` attention still *approximates softmax*, not merely
# "some smooth kernel". And because FAVOR+ acts on `Q` and `K` through a plain
# feature map, it composes cleanly with **rotary position embeddings (notebook
# 04.3)**: rotate `Q` and `K` first, then apply the feature map. Linear cost,
# softmax fidelity, relative positions — that combination is most of what makes
# a MolFormer a MolFormer.

# ---
# ## Checkpoint exercises
#
# Each exercise has starter code and a commented-out solution. Try first, then
# peek.

# +
# Exercise 1 — verify the random-feature identity yourself
# --------------------------------------------------------
# Pick two random vectors q, k of dimension d=4 (scale them by 0.5). Draw
# m = 5000 Gaussian directions and estimate exp(qᵀk) with the positive features
# φ_w(x) = exp(wᵀx − ‖x‖²/2). Print the estimate next to the true exp(qᵀk) and
# confirm they agree to ~2 decimals.

# YOUR CODE HERE

# --- Solution ---
# torch.manual_seed(5)
# q = torch.randn(4) * 0.5; k = torch.randn(4) * 0.5
# W = torch.randn(5000, 4)
# est = (torch.exp(W @ q - (q @ q) / 2) * torch.exp(W @ k - (k @ k) / 2)).mean()
# print(f"estimate {est.item():.3f}   true {math.exp(q @ k):.3f}")

# +
# Exercise 2 — the feature count is an accuracy dial
# --------------------------------------------------
# Using `performer_attn_matrix`, `gaussian_orthogonal_random_matrix`, and the
# caffeine `Q, K, true_softmax` from §E, compute the Frobenius error to softmax
# for m in [8, 64, 512] (average 10 draws each). Confirm the error decreases as
# m grows.

# YOUR CODE HERE

# --- Solution ---
# for m in [8, 64, 512]:
#     errs = [(performer_attn_matrix(Q, K, gaussian_orthogonal_random_matrix(m, D_MODEL))
#              - true_softmax).norm().item() for _ in range(10)]
#     print(f"m={m:>4}  mean error {np.mean(errs):.3f}")

# +
# Exercise 3 — positivity matters
# -------------------------------
# Re-run the §C comparison but increase the feature count to m = 64. Does the
# trigonometric estimator still produce negative values? What fraction? In one
# sentence: does adding features *fix* the negativity, or just make it rarer —
# and why does even a rare negative weight matter inside a softmax denominator?

# YOUR CODE HERE

# --- Solution ---
# torch.manual_seed(2)
# q = torch.randn(8) * 0.6; k = torch.randn(8) * 0.6
# trig = [kernel_estimate_trig(q, k, torch.randn(64, 8)) for _ in range(3000)]
# print(f"m=64 trig fraction negative: {np.mean(np.array(trig) < 0):.2%}")
# # More features makes negatives RARER but never impossible — the trig
# # estimator's support always includes negatives. A single negative weight can
# # drive the attention denominator Σφ(q)·φ(k) toward zero, blowing up the
# # normalized weights. Positive features remove the failure mode entirely.

# ---
# ## What's next
#
# You now have MolFormer's real attention: **`O(L)` cost (04.1) with softmax
# fidelity (this notebook)**. The remaining MolFormer ingredient is *position* —
# **notebook 04.3** builds **rotary position embeddings (RoPE)**, which slot in
# front of the FAVOR+ feature map, and **notebook 04.4** surveys the
# alternatives (ALiBi, relative bias). **Notebook 09** assembles attention,
# RoPE, and MLM into a tiny end-to-end MolFormer.
#
# 📚 **Deep-dive sub-series**
# - **04.1** — Linear attention with ELU+1 (the `O(L)` story).
# - **04.2 (this notebook)** — FAVOR+ / Performer (the softmax-fidelity story).
# - **04.3** — Rotary position embeddings (RoPE).
# - **04.4** — Other position encodings (ALiBi, relative-position bias).
#
# 📚 **References.**
# - Choromanski, K. et al. (2021). *Rethinking Attention with Performers.* ICLR
#   — FAVOR+, positive orthogonal random features.
# - Rahimi, A. & Recht, B. (2008). *Random Features for Large-Scale Kernel
#   Machines.* — the original random-feature idea (trigonometric).
# - Yu, F. et al. (2016). *Orthogonal Random Features.* NeurIPS — the
#   variance-reduction result behind the `OR`.
# - Ross, J. et al. (2022). *Large-Scale Chemical Language Representations
#   Capture Molecular Structure and Properties* (MolFormer) — Performer
#   attention on ~1.1B SMILES.
