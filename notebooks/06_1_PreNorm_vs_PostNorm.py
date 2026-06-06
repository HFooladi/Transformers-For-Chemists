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

# <a href="https://colab.research.google.com/github/HFooladi/Transformers-For-Chemists/blob/main/notebooks/06_1_PreNorm_vs_PostNorm.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 06.1 · Pre-norm vs Post-norm — where you put the LayerNorm decides everything
#
# Notebook 06 built the encoder block in **pre-norm** form and claimed,
# almost in passing, that this placement "keeps the gradient highway clean"
# and is what modern models use. That claim deserves a proper look, because
# the choice between **pre-norm** and **post-norm** is one of the few
# architectural decisions that can make the difference between a deep
# transformer that trains and one that doesn't.
#
# The original Transformer (Vaswani et al., 2017) was **post-norm**:
#
# $$ x \;\leftarrow\; \mathrm{LayerNorm}\big(x + \mathrm{Sublayer}(x)\big). $$
#
# Almost everything since GPT-2 — including MolFormer — switched to
# **pre-norm**:
#
# $$ x \;\leftarrow\; x + \mathrm{Sublayer}\big(\mathrm{LayerNorm}(x)\big). $$
#
# Xiong et al. (2020) explained why: in post-norm the gradients at
# initialization are **badly scaled across depth**, so deep post-norm models
# need a careful learning-rate *warmup* to train at all, while pre-norm
# gradients are well-behaved and train out of the box. This deep-dive
# reproduces that effect on a chemical MLM.
#
# **Scope.** Analysis + tiny-training, in the style of notebook 04.1. The toy
# models are small, but the three effects below — gradient imbalance at init,
# a pre-norm training advantage, and a gap that *grows with depth* — are
# exactly the ones the literature reports at scale.

# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# 1. State the pre-norm and post-norm update rules and where each puts the
#    LayerNorm relative to the residual.
# 2. Measure the **per-layer gradient norm at initialization** and show
#    post-norm is uneven while pre-norm is uniform.
# 3. Train both at depth and compare their loss curves.
# 4. Sweep the **depth** and show the pre-norm advantage *grows* with it.
# 5. Explain why modern models (and MolFormer) ship pre-norm, and where
#    learning-rate warmup fits in.

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
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from utils.smiles_tokenizers import AtomTokenizer, PAD_ID, CLS_ID, SEP_ID, MASK_ID
from utils.transformer_blocks import (
    FeedForward,
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    TokenEmbedding,
)

torch.manual_seed(0)
# -

# ---
# ## 1. The two placements, side by side
#
# Both wrap the same sub-layer (multi-head attention or the feed-forward net)
# with the same LayerNorm and the same residual `+`. The *only* difference is
# the order. We build a single `NormBlock` that can be either.

class NormBlock(nn.Module):
    """One transformer block, either pre-norm or post-norm."""

    def __init__(self, d_model, n_heads, d_ff, mode):
        super().__init__()
        assert mode in ("pre", "post")
        self.mode = mode
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=0.0)
        self.ff = FeedForward(d_model, d_ff, dropout=0.0)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        if self.mode == "pre":
            a, _ = self.attn(self.norm1(x), mask=mask)
            x = x + a                                     # residual stream stays un-normalized
            x = x + self.ff(self.norm2(x))
        else:                                             # post-norm
            a, _ = self.attn(x, mask=mask)
            x = self.norm1(x + a)                         # norm sits ON the residual stream
            x = self.norm2(x + self.ff(x))
        return x

# 💡 **Key Insight.** In **pre-norm** the residual stream `x` is never
# normalized — it flows from the first block to the last untouched, and each
# sub-layer reads a normalized *copy*. In **post-norm** every block
# re-normalizes the whole stream. That single difference is what the rest of
# this notebook is about.

# ---
# ## 2. A small chemical MLM to experiment on
#
# Same self-contained recipe as notebooks 05 and 06: a few hundred BBBP
# molecules, masked-atom prediction. We'll stack `NormBlock`s into a tiny
# encoder, parameterized by depth and norm placement.

# +
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
bbbp_smiles = [s for s in bbbp_smiles if len(s) <= 80][:600]
tokenizer = AtomTokenizer.from_smiles(bbbp_smiles)

D_MODEL, N_HEADS, D_FF = 64, 4, 256
print(f"{len(bbbp_smiles)} molecules, vocab {tokenizer.vocab_size}")


class Encoder(nn.Module):
    def __init__(self, vocab_size, mode, n_layers, d_model=D_MODEL,
                 n_heads=N_HEADS, d_ff=D_FF, max_len=128):
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.blocks = nn.ModuleList(
            [NormBlock(d_model, n_heads, d_ff, mode) for _ in range(n_layers)]
        )
        # Pre-norm models add a final LayerNorm before the head (the residual
        # stream is otherwise never normalized); post-norm does not need one.
        self.final = nn.LayerNorm(d_model) if mode == "pre" else nn.Identity()
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask):
        h = self.pos(self.embed(input_ids))
        for blk in self.blocks:
            h = blk(h, mask=attention_mask)
        return self.head(self.final(h))


def make_mlm_batch(smiles, tok, seed, mask_prob=0.15,
                   special_ids=(PAD_ID, CLS_ID, SEP_ID, MASK_ID)):
    torch.manual_seed(seed)
    idx = torch.randint(0, len(smiles), (32,))
    batch = [smiles[i] for i in idx]
    ids, mask = tok.encode_batch(batch, add_special_tokens=True)
    input_ids, attention_mask = torch.tensor(ids), torch.tensor(mask)
    eligible = torch.ones_like(input_ids, dtype=torch.bool)
    for sid in special_ids:
        eligible &= input_ids != sid
    mlm_mask = (torch.rand_like(input_ids, dtype=torch.float) < mask_prob) & eligible
    corrupted = input_ids.clone()
    corrupted[mlm_mask] = MASK_ID
    labels = torch.full_like(input_ids, -100)
    labels[mlm_mask] = input_ids[mlm_mask]
    return corrupted, labels, attention_mask
# -

# ---
# ## 3. Gradients at initialization — the root cause
#
# Before training anything, let's look at the **gradients at step 0**. We push
# one batch through a freshly-initialized 18-layer encoder, backpropagate the
# loss, and record the gradient norm of each block's query projection. A
# healthy architecture has gradients of *similar* magnitude at every depth; a
# pathological one has gradients that explode or vanish as you move through
# the stack.

# +
def grad_norm_by_layer(mode, n_layers=18, seed=0):
    torch.manual_seed(seed)
    model = Encoder(tokenizer.vocab_size, mode, n_layers)
    corrupted, labels, mask = make_mlm_batch(bbbp_smiles, tokenizer, seed=1)
    logits = model(corrupted, mask)
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
    model.zero_grad()
    loss.backward()
    return [model.blocks[i].attn.w_q.weight.grad.norm().item() for i in range(n_layers)]


g_pre = grad_norm_by_layer("pre")
g_post = grad_norm_by_layer("post")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(range(1, len(g_pre) + 1), g_pre,  marker="o", ms=4, label="pre-norm",  color="#4c72b0")
ax.plot(range(1, len(g_post) + 1), g_post, marker="s", ms=4, label="post-norm", color="#dd8452")
ax.set_yscale("log")
ax.set_xlabel("block (1 = nearest input, 18 = nearest output)")
ax.set_ylabel("gradient norm of W_Q at init")
ax.set_title("Per-layer gradient at initialization")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
print(f"pre-norm  spread (max/min): {max(g_pre)  / min(g_pre):.1f}×")
print(f"post-norm spread (max/min): {max(g_post) / min(g_post):.1f}×")
# -

# ⚠️ **Note.** The pre-norm curve is nearly flat — every block gets a gradient
# of comparable size, so a single global learning rate suits them all. The
# post-norm curve is jagged and spans more than an order of magnitude: some
# blocks get huge gradients, others almost none. A learning rate that's right
# for one block is wrong for another — which is exactly why deep post-norm
# models are finicky to optimize.

# ---
# ## 4. Train both at depth
#
# Now train a 12-layer encoder each way, from step 0, with the *same*
# optimizer and learning rate (no warmup). The pre-norm model should train
# faster and reach a lower loss.

# +
def train(mode, n_layers, steps=250, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    model = Encoder(tokenizer.vocab_size, mode, n_layers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        corrupted, labels, mask = make_mlm_batch(bbbp_smiles, tokenizer, seed=step)
        logits = model(corrupted, mask)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses


pre_losses = train("pre", n_layers=12)
post_losses = train("post", n_layers=12)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
sm = lambda xs: np.convolve(xs, np.ones(15) / 15, mode="valid")
ax.plot(sm(pre_losses),  label="pre-norm",  color="#4c72b0", lw=2)
ax.plot(sm(post_losses), label="post-norm", color="#dd8452", lw=2)
ax.set_xlabel("optimizer step (15-step moving average)")
ax.set_ylabel("MLM loss")
ax.set_title("12-layer encoder — pre-norm trains faster and lower")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
print(f"final loss — pre-norm: {np.mean(pre_losses[-20:]):.3f}   "
      f"post-norm: {np.mean(post_losses[-20:]):.3f}")
# -

# ---
# ## 5. The gap grows with depth
#
# The single comparison above is suggestive; the real signature of the problem
# is how it **scales with depth**. We train both placements at a range of
# depths (shorter budget each) and plot the final loss. Pre-norm should stay
# roughly flat as we add layers; post-norm should fall behind further and
# further.

# +
depths = [2, 4, 8, 16]
pre_final, post_final = [], []
for nl in depths:
    pre_final.append(np.mean(train("pre",  nl, steps=150)[-15:]))
    post_final.append(np.mean(train("post", nl, steps=150)[-15:]))

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(depths, pre_final,  marker="o", label="pre-norm",  color="#4c72b0", lw=2)
ax.plot(depths, post_final, marker="s", label="post-norm", color="#dd8452", lw=2)
ax.set_xlabel("number of layers"); ax.set_ylabel("final MLM loss (150 steps)")
ax.set_title("Pre-norm's advantage grows with depth")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
for nl, p, q in zip(depths, pre_final, post_final):
    print(f"depth {nl:2d}: pre {p:.3f}  post {q:.3f}  gap {q - p:+.3f}")
# -

# 💡 **Key Insight.** A shallow post-norm model is *fine* — at 2 layers the gap
# is tiny, which is why the original (6-layer, carefully-warmed-up)
# Transformer worked beautifully. The trouble shows up as you go deep, exactly
# where modern models live. Pre-norm removes the depth penalty, which is why
# the field switched.

# ---
# ## 6. So what do real models do?
#
# - **Pre-norm everywhere.** GPT-2 and successors, ViT, and **MolFormer** all
#   use pre-norm. It trains deep stacks stably with a single learning rate and
#   little or no warmup — and it's the placement our `EncoderBlock`
#   (notebook 06) implements.
# - **Post-norm + warmup** is still viable, and was the original recipe: a
#   learning-rate *warmup* (start tiny, ramp up over the first few hundred
#   steps) tames the gradient imbalance from §3 long enough for the model to
#   settle. Xiong et al. (2020) showed pre-norm makes that warmup largely
#   unnecessary.
# - **Beyond both:** schemes like DeepNorm (Wang et al., 2022) rescale the
#   residual to push *post*-norm to 1000+ layers — evidence that the residual
#   stream's scale, the thing pre-norm protects, is the crux.

# ⚠️ **Note.** None of this changes the *function* a block computes — pre- and
# post-norm are equally expressive. The entire difference is about
# **optimization**: whether gradient descent can actually find good weights.

# ---
# ## Checkpoint exercises

# +
# Exercise 1 — gradient spread vs depth
# -------------------------------------
# Compute the post-norm gradient spread (max/min over layers) at depths 6, 12,
# and 24 using grad_norm_by_layer. Does the imbalance get worse with depth?

# YOUR CODE HERE

# --- Solution (try the exercise first, then peek) ---
# for nl in (6, 12, 24):
#     g = grad_norm_by_layer("post", n_layers=nl)
#     print(f"depth {nl}: post-norm spread {max(g) / min(g):.1f}×")
# # The spread typically widens with depth — the deeper the stack, the more
# # unevenly post-norm distributes its gradients.

# +
# Exercise 2 — does warmup rescue post-norm?
# ------------------------------------------
# Add a linear learning-rate warmup to `train` (lr scaled by min(1, step/40)
# for the first 40 steps) and retrain the 12-layer post-norm model. Does the
# early part of the loss curve become smoother / lower?

# YOUR CODE HERE

# --- Solution sketch ---
# Copy `train`, and inside the loop set
#   for g in opt.param_groups: g["lr"] = lr * min(1.0, (step + 1) / 40)
# before opt.step(). Warmup mainly helps the *early* steps, where the raw
# gradient imbalance is most destructive.

# +
# Exercise 3 — pre-norm needs a final LayerNorm
# ---------------------------------------------
# The Encoder adds a final LayerNorm for pre-norm but not post-norm. Remove it
# (set self.final = nn.Identity() for both) and inspect the magnitude of the
# pre-norm logits at init (model(...).abs().mean()). Why does an
# un-normalized residual stream make the final logits blow up?

# YOUR CODE HERE

# --- Solution sketch ---
# Without the final norm, the pre-norm residual stream accumulates the output
# of every sub-layer, so its magnitude grows with depth; the head then sees
# large-magnitude inputs and produces large logits (and a large initial loss).
# The final LayerNorm rescales the stream once before the head — which is why
# every real pre-norm model includes it.

# ---
# ## What's next
#
# You now know *why* notebook 06's block is pre-norm, not just *that* it is.
# Back on the **main path**, notebook 07 stacks that pre-norm `EncoderBlock`
# into a full `TransformerEncoder` and trains it on a real property-prediction
# task — deep enough that the choice you just studied genuinely matters.
#
# 📚 **References.**
# - Xiong, R. et al. (2020). *On Layer Normalization in the Transformer
#   Architecture.* — the pre-norm vs post-norm gradient analysis.
# - Vaswani, A. et al. (2017). *Attention Is All You Need.* — original
#   post-norm Transformer (with learning-rate warmup).
# - Ba, J., Kiros, J. & Hinton, G. (2016). *Layer Normalization.*
# - Wang, H. et al. (2022). *DeepNet: Scaling Transformers to 1,000 Layers
#   (DeepNorm).*
