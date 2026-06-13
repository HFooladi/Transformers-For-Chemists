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

# <a href="https://colab.research.google.com/github/HFooladi/Transformers-For-Chemists/blob/main/notebooks/05_1_Head_Specialization.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 05.1 · Head Specialization — do we really need all those heads?
#
# Notebook 05 built multi-head attention and showed, at the end, that even a
# tiny trained model gives its heads *different* patterns. That raises an
# obvious follow-up question, and it's one the field took seriously:
# **are sixteen heads really better than one?** (Michel, Levy & Neubig, 2019.)
# Their surprising answer: many heads in a trained transformer can be
# **removed at test time** with little or no loss in quality. Heads
# specialize, but they also *overlap* — there's redundancy baked in.
#
# This deep-dive makes that concrete on a chemical model. We train a small
# multi-head MLM on SMILES, then:
#
# * **score** each head's importance by ablating it and watching the loss,
# * **measure** how redundant the heads are (pairwise pattern similarity),
# * **prune** the least-important heads and plot the graceful degradation,
# * and look at *which* heads pick up *which* chemical structure.
#
# **Scope.** Analysis + tiny-training, in the style of notebook 04.1. The toy
# model is single-layer so head bookkeeping stays clean; the qualitative
# story matches what large trained transformers do.

# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# 1. Quantify head **diversity** with a pairwise pattern-similarity matrix.
# 2. Define and compute a head **importance** score via ablation.
# 3. Explain head **redundancy** and connect it to the pruning result.
# 4. **Prune** heads by importance and read the loss-vs-heads-kept curve.
# 5. Inspect *which* head attends to *which* chemical motif on real molecules.

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
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    TokenEmbedding,
)
from utils.attention_viz import plot_per_head_grid

torch.manual_seed(0)
# -

# ---
# ## 0. Train a small multi-head MLM
#
# We reuse the self-contained recipe from notebook 05, but with **8 heads** in
# a **single** attention layer so each head has a clear identity we can score
# and prune.

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
bbbp_smiles = [s for s in bbbp_smiles if len(s) <= 80]
train_smiles, eval_smiles = bbbp_smiles[:600], bbbp_smiles[600:760]
tokenizer = AtomTokenizer.from_smiles(bbbp_smiles)

D_MODEL, N_HEADS = 64, 8
print(f"train {len(train_smiles)} / eval {len(eval_smiles)} molecules; "
      f"d_model={D_MODEL}, n_heads={N_HEADS}")

# +
class TinyMLM(nn.Module):
    def __init__(self, vocab_size, d_model=D_MODEL, n_heads=N_HEADS, max_len=128):
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=0.0)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask):
        h = self.pos(self.embed(input_ids))
        a, attn = self.attn(self.norm(h), mask=attention_mask)
        h = h + a
        return self.head(h), attn


def make_mlm_batch(smiles, tok, seed=None, mask_prob=0.15,
                   special_ids=(PAD_ID, CLS_ID, SEP_ID, MASK_ID)):
    if seed is not None:
        torch.manual_seed(seed)
    ids, mask = tok.encode_batch(smiles, add_special_tokens=True)
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


torch.manual_seed(0)
model = TinyMLM(tokenizer.vocab_size)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(600):
    idx = torch.randint(0, len(train_smiles), (32,))
    batch = [train_smiles[i] for i in idx]
    corrupted, labels, attn_mask = make_mlm_batch(batch, tokenizer)
    logits, _ = model(corrupted, attn_mask)
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
    opt.zero_grad(); loss.backward(); opt.step()
model.eval()
print(f"trained — final train loss {loss.item():.3f}")
# -

# A **fixed** evaluation batch (same masking every time) so that every
# ablation below is compared on identical inputs.

# +
eval_corrupted, eval_labels, eval_mask = make_mlm_batch(eval_smiles, tokenizer, seed=123)


@torch.no_grad()
def eval_loss(model):
    logits, _ = model(eval_corrupted, eval_mask)
    return nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), eval_labels.view(-1), ignore_index=-100
    ).item()


base_loss = eval_loss(model)
print(f"baseline eval loss (all {N_HEADS} heads): {base_loss:.4f}")
# -

# ---
# ## 1. How diverse are the trained heads?
#
# First, a recap of notebook 05 §6's diversity metric — but now on a *trained*
# model and *averaged over the evaluation set*. For each molecule we take the
# per-head attention matrices, flatten them, and compute pairwise cosine
# similarity; averaging over molecules gives a stable head-vs-head picture.

# +
@torch.no_grad()
def mean_pairwise_similarity(model, smiles, tok):
    sims = []
    for s in smiles:
        ids, mask = tok.encode_batch([s], add_special_tokens=True)
        _, attn = model(torch.tensor(ids), torch.tensor(mask))
        flat = attn[0].reshape(N_HEADS, -1)
        flat = flat / flat.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        sims.append((flat @ flat.T).numpy())
    return np.mean(sims, axis=0)


sim = mean_pairwise_similarity(model, eval_smiles[:60], tokenizer)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(N_HEADS)); ax.set_yticks(range(N_HEADS))
ax.set_xlabel("head"); ax.set_ylabel("head")
ax.set_title("Mean pairwise head similarity (cosine)")
for i in range(N_HEADS):
    for j in range(N_HEADS):
        ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                color="white" if sim[i, j] < 0.6 else "black", fontsize=8)
fig.colorbar(im, ax=ax, shrink=0.85)
plt.tight_layout()
plt.show()
# -

# 💡 **Key Insight.** Off-diagonal entries near 1.0 are pairs of heads doing
# almost the *same* thing — redundant. Entries near 0 are genuinely distinct
# heads. A trained model usually has both: a few strongly specialized heads
# and clusters of near-duplicates. The duplicates are what makes pruning
# possible.

# ---
# ## 2. Head importance via ablation
#
# How much does each head *matter*? We measure it directly: **ablate** one
# head — switch off its contribution — and see how much the evaluation loss
# rises. A head whose removal barely moves the loss is unimportant; one that
# spikes the loss is doing real work.
#
# The clean way to ablate head `h`: in `MultiHeadAttention` the concatenated
# heads are mixed by `W_O`, and head `h` occupies input columns
# `h·d_k : (h+1)·d_k` of `W_O`. Zeroing those columns removes exactly head
# `h`'s contribution and nothing else.

# +
import copy

d_k = D_MODEL // N_HEADS


def ablate_heads(model, heads_to_kill):
    """Return a copy of `model` with the given heads zeroed out of W_O."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        for h in heads_to_kill:
            m.attn.w_o.weight[:, h * d_k:(h + 1) * d_k] = 0.0
    return m


importance = []
for h in range(N_HEADS):
    importance.append(eval_loss(ablate_heads(model, [h])) - base_loss)
importance = np.array(importance)

order = np.argsort(importance)                          # least → most important
fig, ax = plt.subplots(figsize=(7.5, 4.2))
colours = ["#dd8452" if i in order[:N_HEADS // 2] else "#4c72b0" for i in range(N_HEADS)]
ax.bar(range(N_HEADS), importance, color=colours)
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("head"); ax.set_ylabel("Δ eval loss when ablated")
ax.set_title("Head importance — how much the loss rises if we remove each head")
ax.set_xticks(range(N_HEADS))
plt.tight_layout()
plt.show()
print(f"least important head: {order[0]} (Δloss {importance[order[0]]:+.4f})")
print(f"most  important head: {order[-1]} (Δloss {importance[order[-1]]:+.4f})")
# -

# ⚠️ **Note.** Some bars may be *negative* — removing that head slightly
# *improves* the eval loss. That's not a bug; a head can be net-unhelpful on
# held-out data (mild overfitting or destructive interference). Those are the
# first heads you'd prune.

# ---
# ## 3. Prune the least-important heads
#
# Now the payoff. We rank heads from least to most important and switch them
# off **cumulatively**, recording the eval loss after each removal. If heads
# are redundant, the loss should stay flat for a while before finally climbing
# as we delete the heads that actually matter.

# +
losses_pruned = [base_loss]
for k in range(1, N_HEADS + 1):
    losses_pruned.append(eval_loss(ablate_heads(model, list(order[:k]))))

heads_kept = list(range(N_HEADS, -1, -1))               # N_HEADS down to 0
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(heads_kept, losses_pruned, marker="o", color="#4c72b0", lw=2)
ax.axhline(base_loss, color="gray", ls="--", lw=1, label="all heads")
ax.invert_xaxis()
ax.set_xlabel("heads kept (pruning least-important first)")
ax.set_ylabel("eval MLM loss")
ax.set_title("Graceful degradation: most heads can go before quality drops")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
print("loss as heads are removed (most→fewest kept):")
for kept, lo in zip(heads_kept, losses_pruned):
    print(f"  {kept} heads kept: {lo:.4f}")
# -

# 💡 **Key Insight.** The curve is the whole point of Michel et al. (2019): you
# can usually drop a good fraction of the heads with **almost no loss**, then
# quality falls off a cliff once you start deleting the genuinely important
# ones. Redundancy is real — but it isn't unlimited. (Our toy model is tiny;
# the effect is even more dramatic in large pre-trained transformers, where a
# majority of heads can be pruned at test time.)

# ---
# ## 4. Which head sees which chemistry?
#
# Importance scores tell us *whether* a head matters; the attention patterns
# tell us *what* it does. Let's look at the most-important head's pattern on a
# molecule with both a ring and a carbonyl, and contrast it with a
# low-importance head.

# +
probe = "CC(=O)Oc1ccccc1C(=O)O"                          # aspirin
probe_tokens = ["[CLS]"] + tokenizer.tokenize(probe) + ["[SEP]"]
ids, mask = tokenizer.encode_batch([probe], add_special_tokens=True)
with torch.no_grad():
    _, probe_attn = model(torch.tensor(ids), torch.tensor(mask))

plot_per_head_grid(probe_attn[0], probe_tokens,
                   title="All heads on aspirin (trained)")
plt.show()

top_head, low_head = int(order[-1]), int(order[0])
print(f"Most-important head = {top_head}; least-important head = {low_head}.")
print("Compare their panels above: the important head usually shows clear,")
print("structured focus (e.g. onto ring atoms or the carbonyl), while the")
print("prunable head is diffuse or near-uniform.")
# -

# 🧪 **Chemical Intuition.** In a fully trained MolFormer-scale model these
# specialized heads line up with recognizable chemistry: ring-membership,
# heteroatom neighbourhoods, bond-distance bands, `[CLS]` aggregation. Our toy
# model only hints at it — but the *mechanism* by which interpretable,
# prunable heads arise is exactly what you're seeing.

# ---
# ## Checkpoint exercises

# +
# Exercise 1 — ablate a pair
# --------------------------
# Ablate the TWO most-important heads together. Is the loss increase roughly
# the sum of their individual Δloss values, or more/less? What does that tell
# you about whether they overlap?

# YOUR CODE HERE

# --- Solution (try the exercise first, then peek) ---
# h1, h2 = int(order[-1]), int(order[-2])
# joint = eval_loss(ablate_heads(model, [h1, h2])) - base_loss
# print(f"individual: {importance[h1]:+.4f} + {importance[h2]:+.4f} "
#       f"= {importance[h1] + importance[h2]:+.4f}")
# print(f"joint ablation: {joint:+.4f}")
# # If joint < sum, the heads are partly redundant (they covered for each other).

# +
# Exercise 2 — random pruning baseline
# ------------------------------------
# Instead of pruning least-important-first, prune heads in a RANDOM order and
# plot the loss curve on top of the §3 curve. Importance-ranked pruning should
# stay lower for longer.

# YOUR CODE HERE

# --- Solution ---
# torch.manual_seed(7)
# rand_order = torch.randperm(N_HEADS).tolist()
# rand_losses = [base_loss] + [eval_loss(ablate_heads(model, rand_order[:k]))
#                              for k in range(1, N_HEADS + 1)]
# plt.plot(heads_kept, losses_pruned, marker="o", label="importance-ranked")
# plt.plot(heads_kept, rand_losses,   marker="s", label="random order")
# plt.gca().invert_xaxis(); plt.xlabel("heads kept"); plt.ylabel("eval loss")
# plt.legend(); plt.show()

# +
# Exercise 3 — does more training increase specialization?
# --------------------------------------------------------
# Retrain the model for 3× as many steps and recompute the pairwise similarity
# matrix. Do the off-diagonal similarities go DOWN (heads diverge) as training
# proceeds? (Hint: wrap §0's training in a function of n_steps.)

# YOUR CODE HERE

# --- Solution sketch ---
# Train longer, call mean_pairwise_similarity again, and compare the mean of
# the off-diagonal entries. Typically heads specialize further with training,
# lowering average off-diagonal similarity — though tiny models on tiny data
# can plateau quickly.

# ---
# ## What's next
#
# Head pruning is one lens on transformer efficiency; notebook 04.1 covered
# another (linear attention's `O(N)` cost). Back on the **main path**,
# notebook 06 wraps multi-head attention into a full transformer block, and
# notebook 07 trains the stacked model — at which point the head
# specialization you measured here becomes genuinely chemical.
#
# 📚 **References.**
# - Michel, P., Levy, O. & Neubig, G. (2019). *Are Sixteen Heads Really Better
#   than One?* — head importance scores and test-time pruning.
# - Voita, E. et al. (2019). *Analyzing Multi-Head Self-Attention: Specialized
#   Heads Do the Heavy Lifting, the Rest Can Be Pruned.*
# - Clark, K. et al. (2019). *What Does BERT Look At? An Analysis of BERT's
#   Attention.* — interpretable attention heads.
