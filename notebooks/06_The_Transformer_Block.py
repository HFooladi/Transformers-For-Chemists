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

# <a href="https://colab.research.google.com/github/HFooladi/Transformers-For-Chemists/blob/main/notebooks/06_The_Transformer_Block.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 06 · The Transformer Block
#
# Notebook 05 gave us multi-head attention: several parallel "ways of looking"
# that let tokens **exchange** information. But attention alone has two gaps.
# First, its output for a token is just a *weighted average of other tokens'
# values* — a linear mix, with no place for a token to do non-linear
# processing on what it gathered. Second, a single attention layer is shallow;
# real models stack dozens, and naive stacking destroys the signal.
#
# The **transformer block** closes both gaps with three ingredients:
#
# 1. a **position-wise feed-forward network** — per-token non-linear "thinking",
# 2. **residual connections** — a gradient highway that makes deep stacks
#    trainable,
# 3. **LayerNorm** — keeps activations well-scaled at every depth.
#
# We build each from scratch, assemble the modern **pre-norm** encoder block,
# stack several, and confirm the block is shape-preserving — the one property
# that lets us stack it as many times as we like.

# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# 1. Explain what a block adds beyond attention: per-token computation + depth.
# 2. Build the position-wise **feed-forward network** (Linear → GELU → Linear,
#    `d_ff ≈ 4·d_model`) and give the "attention = talking, FFN = thinking"
#    intuition.
# 3. Demonstrate *why* **residual connections** are necessary with a signal-
#    and gradient-magnitude experiment across a deep stack.
# 4. Explain **LayerNorm** (per-token, across features) and contrast **pre-norm
#    vs post-norm**.
# 5. Assemble a **pre-norm `EncoderBlock`** and run it on a batch of molecules.
# 6. Stack several blocks, confirm **shape invariance**, and watch attention
#    evolve layer by layer.
# 7. Package everything into `FeedForward` and `EncoderBlock` modules and
#    verify against the hand-rolled versions.

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
    EncoderBlock,
    FeedForward,
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    TokenEmbedding,
)
from utils.tokenization_viz import plot_molecule_with_tokens
from utils.attention_viz import (
    animate_attention_over_layers,
    plot_attention_on_smiles,
    plot_per_head_grid,
)

torch.manual_seed(0)  # reproducible random weights
# -

# ---
# ## 1. What a block adds (the map)
#
# Same caffeine setup as notebooks 04 and 05.

# +
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
CORPUS = [
    "CCO", "CC(=O)Oc1ccccc1C(=O)O", CAFFEINE, "BrCCCl", "c1ccc2[nH]ccc2c1",
]
tokenizer = AtomTokenizer.from_smiles(CORPUS)

caffeine_ids, _ = tokenizer.encode_batch([CAFFEINE], add_special_tokens=True)
caffeine_ids = torch.tensor(caffeine_ids)                            # (1, L)
caffeine_tokens = ["[CLS]"] + tokenizer.tokenize(CAFFEINE) + ["[SEP]"]

D_MODEL = 32
N_HEADS = 4
D_FF = 4 * D_MODEL                                                   # = 128
token_embedding = TokenEmbedding(tokenizer.vocab_size, D_MODEL)
positional      = SinusoidalPositionalEncoding(d_model=D_MODEL, max_len=128)

x = positional(token_embedding(caffeine_ids))                        # (1, L, d_model)
L = x.size(1)
print(f"x shape: {tuple(x.shape)}   # (batch, seq_len, d_model)")
# -

# A roadmap of what we're about to build — the **pre-norm** transformer block.
# Data flows bottom to top; the two curved arrows are the residual
# connections that carry the input *around* each sub-layer.

fig, ax = plt.subplots(figsize=(5.2, 7))
boxes = [
    (0.5, "input  (B, L, d_model)", "#d9d9d9"),
    (1.6, "LayerNorm", "#fdd0a2"),
    (2.5, "Multi-Head Attention", "#9ecae1"),
    (3.4, "⊕  add residual", "#c7e9c0"),
    (4.5, "LayerNorm", "#fdd0a2"),
    (5.4, "Feed-Forward (GELU)", "#bcbddc"),
    (6.3, "⊕  add residual", "#c7e9c0"),
    (7.4, "output  (B, L, d_model)", "#d9d9d9"),
]
for y, label, colour in boxes:
    ax.add_patch(plt.Rectangle((0.6, y), 3.4, 0.6, facecolor=colour, edgecolor="black"))
    ax.text(2.3, y + 0.3, label, ha="center", va="center", fontsize=10)
    ax.annotate("", xy=(2.3, y), xytext=(2.3, y - 0.5),
                arrowprops=dict(arrowstyle="->", lw=1.4))
# Residual bypass arrows on the side.
ax.annotate("", xy=(4.1, 3.55), xytext=(4.1, 1.1),
            arrowprops=dict(arrowstyle="->", color="#31a354", lw=2,
                            connectionstyle="arc3,rad=0.45"))
ax.annotate("", xy=(4.1, 6.35), xytext=(4.1, 4.1),
            arrowprops=dict(arrowstyle="->", color="#31a354", lw=2,
                            connectionstyle="arc3,rad=0.45"))
ax.set_xlim(0, 5.2); ax.set_ylim(0, 8.2); ax.axis("off")
ax.set_title("The pre-norm transformer encoder block")
plt.tight_layout()
plt.show()

# 💡 **Key Insight.** A useful mantra: **attention is how tokens *talk*; the
# feed-forward network is how each token *thinks*** about what it heard. A
# transformer block alternates the two, and the residual + norm machinery is
# what makes a *stack* of such blocks actually trainable.

# ---
# ## 2. The position-wise feed-forward network
#
# The FFN is a tiny two-layer MLP — `Linear(d_model → d_ff)`, a non-linearity,
# `Linear(d_ff → d_model)` — applied **independently to every position**. It
# expands to a wider hidden dimension `d_ff` (conventionally `4·d_model`),
# applies a non-linearity, and projects back. We use **GELU**, the smooth
# activation used by BERT and MolFormer.

# +
ffn_lin1 = nn.Linear(D_MODEL, D_FF)
ffn_act  = nn.GELU()
ffn_lin2 = nn.Linear(D_FF, D_MODEL)

h_ffn = ffn_lin2(ffn_act(ffn_lin1(x)))                    # (B, L, d_model)
print(f"FFN: {tuple(x.shape)}  →[{D_MODEL}→{D_FF}]→ GELU →[{D_FF}→{D_MODEL}]→  {tuple(h_ffn.shape)}")
# -

# GELU vs ReLU — GELU is a smooth gate (it lets small negatives through a
# little), which trains slightly better than ReLU's hard cut-off.

z = torch.linspace(-4, 4, 200)
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(z, torch.relu(z), label="ReLU", lw=2, color="#dd8452")
ax.plot(z, torch.nn.functional.gelu(z), label="GELU", lw=2, color="#4c72b0")
ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
ax.set_xlabel("input"); ax.set_ylabel("output")
ax.set_title("GELU is a smooth ReLU")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 🧪 **Chemical Intuition.** "Position-wise" means the FFN never mixes across
# tokens — it transforms each token's vector on its own. So after caffeine's
# carbonyl carbon has *attended* to its neighbouring `=O` and ring `N`, the
# FFN is where it can compute a non-linear feature like "I am an amide
# carbon" from the context it just gathered. Attention moves information
# between atoms; the FFN digests it within each atom.

# ---
# ## 3. Residual connections — the highway that makes depth possible
#
# Stacking many transformations naively is a disaster: each layer rescales the
# signal a little, and over many layers the signal (and its gradient) either
# vanishes to zero or blows up. The fix is the **residual connection**:
# instead of `y = f(x)`, compute `y = x + f(x)`. Now there is always a direct
# path for the signal — and the gradient — to flow straight through.
#
# Let's prove it. We run a 24-layer stack of a deliberately *contractive*
# transform (each layer shrinks its input), once **without** residuals and
# once **with**, tracking the activation norm and the gradient norm at every
# depth.

# +
def run_deep_stack(depth=24, d=64, residual=False, seed=0):
    """Forward a vector through `depth` contractive layers; track signal+grad norms."""
    torch.manual_seed(seed)
    layers = [nn.Linear(d, d) for _ in range(depth)]
    for lin in layers:                                    # small weights → contractive
        nn.init.normal_(lin.weight, std=0.5 / math.sqrt(d))
        nn.init.zeros_(lin.bias)

    h = torch.randn(1, d, requires_grad=True)
    acts = [h]
    cur = h
    for lin in layers:
        f = torch.tanh(lin(cur))
        cur = cur + f if residual else f
        cur.retain_grad()                                 # so we can read its gradient
        acts.append(cur)

    signal = [a.detach().norm().item() for a in acts]
    acts[-1].pow(2).sum().backward()                      # backprop a scalar from the top
    grad = [a.grad.norm().item() if a.grad is not None else float("nan") for a in acts]
    return signal, grad


sig_plain, grad_plain = run_deep_stack(residual=False)
sig_res,   grad_res   = run_deep_stack(residual=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
depths = range(len(sig_plain))
ax1.plot(depths, sig_plain, marker="o", ms=3, label="no residual", color="#dd8452")
ax1.plot(depths, sig_res,   marker="o", ms=3, label="with residual", color="#4c72b0")
ax1.set_yscale("log"); ax1.set_xlabel("layer depth"); ax1.set_ylabel("activation L2 norm")
ax1.set_title("Forward signal vs depth"); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(depths, grad_plain, marker="o", ms=3, label="no residual", color="#dd8452")
ax2.plot(depths, grad_res,   marker="o", ms=3, label="with residual", color="#4c72b0")
ax2.set_yscale("log"); ax2.set_xlabel("layer depth"); ax2.set_ylabel("gradient L2 norm")
ax2.set_title("Backward gradient vs depth"); ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.show()
print(f"no-residual  signal: {sig_plain[0]:.2e} → {sig_plain[-1]:.2e}")
print(f"with-residual signal: {sig_res[0]:.2e} → {sig_res[-1]:.2e}")
# -

# ⚠️ **Note.** Without residuals the orange curves plummet — by layer 24 the
# signal from the input has all but disappeared, and so has its gradient
# (this is the classic *vanishing gradient* problem). With residuals (blue)
# both stay healthy: the `x +` term guarantees an un-attenuated path from
# every layer to every other. **This is the single trick that lets
# transformers be 12, 24, or 100+ layers deep.**

# ---
# ## 4. LayerNorm + pre-norm vs post-norm
#
# **LayerNorm** standardizes each token's vector across its `d_model` features
# to mean 0 and variance 1 (then applies a learned scale and shift). Note what
# it normalizes over: the *feature* axis of each token *independently* — not
# across the batch, not across the sequence.

ln = nn.LayerNorm(D_MODEL)
x_ln = ln(x)
print(f"per-token mean before LN: {x[0].mean(-1)[:4].tolist()}")
print(f"per-token mean after  LN: {[round(v, 4) for v in x_ln[0].mean(-1)[:4].tolist()]}  (≈ 0)")
print(f"per-token std   after  LN: {[round(v, 3) for v in x_ln[0].std(-1)[:4].tolist()]}  (≈ 1)")

# Where you *place* the LayerNorm matters. Two options:
#
# - **Post-norm** (original Transformer): `x → LayerNorm(x + sublayer(x))`.
#   The norm sits *on* the residual stream, so the signal is re-normalized
#   after every block.
# - **Pre-norm** (GPT-2+, ViT, MolFormer): `x → x + sublayer(LayerNorm(x))`.
#   The norm sits *inside* the residual branch, leaving the residual stream
#   itself untouched — preserving the §3 highway.
#
# Let's compare how the gradient survives depth for each placement.

# +
def run_norm_stack(depth=24, d=64, mode="pre", seed=0):
    torch.manual_seed(seed)
    layers = [nn.Linear(d, d) for _ in range(depth)]
    norms = [nn.LayerNorm(d) for _ in range(depth)]
    for lin in layers:
        nn.init.normal_(lin.weight, std=0.5 / math.sqrt(d)); nn.init.zeros_(lin.bias)

    h = torch.randn(1, d, requires_grad=True)
    acts = [h]; cur = h
    for lin, nrm in zip(layers, norms):
        if mode == "pre":
            cur = cur + torch.tanh(lin(nrm(cur)))
        else:                                             # post-norm
            cur = nrm(cur + torch.tanh(lin(cur)))
        cur.retain_grad(); acts.append(cur)
    acts[-1].pow(2).sum().backward()
    return [a.grad.norm().item() if a.grad is not None else float("nan") for a in acts]


grad_pre  = run_norm_stack(mode="pre")
grad_post = run_norm_stack(mode="post")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(grad_pre,  marker="o", ms=3, label="pre-norm",  color="#4c72b0")
ax.plot(grad_post, marker="o", ms=3, label="post-norm", color="#dd8452")
ax.set_yscale("log"); ax.set_xlabel("layer depth"); ax.set_ylabel("gradient L2 norm")
ax.set_title("Pre-norm keeps the gradient highway clean")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# -

# 💡 **Key Insight.** Pre-norm normalizes the *input* to each sub-layer but
# leaves the residual stream un-normalized — so the gradient highway from §3
# survives. Post-norm re-normalizes the stream every layer, which interrupts
# that path and (at depth, without learning-rate warmup) makes training
# fragile. That's why modern models, MolFormer included, use **pre-norm** —
# and it's exactly the placement our `EncoderBlock` uses. The full story is
# the subject of notebook 06.1.

# ---
# ## 5. Assemble the pre-norm `EncoderBlock`
#
# Now we wire the pieces together by hand:
#
# ```
# x = x + MultiHeadAttention(LayerNorm(x))     # talk, then add back
# x = x + FeedForward(LayerNorm(x))            # think, then add back
# ```

# +
attn  = MultiHeadAttention(D_MODEL, N_HEADS, dropout=0.0)
ff    = FeedForward(D_MODEL, D_FF, dropout=0.0)
norm1 = nn.LayerNorm(D_MODEL)
norm2 = nn.LayerNorm(D_MODEL)

# Run the hand-rolled block on a batch mixing ethanol (short) + caffeine.
batch_smiles = ["CCO", CAFFEINE]
batch_ids, batch_mask = tokenizer.encode_batch(batch_smiles, add_special_tokens=True)
batch_ids, batch_mask = torch.tensor(batch_ids), torch.tensor(batch_mask)
xb = positional(token_embedding(batch_ids))               # (2, L, d_model)

a, attn_w = attn(norm1(xb), mask=batch_mask)              # talk
h1 = xb + a
ffo = ff(norm2(h1))                                       # think
block_out = h1 + ffo

print(f"input  shape: {tuple(xb.shape)}")
print(f"output shape: {tuple(block_out.shape)}   # identical → the block is stackable")
print(f"attention shape: {tuple(attn_w.shape)}   # (B, n_heads, L, L)")
# -

plot_per_head_grid(attn_w[1].detach(), caffeine_tokens,
                   title="EncoderBlock attention on caffeine (random weights)")
plt.show()

# ⚠️ **Note.** The output has *exactly* the same shape as the input,
# `(B, L, d_model)`. That invariance is not a coincidence — it's the whole
# design goal. It's what lets us feed a block's output straight into another
# block, as many times as we like.

# ---
# ## 6. Stack blocks and watch attention evolve
#
# Because the block is shape-preserving, stacking is trivial: just loop. We
# build a 4-layer stack of `EncoderBlock`s, push caffeine through, and collect
# each layer's (head-averaged) attention so we can watch how the pattern
# changes with depth.

# +
N_LAYERS = 4
blocks = nn.ModuleList([EncoderBlock(D_MODEL, N_HEADS, D_FF, dropout=0.0)
                        for _ in range(N_LAYERS)])

h = positional(token_embedding(caffeine_ids))             # (1, L, d_model)
per_layer_attn = []
reps = [h.detach().clone()]
for blk in blocks:
    h, w = blk(h)                                         # (1, L, d_model), (1, H, L, L)
    per_layer_attn.append(w[0].mean(0).detach())          # head-averaged (L, L)
    reps.append(h.detach().clone())
print(f"stacked {N_LAYERS} blocks; final shape {tuple(h.shape)} (unchanged)")

# Static small-multiples grid of attention per layer (set static=False for a
# true animation: `from IPython.display import HTML; HTML(anim.to_jshtml())`).
animate_attention_over_layers(per_layer_attn, caffeine_tokens, static=True)
plt.show()
# -

# How much does each token's representation *change* from one layer to the
# next? We measure the average cosine similarity between consecutive layers'
# token vectors — high similarity means the layer barely moved things, low
# means it reshaped them.

# +
drift = []
for i in range(len(reps) - 1):
    a_, b_ = reps[i][0], reps[i + 1][0]                   # (L, d_model)
    cos = torch.nn.functional.cosine_similarity(a_, b_, dim=-1)
    drift.append(cos.mean().item())

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(range(1, len(reps)), drift, marker="o", color="#4c72b0")
ax.set_xlabel("after layer"); ax.set_ylabel("cosine sim to previous layer")
ax.set_title("How much each block reshapes the representation")
ax.set_xticks(range(1, len(reps))); ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# -

# 🔬 **Try This.** Change `N_LAYERS` and re-run. With random weights the
# per-layer changes are mechanical, not meaningful — but the *machinery* is
# exactly what a trained model uses. Notebook 09 trains this stack, and the
# layer-by-layer attention becomes chemically interpretable (early layers
# local, later layers global).

# ---
# ## 7. (Optional) Layer evolution with a trained stack
#
# As in notebook 05, a quick masked-language-modelling run turns the random
# patterns into learned ones. We reuse the same self-contained recipe, now
# with a stack of `EncoderBlock`s. Skip if you only want the mechanism; it
# runs in well under a minute on CPU.

# +
import urllib.request
from pathlib import Path

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
train_tok = AtomTokenizer.from_smiles(bbbp_smiles)
print(f"training corpus: {len(bbbp_smiles)} molecules, vocab {train_tok.vocab_size}")

# +
class TinyEncoderMLM(nn.Module):
    """Embedding + positional + a stack of EncoderBlocks + MLM head."""

    def __init__(self, vocab_size, n_layers=N_LAYERS, d_model=D_MODEL,
                 n_heads=N_HEADS, d_ff=D_FF, max_len=128):
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout=0.0) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask, collect_attn=False):
        h = self.pos(self.embed(input_ids))
        attns = []
        for blk in self.blocks:
            h, w = blk(h, mask=attention_mask)
            if collect_attn:
                attns.append(w)
        return self.head(h), attns


def make_mlm_batch(ids_list, mask_list, mask_prob=0.15,
                   special_ids=(PAD_ID, CLS_ID, SEP_ID, MASK_ID)):
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


torch.manual_seed(0)
model = TinyEncoderMLM(train_tok.vocab_size)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
losses = []
for step in range(400):
    idx = torch.randint(0, len(bbbp_smiles), (32,))
    batch = [bbbp_smiles[i] for i in idx]
    ids, mask = train_tok.encode_batch(batch, add_special_tokens=True)
    corrupted, labels, attn_mask = make_mlm_batch(ids, mask)
    logits, _ = model(corrupted, attn_mask)
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(np.convolve(losses, np.ones(15) / 15, mode="valid"), color="#4c72b0", lw=2)
ax.set_xlabel("optimizer step (15-step moving average)"); ax.set_ylabel("MLM loss")
ax.set_title(f"Tiny MLM training — {N_LAYERS}-layer encoder stack")
ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
print(f"loss: {losses[0]:.3f} → {np.mean(losses[-20:]):.3f}")
# -

# Attention across layers, now **trained**. Compare with the random-weight
# grid in §6 — the trained stack distributes attention with visible structure.

# +
caf_ids2, caf_mask2 = train_tok.encode_batch([CAFFEINE], add_special_tokens=True)
caf_tokens2 = ["[CLS]"] + train_tok.tokenize(CAFFEINE) + ["[SEP]"]
model.eval()
with torch.no_grad():
    _, attns = model(torch.tensor(caf_ids2), torch.tensor(caf_mask2), collect_attn=True)
trained_layers = [w[0].mean(0) for w in attns]            # head-averaged per layer

animate_attention_over_layers(trained_layers, caf_tokens2, static=True)
plt.show()
# -

# ---
# ## 8. The same thing, packaged
#
# `FeedForward` and `EncoderBlock` live in `utils/transformer_blocks.py`. We
# copy our hand-rolled weights into a fresh `EncoderBlock` and confirm it
# reproduces the §5 output exactly.

# +
block = EncoderBlock(D_MODEL, N_HEADS, D_FF, dropout=0.0)
block.attn.load_state_dict(attn.state_dict())
block.ff.load_state_dict(ff.state_dict())
block.norm1.load_state_dict(norm1.state_dict())
block.norm2.load_state_dict(norm2.state_dict())

out_mod, attn_mod = block(xb, mask=batch_mask)
print(f"output shape: {tuple(out_mod.shape)}   # (B, L, d_model)")
print(f"matches hand-rolled block? {torch.allclose(out_mod, block_out, atol=1e-5)}")
print(f"output shape == input shape? {out_mod.shape == xb.shape}")

n_real_eth = int(batch_mask[0].sum().item())
print(f"no head attends to ethanol's [PAD] columns? "
      f"{float(attn_mod[0, :, :, n_real_eth:].max()) == 0.0}")

# Verify the standalone FeedForward module matches our hand-rolled FFN too.
ff_mod = FeedForward(D_MODEL, D_FF, dropout=0.0)
ff_mod.lin1.load_state_dict(ffn_lin1.state_dict())
ff_mod.lin2.load_state_dict(ffn_lin2.state_dict())
print(f"FeedForward module matches hand-rolled FFN? "
      f"{torch.allclose(ff_mod(x), h_ffn, atol=1e-5)}")
# -

# Closing picture: the final block's `[CLS]` attention (head-averaged) painted
# back onto caffeine's SMILES.

plot_attention_on_smiles(attn_mod[1].mean(0)[0].detach(), caffeine_tokens,
                         query_index=0,
                         title="EncoderBlock · [CLS] attention on caffeine (head-averaged)")
plt.show()

# 💡 **Key Insight.** `EncoderBlock` maps `(B, L, d_model)` to the *same*
# shape, so a stack of them is just a `for` loop. That is precisely what
# `TransformerEncoder` does in notebook 07 — embedding + positional encoding +
# `n_layers` of `EncoderBlock` + a pooling head — and then we finally
# **train** it on a real chemical task.

# ---
# ## Checkpoint exercises

# +
# Exercise 1 — the FFN expansion ratio
# ------------------------------------
# Build FeedForward(d_model=32, d_ff=...) for d_ff = 2*d_model and 4*d_model.
# Count the parameters of each (sum of p.numel()). What's the ratio? Confirm
# both leave the output shape unchanged.

# YOUR CODE HERE

# --- Solution (try the exercise first, then peek) ---
# for mult in (2, 4):
#     ff_ex = FeedForward(32, mult * 32, dropout=0.0)
#     n = sum(p.numel() for p in ff_ex.parameters())
#     out = ff_ex(torch.randn(1, 10, 32))
#     print(f"d_ff={mult}×d_model: {n} params, output {tuple(out.shape)}")
# # The 4× FFN has ~2× the parameters of the 2× FFN (both Linears scale with d_ff).

# +
# Exercise 2 — residual ablation
# ------------------------------
# Using run_deep_stack from §3, forward a vector through a 16-layer stack with
# and without residuals. Report the ratio (final signal norm / input signal
# norm) for each. Which one collapses?

# YOUR CODE HERE

# --- Solution ---
# s_plain, _ = run_deep_stack(depth=16, residual=False)
# s_res,   _ = run_deep_stack(depth=16, residual=True)
# print(f"no residual : {s_plain[-1] / s_plain[0]:.2e}  (collapses toward 0)")
# print(f"with residual: {s_res[-1] / s_res[0]:.2e}  (stays O(1) or grows)")

# +
# Exercise 3 — pre-norm vs post-norm placement
# ---------------------------------------------
# For a single sub-layer f(x) = Linear(x), compute both x + f(LayerNorm(x))
# (pre) and LayerNorm(x + f(x)) (post) on the same input. Confirm the two
# outputs differ, and that the pre-norm output's residual stream still
# contains the un-normalized x (its per-token norm varies), while the
# post-norm output has been re-normalized (per-token std ≈ 1).

# YOUR CODE HERE

# --- Solution ---
# torch.manual_seed(1)
# xin = torch.randn(1, 6, 32) * 3.0
# lin = nn.Linear(32, 32); ln_ex = nn.LayerNorm(32)
# pre  = xin + lin(ln_ex(xin))
# post = ln_ex(xin + lin(xin))
# print(f"pre and post differ? {not torch.allclose(pre, post)}")
# print(f"pre-norm per-token std (varies):  {[round(v,2) for v in pre[0].std(-1).tolist()]}")
# print(f"post-norm per-token std (≈ 1):    {[round(v,2) for v in post[0].std(-1).tolist()]}")

# ---
# ## What's next
#
# We now have the full transformer encoder block — attention, feed-forward,
# residuals, LayerNorm — and we've confirmed it stacks. **Notebook 07**
# assembles `TransformerEncoder` (token embedding + positional encoding +
# `n_layers` of `EncoderBlock` + a `[CLS]` pooling head) into a complete model
# and **trains** it on a real property-prediction task. Every random-weight
# picture from notebooks 04–06 becomes chemically meaningful once the weights
# are learned.
#
# 📚 **Deep-dive sub-series**
# - **06.1**: Pre-norm vs post-norm and gradient flow at depth — why placement
#   decides whether a deep transformer trains at all.
#
# 📚 **References.**
# - Vaswani, A. et al. (2017). *Attention Is All You Need.* — the block, FFN,
#   residual, and (post-norm) LayerNorm.
# - He, K. et al. (2016). *Deep Residual Learning for Image Recognition.* —
#   residual connections.
# - Ba, J., Kiros, J. & Hinton, G. (2016). *Layer Normalization.*
# - Xiong, R. et al. (2020). *On Layer Normalization in the Transformer
#   Architecture.* — pre-norm vs post-norm (notebook 06.1).
# - Hendrycks, D. & Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs).*
