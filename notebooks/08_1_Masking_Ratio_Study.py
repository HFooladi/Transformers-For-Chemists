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

# <a href="https://colab.research.google.com/github/HFooladi/Transformers-For-Chemists/blob/main/notebooks/08_1_Masking_Ratio_Study.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 08.1 · Masking Ratio Study — how much should you hide, and why 15%?
#
# Notebook 08 fixed the masking ratio at BERT's **15%** and its first exercise
# took a quick three-point peek. This deep-dive asks the question properly: how
# much of a molecule should you mask during MLM pre-training, and *why* did 15%
# become the default?
#
# The answer is a **trade-off between two forces** — the *signal* you get
# (number of prediction targets per molecule) and the *context* you leave behind
# (how much of the molecule the model can read to make each prediction). We
# isolate each force with a clean, reproducible experiment, then run the full
# pre-train → transfer sweep and confront an honest truth: at toy scale the
# downstream effect is buried in noise, which is itself the lesson about why this
# question needed large models to settle.

# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# 1. Separate the two metrics in play — pre-training masked-accuracy vs
#    **downstream transfer** — and know which one actually matters.
# 2. Quantify the **signal** side: prediction targets per molecule vs ratio.
# 3. Demonstrate the **context** side: accuracy falls as more of the molecule is
#    hidden.
# 4. Explain the signal-vs-context trade-off and why ~15% balances it.
# 5. Read a real masking-ratio sweep honestly — including when an effect is too
#    small to resolve without scale.

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

ensure_environment(["torch", "rdkit", "matplotlib", "tokenizers", "sklearn", "pandas"])

# +
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils.smiles_tokenizers import AtomTokenizer, MASK_ID, PAD_ID
from utils.transformer_blocks import TransformerEncoder
from utils.training_utils import MLMCollator, MLMMaskingConfig, evaluate, train_one_epoch
from utils.data_loading import load_chembl_subset, load_moleculenet
from utils.preprocessing import clean_smiles

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_LEN = 64, 4, 2, 256, 128
print(f"device: {DEVICE}")
# -

# **Scope.** Analysis + tiny-training, in the style of the other deep-dives. The
# models are small, but the *mechanism* — signal vs context — is exactly the one
# the literature analyses at scale.

# +
corpus = load_chembl_subset(n=3000)
tokenizer = AtomTokenizer.from_smiles(corpus)
print(f"corpus: {len(corpus)} ChEMBL SMILES, vocab {tokenizer.vocab_size}")


class TinyMLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
                                          n_layers=N_LAYERS, d_ff=D_FF, max_len=MAX_LEN, dropout=0.1)
        self.head = nn.Linear(D_MODEL, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        seq, _ = self.encoder(input_ids, attention_mask)
        return self.head(seq)


class SmilesCorpus(Dataset):
    def __init__(self, smiles):
        self.encoded = [tokenizer.encode(s, add_special_tokens=True)[:MAX_LEN] for s in smiles]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx]
# -

# ---
# ## 1. Two metrics, and two forces
#
# It's tempting to score a masking ratio by how well the model fills in the
# blanks (**masked-accuracy**). But that's a pre-training proxy; what we actually
# want is **downstream transfer** — does the pretrained encoder help a real task?
# The two can disagree, and the reason is a trade-off between two forces that
# masking controls:
#
# - **Signal**: each masked atom is one prediction target — one gradient. Mask
#   more → more targets per molecule → more learning per pass.
# - **Context**: the model predicts a masked atom from the atoms left visible.
#   Mask more → less context → each prediction is harder, eventually impossible.
#
# 15% is the classic balance point. Let's measure each force.

# ---
# ## 2. The signal side — prediction targets per molecule
#
# More masking means more blanks to fill, i.e. more supervised targets squeezed
# out of every unlabelled molecule.

# +
lengths = [len(tokenizer.tokenize(s)) for s in corpus]
median_len = int(np.median(lengths))
ratios = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
targets = ratios * median_len

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ratios * 100, targets, "-o", color="#55a868")
ax.axvline(15, ls="--", color="gray", label="BERT 15%")
ax.set_xlabel("masking ratio (%)"); ax.set_ylabel(f"targets per molecule (median len {median_len})")
ax.set_title("Signal: more masking → more prediction targets")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
print("targets/molecule:", {f"{int(r*100)}%": round(t, 1) for r, t in zip(ratios, targets)})
# -

# 🧪 **Chemical Intuition.** At 5% you hide barely one or two atoms of a typical
# drug-like molecule — most of a forward pass is just copying visible atoms,
# which teaches little. At 15% you get a handful of genuine prediction targets
# per molecule, so each unlabelled SMILES does real work.

# ---
# ## 3. The context side — accuracy falls as you hide more
#
# Now the opposing force. We train **one** model at the standard 15%, then
# evaluate it while sweeping the *evaluation* masking ratio. Higher eval ratios
# hide more of each molecule, leaving less context to predict from — so accuracy
# should fall. (Single model, only the evaluation masking changes: a clean,
# low-variance probe of the context effect.)

# +
train_smiles, eval_smiles = corpus[:2700], corpus[2700:]

torch.manual_seed(0)
model = TinyMLM(tokenizer.vocab_size).to(DEVICE)
train_loader = DataLoader(SmilesCorpus(train_smiles), batch_size=64, shuffle=True,
                          collate_fn=MLMCollator(tokenizer.vocab_size, MASK_ID, PAD_ID))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for _ in range(6):
    train_one_epoch(model, train_loader, opt, device=DEVICE)

eval_ratios = [0.05, 0.15, 0.30, 0.50, 0.70]
ctx_acc, ctx_std = [], []
for er in eval_ratios:
    accs = []
    for seed in range(3):
        el = DataLoader(SmilesCorpus(eval_smiles), batch_size=64,
                        collate_fn=MLMCollator(tokenizer.vocab_size, MASK_ID, PAD_ID,
                                               config=MLMMaskingConfig(mask_probability=er), seed=seed))
        accs.append(evaluate(model, el, device=DEVICE, task="mlm")["masked_accuracy"])
    ctx_acc.append(np.mean(accs)); ctx_std.append(np.std(accs))
    print(f"eval ratio {er:.2f} | masked accuracy {ctx_acc[-1]:.3f} ± {ctx_std[-1]:.3f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar([r * 100 for r in eval_ratios], ctx_acc, yerr=ctx_std, fmt="-o", capsize=4, color="#4c72b0")
ax.set_xlabel("evaluation masking ratio (%)"); ax.set_ylabel("masked accuracy")
ax.set_title("Context: hide more of the molecule → harder to fill in")
ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
# -

# 💡 **Key Insight.** Same model, same molecules — only the *amount hidden*
# changes, and accuracy slides down as the ratio climbs. The floor (~base rate of
# common atoms like `C`/`c`) is what you'd get by guessing the most frequent
# token; the gap above it is what context buys you, and that gap shrinks as
# context disappears.

# ---
# ## 4. Why ~15%? The trade-off, made visual
#
# Put the two forces together. Too **little** masking: plenty of context, but so
# few targets that pre-training is slow and wasteful. Too **much**: lots of
# targets, but each is near-impossible because the context is gone. The sweet
# spot sits where you get enough targets while leaving enough to predict from —
# empirically ~15–20%.
#
# Seeing is believing: here is caffeine at a gentle 10% mask vs a brutal 50%.

# +
def show_mask(ratio, seed=0):
    toks = ["[CLS]"] + tokenizer.tokenize(CAFFEINE) + ["[SEP]"]
    g = torch.Generator().manual_seed(seed)
    eligible = [i for i, t in enumerate(toks) if t not in ("[CLS]", "[SEP]")]
    k = max(1, round(ratio * len(eligible)))
    masked = set(torch.tensor(eligible)[torch.randperm(len(eligible), generator=g)[:k]].tolist())
    fig, ax = plt.subplots(figsize=(min(0.45 * len(toks) + 1, 14), 1.4))
    for i, t in enumerate(toks):
        hid = i in masked
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor="#ffd24c" if hid else "#eeeeee",
                                   edgecolor="white", lw=1.5))
        ax.text(i + 0.5, 0.5, "[M]" if hid else t, ha="center", va="center",
                fontsize=8, fontweight="bold" if hid else "normal")
    ax.set_xlim(0, len(toks)); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(f"caffeine at {int(ratio*100)}% masking ({k} of {len(eligible)} atoms hidden)")
    plt.tight_layout(); plt.show()


show_mask(0.10)
show_mask(0.50)
# -

# 🧪 **Chemical Intuition.** At 10% the molecule is still obviously caffeine —
# the ring system and carbonyls anchor every guess. At 50% it's a sketch with
# half its atoms erased; even an expert would struggle to reconstruct it, and so
# does the model. That collapse of context is why very high ratios train poorly.

# ---
# ## 5. The honest downstream sweep
#
# The metric we *truly* care about is downstream transfer. So we run the full
# experiment: pre-train a fresh encoder at each ratio, then fine-tune it on a
# small BBBP set and record test ROC-AUC.

# +
from sklearn.model_selection import train_test_split

smiles_raw, labels_raw = load_moleculenet("bbbp")
pairs = [(clean_smiles(s), int(y)) for s, y in zip(smiles_raw, labels_raw)]
pairs = [(s, y) for s, y in pairs if s is not None]
Xtr, Xtmp, ytr, ytmp = train_test_split([s for s, _ in pairs], [y for _, y in pairs],
                                         test_size=0.2, stratify=[y for _, y in pairs], random_state=0)
Xval, Xtest, yval, ytest = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=0)
Xsmall, ysmall = train_test_split(Xtr, ytr, train_size=200, stratify=ytr, random_state=0)[0::2]


def clf_loader(xs, ys, shuffle=False):
    class _DS(Dataset):
        def __len__(self): return len(xs)
        def __getitem__(self, i): return xs[i], ys[i]

    def _coll(b):
        ids, m = tokenizer.encode_batch([z[0] for z in b], add_special_tokens=True, max_length=MAX_LEN)
        return torch.tensor(ids), torch.tensor(m), torch.tensor([z[1] for z in b])
    return DataLoader(_DS(), batch_size=32, shuffle=shuffle, collate_fn=_coll)


val_loader = clf_loader(Xval, yval)
test_loader = clf_loader(Xtest, ytest)


def pretrain_at(ratio, epochs=3, seed=0):
    torch.manual_seed(seed)
    m = TinyMLM(tokenizer.vocab_size).to(DEVICE)
    ld = DataLoader(SmilesCorpus(corpus), batch_size=64, shuffle=True,
                    collate_fn=MLMCollator(tokenizer.vocab_size, MASK_ID, PAD_ID,
                                           config=MLMMaskingConfig(mask_probability=ratio)))
    o = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(epochs):
        train_one_epoch(m, ld, o, device=DEVICE)
    return {k: v.clone() for k, v in m.encoder.state_dict().items()}


def transfer(encoder_state, seed=0):
    torch.manual_seed(seed)
    enc = TransformerEncoder(tokenizer.vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
                             n_layers=N_LAYERS, d_ff=D_FF, max_len=MAX_LEN, dropout=0.1)
    enc.load_state_dict(encoder_state)
    proj, drop = nn.Linear(D_MODEL, 1), nn.Dropout(0.1)

    class Clf(nn.Module):
        def __init__(self): super().__init__(); self.enc = enc; self.drop = drop; self.proj = proj
        def forward(self, i, a):
            h, _ = self.enc(i, a); return self.proj(self.drop(h[:, 0, :])).squeeze(-1)

    m = Clf().to(DEVICE)
    o = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=0.01)
    lf = nn.BCEWithLogitsLoss()
    tl = clf_loader(Xsmall, ysmall, shuffle=True)
    best, best_state, since = -1.0, None, 0
    for _ in range(20):
        m.train()
        for i, a, y in tl:
            i, a, y = i.to(DEVICE), a.to(DEVICE), y.to(DEVICE)
            o.zero_grad(); lf(m(i, a), y.float()).backward(); o.step()
        va = evaluate(m, val_loader, device=DEVICE, task="classification")["roc_auc"]
        if va > best:
            best, since = va, 0
            best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        else:
            since += 1
        if since >= 5:
            break
    m.load_state_dict(best_state)
    return evaluate(m, test_loader, device=DEVICE, task="classification")["roc_auc"]


sweep = [0.05, 0.15, 0.30, 0.50]
means, lows, highs = [], [], []
for r in sweep:
    enc = pretrain_at(r)
    scores = [transfer(enc, seed=s) for s in (0, 1)]
    means.append(np.mean(scores)); lows.append(min(scores)); highs.append(max(scores))
    print(f"ratio {r:.2f} | transfer AUC {np.mean(scores):.3f}  (seeds {[round(s, 3) for s in scores]})")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.errorbar([r * 100 for r in sweep], means,
            yerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)],
            fmt="-o", capsize=5, color="#8172b3")
ax.axvline(15, ls="--", color="gray", label="BERT 15%")
ax.set_xlabel("pre-training masking ratio (%)"); ax.set_ylabel("BBBP transfer ROC-AUC")
ax.set_title("Downstream transfer vs masking ratio (2 seeds, min–max band)")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
# -

# ⚠️ **The honest read.** At this toy scale the curve is **dominated by
# seed noise** — the per-seed spread (the band) is wider than the differences
# between ratios. We genuinely cannot resolve a few-percent masking-ratio effect
# with a 100k-parameter model on 3,000 molecules and 200 labels. That is the
# real lesson: small effects need scale (more data, bigger models, many seeds) to
# measure. The *mechanism* (§2–§4) is robust and visible here; the precise
# optimum is not.

# 💡 **Key Insight.** What the literature found *with* scale: Devlin et al. (2019)
# settled on 15% for BERT, and it stuck for years. Wettig et al. (2023) later
# showed the optimum isn't universal — **larger** models and tasks often prefer
# *higher* ratios (up to ~40%), because a bigger model can still exploit reduced
# context while reaping more targets. The trade-off you measured here is exactly
# the one that shifts with scale.

# ---
# ## Checkpoint exercises

# +
# Exercise 1 — finer grid near the default
# ----------------------------------------
# Re-run the §5 sweep on {0.10, 0.15, 0.20, 0.25} with 3 seeds and more pretrain
# epochs. Does a peak emerge from the noise, or does the band still swamp it?

# YOUR CODE HERE

# --- Solution (try the exercise first, then peek) ---
# for r in (0.10, 0.15, 0.20, 0.25):
#     enc = pretrain_at(r, epochs=5)
#     scores = [transfer(enc, seed=s) for s in range(3)]
#     print(r, round(np.mean(scores), 3), round(np.std(scores), 3))

# +
# Exercise 2 — context effect on a stronger model
# ------------------------------------------------
# Repeat the §3 eval-ratio sweep with N_LAYERS=4 / more pretrain epochs. Does a
# bigger model hold accuracy better at high eval ratios (more capacity to exploit
# limited context)?

# YOUR CODE HERE

# --- Solution ---
# Rebuild TinyMLM with a 4-layer TransformerEncoder, pretrain at 15%, and rerun
# the eval_ratios loop from §3. The high-ratio accuracy should sag less.

# +
# Exercise 3 — 80/10/10 vs 100% mask
# ----------------------------------
# At ratio 0.15, compare the default MLMMaskingConfig against
# MLMMaskingConfig(mask_token_fraction=1.0, random_token_fraction=0.0,
# keep_token_fraction=0.0) on downstream transfer.

# YOUR CODE HERE

# --- Solution ---
# cfg = MLMMaskingConfig(mask_probability=0.15, mask_token_fraction=1.0,
#                        random_token_fraction=0.0, keep_token_fraction=0.0)
# # rebuild pretrain_at to pass this cfg to MLMCollator, then compare transfer().
# -

# ---
# ## What's next
#
# Back on the main path: notebook 09 uses the 15% default to build the tiny
# MolFormer, and notebook 10 rebuilds it with HuggingFace. The masking ratio is
# one of several pre-training knobs — the position-encoding and attention
# deep-dives (04.1–04.4) cover others MolFormer actually tunes.
#
# 📚 **References.**
# - Devlin, J. et al. (2019). *BERT.* — the 15% / 80-10-10 convention.
# - Wettig, A. et al. (2023). *Should You Mask 15% in Masked Language Modeling?*
#   — larger models often prefer higher ratios.
# - Ross, J. et al. (2022). *MolFormer.* — masked pre-training on SMILES.
