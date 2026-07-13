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

# <a href="https://colab.research.google.com/github/HFooladi/Transformers-For-Chemists/blob/main/notebooks/09_Tiny_MolFormer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 09 · Tiny MolFormer
#
# This is where the course pays off. Notebook 07 trained a classifier **from
# scratch** and needed labels; notebook 08 **pre-trained** an encoder on
# unlabelled SMILES but never used the result. A real chemical foundation model
# joins the two: pre-train once on a large unlabelled corpus, then fine-tune the
# learned encoder on whatever small labelled task you have.
#
# We do exactly that — pre-train the MLM encoder on ChEMBL (notebook 08), then
# fine-tune it on a **tiny** labelled BBBP set, with an identical random-init
# model as the control. The pretrained encoder gets a head start, so it wins —
# most clearly when labels are scarce, the whole reason MolFormer-style models
# exist.

# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# 1. Assemble the full **pre-train → fine-tune** pipeline end to end.
# 2. **Save** a pretrained encoder's weights and **transplant** them into a
#    fresh classifier.
# 3. Build a `MoleculeClassifier` from a pretrained encoder + a new head.
# 4. Fine-tune with a sensible recipe (lower learning rate for the pretrained
#    body; freeze vs. full fine-tune; avoiding catastrophic forgetting).
# 5. Run a **from-scratch control** under an identical recipe and make the
#    comparison robust by averaging over seeds.
# 6. Trace *where* pretraining pays off via an AUC-vs-#labels curve, watch the
#    pretrained model's head start in its learning curve, and confirm the whole
#    model is tiny enough for a free Colab.

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
from utils.training_utils import (
    MLMCollator,
    count_parameters,
    evaluate,
    train_one_epoch,
)
from utils.data_loading import load_chembl_subset, load_moleculenet
from utils.preprocessing import clean_smiles

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Shared dimensions — IDENTICAL to notebook 08 so the pretrained encoder
# transplants cleanly into the classifier below.
D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_LEN = 64, 4, 2, 256, 128
print(f"device: {DEVICE}")
# -

# ---
# ## 1. The plan: pre-train once, fine-tune many times
#
# ```
#  Stage 1 (no labels)                 Stage 2 (few labels)
#  ───────────────────                 ─────────────────────
#  ChEMBL SMILES                       BBBP (small labelled set)
#       │                                     │
#   MLM pre-training  ──▶ save encoder ──▶ load encoder + fresh [CLS] head ──▶ fine-tune
#  (notebook 08)            (body only)               │
#                                                     ▼
#                              vs.  random-init encoder + head  (control)
# ```
#
# The experiment is a controlled A/B: the **same architecture** fine-tuned two
# ways — once starting from the pretrained encoder, once from random weights.
# Any difference is the value of pre-training.

# ---
# ## 2. Stage 1 — pre-train the MLM encoder on ChEMBL
#
# This is exactly notebook 08, now treated as a reusable step. We build the
# `TinyMLM` (encoder body + per-position head), mask ~15% of atoms with
# `MLMCollator`, and train for a few epochs.

# +
corpus = load_chembl_subset(n=5000)
tokenizer = AtomTokenizer.from_smiles(corpus)
print(f"pre-training corpus: {len(corpus)} ChEMBL SMILES, vocab {tokenizer.vocab_size}")


class TinyMLM(nn.Module):
    """TransformerEncoder body + a per-position MLM head (from notebook 08)."""

    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
            d_ff=D_FF, max_len=MAX_LEN, dropout=0.1)
        self.head = nn.Linear(D_MODEL, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        seq, _ = self.encoder(input_ids, attention_mask)
        return self.head(seq)


class SmilesCorpus(Dataset):
    def __init__(self, smiles, tokenizer, max_len=MAX_LEN):
        self.encoded = [tokenizer.encode(s, add_special_tokens=True)[:max_len] for s in smiles]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx]


torch.manual_seed(0)
pretrained = TinyMLM(tokenizer.vocab_size).to(DEVICE)
mlm_loader = DataLoader(SmilesCorpus(corpus, tokenizer), batch_size=64, shuffle=True,
                        collate_fn=MLMCollator(tokenizer.vocab_size, MASK_ID, PAD_ID))
mlm_opt = torch.optim.AdamW(pretrained.parameters(), lr=1e-3)

for epoch in range(1, 9):
    loss = train_one_epoch(pretrained, mlm_loader, mlm_opt, device=DEVICE)
    print(f"pre-train epoch {epoch} | MLM loss {loss:.3f}")
acc = evaluate(pretrained, mlm_loader, device=DEVICE, task="mlm")["masked_accuracy"]
print(f"final masked accuracy: {acc:.3f}")
# -

# ---
# ## 3. Save the learned encoder
#
# We save **only the encoder body** — the embeddings, positional encoding, and
# transformer blocks — not the MLM head. The head was a scaffold for the
# fill-in-the-blank game; the *body* holds the transferable chemistry.

ENCODER_PATH = "tiny_molformer_encoder.pt"
torch.save(pretrained.encoder.state_dict(), ENCODER_PATH)
print(f"saved encoder to {ENCODER_PATH}")
print(f"encoder parameters: {count_parameters(pretrained.encoder):,}")
print(f"full TinyMLM parameters (with head): {count_parameters(pretrained):,}")

# ⚠️ **Note.** The head is task-specific and gets thrown away. For fine-tuning
# we keep the body and attach a *fresh* head suited to the new task — a
# classification head here, but it could equally be a regression or MLM head.

# ---
# ## 4. Stage 2 — a tiny labelled BBBP set (the low-label regime)
#
# Transfer learning helps most when labels are scarce — the realistic situation
# in chemistry, where an assay might yield a few hundred measurements. So we
# deliberately fine-tune on a **small** slice of BBBP.

# +
from sklearn.model_selection import train_test_split

smiles_raw, labels_raw = load_moleculenet("bbbp")
pairs = [(clean_smiles(s), int(y)) for s, y in zip(smiles_raw, labels_raw)]
pairs = [(s, y) for s, y in pairs if s is not None]
X_all = [s for s, _ in pairs]
y_all = [y for _, y in pairs]

X_train_full, X_tmp, y_train_full, y_tmp = train_test_split(
    X_all, y_all, test_size=0.20, stratify=y_all, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=0)

# The low-label regime: keep only ~200 labelled training molecules.
N_LABELS = 200
X_small, _, y_small, _ = train_test_split(
    X_train_full, y_train_full, train_size=N_LABELS, stratify=y_train_full, random_state=0)
print(f"fine-tune on {len(X_small)} labelled molecules "
      f"(val {len(X_val)}, test {len(X_test)})")
# -

# ⚠️ **Note — vocabulary alignment.** We must reuse the **ChEMBL tokenizer** from
# §2, *not* refit one on BBBP. The pretrained encoder's embedding rows are
# indexed by that vocabulary; a different tokenizer would scramble the mapping
# and the transplanted weights would be meaningless. BBBP atoms unseen during
# pre-training simply map to `[UNK]` — an accepted limitation of a tiny vocab.

# +
class SmilesDataset(Dataset):
    def __init__(self, smiles, labels):
        self.smiles, self.labels = smiles, labels

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], self.labels[idx]


def collate_classification(batch):
    smis = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    ids, mask = tokenizer.encode_batch(smis, add_special_tokens=True, max_length=MAX_LEN)
    return torch.tensor(ids), torch.tensor(mask), torch.tensor(labels)


val_loader = DataLoader(SmilesDataset(X_val, y_val), batch_size=32, collate_fn=collate_classification)
test_loader = DataLoader(SmilesDataset(X_test, y_test), batch_size=32, collate_fn=collate_classification)
# -

# ---
# ## 5. Two classifiers: pretrained vs from-scratch
#
# Both are the *same* architecture — the `TransformerEncoder` body plus a
# `[CLS]`-pooling head from notebook 07. The only difference is the encoder's
# starting weights. `make_classifier` optionally **freezes** the encoder so only
# the head trains (used in the exercises).

# +
class ClassificationHead(nn.Module):
    def __init__(self, d_model=D_MODEL, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, 1)

    def forward(self, sequence_output):
        cls_vec = sequence_output[:, 0, :]                   # (B, d_model) — [CLS]
        return self.proj(self.dropout(cls_vec)).squeeze(-1)  # (B,)


class MoleculeClassifier(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, input_ids, attention_mask=None, return_attention=False):
        seq, attns = self.encoder(input_ids, attention_mask, return_attention=return_attention)
        logit = self.head(seq)
        return (logit, attns) if return_attention else logit


def make_classifier(pretrained: bool, seed: int = 0, freeze: bool = False):
    """Fresh classifier; if pretrained, load the saved encoder; if freeze, the
    encoder is fixed and only the head learns."""
    torch.manual_seed(seed)
    encoder = TransformerEncoder(
        tokenizer.vocab_size, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=D_FF, max_len=MAX_LEN, dropout=0.1)
    if pretrained:
        encoder.load_state_dict(torch.load(ENCODER_PATH))
    if freeze:
        for p in encoder.parameters():
            p.requires_grad = False
    return MoleculeClassifier(encoder, ClassificationHead()).to(DEVICE)


print("pretrained classifier:", count_parameters(make_classifier(True)), "params")
print("from-scratch classifier:", count_parameters(make_classifier(False)), "params (identical)")
# -

# 💡 **Key Insight.** Identical capacity, identical optimizer, identical data —
# the *only* knob is whether the encoder starts from pretrained chemistry or from
# noise. That makes this a clean controlled experiment.

def finetune(model, lr, epochs=25, patience=6, seed=0):
    """Fine-tune a classifier (only its trainable params); return
    (best_val_auc, test_auc, val_history). Works for full fine-tune and, when
    the encoder is frozen, for a linear probe."""
    torch.manual_seed(seed)
    train_loader = DataLoader(SmilesDataset(X_small, y_small), batch_size=32,
                              shuffle=True, collate_fn=collate_classification)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    history, best_auc, best_state, since = [], -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        for ids, mask, lab in train_loader:
            ids, mask, lab = ids.to(DEVICE), mask.to(DEVICE), lab.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(ids, mask), lab.float())
            loss.backward(); optimizer.step()
        va = evaluate(model, val_loader, device=DEVICE, task="classification")["roc_auc"]
        history.append(va)
        if va > best_auc:
            best_auc, since = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if since >= patience:
            break
    model.load_state_dict(best_state)
    test_auc = evaluate(model, test_loader, device=DEVICE, task="classification")["roc_auc"]
    return best_auc, test_auc, history

# ⚠️ **Note — the recipe.** A pretrained body is already good, so we nudge it
# gently with a **lower learning rate** (5e-4) than the from-scratch model
# (3e-4, the notebook-07 default). Too high a lr on a pretrained model erases
# what it learned — "catastrophic forgetting." When labels are *very* scarce,
# freezing the encoder and training only the head (`make_classifier(..., freeze=True)`)
# is the safest option; with a few hundred labels a gentle full fine-tune wins.

# ---
# ## 6. Head-to-head, averaged over seeds
#
# At this scale a single run is **noisy** — one lucky from-scratch run can match
# an unlucky pretrained one. The honest comparison averages over several seeds
# and reports the spread.

# +
SEEDS = [0, 1, 2]
pre_scores = [finetune(make_classifier(True, s), lr=5e-4, seed=s)[1] for s in SEEDS]
scr_scores = [finetune(make_classifier(False, s), lr=3e-4, seed=s)[1] for s in SEEDS]
print(f"pretrained  : {np.mean(pre_scores):.3f} ± {np.std(pre_scores):.3f}  {[round(a, 3) for a in pre_scores]}")
print(f"from-scratch: {np.mean(scr_scores):.3f} ± {np.std(scr_scores):.3f}  {[round(a, 3) for a in scr_scores]}")

fig, ax = plt.subplots(figsize=(5, 4.5))
ax.bar(["pretrained", "from-scratch"], [np.mean(pre_scores), np.mean(scr_scores)],
       yerr=[np.std(pre_scores), np.std(scr_scores)], capsize=8,
       color=["#4c72b0", "#c44e52"], edgecolor="white")
ax.set_ylabel("test ROC-AUC"); ax.set_ylim(0.5, 1.0)
ax.set_title(f"Fine-tuning gap at {N_LABELS} labels (mean ± std, {len(SEEDS)} seeds)")
plt.tight_layout(); plt.show()
# -

# 💡 **Key Insight.** Starting from the pretrained encoder gives a clear edge at
# 200 labels — the pretrained model's *worst* seed is about the from-scratch
# model's *best*. Pre-training drops the model into a good region of weight
# space, so the few labels we have go toward refining chemistry the model already
# half-knows, instead of discovering it from noise.

# ---
# ## 7. Where pretraining pays off most — AUC vs #labels
#
# The transfer advantage is largest when labels are few and shrinks as data
# grows. We sweep the fine-tuning set size (averaging two seeds per point to tame
# the noise) and plot both models.

# +
LABEL_GRID = [200, 400, 800]
pre_curve, scr_curve = [], []
for n in LABEL_GRID:
    Xn, _, yn, _ = train_test_split(
        X_train_full, y_train_full, train_size=n, stratify=y_train_full, random_state=0)
    X_small, y_small = Xn, yn   # finetune() reads these globals
    pre_curve.append(np.mean([finetune(make_classifier(True, s), 5e-4, seed=s)[1] for s in (0, 1)]))
    scr_curve.append(np.mean([finetune(make_classifier(False, s), 3e-4, seed=s)[1] for s in (0, 1)]))
    print(f"{n:4d} labels | pretrained {pre_curve[-1]:.3f} | from-scratch {scr_curve[-1]:.3f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(LABEL_GRID, pre_curve, "-o", color="#4c72b0", label="pretrained")
ax.plot(LABEL_GRID, scr_curve, "-o", color="#c44e52", label="from-scratch")
ax.set_xlabel("# labelled training molecules"); ax.set_ylabel("test ROC-AUC")
ax.set_title("Transfer advantage shrinks as labels grow")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()

# Restore the default fine-tune set for later cells.
X_small, y_small = train_test_split(
    X_train_full, y_train_full, train_size=N_LABELS, stratify=y_train_full, random_state=0)[0::2]
# -

# 💡 **Key Insight.** This is the textbook transfer-learning signature: the gap
# is widest on the left (few labels) and narrows as labels grow. A chemical
# foundation model is most valuable exactly where chemistry usually lives — the
# low-data regime; with enough labels a from-scratch model eventually catches up.

# ---
# ## 8. Transfer in action — the head start
#
# Pre-training shows up vividly in the **learning curve**: the pretrained model's
# validation AUC starts higher and climbs faster, because it begins fine-tuning
# from chemistry it already knows rather than from noise.

# +
_, _, hist_pre = finetune(make_classifier(True, 0), lr=5e-4, epochs=20, patience=20, seed=0)
_, _, hist_scr = finetune(make_classifier(False, 0), lr=3e-4, epochs=20, patience=20, seed=0)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(range(1, len(hist_pre) + 1), hist_pre, "-o", color="#4c72b0", label="pretrained")
ax.plot(range(1, len(hist_scr) + 1), hist_scr, "-o", color="#c44e52", label="from-scratch")
ax.set_xlabel("fine-tuning epoch"); ax.set_ylabel("validation ROC-AUC")
ax.set_title(f"Pretrained gets a head start ({N_LABELS} labels)")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
# -

# 🧪 **Chemical Intuition.** The from-scratch model spends its first epochs just
# learning that SMILES has atoms, bonds, and rings — structure the pretrained
# model already absorbed from 5,000 unlabelled ChEMBL molecules. Fine-tuning then
# only has to connect that structure to brain penetration, which takes far fewer
# labelled examples.

# ---
# ## Checkpoint exercises

# +
# Exercise 1 — freeze vs full fine-tune
# -------------------------------------
# Compare make_classifier(True, freeze=True) (linear probe — train only the head)
# against the full fine-tune at N_LABELS=200. Which reaches higher test AUC?
# Note: the [CLS] vector of a PURE-MLM model is a weak summary (no sentence-level
# objective trained it), so the probe usually trails full fine-tuning here.

# YOUR CODE HERE

# --- Solution (try the exercise first, then peek) ---
# probe = finetune(make_classifier(True, 0, freeze=True), lr=1e-3, seed=0)[1]
# full  = finetune(make_classifier(True, 0), lr=5e-4, seed=0)[1]
# print(f"frozen probe {probe:.3f}  vs  full fine-tune {full:.3f}")

# +
# Exercise 2 — how low can labels go?
# -----------------------------------
# Push N_LABELS down to 50 and 30 (full fine-tune). Does pretraining still win?
# Does the from-scratch variance explode? Average over a few seeds to be sure.

# YOUR CODE HERE

# --- Solution ---
# for n in (30, 50):
#     Xn, _, yn, _ = train_test_split(X_train_full, y_train_full, train_size=n,
#                                     stratify=y_train_full, random_state=0)
#     X_small, y_small = Xn, yn
#     pre = [finetune(make_classifier(True, s), 5e-4, seed=s)[1] for s in range(3)]
#     scr = [finetune(make_classifier(False, s), 3e-4, seed=s)[1] for s in range(3)]
#     print(n, np.mean(pre), np.mean(scr))

# +
# Exercise 3 — fine-tune learning-rate sweep
# -------------------------------------------
# Sweep the pretrained model's lr over {1e-4, 5e-4, 2e-3}. Show that too high a
# lr destroys the pretrained features (AUC drops toward the from-scratch level).

# YOUR CODE HERE

# --- Solution ---
# for lr in (1e-4, 5e-4, 2e-3):
#     auc = finetune(make_classifier(True, 0), lr=lr, seed=0)[1]
#     print(f"lr={lr}: test AUC {auc:.3f}")
# # 2e-3 forgets; 5e-4 is the sweet spot.
# -

# ---
# ## What's next
#
# You have built — and trained — a tiny but complete MolFormer: tokenizer,
# encoder, MLM pre-training, and transfer to a downstream task, every line from
# scratch. **Notebook 10** rebuilds this exact model with the HuggingFace
# `transformers` stack, so you can see how each hand-written component maps to
# one production-grade line and how to load a *real* pretrained chemical model.
#
# 📚 **Deep-dive sub-series**
# - **08.1**: How the MLM masking ratio affects downstream transfer.
# - **09.1**: GNN vs encoder-transformer head-to-head on BBBP.
#
# 📚 **References.**
# - Devlin, J. et al. (2019). *BERT.* — pre-train then fine-tune.
# - Howard, J. & Ruder, S. (2018). *ULMFiT.* — transfer learning + discriminative
#   (lower) fine-tuning learning rates.
# - Ross, J. et al. (2022). *MolFormer.* — the chemical foundation model this
#   course builds toward.
# - Wu, Z. et al. (2018). *MoleculeNet.* — the BBBP task.
