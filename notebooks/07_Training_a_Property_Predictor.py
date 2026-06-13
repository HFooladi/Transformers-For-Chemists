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

# <a href="https://colab.research.google.com/github/HFooladi/Transformers-For-Chemists/blob/main/notebooks/07_Training_a_Property_Predictor.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 07 · Training a Property Predictor
#
# Notebooks 03–06 built every piece of a transformer encoder — token
# embeddings, positional encoding, attention, the feed-forward network, the
# pre-norm block — and confirmed the block **stacks**. But every picture we
# drew used *random* weights, so the attention patterns were structured yet
# meaningless. This notebook finally **trains** the thing.
#
# We assemble the pieces into a complete `TransformerEncoder`, bolt a tiny
# `[CLS]`-pooling **classification head** on top, and train it to predict
# whether a molecule crosses the **blood–brain barrier** (the BBBP task) — a
# real question in CNS drug design. Along the way we meet, for the first time
# in the course, the whole supervised-training toolkit: a loss, an optimizer, a
# train/validation/test split, a `DataLoader`, learning curves, early stopping,
# and honest evaluation with ROC-AUC and a confusion matrix.

# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# 1. Assemble `TransformerEncoder` = `TokenEmbedding` + positional encoding +
#    `n_layers` × `EncoderBlock`, threading the padding mask through the stack.
# 2. Explain `[CLS]` pooling and build a classification head on the `[CLS]`
#    vector — and see why pooling belongs in the *head*, not the encoder.
# 3. Set up a stratified **train / val / test** split and PyTorch `DataLoader`s
#    over tokenized SMILES.
# 4. Write a training loop with **AdamW** + **`BCEWithLogitsLoss`** and walk
#    through one optimizer step.
# 5. Read **learning curves**, recognise overfitting, and apply early stopping.
# 6. Evaluate with **accuracy + ROC-AUC** and read a **confusion matrix** and
#    **ROC curve**.
# 7. Interpret attention on **correctly vs. incorrectly** classified molecules,
#    then package everything into the reusable `TransformerEncoder` /
#    `train_one_epoch` / `evaluate` utilities and verify they match.

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

from utils.smiles_tokenizers import AtomTokenizer, CLS_ID, PAD_ID, SEP_ID
from utils.transformer_blocks import (
    EncoderBlock,
    SinusoidalPositionalEncoding,
    TokenEmbedding,
    TransformerEncoder,
)
from utils.training_utils import evaluate, train_one_epoch
from utils.data_loading import load_moleculenet
from utils.preprocessing import clean_smiles
from utils.attention_viz import animate_attention_over_layers, plot_attention_on_smiles
from utils.tokenization_viz import plot_molecule_with_tokens

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}")
# -

# ---
# ## 1. The task: does it cross the blood–brain barrier?
#
# The **BBBP** (Blood–Brain Barrier Penetration) dataset labels each molecule
# `1` if it penetrates the brain and `0` if it doesn't. It's a small, classic
# binary-classification benchmark from MoleculeNet — perfect for a from-scratch
# transformer that has to fit on a free Colab.

# +
smiles_raw, labels_raw = load_moleculenet("bbbp")
labels_raw = [int(y) for y in labels_raw]
n_pos = sum(labels_raw)
print(f"BBBP: {len(smiles_raw)} molecules — {n_pos} penetrant (1), "
      f"{len(smiles_raw) - n_pos} non-penetrant (0)")

fig, ax = plt.subplots(figsize=(4.5, 4))
ax.bar(["non-penetrant\n(0)", "penetrant\n(1)"],
       [len(smiles_raw) - n_pos, n_pos], color=["#c44e52", "#4c72b0"])
ax.set_ylabel("number of molecules")
ax.set_title("BBBP class balance")
plt.tight_layout(); plt.show()
# -

# 🧪 **Chemical Intuition.** The rule of thumb: small, lipophilic, roughly
# neutral molecules slip across the barrier; large, very polar, or charged ones
# don't. All of that is *latent in the SMILES string* — ring count, heteroatom
# density, formal charges, chain length. The model's job is to learn to read
# those cues. Notice the classes are imbalanced (more penetrant than not),
# which is exactly why we'll report **ROC-AUC**, not just accuracy.

# Let's look at one molecule from each class together with its atom-level
# tokenization — the sequence the transformer actually sees.

pos_example = smiles_raw[labels_raw.index(1)]
neg_example = smiles_raw[labels_raw.index(0)]
tmp_tok = AtomTokenizer.from_smiles([pos_example, neg_example])
for smi, lab in [(pos_example, "penetrant (1)"), (neg_example, "non-penetrant (0)")]:
    plot_molecule_with_tokens(smi, tmp_tok.tokenize(smi), title=f"BBBP — {lab}")
    plt.show()

# ---
# ## 2. From SMILES to model-ready batches
#
# Raw MoleculeNet rows are messy: BBBP contains salts and co-crystals written as
# multi-fragment SMILES (e.g. `[Cl].CC(C)NCC(O)...` — the very first row is the
# β-blocker propranolol hydrochloride). We clean each one to its **largest
# fragment, canonicalized**, dropping anything RDKit can't parse.

# +
# clean_smiles parses each SMILES, keeps the largest fragment, and re-emits a
# canonical string — or returns None if RDKit can't parse it. Cleaning row by
# row (rather than on the whole column at once) keeps each label paired with its
# molecule, so dropping a bad row never misaligns the labels.
n_multi = sum("." in s for s in smiles_raw)
kept = [(clean_smiles(s), y) for s, y in zip(smiles_raw, labels_raw)]
kept = [(s, y) for s, y in kept if s is not None]
n_bad = len(smiles_raw) - len(kept)

X_all = [s for s, _ in kept]
y_all = [y for _, y in kept]
print(f"dropped {n_bad} unparseable rows; {n_multi} rows were multi-fragment "
      f"(salts/co-crystals → largest fragment kept)")
print(f"clean dataset: {len(X_all)} molecules")
print(f"example cleaned SMILES: {X_all[0]}")
# -

# ⚠️ **Note.** We will train the tokenizer on the **training split only** (next
# section). Fitting the vocabulary on molecules the model is later evaluated on
# would leak information — a subtle but real form of data snooping.

# ---
# ## 3. Train / validation / test split
#
# Three disjoint sets, each with the same class balance (**stratified**):
# *train* fits the weights, *validation* picks when to stop, *test* is touched
# exactly once at the end for the honest number.

# +
from sklearn.model_selection import train_test_split

X_train, X_tmp, y_train, y_tmp = train_test_split(
    X_all, y_all, test_size=0.20, stratify=y_all, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=0)

for name, ys in [("train", y_train), ("val", y_val), ("test", y_test)]:
    print(f"{name:>5}: {len(ys):4d} molecules, {100 * np.mean(ys):.1f}% penetrant")
# -

# 💡 **Key Insight.** A *random* stratified split is the friendly setting. The
# harder, more honest BBBP benchmark uses a **scaffold split** (train and test
# share no molecular scaffolds), which we revisit in the GNN-vs-transformer
# deep-dive (09.1). For learning the training mechanics, a stratified random
# split is the right place to start.

# Now build the tokenizer on the training split, then a tiny `Dataset` that
# hands back `(smiles, label)` pairs. A `collate_fn` tokenizes and pads each
# batch on the fly to the batch's longest sequence.

# +
MAX_LEN = 128
tokenizer = AtomTokenizer.from_smiles(X_train)
print(f"vocabulary size (atom-level, train only): {tokenizer.vocab_size}")


class SmilesDataset(Dataset):
    def __init__(self, smiles, labels):
        self.smiles = smiles
        self.labels = labels

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], self.labels[idx]


def collate_classification(batch):
    smis = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    ids, mask = tokenizer.encode_batch(smis, add_special_tokens=True, max_length=MAX_LEN)
    return torch.tensor(ids), torch.tensor(mask), torch.tensor(labels)


BATCH_SIZE = 32
train_loader = DataLoader(SmilesDataset(X_train, y_train), batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate_classification)
val_loader = DataLoader(SmilesDataset(X_val, y_val), batch_size=BATCH_SIZE,
                        collate_fn=collate_classification)
test_loader = DataLoader(SmilesDataset(X_test, y_test), batch_size=BATCH_SIZE,
                         collate_fn=collate_classification)

# Sanity-check one batch and a decoded round-trip.
ids_b, mask_b, lab_b = next(iter(train_loader))
print(f"batch input_ids {tuple(ids_b.shape)}, mask {tuple(mask_b.shape)}, "
      f"labels {tuple(lab_b.shape)}")
print(f"first row decodes to: {tokenizer.decode(ids_b[0].tolist())}")
# -

# ---
# ## 4. The model: `TransformerEncoder` + a `[CLS]`-pooling head
#
# The plan, in one picture:
#
# ```
#  input_ids ─→ TokenEmbedding ─→ +PositionalEncoding ─→ EncoderBlock × N
#                                                              │
#                                            take row 0 = [CLS] vector
#                                                              │
#                                                  Dropout → Linear → logit
# ```
#
# We first build the encoder body **by hand** — a literal `for` loop over the
# `EncoderBlock`s from notebook 06 — so nothing is hidden. In §9 we'll swap in
# the packaged `TransformerEncoder` and confirm they're identical.

# +
D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT = 64, 4, 2, 256, 0.1


class HandRolledEncoder(nn.Module):
    """Embedding + positional encoding + a stack of EncoderBlocks.

    Returns the *full sequence* (B, L, d_model) and the per-layer attention —
    no pooling. Reading the sequence is the head's job.
    """

    def __init__(self, vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, d_ff=D_FF, max_len=MAX_LEN, dropout=DROPOUT):
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None, return_attention=False):
        h = self.dropout(self.pos(self.embed(input_ids)))   # (B, L, d_model)
        attns = []
        for blk in self.blocks:
            h, w = blk(h, mask=attention_mask, return_attention=return_attention)
            if return_attention:
                attns.append(w)                              # (B, n_heads, L, L)
        return (h, attns) if return_attention else (h, None)


class ClassificationHead(nn.Module):
    """Read the [CLS] vector (row 0) and map it to a single logit."""

    def __init__(self, d_model=D_MODEL, dropout=DROPOUT):
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
        seq, attns = self.encoder(input_ids, attention_mask,
                                  return_attention=return_attention)
        logit = self.head(seq)
        return (logit, attns) if return_attention else logit


torch.manual_seed(0)
model = MoleculeClassifier(
    HandRolledEncoder(tokenizer.vocab_size), ClassificationHead()).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
logit_b = model(ids_b.to(DEVICE), mask_b.to(DEVICE))
print(f"model parameters: {n_params:,}")
print(f"forward output shape: {tuple(logit_b.shape)}   # (B,) one logit per molecule")
# -

# 💡 **Key Insight.** The `[CLS]` token (id 2, always at position 0 thanks to
# `add_special_tokens=True`) is a **learnable summary slot**. It carries no atom
# of its own; instead, through attention it pulls in information from every real
# token, and its final-layer vector becomes the molecule "fingerprint" the head
# reads. Keeping pooling in the *head* — not the encoder — is deliberate: the
# same encoder body will drive a per-position MLM head in notebook 08 unchanged.

# ---
# ## 5. Loss, optimizer, and one training step
#
# For binary classification we use **`BCEWithLogitsLoss`**, which takes *raw
# logits* (not probabilities) and applies the sigmoid internally — numerically
# stabler than `sigmoid` followed by `BCELoss`. The optimizer is **AdamW**
# (Adam with decoupled weight decay), the modern default for transformers.

# +
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

# One annotated step, to make the cycle concrete.
model.train()
ids_b, mask_b, lab_b = next(iter(train_loader))
ids_b, mask_b, lab_b = ids_b.to(DEVICE), mask_b.to(DEVICE), lab_b.to(DEVICE)

optimizer.zero_grad()                                  # 1. clear old gradients
logit = model(ids_b, mask_b)                           # 2. forward
loss_before = loss_fn(logit, lab_b.float())            # 3. measure error
loss_before.backward()                                 # 4. backprop
optimizer.step()                                       # 5. nudge the weights

with torch.no_grad():
    loss_after = loss_fn(model(ids_b, mask_b), lab_b.float())
print(f"loss on this batch: {loss_before.item():.4f} → {loss_after.item():.4f} "
      f"(one step already moved it)")
# -

# 🧪 **Chemical Intuition.** Each step nudges the weights so the substructure
# patterns that co-occur with brain penetration push the `[CLS]` logit up, and
# those that co-occur with exclusion push it down. Repeat over the whole
# training set, many times, and the encoder learns chemistry-relevant features.

# ---
# ## 6. The training loop + learning curves
#
# We hand-roll a `train_one_epoch` / `evaluate` pair (§9 verifies them against
# the packaged versions), loop over epochs, and after each epoch record train
# loss, validation loss, validation accuracy, and validation ROC-AUC. We keep
# the weights from the **best validation ROC-AUC** epoch — that's **early
# stopping**.

# +
from sklearn.metrics import accuracy_score, roc_auc_score


def run_epoch(model, loader, optimizer=None):
    """One pass. If optimizer is given, train; else evaluate. Returns
    (mean_loss, accuracy, roc_auc)."""
    train = optimizer is not None
    model.train() if train else model.eval()
    total, n, logits_all, labels_all = 0.0, 0, [], []
    for ids, mask, lab in loader:
        ids, mask, lab = ids.to(DEVICE), mask.to(DEVICE), lab.to(DEVICE)
        with torch.set_grad_enabled(train):
            logit = model(ids, mask)
            loss = loss_fn(logit, lab.float())
        if train:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += loss.item(); n += 1
        logits_all.append(logit.detach().cpu()); labels_all.append(lab.cpu())
    logits_all = torch.cat(logits_all); labels_all = torch.cat(labels_all)
    probs = torch.sigmoid(logits_all).numpy(); labs = labels_all.numpy()
    acc = accuracy_score(labs, probs >= 0.5)
    auc = roc_auc_score(labs, probs) if len(set(labs)) > 1 else float("nan")
    return total / n, acc, auc


# Fresh model so the curves start from scratch.
torch.manual_seed(0)
model = MoleculeClassifier(
    HandRolledEncoder(tokenizer.vocab_size), ClassificationHead()).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

EPOCHS, PATIENCE = 15, 3
history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}
best_auc, best_state, best_epoch, since_improved = -1.0, None, 0, 0

for epoch in range(1, EPOCHS + 1):
    tr_loss, _, _ = run_epoch(model, train_loader, optimizer)
    va_loss, va_acc, va_auc = run_epoch(model, val_loader)
    history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
    history["val_acc"].append(va_acc); history["val_auc"].append(va_auc)
    flag = ""
    if va_auc > best_auc:
        best_auc, best_epoch, since_improved = va_auc, epoch, 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        flag = "  ← best"
    else:
        since_improved += 1
    print(f"epoch {epoch:2d} | train {tr_loss:.3f} | val {va_loss:.3f} | "
          f"acc {va_acc:.3f} | auc {va_auc:.3f}{flag}")
    if since_improved >= PATIENCE:
        print(f"early stopping (no val-AUC gain for {PATIENCE} epochs)")
        break

model.load_state_dict(best_state)   # restore best-validation weights
print(f"restored best epoch {best_epoch} (val AUC {best_auc:.3f})")

# +
ep = range(1, len(history["train_loss"]) + 1)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8))
axL.plot(ep, history["train_loss"], "-o", color="#4c72b0", label="train loss")
axL.plot(ep, history["val_loss"], "-o", color="#c44e52", label="val loss")
axL.axvline(best_epoch, ls="--", color="gray", label="best epoch")
axL.set_xlabel("epoch"); axL.set_ylabel("BCE loss"); axL.set_title("Learning curves")
axL.legend(); axL.grid(alpha=0.3)

axR.plot(ep, history["val_acc"], "-o", color="#55a868", label="val accuracy")
axR.plot(ep, history["val_auc"], "-o", color="#8172b3", label="val ROC-AUC")
axR.axvline(best_epoch, ls="--", color="gray")
axR.set_xlabel("epoch"); axR.set_ylabel("score"); axR.set_ylim(0, 1)
axR.set_title("Validation metrics"); axR.legend(); axR.grid(alpha=0.3)
plt.tight_layout(); plt.show()
# -

# ⚠️ **Note.** Overfitting has a signature: **train loss keeps falling while
# validation loss turns back up.** That divergence is the moment to stop —
# which is why we kept the best-validation weights rather than the last ones.
# (Exercise 3 makes this dramatic on purpose.)

# ---
# ## 7. Test-set evaluation: confusion matrix + ROC curve
#
# One pass over the held-out test set, the number we actually report.

# +
from sklearn.metrics import confusion_matrix, roc_curve

model.eval()
test_logits, test_labels = [], []
with torch.no_grad():
    for ids, mask, lab in test_loader:
        test_logits.append(model(ids.to(DEVICE), mask.to(DEVICE)).cpu())
        test_labels.append(lab)
test_probs = torch.sigmoid(torch.cat(test_logits)).numpy()
test_labs = torch.cat(test_labels).numpy()
test_pred = (test_probs >= 0.5).astype(int)

acc = accuracy_score(test_labs, test_pred)
auc = roc_auc_score(test_labs, test_probs)
print(f"TEST accuracy: {acc:.3f}   |   TEST ROC-AUC: {auc:.3f}")

cm = confusion_matrix(test_labs, test_pred)
fpr, tpr, _ = roc_curve(test_labs, test_probs)

fig, (axc, axr) = plt.subplots(1, 2, figsize=(12, 5))
im = axc.imshow(cm, cmap="Blues")
axc.set_xticks([0, 1]); axc.set_xticklabels(["pred 0", "pred 1"])
axc.set_yticks([0, 1]); axc.set_yticklabels(["true 0", "true 1"])
for i in range(2):
    for j in range(2):
        axc.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
axc.set_title("Confusion matrix (test)")
fig.colorbar(im, ax=axc, shrink=0.8)

axr.plot(fpr, tpr, color="#4c72b0", lw=2, label=f"ROC (AUC = {auc:.3f})")
axr.plot([0, 1], [0, 1], ls="--", color="gray", label="random")
axr.set_xlabel("false positive rate"); axr.set_ylabel("true positive rate")
axr.set_title("ROC curve (test)"); axr.legend(); axr.grid(alpha=0.3)
plt.tight_layout(); plt.show()
# -

# ---
# ## 8. What did it learn? Attention on a right vs. a wrong call
#
# The model is no longer random, so its attention is finally *meaningful*. Let's
# paint the `[CLS]` token's attention (head-averaged, last layer) back onto the
# SMILES — for one molecule it got **right** and one it got **wrong**.

# +
# Find a confident-correct and a confident-wrong test molecule.
order = np.argsort(-np.abs(test_probs - 0.5))   # most confident first
correct_idx = next(i for i in order if test_pred[i] == test_labs[i])
wrong_idx = next(i for i in order if test_pred[i] != test_labs[i])


def cls_attention_on(smiles):
    ids, mask = tokenizer.encode_batch([smiles], add_special_tokens=True, max_length=MAX_LEN)
    toks = ["[CLS]"] + tokenizer.tokenize(smiles) + ["[SEP]"]
    with torch.no_grad():
        _, attns = model(torch.tensor(ids).to(DEVICE),
                         torch.tensor(mask).to(DEVICE), return_attention=True)
    last = attns[-1][0].mean(0).cpu()            # (L, L) head-averaged, last layer
    return last, toks


for idx, tag in [(correct_idx, "correct"), (wrong_idx, "wrong")]:
    smi = X_test[idx]
    last, toks = cls_attention_on(smi)
    plot_attention_on_smiles(
        last[0][: len(toks)], toks, query_index=0,
        title=f"[CLS] attention — {tag} "
              f"(true {test_labs[idx]}, pred {test_pred[idx]}, p={test_probs[idx]:.2f})")
    plt.show()
# -

# How the focus evolves layer by layer, for the molecule the model got right:

smi = X_test[correct_idx]
ids, mask = tokenizer.encode_batch([smi], add_special_tokens=True, max_length=MAX_LEN)
toks = ["[CLS]"] + tokenizer.tokenize(smi) + ["[SEP]"]
with torch.no_grad():
    _, attns = model(torch.tensor(ids).to(DEVICE),
                     torch.tensor(mask).to(DEVICE), return_attention=True)
per_layer = [w[0].mean(0).cpu()[: len(toks), : len(toks)] for w in attns]
animate_attention_over_layers(per_layer, toks, static=True)
plt.show()

# 🔬 **Try This.** Sort the test set by `|prob − 0.5|` to find the molecules the
# model is most and least confident about. Do the confident-correct ones
# concentrate `[CLS]` attention on chemically sensible atoms (heteroatoms, polar
# groups, ring systems), while the wrong ones look diffuse or misplaced?

# 🧪 **Chemical Intuition.** With only ~1,500 training molecules and two layers,
# this model is tiny — don't expect textbook-clean attention. The point is that
# the *machinery* now carries learned signal: the same code that drew random
# heatmaps in notebooks 04–06 is doing real chemistry here.

# ---
# ## 9. The same thing, packaged
#
# `TransformerEncoder` (in `utils/transformer_blocks.py`) is exactly the
# hand-rolled body from §4 — embedding + positional encoding + a loop over
# `EncoderBlock`s. We copy our trained weights into it and confirm the output is
# identical to the last decimal, then re-run the packaged `train_one_epoch` /
# `evaluate` to confirm they reproduce our loop.

# +
packaged_encoder = TransformerEncoder(
    tokenizer.vocab_size, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
    d_ff=D_FF, max_len=MAX_LEN, dropout=DROPOUT).to(DEVICE)
packaged_model = MoleculeClassifier(packaged_encoder, model.head).to(DEVICE)

# The hand-rolled encoder and TransformerEncoder share submodule names
# (embed / pos / blocks), so the trained weights transfer directly.
packaged_encoder.load_state_dict(model.encoder.state_dict())

model.eval(); packaged_model.eval()
with torch.no_grad():
    a = model(ids_b, mask_b)
    b = packaged_model(ids_b, mask_b)
print(f"packaged TransformerEncoder matches hand-rolled? "
      f"{torch.allclose(a, b, atol=1e-5)}")

# And the packaged training utilities reproduce our run. train_one_epoch /
# evaluate accept (input_ids, attention_mask, labels) batches directly.
metrics = evaluate(packaged_model, test_loader, device=DEVICE, task="classification")
print(f"utils.evaluate on test: acc {metrics['accuracy']:.3f}, "
      f"auc {metrics['roc_auc']:.3f}  (matches §7)")
one_epoch_loss = train_one_epoch(packaged_model, train_loader, optimizer, device=DEVICE)
print(f"utils.train_one_epoch ran one epoch, mean loss {one_epoch_loss:.3f}")
# -

# 💡 **Key Insight.** The encoder is **task-agnostic**. Swap the head and the
# same body does regression (predict a continuous property), masked language
# modelling (notebook 08), or fine-tuning a pretrained model (notebook 09).
# That reusability is the whole reason we kept pooling out of the encoder.

# ---
# ## Checkpoint exercises

# +
# Exercise 1 — a regression head
# ------------------------------
# The encoder doesn't care what it predicts. Swap the classification head for a
# 1-unit linear with NO sigmoid, use nn.MSELoss, and train on ESOL (aqueous
# solubility, a continuous target): smi, y = load_moleculenet("esol"). Report
# RMSE on a held-out split.

# YOUR CODE HERE

# --- Solution (try the exercise first, then peek) ---
# smi, ytgt = load_moleculenet("esol")
# # ... clean + split + tokenizer exactly as §2-3, labels are floats ...
# class RegressionHead(nn.Module):
#     def __init__(self, d_model=D_MODEL):
#         super().__init__(); self.proj = nn.Linear(d_model, 1)
#     def forward(self, seq): return self.proj(seq[:, 0, :]).squeeze(-1)
# # train with nn.MSELoss(); RMSE = sqrt(mean((pred - true)**2)).
# # The encoder code is unchanged — only the head and loss differ.

# +
# Exercise 2 — [CLS] pooling vs mean pooling
# ------------------------------------------
# Replace the [CLS] read (sequence_output[:, 0, :]) with a MASK-AWARE mean over
# real tokens, and compare validation ROC-AUC. Why must the mean ignore padding?

# YOUR CODE HERE

# --- Solution ---
# def masked_mean(seq, mask):
#     m = mask.unsqueeze(-1).float()                  # (B, L, 1)
#     return (seq * m).sum(1) / m.sum(1).clamp(min=1) # (B, d_model)
# # Build a head that pools with masked_mean(seq, attention_mask) instead of
# # seq[:, 0]. Train both and compare val AUC. Padding must be excluded or the
# # [PAD] vectors would dilute the molecule representation.

# +
# Exercise 3 — overfit on purpose
# -------------------------------
# Shrink the training set to 50 molecules, set dropout=0.0, and train for 100
# epochs with no early stopping. Plot train vs val loss and mark the epoch where
# val loss starts rising — the textbook overfitting point.

# YOUR CODE HERE

# --- Solution ---
# X_small, y_small = X_train[:50], y_train[:50]
# # rebuild train_loader on (X_small, y_small), rebuild model with DROPOUT=0.0,
# # run 100 epochs WITHOUT early stopping, store train/val loss, then:
# #   best = int(np.argmin(history["val_loss"])) + 1
# # train loss → ~0 while val loss bottoms out at `best` then climbs.
# -

# ---
# ## What's next
#
# We trained an encoder from scratch and it learned real signal — but it needed
# **labels**, and labelled chemistry data is scarce. **Notebook 08** keeps this
# exact encoder body and swaps the supervised head for a **masked-language-
# modelling** head, learning chemistry from *unlabelled* SMILES (the way BERT
# and MolFormer do). **Notebook 09** then combines the two: pre-train with MLM,
# fine-tune on a task like BBBP, and show the pretrained model beats training
# from scratch.
#
# 📚 **Deep-dive sub-series**
# - **02.1**: How tokenizer choice changes downstream property prediction.
# - **09.1**: GNN vs encoder-transformer head-to-head on MoleculeNet (with the
#   harder scaffold split).
#
# 📚 **References.**
# - Vaswani, A. et al. (2017). *Attention Is All You Need.* — the encoder stack.
# - Devlin, J. et al. (2019). *BERT.* — the `[CLS]` pooling convention.
# - Wu, Z. et al. (2018). *MoleculeNet: A Benchmark for Molecular Machine
#   Learning.* — the BBBP task and the scaffold-split caveat.
# - Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization.*
#   — AdamW.
# - Ross, J. et al. (2022). *Large-Scale Chemical Language Representations
#   (MolFormer).* — the model this course builds toward.
