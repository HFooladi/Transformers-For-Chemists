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

# <a href="https://colab.research.google.com/github/HFooladi/Transformers-For-Chemists/blob/main/notebooks/08_Masked_Language_Modeling.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 08 · Masked Language Modeling
#
# Notebook 07 trained an encoder — but it needed **labels**, and labelled
# chemistry data is scarce and expensive. *Unlabelled* SMILES, on the other
# hand, are essentially infinite: ChEMBL, ZINC, PubChem hold hundreds of
# millions of molecules with no property attached.
#
# **Masked language modelling (MLM)** is how BERT and MolFormer turn that
# unlabelled flood into a teacher. The recipe is a fill-in-the-blank game: hide
# ~15% of the atoms in a SMILES string and train the model to predict what was
# hidden, using only the surrounding context. No labels required. To win the
# game the model has to internalise chemistry — valence, aromaticity, which
# atoms follow which — and those learned representations are exactly what we
# fine-tune later.
#
# We reuse the **same `TransformerEncoder` body from notebook 07**, swap the
# classification head for a per-position MLM head, and pre-train on a small
# ChEMBL subset. This notebook stays focused on the MLM objective itself;
# notebook 09 combines pre-training and fine-tuning into a tiny MolFormer.

# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# 1. Explain self-supervised MLM and why it fits chemistry (cheap unlabelled SMILES).
# 2. Build the BERT **80/10/10** masking scheme and visualise masked atoms.
# 3. Implement an `MLMCollator` that returns `(input_ids, labels)` with
#    `labels = -100` on every position we are *not* predicting.
# 4. Attach an **MLM prediction head** (`d_model → vocab_size`) and understand
#    weight tying.
# 5. Compute **masked cross-entropy** with `ignore_index=-100` and see why only
#    masked positions contribute.
# 6. Train the encoder + MLM head and read the loss + masked-accuracy curves.
# 7. Run a qualitative **"fill in the masked atom"** demo with top-k predictions,
#    then package `MLMCollator` and verify it.

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

ensure_environment(["torch", "rdkit", "matplotlib", "tokenizers", "pandas"])

# +
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from utils.smiles_tokenizers import AtomTokenizer, CLS_ID, MASK_ID, PAD_ID, SEP_ID, SPECIAL_TOKENS
from utils.transformer_blocks import TransformerEncoder
from utils.training_utils import MLMCollator, MLMMaskingConfig, evaluate
from utils.data_loading import load_chembl_subset

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
print(f"device: {DEVICE}")
# -

# ---
# ## 1. Why mask atoms? Self-supervision for molecules
#
# In supervised learning (notebook 07) every molecule comes with an answer key.
# In **self-supervised** learning the data *is* its own answer key: we hide part
# of the input and ask the model to reconstruct it. Nothing but raw SMILES is
# needed. Our corpus is a small, committed subset of **ChEMBL** — real drug-like
# molecules, canonicalised, no labels.

# +
corpus = load_chembl_subset(n=5000)
print(f"corpus: {len(corpus)} ChEMBL SMILES (unlabelled)")
print("examples:")
for s in corpus[:4]:
    print("  ", s)

tokenizer = AtomTokenizer.from_smiles(corpus)
print(f"\natom-level vocabulary: {tokenizer.vocab_size} tokens")

lengths = [len(tokenizer.tokenize(s)) for s in corpus]
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(lengths, bins=40, color="#4c72b0", edgecolor="white")
ax.set_xlabel("number of atom tokens"); ax.set_ylabel("molecules")
ax.set_title("ChEMBL subset — token-length distribution")
ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
print(f"median length {int(np.median(lengths))} tokens, max {max(lengths)}")
# -

# 🧪 **Chemical Intuition.** Predicting a hidden atom from its neighbours is not
# a trivial trick. To guess that the masked atom between a carbonyl `C(=O)` and a
# ring `N` is most likely another carbon (an amide), the model must learn
# valence rules, aromaticity, and common functional-group grammar. Those are
# precisely the chemistry regularities we want baked into the embeddings before
# we ever show the model a labelled task.

# ---
# ## 2. Masking, made visible
#
# Let's see the game on caffeine. We tokenize it, add `[CLS]`/`[SEP]`, choose a
# few atom positions, and replace them with `[MASK]`. The model will see the
# corrupted row; the original atoms (shown in red below) are the answer key.

# +
caffeine_tokens = ["[CLS]"] + tokenizer.tokenize(CAFFEINE) + ["[SEP]"]


def show_masking(original, corrupted, masked_positions, title=None):
    """Draw the token strip: corrupted token in each cell, with the original
    atom in red beneath every masked position."""
    n = len(original)
    fig, ax = plt.subplots(figsize=(min(0.45 * n + 1, 14), 2.0))
    for i, (orig, corr) in enumerate(zip(original, corrupted)):
        masked = i in masked_positions
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor="#ffd24c" if masked else "#eeeeee",
                                   edgecolor="white", lw=1.5))
        ax.text(i + 0.5, 0.5, corr, ha="center", va="center", fontsize=9,
                fontweight="bold" if masked else "normal")
        if masked:
            ax.text(i + 0.5, -0.45, orig, ha="center", va="center", fontsize=8, color="#c44e52")
    ax.text(-0.3, -0.45, "truth:", ha="right", va="center", fontsize=8, color="#c44e52")
    ax.set_xlim(-2, n); ax.set_ylim(-0.9, 1.1); ax.axis("off")
    if title:
        ax.set_title(title)
    plt.tight_layout(); plt.show()


# Mask three positions by hand for the illustration: a ring N, a ring C, and
# the carbonyl O.
mask_positions = [6, 12, 15]
corrupted_tokens = list(caffeine_tokens)
for p in mask_positions:
    corrupted_tokens[p] = "[MASK]"
show_masking(caffeine_tokens, corrupted_tokens, mask_positions,
             title="MLM on caffeine — yellow = hidden, red = the answer")
# -

# ⚠️ **Note.** We never mask the structural special tokens `[CLS]`, `[SEP]`, or
# `[PAD]`. Masking them would be pointless (they carry no chemistry) and would
# corrupt the `[CLS]` summary slot. The eligible set is always the *real,
# non-special* atom positions.

# ---
# ## 3. The 80/10/10 rule
#
# BERT's masking is subtler than "replace 15% with `[MASK]`". Of the ~15% of
# tokens selected for prediction:
#
# - **80%** become `[MASK]`,
# - **10%** are replaced by a **random** vocabulary token,
# - **10%** are **left unchanged**.
#
# `MLMMaskingConfig` holds these fractions.

# +
config = MLMMaskingConfig()
print(config)

# Visualise the split for a 100-token molecule.
n_sel = 100 * config.mask_probability
parts = [n_sel * config.mask_token_fraction,
         n_sel * config.random_token_fraction,
         n_sel * config.keep_token_fraction]
fig, ax = plt.subplots(figsize=(7.5, 1.8))
left = 0
for size, label, color in zip(
        parts, ["[MASK] (80%)", "random (10%)", "keep (10%)"],
        ["#4c72b0", "#dd8452", "#55a868"]):
    ax.barh(0, size, left=left, color=color, edgecolor="white")
    if size > 0.4:
        ax.text(left + size / 2, 0, label, ha="center", va="center",
                color="white", fontsize=9)
    left += size
ax.set_xlim(0, n_sel); ax.set_yticks([])
ax.set_xlabel("masked tokens (of ~15 selected per 100)")
ax.set_title("BERT 80 / 10 / 10 split"); plt.tight_layout(); plt.show()
# -

# 💡 **Key Insight.** Why not 100% `[MASK]`? Because at fine-tune and inference
# time there is **no `[MASK]` token** — real molecules have no blanks. If the
# model only ever saw `[MASK]` it would learn to do nothing useful for
# unmasked positions. The 10%-random and 10%-keep positions force it to build a
# robust contextual representation of *every* token, masked or not, so the
# learned features transfer. (The masking-ratio deep-dive **08.1** studies how
# much to mask.)

# ---
# ## 4. The `MLMCollator` — by hand, then packaged
#
# A *collator* turns a list of variable-length token sequences into a single
# padded, corrupted batch. We build it by hand first so every step is visible.
# The key output is `labels`: the original token id at each selected position,
# and `-100` everywhere else (PyTorch's `cross_entropy` skips `-100`).

# +
SPECIAL_IDS = tuple(range(len(SPECIAL_TOKENS)))   # all special-token ids


def mlm_collate_by_hand(batch, vocab_size, config=config, generator=None):
    """batch: list of token-id lists. Returns (input_ids, attention_mask, labels)."""
    target_len = max(len(seq) for seq in batch)
    input_ids = torch.full((len(batch), target_len), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), target_len), dtype=torch.long)
    for i, seq in enumerate(batch):
        input_ids[i, : len(seq)] = torch.tensor(seq)
        attention_mask[i, : len(seq)] = 1
    original = input_ids.clone()

    # 1. eligible = real, non-special positions
    eligible = attention_mask.bool()
    for sid in SPECIAL_IDS:
        eligible &= input_ids != sid
    # 2. select ~15% of eligible positions
    selected = (torch.rand(input_ids.shape, generator=generator) < config.mask_probability) & eligible
    # 3. split selected 80 / 10 / 10
    roll = torch.rand(input_ids.shape, generator=generator)
    to_mask = selected & (roll < config.mask_token_fraction)
    to_random = selected & (roll >= config.mask_token_fraction) & \
        (roll < config.mask_token_fraction + config.random_token_fraction)
    input_ids[to_mask] = MASK_ID
    n_rand = int(to_random.sum())
    if n_rand:
        input_ids[to_random] = torch.randint(len(SPECIAL_IDS), vocab_size,
                                              (n_rand,), generator=generator)
    # 4. labels: original id at selected positions, -100 elsewhere
    labels = torch.full_like(input_ids, -100)
    labels[selected] = original[selected]
    return input_ids, attention_mask, labels


# Inspect one collated batch.
demo = [tokenizer.encode(s, add_special_tokens=True) for s in corpus[:4]]
g = torch.Generator().manual_seed(0)
inp, att, lab = mlm_collate_by_hand(demo, tokenizer.vocab_size, generator=g)
n_masked = int((lab != -100).sum())
n_real = int(att.sum())
print(f"batch shape {tuple(inp.shape)}")
print(f"masked {n_masked} of {n_real} real tokens "
      f"({100 * n_masked / n_real:.1f}% — target ~15%)")
print("labels are -100 everywhere except the selected positions:")
print(lab[0].tolist())
# -

# ⚠️ **Note.** `labels` stores the original id at *all* selected positions —
# including the 10%-random and 10%-kept ones. That's the BERT convention: the
# model is graded on those too, which is what makes the keep/random tricks work.

# ---
# ## 5. The MLM prediction head
#
# The head maps every position's `d_model` vector to a distribution over the
# vocabulary — a plain `Linear(d_model, vocab_size)`. Run it on the whole
# sequence and you get one prediction per token.

# +
D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_LEN = 64, 4, 2, 256, 128


class TinyMLM(nn.Module):
    """TransformerEncoder body + a per-position MLM head."""

    def __init__(self, vocab_size, tie_weights=False):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
            d_ff=D_FF, max_len=MAX_LEN, dropout=0.1)
        self.head = nn.Linear(D_MODEL, vocab_size)
        if tie_weights:
            # Share the embedding matrix with the output projection.
            self.head.weight = self.encoder.embed.embedding.weight

    def forward(self, input_ids, attention_mask=None):
        seq, _ = self.encoder(input_ids, attention_mask)   # (B, L, d_model)
        return self.head(seq)                              # (B, L, vocab_size)


model = TinyMLM(tokenizer.vocab_size).to(DEVICE)
logits = model(inp.to(DEVICE), att.to(DEVICE))
print(f"MLM logits shape: {tuple(logits.shape)}   # (B, L, vocab_size)")
# -

# ⚠️ **Note — weight tying.** A common trick (BERT, MolFormer) is to **tie** the
# head's weight to the embedding matrix (`head.weight = embed.embedding.weight`),
# saving parameters and coupling the input/output representations. We keep them
# **untied** here for a clean, stable loss curve: `TokenEmbedding` scales its
# output by `√d_model`, so naive tying produces very large logits early in
# training. Tying done right needs care with that scaling — Exercise 2 explores
# the comparison.

# ---
# ## 6. Masked cross-entropy
#
# The loss flattens the `(B, L, vocab)` logits to `(B·L, vocab)` and the labels
# to `(B·L,)`, then calls `cross_entropy` with `ignore_index=-100`. Every
# non-selected position is `-100`, so it contributes **nothing** to the loss.

# +
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    lab.to(DEVICE).view(-1),
    ignore_index=-100,
)
print(f"masked cross-entropy on the demo batch: {loss.item():.3f}")
print(f"(random-guess baseline ≈ ln(vocab) = {np.log(tokenizer.vocab_size):.3f})")

# Proof that unmasked positions don't matter: scramble the logits everywhere
# except the masked positions — the loss is unchanged.
masked = lab.to(DEVICE).view(-1) != -100
scrambled = logits.view(-1, logits.size(-1)).clone()
scrambled[~masked] = torch.randn_like(scrambled[~masked]) * 100
loss2 = F.cross_entropy(scrambled, lab.to(DEVICE).view(-1), ignore_index=-100)
print(f"loss after scrambling only the UNMASKED logits: {loss2.item():.3f}  (identical)")
# -

# 💡 **Key Insight.** The model is graded *purely* on its fill-in-the-blank
# skill. The bulk of the sequence — the visible context — is the model's input,
# not its target. That asymmetry is what makes one molecule yield many cheap
# training signals.

# ---
# ## 7. Pre-training the model
#
# Now the real run: a `DataLoader` whose `collate_fn` is the packaged
# `MLMCollator` (it re-masks every batch, so the model sees fresh blanks each
# epoch), the `TinyMLM` model, AdamW, and a transparent loop that records loss
# and **masked accuracy** (fraction of masked atoms predicted correctly).

# +
class SmilesCorpus(Dataset):
    def __init__(self, smiles, tokenizer, max_len=MAX_LEN):
        self.encoded = [tokenizer.encode(s, add_special_tokens=True)[:max_len] for s in smiles]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx]


collator = MLMCollator(tokenizer.vocab_size, MASK_ID, PAD_ID, config=config)
loader = DataLoader(SmilesCorpus(corpus, tokenizer), batch_size=64,
                    shuffle=True, collate_fn=collator)

torch.manual_seed(0)
model = TinyMLM(tokenizer.vocab_size).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

EPOCHS = 8
loss_hist, acc_hist = [], []
for epoch in range(1, EPOCHS + 1):
    model.train()
    ep_loss, ep_correct, ep_masked = 0.0, 0, 0
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        logits = model(ids, mask)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               labels.view(-1), ignore_index=-100)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        ep_loss += loss.item()
        sel = labels != -100
        ep_correct += int((logits.argmax(-1)[sel] == labels[sel]).sum())
        ep_masked += int(sel.sum())
    loss_hist.append(ep_loss / len(loader))
    acc_hist.append(ep_correct / ep_masked)
    print(f"epoch {epoch:2d} | MLM loss {loss_hist[-1]:.3f} | "
          f"masked accuracy {acc_hist[-1]:.3f}")
# -

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.5))
ep = range(1, EPOCHS + 1)
axL.plot(ep, loss_hist, "-o", color="#4c72b0")
axL.axhline(np.log(tokenizer.vocab_size), ls="--", color="gray", label="random guess")
axL.set_xlabel("epoch"); axL.set_ylabel("masked cross-entropy")
axL.set_title("MLM loss"); axL.legend(); axL.grid(alpha=0.3)
axR.plot(ep, acc_hist, "-o", color="#55a868")
axR.axhline(1 / tokenizer.vocab_size, ls="--", color="gray", label="random guess")
axR.set_xlabel("epoch"); axR.set_ylabel("masked accuracy")
axR.set_title("Fraction of masked atoms recovered"); axR.legend(); axR.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# This is **pure pre-training** — no property labels anywhere. We are *not*
# adding a classification head; that's the job of notebook 09. The payoff here
# is a model that has learned chemical structure, which we inspect next.

# ---
# ## 8. "Fill in the masked atom"
#
# The real test: hide a few chemically interesting atoms in caffeine, run the
# trained model, and read off its **top-k** guesses for each blank. We compare
# against an **untrained** copy to see what training bought us.

# +
def predict_masked(model, tokens, mask_positions, k=5):
    """Mask the given positions, run the model, return top-k (token, prob) per blank."""
    ids = [tokenizer.token_to_id[t] if t in tokenizer.token_to_id else CLS_ID for t in tokens]
    ids = torch.tensor(ids).unsqueeze(0)
    corrupted = ids.clone()
    for p in mask_positions:
        corrupted[0, p] = MASK_ID
    attn = torch.ones_like(ids)
    model.eval()
    with torch.no_grad():
        logits = model(corrupted.to(DEVICE), attn.to(DEVICE))
    probs = F.softmax(logits[0], dim=-1)
    out = {}
    for p in mask_positions:
        top = torch.topk(probs[p], k)
        out[p] = [(tokenizer.id_to_token[i.item()], v.item())
                  for v, i in zip(top.values, top.indices)]
    return out


mask_positions = [6, 12, 15]   # ring N, ring C, carbonyl O
untrained = TinyMLM(tokenizer.vocab_size).to(DEVICE)

print("position | truth | trained top-5                      | untrained top-5")
trained_pred = predict_masked(model, caffeine_tokens, mask_positions)
untrained_pred = predict_masked(untrained, caffeine_tokens, mask_positions)
for p in mask_positions:
    truth = caffeine_tokens[p]
    t = " ".join(f"{tok}:{pr:.2f}" for tok, pr in trained_pred[p][:3])
    u = " ".join(f"{tok}:{pr:.2f}" for tok, pr in untrained_pred[p][:3])
    print(f"  {p:>3}    |  {truth:>4} | {t:<34} | {u}")
# -

# Bar chart of the trained model's distribution at the first masked position.
p = mask_positions[0]
toks = [t for t, _ in trained_pred[p]]
vals = [v for _, v in trained_pred[p]]
fig, ax = plt.subplots(figsize=(6.5, 3.6))
colors = ["#55a868" if t == caffeine_tokens[p] else "#4c72b0" for t in toks]
ax.bar(toks, vals, color=colors, edgecolor="white")
ax.set_ylabel("probability")
ax.set_title(f"Trained model — top-5 for masked position {p} "
             f"(truth = '{caffeine_tokens[p]}', green if correct)")
plt.tight_layout(); plt.show()

# 🧪 **Chemical Intuition.** Before training the distribution is essentially
# flat — the untrained model has no idea. After a few epochs the true atom (or a
# chemically plausible substitute, e.g. predicting `O` where an `N` sat in a
# similar environment) rises to the top. The model has learned local chemical
# grammar purely from filling in blanks.

# 🔬 **Try This.** Mask an aromatic atom (lowercase `c`) and an aliphatic one
# (uppercase `C`) and compare how confident the model is. Aromatic positions are
# strongly constrained by the ring context — is the model more sure about them?

# ---
# ## 9. The same masking, packaged
#
# The `MLMCollator` we trained with lives in `utils/training_utils.py`. Given
# the same seed it reproduces our hand-rolled masking exactly — same selected
# positions, same labels.

# +
seq_batch = [tokenizer.encode(s, add_special_tokens=True) for s in corpus[:4]]

g1 = torch.Generator().manual_seed(123)
hand_inp, hand_att, hand_lab = mlm_collate_by_hand(seq_batch, tokenizer.vocab_size, generator=g1)

packaged = MLMCollator(tokenizer.vocab_size, MASK_ID, PAD_ID, config=config, seed=123)
pk = packaged(seq_batch)

print(f"input_ids match? {torch.equal(hand_inp, pk['input_ids'])}")
print(f"labels match?    {torch.equal(hand_lab, pk['labels'])}")

# utils.evaluate also reports masked accuracy directly on an MLM loader.
metrics = evaluate(model, loader, device=DEVICE, task="mlm")
print(f"utils.evaluate (mlm): loss {metrics['loss']:.3f}, "
      f"masked_accuracy {metrics['masked_accuracy']:.3f}")
# -

# ---
# ## Checkpoint exercises

# +
# Exercise 1 — masking-ratio sweep
# --------------------------------
# Re-run a short pre-training (e.g. 3 epochs) with mask_probability in
# {0.10, 0.15, 0.30} by passing a custom MLMMaskingConfig to MLMCollator. Plot
# final masked accuracy vs ratio. Too little → few signals per molecule; too
# much → not enough context to predict from. (This previews deep-dive 08.1.)

# YOUR CODE HERE

# --- Solution (try the exercise first, then peek) ---
# for p in (0.10, 0.15, 0.30):
#     cfg = MLMMaskingConfig(mask_probability=p)
#     coll = MLMCollator(tokenizer.vocab_size, MASK_ID, PAD_ID, config=cfg)
#     ld = DataLoader(SmilesCorpus(corpus, tokenizer), batch_size=64, shuffle=True, collate_fn=coll)
#     # ... train a fresh TinyMLM for 3 epochs, record final masked accuracy ...

# +
# Exercise 2 — tied vs untied head
# --------------------------------
# Train TinyMLM(vocab, tie_weights=True) and TinyMLM(vocab, tie_weights=False)
# for equal steps. Compare parameter counts (sum(p.numel() for p in m.parameters()))
# and final loss. The tied model has fewer params; does it train as cleanly?

# YOUR CODE HERE

# --- Solution ---
# for tie in (False, True):
#     m = TinyMLM(tokenizer.vocab_size, tie_weights=tie)
#     n = sum(p.numel() for p in m.parameters())
#     print(f"tie={tie}: {n:,} params")
# # Tying removes one (vocab × d_model) matrix; watch the loss curve for stability.

# +
# Exercise 3 — which atom type is hardest to predict?
# ---------------------------------------------------
# Over a held-out batch, group masked positions by their true token and compute
# per-token-type accuracy. Which token type does the model recover worst (e.g.
# ring-closure digits vs common 'C')? Why might that be?

# YOUR CODE HERE

# --- Solution ---
# from collections import defaultdict
# correct, total = defaultdict(int), defaultdict(int)
# # run the trained model on a masked batch; for each masked position p:
# #   tok = tokenizer.id_to_token[label]; total[tok]+=1; correct[tok]+= (pred==label)
# # then sort tokens by correct[t]/total[t]. Rare/ambiguous tokens score worst.
# -

# ---
# ## What's next
#
# We pre-trained an encoder with **no labels** and watched it learn to fill in
# masked atoms. **Notebook 09 (Tiny MolFormer)** finally joins the two halves of
# the course: pre-train this MLM encoder on ChEMBL, then **fine-tune** it on a
# labelled task like BBBP (notebook 07) — and show the pretrained model beats
# training from scratch. That pretrain → fine-tune recipe is the whole point of
# a chemical foundation model.
#
# 📚 **Deep-dive sub-series**
# - **08.1**: An empirical sweep of MLM masking ratios — how much should you
#   hide, and why 15% became the default.
#
# 📚 **References.**
# - Devlin, J. et al. (2019). *BERT.* — masked language modelling and the
#   80/10/10 corruption scheme.
# - Taylor, W. L. (1953). *"Cloze procedure": a new tool for measuring
#   readability.* — the fill-in-the-blank task MLM descends from.
# - Press, O. & Wolf, L. (2017); Inan, H. et al. (2017). — output/input embedding
#   weight tying.
# - Ross, J. et al. (2022). *MolFormer.* — the MLM-pretrained chemical model this
#   course builds toward.
