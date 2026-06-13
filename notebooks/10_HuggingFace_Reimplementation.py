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

# <a href="https://colab.research.google.com/github/HFooladi/Transformers-For-Chemists/blob/main/notebooks/10_HuggingFace_Reimplementation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 10 · HuggingFace Reimplementation
#
# You have built every line of a tiny MolFormer — the tokenizer, the encoder,
# the MLM head, the training loop, the transfer recipe. Now meet the production
# stack. The **HuggingFace `transformers`** library packages exactly these
# pieces, battle-tested and optimised, behind a handful of classes.
#
# This notebook maps each from-scratch component to its HuggingFace equivalent,
# rebuilds the tiny MolFormer with `BertForMaskedLM` and
# `BertForSequenceClassification`, checks that it learns like ours, and points
# you to *real* pretrained chemical models on the Hub. Nothing here is new
# conceptually — it's the same model you already understand, written the way you
# would write it in practice.

# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# 1. Map every from-scratch component to its HuggingFace class.
# 2. Build a HF tokenizer **from our own atom vocabulary**, so the comparison is
#    apples-to-apples.
# 3. Configure a tiny BERT with `BertConfig` to match our dimensions.
# 4. Pre-train with `BertForMaskedLM` + `DataCollatorForLanguageModeling`.
# 5. Confirm the HF model learns like the hand-built one (matching loss curves).
# 6. Fine-tune with `BertForSequenceClassification` and transplant the
#    pretrained body.
# 7. Know how to load a *real* pretrained chemical model (ChemBERTa / MoLFormer)
#    from the Hub.

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

ensure_environment(["torch", "rdkit", "matplotlib", "tokenizers", "transformers", "sklearn", "pandas"])

# +
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
)

from utils.smiles_tokenizers import AtomTokenizer
from utils.transformer_blocks import TransformerEncoder
from utils.training_utils import MLMCollator, count_parameters
from utils.data_loading import load_chembl_subset, load_moleculenet
from utils.preprocessing import clean_smiles

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
print(f"device: {DEVICE}")
# -

# ---
# ## 1. The map: your class ↔ the HuggingFace class
#
# Everything you built has a direct counterpart:
#
# | From scratch (this course)                       | HuggingFace                          |
# | ------------------------------------------------ | ------------------------------------ |
# | `AtomTokenizer`                                  | `PreTrainedTokenizerFast` (WordLevel)|
# | `TokenEmbedding` + `SinusoidalPositionalEncoding`| `BertEmbeddings` (learned positions) |
# | `EncoderBlock` (pre-norm MHA + FFN)              | `BertLayer` (post-norm)              |
# | `TransformerEncoder`                             | `BertModel`                          |
# | `TinyMLM` (encoder + Linear head)                | `BertForMaskedLM`                    |
# | `MoleculeClassifier` (encoder + `[CLS]` head)    | `BertForSequenceClassification`      |
# | `MLMCollator`                                    | `DataCollatorForLanguageModeling`    |
# | hand-written `train_one_epoch`                   | the same loop (or `Trainer`)         |
#
# ⚠️ **Note.** HF's BERT differs from our encoder in two details: it uses
# **learned** positional embeddings (ours are sinusoidal) and **post-norm**
# blocks (ours are pre-norm, see notebook 06.1). So the weights are *not*
# transferable between the two — the point here is **conceptual equivalence**,
# not a bit-for-bit copy.

# ---
# ## 2. Build a HF tokenizer from OUR vocabulary
#
# To compare fairly, the HF model should see the *same tokens* as ours. We wrap
# our `AtomTokenizer`'s vocabulary in a HuggingFace `WordLevel` tokenizer. The
# trick: pre-tokenize by **whitespace** and feed it our atoms already
# space-joined (`" ".join(atom_tokenizer.tokenize(smiles))`), which reproduces
# our atom segmentation exactly.

# +
corpus = load_chembl_subset(n=3000)
atom_tokenizer = AtomTokenizer.from_smiles(corpus)

hf_core = Tokenizer(WordLevel(vocab=atom_tokenizer.token_to_id, unk_token="[UNK]"))
hf_core.pre_tokenizer = WhitespaceSplit()
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=hf_core,
    pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
    sep_token="[SEP]", mask_token="[MASK]", bos_token="[BOS]", eos_token="[EOS]")

# Our pinned special-token ids must survive the round-trip into HF.
print(f"pad={hf_tokenizer.pad_token_id}  unk={hf_tokenizer.unk_token_id}  "
      f"cls={hf_tokenizer.cls_token_id}  sep={hf_tokenizer.sep_token_id}  "
      f"mask={hf_tokenizer.mask_token_id}")
assert (hf_tokenizer.pad_token_id, hf_tokenizer.mask_token_id) == (0, 4)


def hf_encode(smiles, add_special_tokens=True):
    """Encode one SMILES with the HF tokenizer, matching our [CLS]..[SEP] framing."""
    ids = hf_tokenizer(" ".join(atom_tokenizer.tokenize(smiles)), add_special_tokens=False)["input_ids"]
    if add_special_tokens:
        ids = [hf_tokenizer.cls_token_id] + ids + [hf_tokenizer.sep_token_id]
    return ids


print("HF round-trip == our tokenizer?",
      hf_encode(CAFFEINE) == atom_tokenizer.encode(CAFFEINE, add_special_tokens=True))
# -

# 💡 **Key Insight.** Because we reused our own vocabulary, every later
# comparison is apples-to-apples: the only thing that changes between our model
# and HF's is the *implementation*, not the tokens.

# ---
# ## 3. Configure the model — three lines
#
# `BertConfig` is the entire architecture spec. We set it to our tiny
# dimensions, and `BertForMaskedLM` builds the model.

# +
config = BertConfig(
    vocab_size=atom_tokenizer.vocab_size,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=256,
    max_position_embeddings=128,
    pad_token_id=0,
)
hf_mlm = BertForMaskedLM(config).to(DEVICE)

# Our hand-built TinyMLM, for comparison (from notebooks 08–09).
class TinyMLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model=64, n_heads=4, n_layers=2,
                                          d_ff=256, max_len=128, dropout=0.1)
        self.head = nn.Linear(64, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        seq, _ = self.encoder(input_ids, attention_mask)
        return self.head(seq)


our_mlm = TinyMLM(atom_tokenizer.vocab_size).to(DEVICE)
print(f"HF  BertForMaskedLM : {count_parameters(hf_mlm):,} params")
print(f"our TinyMLM         : {count_parameters(our_mlm):,} params")
# -

# 💡 **Key Insight.** Notebooks 03–08 spent hundreds of lines building what is
# now three lines of `BertConfig`. HF is a little larger mostly because it adds
# **learned** positional embeddings and a couple of extra LayerNorms — but it's
# the same architecture family.

# ---
# ## 4. Shape sanity check
#
# The contract is identical to our `TinyMLM`: feed token ids, get
# `(batch, seq_len, vocab_size)` logits.

demo_ids = torch.tensor([hf_encode(CAFFEINE)]).to(DEVICE)
demo_mask = torch.ones_like(demo_ids)
out = hf_mlm(input_ids=demo_ids, attention_mask=demo_mask)
print(f"HF logits shape:  {tuple(out.logits.shape)}   # (B, L, vocab_size)")
print(f"our logits shape: {tuple(our_mlm(demo_ids, demo_mask).shape)}")

# ---
# ## 5. Pre-train with HuggingFace
#
# `DataCollatorForLanguageModeling` is HF's `MLMCollator`: it masks ~15% of
# tokens with the same 80/10/10 scheme and builds `labels` with `-100` on
# unmasked positions. We feed it our encoded SMILES and train with a transparent
# loop — `outputs.loss` *is* the masked cross-entropy you computed by hand in
# notebook 08.

# +
class EncodedCorpus(Dataset):
    def __init__(self, smiles):
        self.examples = [{"input_ids": hf_encode(s)[:128]} for s in smiles]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


hf_collator = DataCollatorForLanguageModeling(tokenizer=hf_tokenizer, mlm=True, mlm_probability=0.15)

# One collated batch: same shape and masking semantics as our MLMCollator.
peek = hf_collator([{"input_ids": hf_encode(s)} for s in corpus[:4]])
n_masked = int((peek["labels"] != -100).sum())
print(f"HF collator → keys {list(peek.keys())}, masked {n_masked} tokens (~15% of reals)")

EPOCHS = 4
hf_loader = DataLoader(EncodedCorpus(corpus), batch_size=64, shuffle=True, collate_fn=hf_collator)
hf_opt = torch.optim.AdamW(hf_mlm.parameters(), lr=1e-3)
hf_losses = []
for epoch in range(1, EPOCHS + 1):
    hf_mlm.train()
    total = 0.0
    for batch in hf_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = hf_mlm(**batch)            # HF computes the masked-LM loss for us
        hf_opt.zero_grad(); out.loss.backward(); hf_opt.step()
        total += out.loss.item()
    hf_losses.append(total / len(hf_loader))
    print(f"HF pre-train epoch {epoch} | MLM loss {hf_losses[-1]:.3f}")
# -

# ---
# ## 6. Does it learn like ours?
#
# Train our hand-built `TinyMLM` on the same corpus with our `MLMCollator`, and
# overlay the two loss curves. They won't be identical (different positional
# encodings, norm placement, and init), but both descend from ≈ ln(vocab) to a
# similar place — the objective, not the implementation detail, drives learning.

# +
from utils.smiles_tokenizers import MASK_ID, PAD_ID


class IdCorpus(Dataset):
    def __init__(self, smiles):
        self.encoded = [atom_tokenizer.encode(s, add_special_tokens=True)[:128] for s in smiles]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx]


from utils.training_utils import train_one_epoch

torch.manual_seed(0)
our_mlm = TinyMLM(atom_tokenizer.vocab_size).to(DEVICE)
our_loader = DataLoader(IdCorpus(corpus), batch_size=64, shuffle=True,
                        collate_fn=MLMCollator(atom_tokenizer.vocab_size, MASK_ID, PAD_ID))
our_opt = torch.optim.AdamW(our_mlm.parameters(), lr=1e-3)
our_losses = [train_one_epoch(our_mlm, our_loader, our_opt, device=DEVICE) for _ in range(EPOCHS)]

fig, ax = plt.subplots(figsize=(7, 4.5))
ep = range(1, EPOCHS + 1)
ax.plot(ep, hf_losses, "-o", color="#dd8452", label="HuggingFace BertForMaskedLM")
ax.plot(ep, our_losses, "-o", color="#4c72b0", label="our TinyMLM")
ax.axhline(np.log(atom_tokenizer.vocab_size), ls="--", color="gray", label="random guess")
ax.set_xlabel("epoch"); ax.set_ylabel("MLM loss"); ax.set_title("Same objective, two implementations")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
print(f"final loss — HF {hf_losses[-1]:.3f}  |  ours {our_losses[-1]:.3f}")
# -

# 💡 **Key Insight.** Two independent implementations, the same masked-LM
# objective, the same ballpark result. Once you understand the objective, the
# framework is just packaging.

# ---
# ## 7. Fine-tune with `BertForSequenceClassification`
#
# The classification analogue of notebook 09: build a sequence classifier,
# **transplant the pretrained body** from the MLM model, and fine-tune on a
# small BBBP set. The ids come from the same vocabulary, so we reuse our
# tokenizer to batch them.

# +
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

smiles_raw, labels_raw = load_moleculenet("bbbp")
pairs = [(clean_smiles(s), int(y)) for s, y in zip(smiles_raw, labels_raw)]
pairs = [(s, y) for s, y in pairs if s is not None]
Xtr, Xte, ytr, yte = train_test_split([s for s, _ in pairs], [y for _, y in pairs],
                                       test_size=0.2, stratify=[y for _, y in pairs], random_state=0)
Xtr, ytr = train_test_split(Xtr, ytr, train_size=400, stratify=ytr, random_state=0)[0::2]

clf = BertForSequenceClassification(BertConfig(
    vocab_size=atom_tokenizer.vocab_size, hidden_size=64, num_hidden_layers=2,
    num_attention_heads=4, intermediate_size=256, max_position_embeddings=128,
    pad_token_id=0, num_labels=2)).to(DEVICE)
# Transplant the pretrained encoder body. strict=False because the classifier
# adds a pooler that the MLM model doesn't have (left at its fresh init).
missing = clf.bert.load_state_dict(hf_mlm.bert.state_dict(), strict=False)
print(f"transplanted pretrained body (fresh: {missing.missing_keys})")


def clf_batch(smiles, labels):
    ids, mask = atom_tokenizer.encode_batch(smiles, add_special_tokens=True, max_length=128)
    return torch.tensor(ids).to(DEVICE), torch.tensor(mask).to(DEVICE), torch.tensor(labels).to(DEVICE)


opt = torch.optim.AdamW(clf.parameters(), lr=5e-4)
for epoch in range(6):
    clf.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 32):
        idx = perm[i:i + 32].tolist()
        ids, mask, lab = clf_batch([Xtr[j] for j in idx], [ytr[j] for j in idx])
        out = clf(input_ids=ids, attention_mask=mask, labels=lab)
        opt.zero_grad(); out.loss.backward(); opt.step()

clf.eval()
with torch.no_grad():
    ids, mask, _ = clf_batch(Xte, yte)
    probs = clf(input_ids=ids, attention_mask=mask).logits.softmax(-1)[:, 1].cpu().numpy()
print(f"HF fine-tuned BBBP test ROC-AUC: {roc_auc_score(yte, probs):.3f}")
# -

# 💡 **Key Insight.** This is notebook 09's experiment in HuggingFace form:
# `load_state_dict` is the manual version of `from_pretrained`, and a few lines
# of `BertForSequenceClassification` replace the whole hand-built classifier.

# ---
# ## 8. The bridge — real pretrained chemical models
#
# Everything so far used *our* 5,000-molecule pre-training. The Hub hosts models
# pretrained on tens of millions of molecules, behind the identical API — for
# example `DeepChem/ChemBERTa-77M-MLM` (RoBERTa, MLM on ~77M PubChem SMILES) and
# `ibm/MoLFormer-XL-both-10pct` (the full MolFormer). Loading one is a single
# `from_pretrained` call.

# Optional: downloads from the Hub, so it's wrapped for offline safety. Uncomment
# the body to run it where you have internet (e.g. Colab).
try:
    if os.environ.get("RUN_HUB_DEMO") == "1":
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        name = "DeepChem/ChemBERTa-77M-MLM"
        hub_tok = AutoTokenizer.from_pretrained(name)
        hub_model = AutoModelForMaskedLM.from_pretrained(name)
        enc = hub_tok(CAFFEINE, return_tensors="pt")
        with torch.no_grad():
            hidden = hub_model.roberta(**enc).last_hidden_state
        print(f"ChemBERTa caffeine embedding: {tuple(hidden.shape)}")
    else:
        print("Hub demo skipped (set RUN_HUB_DEMO=1 with internet to run it).")
except Exception as e:  # offline / not installed — don't fail the notebook
    print(f"skipping Hub download: {e}")

# ⚠️ **Note.** That `from_pretrained` line is the same API you just used — only
# now the weights come from someone else's 77-million-molecule pre-training run.
# You understand every component inside it.

# ---
# ## Checkpoint exercises

# +
# Exercise 1 — use the HF Trainer
# -------------------------------
# Replace the manual loop in §5 with transformers.Trainer + TrainingArguments
# (data_collator=hf_collator). Confirm it reaches a comparable MLM loss.

# YOUR CODE HERE

# --- Solution (try the exercise first, then peek) ---
# from transformers import Trainer, TrainingArguments
# args = TrainingArguments(output_dir="hf_mlm", per_device_train_batch_size=64,
#                          num_train_epochs=4, learning_rate=1e-3, report_to=[])
# Trainer(model=BertForMaskedLM(config), args=args,
#         train_dataset=EncodedCorpus(corpus), data_collator=hf_collator).train()

# +
# Exercise 2 — where does HF spend its extra parameters?
# ------------------------------------------------------
# Print clf.named_parameters() grouped by top-level module and compare with our
# model. Which submodules (positions, pooler) exist in HF but not in ours?

# YOUR CODE HERE

# --- Solution ---
# from collections import defaultdict
# sizes = defaultdict(int)
# for n, p in hf_mlm.named_parameters():
#     sizes[n.split(".")[1]] += p.numel()
# print(dict(sizes))   # note position_embeddings + extra LayerNorms

# +
# Exercise 3 — DataCollator vs our MLMCollator
# --------------------------------------------
# Seed both collators and compare the fraction of tokens masked and the 80/10/10
# split on the same batch. Confirm they implement the same scheme.

# YOUR CODE HERE

# --- Solution ---
# b1 = hf_collator([{"input_ids": atom_tokenizer.encode(s, add_special_tokens=True)} for s in corpus[:8]])
# b2 = MLMCollator(atom_tokenizer.vocab_size, MASK_ID, PAD_ID, seed=0)(
#         [atom_tokenizer.encode(s, add_special_tokens=True) for s in corpus[:8]])
# print((b1["labels"] != -100).float().mean(), (b2["labels"] != -100).float().mean())
# -

# ---
# ## What's next
#
# That completes the core course. You walked the whole arc — **SMILES → tokens →
# embeddings → attention → multi-head → the transformer block → supervised
# training → masked-language-model pre-training → a tiny MolFormer → the
# production HuggingFace stack** — building every piece from scratch and then
# recognising it in the tools practitioners actually use.
#
# Where to go from here:
#
# 📚 **Deep-dive sub-series**
# - **04.1–04.4**: the linear/Performer attention and position encodings
#   MolFormer actually uses.
# - **05.1 / 06.1**: head specialization; pre-norm vs post-norm.
# - **08.1**: how the MLM masking ratio affects downstream transfer.
# - **09.1**: GNN vs encoder-transformer head-to-head.
#
# And the sister course, **[GNNs-For-Chemists](https://github.com/HFooladi/GNNs-For-Chemists)**,
# for the graph view of the same molecules.
#
# 📚 **References.**
# - Wolf, T. et al. (2020). *Transformers: State-of-the-Art Natural Language
#   Processing.* — the `transformers` library.
# - Devlin, J. et al. (2019). *BERT.*
# - Chithrananda, S. et al. (2020). *ChemBERTa.* — a SMILES RoBERTa on the Hub.
# - Ross, J. et al. (2022). *MolFormer.*
