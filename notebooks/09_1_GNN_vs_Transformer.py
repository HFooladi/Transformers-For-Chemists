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

# <a href="https://colab.research.google.com/github/HFooladi/Transformers-For-Chemists/blob/main/notebooks/09_1_GNN_vs_Transformer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 09.1 · GNN vs Transformer — graph neighbours or SMILES sequence?
#
# This whole course teaches a transformer that reads a molecule as a **string**
# of atom tokens in SMILES order. The sister course
# [GNNs-For-Chemists](https://github.com/HFooladi/GNNs-For-Chemists) reads the
# same molecule as a **graph** of atoms joined by bonds. Two very different
# inductive biases — so which wins?
#
# We build a tiny **graph neural network from scratch** in pure PyTorch (RDKit
# supplies the atoms and bonds; no `torch_geometric`, no external repo), train it
# on BBBP alongside the transformer on the *same* split, and compare them head to
# head — then unpack *what each architecture actually exploits*.

# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# 1. Turn a molecule into an atom-feature matrix + adjacency with RDKit.
# 2. Implement a GCN-style **message-passing** layer from scratch.
# 3. **Batch variable-size graphs** without `torch_geometric` (block-diagonal
#    adjacency + a batch index).
# 4. Train the GNN on BBBP and compare ROC-AUC and parameter count to the
#    transformer.
# 5. Articulate the conceptual contrast: permutation-invariant graph vs
#    order-dependent SMILES sequence — and why MolFormer is a sequence model.

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

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from utils.smiles_tokenizers import AtomTokenizer
from utils.transformer_blocks import TransformerEncoder
from utils.training_utils import count_parameters
from utils.data_loading import load_moleculenet
from utils.preprocessing import clean_smiles

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_LEN = 64, 4, 2, 256, 128
print(f"device: {DEVICE}")
# -

# ---
# ## 1. Two ways to read a molecule
#
# The transformer sees caffeine as a **sequence** of atom tokens (SMILES order);
# the GNN sees it as a **graph** — an unordered set of atoms plus an adjacency
# matrix saying which bond to which. Same molecule, two data structures.

# +
tok = AtomTokenizer.from_smiles([CAFFEINE])
caf_tokens = tok.tokenize(CAFFEINE)
caf_atoms = [a.GetSymbol() for a in Chem.MolFromSmiles(CAFFEINE).GetAtoms()]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [2, 1]})
# left: the SMILES token strip
a1.set_xlim(0, len(caf_tokens)); a1.set_ylim(0, 1); a1.axis("off")
for i, t in enumerate(caf_tokens):
    a1.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor="#dbe7ff", edgecolor="white"))
    a1.text(i + 0.5, 0.5, t, ha="center", va="center", fontsize=7)
a1.set_title("Transformer's view: SMILES token sequence (ordered)")
# right: adjacency of the atom graph
mol = Chem.MolFromSmiles(CAFFEINE); n = mol.GetNumAtoms()
A = np.eye(n)
for b in mol.GetBonds():
    A[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] = A[b.GetEndAtomIdx(), b.GetBeginAtomIdx()] = 1
a2.imshow(A, cmap="Blues")
a2.set_xticks(range(n)); a2.set_xticklabels(caf_atoms, fontsize=6)
a2.set_yticks(range(n)); a2.set_yticklabels(caf_atoms, fontsize=6)
a2.set_title("GNN's view: atom adjacency (unordered)")
plt.tight_layout(); plt.show()
# -

# 🧪 **Chemical Intuition.** The SMILES order is an arbitrary traversal of the
# molecule — write the same caffeine starting from a different atom and the token
# sequence changes, though the molecule doesn't. The graph has *no* canonical
# order: it encodes only which atoms are bonded. That asymmetry — the GNN is
# permutation-invariant by construction, the transformer must *learn* to be — is
# the heart of this comparison.

# ---
# ## 2. Molecule → features + adjacency (RDKit, from scratch)
#
# Each atom becomes a feature vector: a one-hot of its element (over the common
# organic set) plus a few flags. Bonds become a symmetric adjacency, to which we
# add self-loops and apply the standard GCN normalisation
# $\hat{A} = D^{-1/2}(A+I)D^{-1/2}$ so message passing doesn't blow up the scale.

# +
ELEMENTS = ["C", "N", "O", "F", "S", "Cl", "Br", "P", "I"]   # + an "other" bucket
F_DIM = len(ELEMENTS) + 1 + 4    # element one-hot + [degree, aromatic, charge, numH]


def mol_to_graph(smiles):
    """SMILES → (atom features X (n, F_DIM), normalised adjacency Â (n, n)) or None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    n = mol.GetNumAtoms()
    X = np.zeros((n, F_DIM), dtype=np.float32)
    for atom in mol.GetAtoms():
        i, sym = atom.GetIdx(), atom.GetSymbol()
        X[i, ELEMENTS.index(sym) if sym in ELEMENTS else len(ELEMENTS)] = 1.0
        X[i, len(ELEMENTS) + 0] = atom.GetDegree() / 4.0
        X[i, len(ELEMENTS) + 1] = float(atom.GetIsAromatic())
        X[i, len(ELEMENTS) + 2] = atom.GetFormalCharge()
        X[i, len(ELEMENTS) + 3] = atom.GetTotalNumHs() / 4.0
    A = np.eye(n, dtype=np.float32)                       # self-loops (A + I)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        A[i, j] = A[j, i] = 1.0
    deg_inv_sqrt = 1.0 / np.sqrt(A.sum(1))                # D^{-1/2}
    A_hat = deg_inv_sqrt[:, None] * A * deg_inv_sqrt[None, :]
    return torch.tensor(X), torch.tensor(A_hat)


Xc, Ac = mol_to_graph(CAFFEINE)
print(f"caffeine → features {tuple(Xc.shape)}, normalised adjacency {tuple(Ac.shape)}")
# -

# ⚠️ **Note.** This feature set is deliberately minimal — hydrogens are implicit,
# stereochemistry and bond orders are dropped. A production GNN adds bond
# features, richer atom descriptors, and edge types; the exercises explore some.

# ---
# ## 3. A message-passing layer, by hand
#
# One graph-convolution layer: project every atom's features, then mix each atom
# with its neighbours by multiplying with $\hat{A}$. That's the entire
# Kipf–Welling GCN update, $H' = \mathrm{ReLU}(\hat{A}\,H\,W)$.

class GCNLayer(nn.Module):
    def __init__(self, f_in, f_out):
        super().__init__()
        self.lin = nn.Linear(f_in, f_out)

    def forward(self, X, A_hat):
        return torch.relu(A_hat @ self.lin(X))    # (n, f_out)

# 💡 **Key Insight.** One layer lets each atom see its **immediate bonded
# neighbours**; stacking `k` layers grows the receptive field to a `k`-hop
# neighbourhood. It's the graph analogue of attention's receptive field — but
# defined by *bonds*, not by position in a string.

# ---
# ## 4. Batching variable-size graphs without torch_geometric
#
# Molecules have different atom counts, so we can't just stack them into a
# tensor. The standard trick: lay every molecule's adjacency on the **diagonal**
# of one big block-diagonal matrix, concatenate all atom features into one tall
# matrix, and keep a `batch_index` telling us which molecule each atom belongs to.

def collate_graphs(batch):
    sizes = [x.shape[0] for x, _, _ in batch]
    N = sum(sizes)
    X = torch.zeros(N, F_DIM)
    A = torch.zeros(N, N)                     # block-diagonal: no edges between molecules
    batch_index = torch.zeros(N, dtype=torch.long)
    offset = 0
    for gi, (x, a, _) in enumerate(batch):
        n = x.shape[0]
        X[offset:offset + n] = x
        A[offset:offset + n, offset:offset + n] = a
        batch_index[offset:offset + n] = gi
        offset += n
    labels = torch.tensor([y for _, _, y in batch], dtype=torch.float)
    return X, A, batch_index, labels

# 💡 **Key Insight.** Because the off-diagonal blocks are zero, message passing
# over the big matrix never leaks information between molecules — it's exactly
# equivalent to processing each graph separately, just vectorised. The
# `batch_index` is what lets us pool each molecule's atoms back into one vector.

# ---
# ## 5. The GNN classifier
#
# Stack a few `GCNLayer`s, **mean-pool** each molecule's atom vectors (a
# permutation-invariant readout, using `batch_index`), and map to a logit.

# +
class TinyGNN(nn.Module):
    def __init__(self, f_dim=F_DIM, hidden=64, n_layers=3):
        super().__init__()
        dims = [f_dim] + [hidden] * n_layers
        self.layers = nn.ModuleList([GCNLayer(dims[i], dims[i + 1]) for i in range(n_layers)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, X, A_hat, batch_index):
        for layer in self.layers:
            X = layer(X, A_hat)                          # (N_atoms, hidden)
        n_graphs = int(batch_index.max()) + 1
        pooled = torch.zeros(n_graphs, X.shape[1], device=X.device)
        counts = torch.zeros(n_graphs, 1, device=X.device)
        pooled.index_add_(0, batch_index, X)             # sum atoms per molecule
        counts.index_add_(0, batch_index, torch.ones(X.shape[0], 1, device=X.device))
        pooled = pooled / counts.clamp(min=1)            # mean pool
        return self.head(pooled).squeeze(-1)             # (n_graphs,)


print(f"TinyGNN parameters: {count_parameters(TinyGNN()):,}")
# -

# ---
# ## 6. Train both models on the SAME BBBP split
#
# Load BBBP once, clean and split it, then build graph views (for the GNN) and
# token views (for the transformer) from the *identical* molecules, so the
# comparison is fair.

# +
smiles_raw, labels_raw = load_moleculenet("bbbp")
pairs = [(clean_smiles(s), int(y)) for s, y in zip(smiles_raw, labels_raw)]
pairs = [(s, y) for s, y in pairs if s is not None and mol_to_graph(s) is not None]
X_all = [s for s, _ in pairs]
y_all = [y for _, y in pairs]

X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=0)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=0)
print(f"train {len(X_tr)} | val {len(X_va)} | test {len(X_te)} molecules")


def graph_loader(smiles, labels, shuffle=False):
    graphs = [(*mol_to_graph(s), y) for s, y in zip(smiles, labels)]
    return DataLoader(graphs, batch_size=32, shuffle=shuffle, collate_fn=collate_graphs)


def auc_gnn(model, loader):
    model.eval(); probs, ys = [], []
    with torch.no_grad():
        for X, A, b, y in loader:
            probs += torch.sigmoid(model(X.to(DEVICE), A.to(DEVICE), b.to(DEVICE))).cpu().tolist()
            ys += y.tolist()
    return roc_auc_score(ys, probs)


def train_gnn(seed):
    torch.manual_seed(seed)
    model = TinyGNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    tr_loader = graph_loader(X_tr, y_tr, shuffle=True)
    va_loader, te_loader = graph_loader(X_va, y_va), graph_loader(X_te, y_te)
    history, best, best_state, since = [], -1.0, None, 0
    for epoch in range(30):
        model.train()
        for X, A, b, y in tr_loader:
            X, A, b, y = X.to(DEVICE), A.to(DEVICE), b.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss_fn(model(X, A, b), y).backward(); opt.step()
        va = auc_gnn(model, va_loader); history.append(va)
        if va > best:
            best, since = va, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if since >= 6:
            break
    model.load_state_dict(best_state)
    return auc_gnn(model, te_loader), history
# -

# Now the transformer — the notebook-07 recipe (atom tokenizer fitted on the
# training split, `TransformerEncoder` + `[CLS]` head), trained from scratch on
# the same molecules.

# +
class ClassificationHead(nn.Module):
    def __init__(self, d_model=D_MODEL, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout); self.proj = nn.Linear(d_model, 1)

    def forward(self, seq):
        return self.proj(self.dropout(seq[:, 0, :])).squeeze(-1)


class MoleculeClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
                                          n_layers=N_LAYERS, d_ff=D_FF, max_len=MAX_LEN, dropout=0.1)
        self.head = ClassificationHead()

    def forward(self, input_ids, attention_mask):
        seq, _ = self.encoder(input_ids, attention_mask)
        return self.head(seq)


tokenizer = AtomTokenizer.from_smiles(X_tr)


def token_loader(smiles, labels, shuffle=False):
    class _DS(Dataset):
        def __len__(self): return len(smiles)
        def __getitem__(self, i): return smiles[i], labels[i]

    def _coll(b):
        ids, m = tokenizer.encode_batch([z[0] for z in b], add_special_tokens=True, max_length=MAX_LEN)
        return torch.tensor(ids), torch.tensor(m), torch.tensor([z[1] for z in b], dtype=torch.float)
    return DataLoader(_DS(), batch_size=32, shuffle=shuffle, collate_fn=_coll)


def auc_tf(model, loader):
    model.eval(); probs, ys = [], []
    with torch.no_grad():
        for ids, m, y in loader:
            probs += torch.sigmoid(model(ids.to(DEVICE), m.to(DEVICE))).cpu().tolist()
            ys += y.tolist()
    return roc_auc_score(ys, probs)


def train_transformer(seed):
    torch.manual_seed(seed)
    model = MoleculeClassifier(tokenizer.vocab_size).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    tr_loader = token_loader(X_tr, y_tr, shuffle=True)
    va_loader, te_loader = token_loader(X_va, y_va), token_loader(X_te, y_te)
    history, best, best_state, since = [], -1.0, None, 0
    for epoch in range(15):
        model.train()
        for ids, m, y in tr_loader:
            ids, m, y = ids.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss_fn(model(ids, m), y).backward(); opt.step()
        va = auc_tf(model, va_loader); history.append(va)
        if va > best:
            best, since = va, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if since >= 4:
            break
    model.load_state_dict(best_state)
    return auc_tf(model, te_loader), history


gnn_scores, gnn_hist = [], None
tf_scores, tf_hist = [], None
for s in (0, 1):
    a, h = train_gnn(s); gnn_scores.append(a); gnn_hist = gnn_hist or h
    a, h = train_transformer(s); tf_scores.append(a); tf_hist = tf_hist or h
print(f"GNN         test AUC: {np.mean(gnn_scores):.3f}  {[round(a, 3) for a in gnn_scores]}")
print(f"Transformer test AUC: {np.mean(tf_scores):.3f}  {[round(a, 3) for a in tf_scores]}")
# -

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(range(1, len(gnn_hist) + 1), gnn_hist, "-o", color="#55a868", label="GNN")
ax.plot(range(1, len(tf_hist) + 1), tf_hist, "-o", color="#4c72b0", label="Transformer")
ax.set_xlabel("epoch"); ax.set_ylabel("validation ROC-AUC")
ax.set_title("Training both from scratch on BBBP"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# ---
# ## 7. Head-to-head
#
# Test ROC-AUC and parameter count, side by side.

# +
gnn_params = count_parameters(TinyGNN())
tf_params = count_parameters(MoleculeClassifier(tokenizer.vocab_size))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
a1.bar(["GNN", "Transformer"], [np.mean(gnn_scores), np.mean(tf_scores)],
       yerr=[np.std(gnn_scores), np.std(tf_scores)], capsize=8,
       color=["#55a868", "#4c72b0"], edgecolor="white")
a1.set_ylabel("test ROC-AUC"); a1.set_ylim(0.5, 1.0); a1.set_title("Accuracy (mean ± std, 2 seeds)")
a2.bar(["GNN", "Transformer"], [gnn_params, tf_params], color=["#55a868", "#4c72b0"], edgecolor="white")
a2.set_ylabel("parameters"); a2.set_title("Model size")
for i, v in enumerate([gnn_params, tf_params]):
    a2.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
plt.tight_layout(); plt.show()
print(f"GNN: {gnn_params:,} params | Transformer: {tf_params:,} params")
# -

# ⚠️ **Note.** Read this as a trade-off, not a leaderboard. The transformer edges
# ahead on ROC-AUC, but it spends **~11× more parameters** to get there; the
# 9k-parameter GNN, with the molecular graph baked in, lands surprisingly close
# for a fraction of the size. On only ~1,300 BBBP molecules both numbers are
# seed- and split-sensitive — what's robust is the *shape* of the trade: explicit
# graph prior and efficiency (GNN) vs. raw capacity and a global view
# (transformer). The transformer's decisive edge shows up elsewhere — pre-training
# (§8).

# ---
# ## 8. What each architecture exploits
#
# | | **GNN** | **Transformer** |
# | --- | --- | --- |
# | Input | atom–bond **graph** | SMILES **token sequence** |
# | Symmetry | permutation-invariant *by construction* | must *learn* it (SMILES order matters) |
# | Inductive bias | locality — messages flow along bonds | none built in; all-pairs attention |
# | Receptive field | `k` hops after `k` layers | global in a single layer |
# | Long-range / global features | needs depth (risks over-smoothing) | one attention layer reaches everything |
# | Unlabelled pre-training | needs graph pretext tasks | trivial & powerful (MLM on raw SMILES) |
#
# 💡 **Key Insight.** The GNN bakes the molecular graph *in*, so it's
# parameter-efficient and never has to discover that bonded atoms are related.
# The transformer throws that prior away and reads a string — but in exchange it
# gets a global receptive field *and*, crucially, the ability to **pre-train on
# essentially unlimited unlabelled SMILES** (notebooks 08–09). That trade — give
# up the explicit graph prior, gain cheap massive pre-training — is exactly why
# MolFormer and most chemical "foundation models" are sequence transformers, not
# GNNs. For the graph side in depth, see the sister
# [GNNs-For-Chemists](https://github.com/HFooladi/GNNs-For-Chemists) course.

# ---
# ## Checkpoint exercises

# +
# Exercise 1 — add bond information
# ---------------------------------
# Extend mol_to_graph so the adjacency is weighted by bond order (single=1,
# double=2, aromatic=1.5) before normalisation. Does the GNN's test AUC move?

# YOUR CODE HERE

# --- Solution (try the exercise first, then peek) ---
# In mol_to_graph, replace A[i, j] = A[j, i] = 1.0 with:
#   w = b.GetBondTypeAsDouble()       # 1.0 / 2.0 / 3.0 / 1.5 (aromatic)
#   A[i, j] = A[j, i] = w
# then keep the same D^{-1/2} normalisation and retrain.

# +
# Exercise 2 — depth and over-smoothing
# -------------------------------------
# Train TinyGNN(n_layers=2, 4, 6, 8). Beyond a few layers, repeated neighbour
# averaging makes all atom vectors look alike ("over-smoothing") and test AUC
# usually drops. Plot AUC vs depth.

# YOUR CODE HERE

# --- Solution ---
# for L in (2, 4, 6, 8):
#     torch.manual_seed(0)
#     # rebuild TinyGNN(n_layers=L), train with the §6 loop, record test AUC.
#     # Expect a peak at small L, then decline.

# +
# Exercise 3 — scaffold split
# ---------------------------
# Re-split BBBP by Murcko scaffold (rdkit.Chem.Scaffolds.MurckoScaffold) so train
# and test share no scaffolds — the honest, harder benchmark. Does the GNN vs
# transformer gap change?

# YOUR CODE HERE

# --- Solution ---
# from rdkit.Chem.Scaffolds import MurckoScaffold
# scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(s) for s in X_all]
# # group molecules by scaffold, put whole scaffolds into train/test (no overlap),
# # then rerun §6. Both models usually drop vs the random split.
# -

# ---
# ## What's next
#
# Back on the main path. You've now seen both halves of molecular machine
# learning — molecules as **sequences** (this course) and molecules as **graphs**
# (the GNN view) — and why the sequence transformer, despite discarding the graph
# prior, became the backbone of chemical foundation models: it pre-trains.
#
# 📚 **References.**
# - Kipf, T. & Welling, M. (2017). *Semi-Supervised Classification with Graph
#   Convolutional Networks.* — the GCN layer.
# - Gilmer, J. et al. (2017). *Neural Message Passing for Quantum Chemistry.* —
#   the message-passing framework.
# - Wu, Z. et al. (2018). *MoleculeNet.* — BBBP and the scaffold split.
# - Ross, J. et al. (2022). *MolFormer.* — why a sequence model.
# - The sister course: **[GNNs-For-Chemists](https://github.com/HFooladi/GNNs-For-Chemists)**.
