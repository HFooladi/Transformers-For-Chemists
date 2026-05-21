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
# 🚧 **Coming soon.**
#
# Notebook 04.1 built **linear attention** with the simplest correct feature
# map — `φ(x) = elu(x) + 1` from Katharopoulos et al. (2020). It is great
# pedagogy: five lines of PyTorch, strictly positive, no random sampling,
# and it carries the full `O(L)` complexity story.
#
# But it is **not what MolFormer actually uses.** Ross et al. (2022) use
# the **Performer** feature map (Choromanski et al., 2021) — *Fast
# Attention Via positive Orthogonal Random features* (FAVOR+). FAVOR+ uses
# **random orthogonal projections** to construct a feature map that
# *provably approximates* the softmax kernel, with controllable variance,
# at the same `O(L)` cost.
#
# This supplementary notebook will fill in:
#
# 1. **The softmax-as-Gaussian-integral identity** that motivates random
#    features.
# 2. **Positive random features** — why naive Gaussian features can give
#    negative attention weights, and how FAVOR+ fixes it.
# 3. **Orthogonal random features** — the variance-reduction trick that
#    makes the approximation work in practice.
# 4. A **`PerformerAttention`** module mirroring `LinearAttention`'s
#    interface, drop-in interchangeable.
# 5. A **direct comparison** with both softmax attention and ELU+1 linear
#    attention on caffeine: which one tracks softmax most faithfully, at
#    what number of random features?
# 6. Forward-pass timing vs. random-feature dimension `m`.
#
# Until this notebook is written, the working understanding to carry into
# the rest of the course is: **MolFormer's attention has the same `O(L)`
# complexity as the ELU+1 version you built in 04.1**, but a more careful
# choice of `φ` that approximates softmax closely. The complexity story is
# the same; the approximation story is different.
#
# 📚 **References to read in the meantime.**
# - Choromanski, K. et al. (2021). *Rethinking Attention with Performers.*
#   ICLR.
# - Ross, J. et al. (2022). *Large-Scale Chemical Language Representations
#   Capture Molecular Structure and Properties.* Nature Machine
#   Intelligence — see the architecture section on attention.

# ## Setup
#
# (Boilerplate kept identical to notebook 04.1 so future cells can be
# added without re-staging.)

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
# -

print("This notebook is a placeholder. See 04.1 for the linear-attention story; ")
print("the FAVOR+ / Performer feature map will be filled in here.")
