"""Colab setup helpers.

Each notebook in the course starts with a cell that clones this repository and
adds ``Transformers-For-Chemists/notebooks`` to ``sys.path`` so the ``utils``
package becomes importable. The functions here centralize that boilerplate so
notebooks can stay readable.

Typical first-cell usage in a Colab notebook::

    !git clone -q https://github.com/HFooladi/Transformers-For-Chemists.git
    import sys
    sys.path.append("Transformers-For-Chemists/notebooks")
    from utils.colab_setup import ensure_environment
    ensure_environment(["torch", "rdkit"])  # installs only what's missing
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from typing import Iterable


REPO_URL = "https://github.com/HFooladi/Transformers-For-Chemists.git"

# A few packages whose import name differs from their pip name.
_PIP_NAME = {
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "cv2": "opencv-python",
}


def is_colab() -> bool:
    """Return True if the current runtime is Google Colab."""
    return "google.colab" in sys.modules


def ensure_environment(packages: Iterable[str]) -> None:
    """Pip-install any packages from ``packages`` that aren't already importable.

    Uses each entry as the *import* name and looks up the pip name via a small
    override table for the common mismatches (``PIL`` → ``pillow``, etc.).
    Installs are quiet (``pip install -q``) and only the missing ones are
    fetched, so re-running the setup cell is cheap.
    """
    missing = []
    for name in packages:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(_PIP_NAME.get(name, name))

    if not missing:
        return

    print(f"Installing: {', '.join(missing)}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *missing]
    )
