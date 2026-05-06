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
    ensure_environment(["torch", "rdkit"])  # warns + installs missing pieces
"""

from __future__ import annotations

from typing import Iterable


REPO_URL = "https://github.com/HFooladi/Transformers-For-Chemists.git"


def ensure_environment(packages: Iterable[str]) -> None:
    """Pip-install any missing packages from ``packages`` (Colab-friendly).

    The function is designed to be called from a notebook cell. It checks
    whether each package can be imported and only installs the ones that are
    missing. Installs are done quietly with ``pip install -q``.

    Parameters
    ----------
    packages
        Iterable of pip package names. The package name is also used as the
        import name; pass ``("rdkit",)`` rather than ``("rdkit-pypi",)``.
    """
    raise NotImplementedError("Phase 2: implement in notebook 01.")


def is_colab() -> bool:
    """Return ``True`` if the current runtime is Google Colab."""
    raise NotImplementedError("Phase 2: implement in notebook 01.")
