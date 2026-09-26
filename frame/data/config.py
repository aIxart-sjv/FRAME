"""Where datasets live on disk (Phase 3).

Datasets are large and are never stored in the repository. Every path in a
manifest is RELATIVE to its dataset directory:

    $FRAME_DATA_ROOT/<dataset>/<relative path from the manifest record>

`FRAME_DATA_ROOT` defaults to ``~/.cache/frame_data`` (outside the repo, like
the model-weight caches of earlier phases).
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV = "FRAME_DATA_ROOT"
DEFAULT_DATA_ROOT = Path.home() / ".cache" / "frame_data"


def data_root() -> Path:
    """The dataset root directory (``$FRAME_DATA_ROOT`` or the default)."""
    return Path(os.environ.get(DATA_ROOT_ENV, str(DEFAULT_DATA_ROOT)))


def dataset_dir(dataset: str, root: Path | None = None) -> Path:
    """Directory holding one dataset's files."""
    return (root if root is not None else data_root()) / dataset
