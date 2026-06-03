from collections.abc import Iterator
from pathlib import Path
import shutil
from uuid import uuid4

import pytest


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    root = Path(".test_tmp")
    path = root / f"case-{uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        try:
            root.rmdir()
        except OSError:
            pass
