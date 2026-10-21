"""Keep each test's scratch files in its own automatically cleaned temp folder."""

from pathlib import Path
import tempfile

import pytest


@pytest.fixture
def tmp_path():
    root = Path(__file__).resolve().parents[1] / "temp"
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="test-", dir=root) as directory:
        yield Path(directory)
