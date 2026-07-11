import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_media(tmp_path_factory) -> Path:
    """Directory with the generated song.mp4 / song.mp3 test media."""
    out = tmp_path_factory.mktemp("media")
    subprocess.run(
        [sys.executable, str(FIXTURES / "make_fixture.py"), str(out)],
        check=True,
    )
    return out
