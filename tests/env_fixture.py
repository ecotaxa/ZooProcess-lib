import os
from glob import glob
from os import getenv
from pathlib import Path
from typing import Generator, Any

import pytest

ZOOSCAN_HOME = "ZOOSCAN_PROJECTS"

HERE = Path(__file__).parent


@pytest.fixture(scope="session")
def projects() -> Generator[Path, Any, None]:
    # Setup
    path = read_home()
    yield path
    # Teardown


def read_home():
    _loadenv()
    env = getenv(ZOOSCAN_HOME)
    assert env is not None, f"This test needs {ZOOSCAN_HOME} environment variable set, you might provide a foo.env file."
    path = Path(env)
    assert path.exists(), f"{path} read from {ZOOSCAN_HOME} environment variable does not exist"
    return path


def _loadenv() -> None:
    """ Import all env files in present directory into process environment"""
    for an_env in glob(f"*.env", root_dir=HERE):
        path = Path(an_env)
        with open(path, "r") as f:
            lines = f.readlines()
            for a_line in lines:
                key, val = a_line.strip().split("=", maxsplit=1)
                os.environ[key] = val
