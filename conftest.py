"""Shared pytest configuration.

Provides repo-wide fixtures so individual test modules stay focused on assertions.
`pythonpath = ["."]` in pyproject.toml already makes `src` importable; this file is
the place for cross-module fixtures as the suite grows.
"""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
OUTPUTS_DIR = REPO_ROOT / "tests" / "outputs"


def pytest_addoption(parser):
    """Adds --run-bedrock so live-credential tests are opt-in even locally."""
    parser.addoption(
        "--run-bedrock",
        action="store_true",
        default=False,
        help="run tests marked 'bedrock' that require live AWS credentials",
    )


def pytest_collection_modifyitems(config, items):
    """Skips bedrock-marked tests unless --run-bedrock (or -m bedrock) is given."""
    if config.getoption("--run-bedrock") or "bedrock" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="needs live AWS Bedrock credentials; pass --run-bedrock to enable")
    for item in items:
        if "bedrock" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Absolute path to tests/fixtures (golden datasets, copy fixtures, benchmarks)."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def outputs_dir() -> Path:
    """Absolute path to tests/outputs, created on demand for evaluation run artifacts."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR
