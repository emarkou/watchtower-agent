from __future__ import annotations

import tempfile

import pytest

from data.synthetic import SyntheticDataLoader, SyntheticDataLoaderNoDrift
from agent.run_store import RunStore


@pytest.fixture
def loader():
    return SyntheticDataLoader()


@pytest.fixture
def loader_no_drift():
    return SyntheticDataLoaderNoDrift()


@pytest.fixture
def ref_data(loader):
    return loader.load_reference()


@pytest.fixture
def cur_data(loader):
    return loader.load_current_window()


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test_runs.db")
    return RunStore(db_path)


@pytest.fixture
def run_id():
    return "test-run-00000000"
