"""Fixtures: one tiny random bundle per session, its reference made by the independent oracle."""

from __future__ import annotations

from pathlib import Path

import pytest
from oracle import write_bundle

from mlime_coreml.bundle import Bundle, read_bundle


@pytest.fixture(name="bundle_dir", scope="session")
def bundle_dir_fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("bundle")
    write_bundle(directory)
    return directory


@pytest.fixture(name="bundle", scope="session")
def bundle_fixture(bundle_dir: Path) -> Bundle:
    return read_bundle(bundle_dir)
