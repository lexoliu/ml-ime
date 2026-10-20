"""`mlime e2e stamp`: a v2 run's kernel directory from the committed e2e kernel.

v2 is v1's recipe on a different stream, so the copy may differ from the
committed kernel only in the run's own fields, and its metadata only in the
run's slug and the sources a v2 leg mounts.
"""

import difflib
import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mlime.__main__ import app
from mlime.stamp import E2E_V2_SOURCES, V1_RECIPE, E2EV2Run, literal, stamp, stamp_e2e_v2

TEMPLATE = Path(__file__).resolve().parents[2] / "kaggle" / "e2e"


def changed_lines(before: str, after: str) -> list[str]:
    """The lines of *after* that are not in *before*, by a line diff."""
    return [
        line[1:]
        for line in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0)
        if line.startswith("+") and not line.startswith("+++")
    ]


def test_a_v2_kernel_is_v1s_recipe_with_only_the_runs_fields_stamped(tmp_path: Path) -> None:
    out = tmp_path / "a100"
    result = CliRunner().invoke(app, ["e2e", "stamp", "mlime-e2e-v2-a100", "--out", str(out)])
    assert result.exit_code == 0, result.output
    template = (TEMPLATE / "kernel.py").read_text()
    kernel = (out / "kernel.py").read_text()
    assert changed_lines(template, kernel) == ["PROFILE_STEPS = 30", "TRACE_STEPS = 3"]
    for name, value in V1_RECIPE.items():
        assert literal(kernel, name) == value == literal(template, name), name
        assert f"  {name} = {value!r}" in result.output
    assert literal(kernel, "TRAIN_SHARDS") is None

    metadata = json.loads((out / "kernel-metadata.json").read_text())
    committed = json.loads((TEMPLATE / "kernel-metadata.json").read_text())
    assert metadata["id"] == "lexoliu/mlime-e2e-v2-a100-s0"
    assert metadata["title"] == "mlime-e2e-v2-a100-s0"
    assert metadata["dataset_sources"] == list(E2E_V2_SOURCES)
    assert metadata["kernel_sources"] == []
    unchanged = set(committed) - {"id", "title", "dataset_sources", "kernel_sources"}
    assert {key: metadata[key] for key in unchanged} == {key: committed[key] for key in unchanged}


def test_a_check_narrows_the_epoch_and_the_profile_window(tmp_path: Path) -> None:
    out = tmp_path / "check"
    argv = ["e2e", "stamp", "mlime-e2e-v2-t4-check", "--out", str(out)]
    argv += ["--profile-steps", "0", "--trace-steps", "0"]
    argv += ["--train-shard", "news-00002.parquet", "--train-shard", "wiki-00003.parquet"]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 0, result.output
    kernel = (out / "kernel.py").read_text()
    assert literal(kernel, "TRAIN_SHARDS") == ("news-00002.parquet", "wiki-00003.parquet")
    assert literal(kernel, "PROFILE_STEPS") == 0
    assert literal(kernel, "TRACE_STEPS") == 0
    assert json.loads((out / "kernel-metadata.json").read_text())["id"] == (
        "lexoliu/mlime-e2e-v2-t4-check-s0"
    )
    with pytest.raises(ValueError, match="trace covers steps of the profile window"):
        E2EV2Run(run="mlime-e2e-v2-t4-check", profile_steps=2, trace_steps=3)


def test_a_template_off_v1s_recipe_is_refused_by_field(tmp_path: Path) -> None:
    template = tmp_path / "e2e"
    shutil.copytree(TEMPLATE, template)
    source = (template / "kernel.py").read_text()
    (template / "kernel.py").write_text(stamp(source, TOKEN_BUDGET=12288, AUX_WEIGHT=0.5))
    with pytest.raises(ValueError, match=r"\['AUX_WEIGHT', 'TOKEN_BUDGET'\]"):
        stamp_e2e_v2(template, tmp_path / "out", E2EV2Run(run="mlime-e2e-v2-a100"))
    assert not (tmp_path / "out").exists()


def test_a_stamp_lands_as_a_literal_or_is_refused() -> None:
    source = (TEMPLATE / "kernel.py").read_text()
    stamped = stamp(source, COMMIT="c" * 40, TRAIN_SHARDS=("a\\b.parquet",))
    assert literal(stamped, "COMMIT") == "c" * 40
    assert literal(stamped, "TRAIN_SHARDS") == ("a\\b.parquet",)
    with pytest.raises(KeyError, match="no `MISSING =` line"):
        stamp(source, MISSING=1)
    # HELD_OUT's default spans lines: rewriting its first one would break it.
    with pytest.raises(SyntaxError):
        stamp(source, HELD_OUT=("x.parquet",))
