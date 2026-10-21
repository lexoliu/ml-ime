"""`mlime e2e stamp`: a v2 run's kernel directory from the committed e2e kernel.

v2 is v1's recipe on a different stream, so the copy may differ from the
committed kernel only in the run's own fields, and its metadata only in the
run's slug and the sources a v2 leg mounts.
"""

import difflib
import json
import shutil
from pathlib import Path

import click
import pytest
from typer.testing import CliRunner

from mlime.__main__ import app
from mlime.stamp import (
    E2E_V2_SOURCES,
    V1_RECIPE,
    E2EV2Run,
    E2EV2Stage,
    literal,
    stamp,
    stamp_e2e_v2,
)

TEMPLATE = Path(__file__).resolve().parents[2] / "kaggle" / "e2e"


def error(output: str) -> str:
    """A refusal's text with its styling, the error box's borders and wrapping taken out.

    Rich colours the box wherever it detects CI, so the styling goes first.
    """
    return " ".join(click.unstyle(output).replace("│", " ").split())


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


#: #132's weight-0.05 stage of v2, as `mlime e2e stamp` takes it.
STAGE_ARGV = [
    "--init-from",
    "mlime-e2e-v2-g4",
    "--ranking",
    "--rank-segmentations",
    "8",
    "--rank-beam",
    "4",
    "--rank-negatives",
    "8",
    "--rank-weight",
    "0.05",
    "--base-lr",
    "2e-5",
    "--new-lr",
    "2e-5",
    "--token-budget",
    "12288",
    "--accumulate",
    "2",
    "--train-shard",
    "news-00000.parquet",
    "--train-shard",
    "wiki-rest-00070.parquet",
]


def test_a_stage_is_v1s_recipe_with_only_the_stages_fields_stamped(tmp_path: Path) -> None:
    out = tmp_path / "rankw005"
    argv = ["e2e", "stamp", "mlime-e2e-v2-g4-rankw005", "--out", str(out), *STAGE_ARGV]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 0, result.output
    template = (TEMPLATE / "kernel.py").read_text()
    kernel = (out / "kernel.py").read_text()
    ranking = {"segmentations": 8, "beam": 4, "negatives": 8, "weight": 0.05}
    stage = {
        "INIT_FROM": "mlime-e2e-v2-g4",
        "RANKING": ranking,
        "LEARNING_RATES": (2e-5, 2e-5),
        "TOKEN_BUDGET": 12288,
        "ACCUMULATE": 2,
    }
    shards = ("news-00000.parquet", "wiki-rest-00070.parquet")
    run = {"PROFILE_STEPS": 30, "TRACE_STEPS": 3, "TRAIN_SHARDS": shards}
    assert sorted(changed_lines(template, kernel)) == sorted(
        f"{name} = {value!r}" for name, value in {**stage, **run}.items()
    )
    for name, value in V1_RECIPE.items():
        stamped = stage.get(name, value)
        assert literal(kernel, name) == stamped, name
        assert f"  {name} = {stamped!r}" in result.output
    assert "a stage of mlime-e2e-v2-g4" in result.output

    metadata = json.loads((out / "kernel-metadata.json").read_text())
    assert metadata["id"] == "lexoliu/mlime-e2e-v2-g4-rankw005-s0"
    assert metadata["dataset_sources"] == list(E2E_V2_SOURCES)
    assert metadata["kernel_sources"] == []


def test_a_stage_without_ranking_keeps_v1s_budget_and_rates(tmp_path: Path) -> None:
    """A stage changes only what it names; the rest of its recipe is v1's."""
    recipe = stamp_e2e_v2(
        TEMPLATE,
        tmp_path / "control",
        E2EV2Run(run="mlime-e2e-v2-g4-control", stage=E2EV2Stage(init_from="mlime-e2e-v2-g4")),
    )
    assert recipe == {**V1_RECIPE, "INIT_FROM": "mlime-e2e-v2-g4"}


@pytest.mark.parametrize(
    "options",
    [
        ["--ranking"],
        ["--ranking", "--rank-weight", "0.05"],
        ["--base-lr", "2e-5", "--new-lr", "2e-5"],
        ["--token-budget", "12288"],
        ["--accumulate", "2"],
    ],
)
def test_a_stages_options_without_its_start_are_refused(tmp_path: Path, options: list[str]) -> None:
    """A plain v2 run is v1's recipe; ranking above all needs a finished run to start from."""
    out = tmp_path / "plain"
    argv = ["e2e", "stamp", "mlime-e2e-v2-g4-rankw005", "--out", str(out), *options]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 2, result.output
    assert "pass --init-from" in error(result.output)
    assert not out.exists()


def test_a_stage_starts_from_another_finished_v2_run_with_whole_rates(tmp_path: Path) -> None:
    out = tmp_path / "stage"
    argv = ["e2e", "stamp", "mlime-e2e-v2-g4-rankw005", "--out", str(out)]
    half = ["--init-from", "mlime-e2e-v2-g4", "--base-lr", "2e-5"]
    result = CliRunner().invoke(app, [*argv, *half])
    assert result.exit_code == 2, result.output
    assert "learning rates are a pair" in error(result.output)
    assert not out.exists()
    with pytest.raises(ValueError, match="not a v2 run name"):
        E2EV2Stage(init_from="mlime-e2e-final")
    with pytest.raises(ValueError, match="cannot start from itself"):
        E2EV2Run(run="mlime-e2e-v2-g4", stage=E2EV2Stage(init_from="mlime-e2e-v2-g4"))


def test_a_ranking_stage_stamps_its_search_graphs_beside_its_recipe(tmp_path: Path) -> None:
    """The graphs change how the search runs, not the recipe it trains."""
    out = tmp_path / "graphs"
    argv = [
        "e2e",
        "stamp",
        "mlime-e2e-v2-g4-graphs",
        "--out",
        str(out),
        *STAGE_ARGV,
        "--search-graphs",
    ]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 0, result.output
    plain = tmp_path / "eager"
    eager = ["e2e", "stamp", "mlime-e2e-v2-g4-graphs", "--out", str(plain), *STAGE_ARGV]
    assert CliRunner().invoke(app, eager).exit_code == 0
    assert changed_lines((plain / "kernel.py").read_text(), (out / "kernel.py").read_text()) == [
        "SEARCH_GRAPHS = True"
    ]


def test_search_graphs_without_a_ranking_stage_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only a ranking stage"):
        E2EV2Run(
            run="mlime-e2e-v2-g4-control",
            stage=E2EV2Stage(init_from="mlime-e2e-v2-g4"),
            search_graphs=True,
        )
    with pytest.raises(ValueError, match="only a ranking stage"):
        E2EV2Run(run="mlime-e2e-v2-a100", search_graphs=True)
