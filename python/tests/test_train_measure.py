"""Reading a run's steady state back from the files it left behind."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from mlime.train.measure import cpu_usage, gpu_means, rate, steady_steps, trace_summary


def step(number: int, at: float, recompiles: int = 0, examples: int = 10) -> dict[str, Any]:
    return {
        "event": "step",
        "step": number,
        "time": at,
        "seconds": at - 1000.0,
        "loss": 1.0,
        "examples": examples,
        "tokens": 100,
        "context_tokens": 50,
        "recompiles": recompiles,
    }


def test_the_steady_span_starts_after_the_window_and_the_last_recompile() -> None:
    steps = [
        step(10, 1010.0),
        step(20, 1020.0, recompiles=1),
        step(30, 1030.0),
        step(40, 1040.0),
        step(50, 1050.0),
    ]
    assert [r["step"] for r in steady_steps(steps, {"window": [1005.0, 1025.0]})] == [30, 40, 50]
    assert [r["step"] for r in steady_steps(steps, {"window": [1005.0, 1035.0]})] == [40, 50]
    assert [r["step"] for r in steady_steps(steps, {"window": [1005.0, 1025.0]}, 40)] == [40, 50]
    steps[3]["recompiles"] = 2
    with pytest.raises(ValueError, match="a span needs two"):
        steady_steps(steps, {"window": [1005.0, 1025.0]})


def test_a_window_that_never_closed_is_refused() -> None:
    with pytest.raises(ValueError, match="never closed"):
        steady_steps([step(10, 1010.0), step(20, 1020.0)], {"window": [1005.0]})


def test_the_rate_covers_the_steps_after_the_spans_first_record() -> None:
    measured = rate(
        [step(30, 1030.0, examples=99), step(40, 1040.0), step(50, 1050.0, examples=30)]
    )
    assert measured["seconds_per_step"] == 1.0
    assert measured["steps_per_second"] == 1.0
    # The first record is the starting point: its own step is not in the span.
    assert measured["examples_per_second_per_rank"] == 20.0
    assert measured["padded_tokens_per_second_per_rank"] == 150.0


def gpu_line(index: int, at: datetime, utilisation: int, throttle: str) -> str:
    stamp = at.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
    return f"{index}, {stamp}, {utilisation}, 1000, 1200, 60, 100.0, {throttle}"


def test_gpu_means_cover_only_the_span(tmp_path: Path) -> None:
    log = tmp_path / "gpu.csv"
    base = datetime(2026, 10, 19, 15, 0, 0)
    lines = [
        gpu_line(0, base.replace(second=0), 10, "0x0000000000000000"),  # before the span
        gpu_line(0, base.replace(second=5), 60, "0x0000000000000004"),
        gpu_line(0, base.replace(second=6), 80, "0x0000000000000000"),
        gpu_line(0, base.replace(second=30), 0, "0x0000000000000000"),  # after it
        "0, 2026/10/19 15:00:06.5",  # the line nvidia-smi was writing when stopped
    ]
    log.write_text("\n".join(lines) + "\n")
    since = base.replace(second=4).timestamp()
    until = base.replace(second=7).timestamp()
    means = gpu_means(log, since, until)
    assert means["0"]["utilization"] == 70.0
    assert means["0"]["samples"] == 2
    assert means["0"]["throttle_share"] == {"0x0000000000000004": 0.5, "0x0000000000000000": 0.5}


def test_cpu_usage_splits_each_process_into_its_main_thread_and_the_rest(tmp_path: Path) -> None:
    hertz = os.sysconf("SC_CLK_TCK")
    log = tmp_path / "cpu.jsonl"
    samples = [
        {"pid": 7, "command": "python3 -m mlime train e2e"},
        {"time": 100.0, "busy": 0, "total": 0, "threads": [[7, 7, 0], [7, 8, 0]]},
        {"time": 105.0, "busy": 20, "total": 50, "threads": [[7, 7, 5 * hertz], [7, 8, 0]]},
        {
            "time": 110.0,
            "busy": 50,
            "total": 100,
            "threads": [[7, 7, 10 * hertz], [7, 8, 5 * hertz]],
        },
        {"time": 200.0, "busy": 90, "total": 100, "threads": [[7, 7, 99 * hertz], [7, 8, 0]]},
    ]
    log.write_text("".join(json.dumps(s) + "\n" for s in samples))
    usage = cpu_usage(log, 100.0, 110.0)
    assert usage["busy_percent"] == 50.0
    assert usage["processes"] == [
        {"pid": 7, "command": "python3 -m mlime train e2e", "main_cores": 1.0, "other_cores": 0.5}
    ]


def annotation(name: str, at: float, duration: float = 0.0, tid: int = 1) -> dict[str, Any]:
    return {"cat": "user_annotation", "name": name, "ts": at, "dur": duration, "tid": tid}


def test_a_trace_charges_idle_time_and_launches_to_phases_and_regions(tmp_path: Path) -> None:
    """Two steps of marks, a search region, five kernels and four host launches.

    Times are microseconds. Device work is 200 ms of the 500 ms span; each gap
    is charged to the phase and innermost region around its midpoint.
    """
    marks = [
        ("batch_wait", 0),
        ("forward", 100_000),
        ("optimizer", 200_000),
        ("batch_wait", 300_000),
        ("forward", 400_000),
        ("optimizer", 500_000),
    ]
    events = [annotation(f"phase.{name}", at) for name, at in marks]
    events.append(annotation("rank.beam", 20_000, 60_000))
    for start, duration in (
        (10_000, 30_000),
        (50_000, 10_000),
        (120_000, 50_000),
        (320_000, 60_000),
        (420_000, 50_000),
    ):
        events.append({"cat": "kernel", "name": "k", "ts": start, "dur": duration, "tid": 7})
    events += [
        {"cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 25_000, "dur": 5, "tid": 1},
        {"cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 26_000, "dur": 5, "tid": 1},
        {"cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": 30_000, "dur": 5, "tid": 1},
        {"cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 110_000, "dur": 5, "tid": 1},
        # Another thread's launch is not the trainer's.
        {"cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 111_000, "dur": 5, "tid": 2},
    ]
    trace = tmp_path / "trace-rank0.json"
    trace.write_text(json.dumps({"traceEvents": events}))
    summary = trace_summary(trace)
    assert summary["steps"] == 2
    assert summary["span_seconds"] == 0.5
    assert summary["device_busy_fraction"] == 0.4
    assert summary["phase_seconds"] == {"forward": 0.2, "optimizer": 0.2, "batch_wait": 0.1}
    assert summary["kernel_launches_per_step"] == {
        "phase": {"forward": 1.0, "optimizer": 0.5},
        "region": {"rank.beam": 1.0, "-": 0.5},
    }
    assert summary["graph_launches_per_step"] == {
        "phase": {"forward": 0.5},
        "region": {"rank.beam": 0.5},
    }
    assert summary["idle_ms_per_step"] == [
        {"phase": "batch_wait", "region": "-", "ms": 75.0},
        {"phase": "forward", "region": "-", "ms": 35.0},
        {"phase": "optimizer", "region": "-", "ms": 35.0},
        {"phase": "forward", "region": "rank.beam", "ms": 5.0},
    ]
