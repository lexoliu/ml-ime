"""Reading a run's steady state back from the files it left behind."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from mlime.train.measure import (
    GpuSampler,
    HostSampler,
    gpu_means,
    host_usage,
    rate,
    steady_steps,
    trace_summary,
)


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


@pytest.fixture
def eastern_clock() -> Iterator[None]:
    """A reader whose local zone is not the VM's UTC, as on the Mac."""
    before = os.environ.get("TZ")
    os.environ["TZ"] = "America/New_York"
    time.tzset()
    yield
    if before is None:
        del os.environ["TZ"]
    else:
        os.environ["TZ"] = before
    time.tzset()


@pytest.mark.usefixtures("eastern_clock")
def test_gpu_means_cover_only_the_span(tmp_path: Path) -> None:
    """The log's clock is UTC, read on a machine whose zone is not."""
    log = tmp_path / "gpu.csv"
    base = datetime(2026, 10, 19, 15, 0, 0, tzinfo=UTC)
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


def test_the_gpu_sampler_stamps_its_samples_in_utc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launched: dict[str, Any] = {}

    class Started:
        returncode = None

        def poll(self) -> None:
            return None

    def popen(argv: list[str], **options: Any) -> Started:
        launched.update(options, argv=argv)
        return Started()

    monkeypatch.setattr(subprocess, "Popen", popen)
    GpuSampler(tmp_path / "gpu.csv").__enter__()
    assert launched["argv"][0] == "nvidia-smi"
    assert launched["env"]["TZ"] == "UTC"


@pytest.mark.skipif(not Path("/proc/meminfo").is_file(), reason="reads Linux /proc")
def test_the_host_sampler_follows_a_trainer_through_proc(tmp_path: Path) -> None:
    """A busy child named like a trainer: its threads tick and its memory is resident."""
    trainer = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time\nend = time.time() + 30\nwhile time.time() < end: pass",
            "mlime",
        ]
    )
    log = tmp_path / "host.jsonl"
    try:
        with HostSampler(log, interval=0.2):
            time.sleep(1.5)
    finally:
        trainer.kill()
        trainer.wait()
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert lines[0]["memory_total"] > 0
    assert lines[0]["cores"] == os.cpu_count()
    assert lines[0]["clock_ticks"] == os.sysconf("SC_CLK_TCK")
    assert lines[1]["pid"] == trainer.pid
    assert lines[1]["command"].endswith(" mlime")
    samples = [line for line in lines if "time" in line]
    usage = host_usage(log, samples[0]["time"], samples[-1]["time"])
    assert 0 < usage["least_available_bytes"] <= usage["memory_total_bytes"]
    [process] = usage["processes"]
    assert process["pid"] == trainer.pid
    assert process["main_cores"] > 0.5
    assert process["peak_resident_bytes"] > 1 << 20


def test_host_usage_splits_cores_by_thread_and_keeps_the_memory_low_and_high(
    tmp_path: Path,
) -> None:
    # The VM's tick rate and cores, which the machine reading the log need not share.
    hertz = 250
    log = tmp_path / "host.jsonl"
    samples = [
        {"memory_total": 1000, "cores": 12, "clock_ticks": hertz},
        {"pid": 7, "command": "python3 -m mlime train e2e"},
        {
            "time": 100.0,
            "busy": 0,
            "total": 0,
            "available": 600,
            "threads": [[7, 7, 0], [7, 8, 0]],
            "resident": [[7, 300]],
        },
        {
            "time": 105.0,
            "busy": 20,
            "total": 50,
            "available": 400,
            "threads": [[7, 7, 5 * hertz], [7, 8, 0]],
            "resident": [[7, 500]],
        },
        {
            "time": 110.0,
            "busy": 50,
            "total": 100,
            "available": 500,
            "threads": [[7, 7, 10 * hertz], [7, 8, 5 * hertz]],
            "resident": [[7, 450]],
        },
        {
            "time": 200.0,
            "busy": 90,
            "total": 100,
            "available": 10,
            "threads": [[7, 7, 99 * hertz], [7, 8, 0]],
            "resident": [[7, 990]],
        },
    ]
    log.write_text("".join(json.dumps(s) + "\n" for s in samples))
    usage = host_usage(log, 100.0, 110.0)
    assert usage["busy_percent"] == 50.0
    assert usage["cores"] == 12
    assert usage["memory_total_bytes"] == 1000
    assert usage["least_available_bytes"] == 400
    assert usage["processes"] == [
        {
            "pid": 7,
            "command": "python3 -m mlime train e2e",
            "main_cores": 1.0,
            "other_cores": 0.5,
            "peak_resident_bytes": 500,
        }
    ]


def test_a_host_log_without_its_machine_line_is_refused(tmp_path: Path) -> None:
    log = tmp_path / "host.jsonl"
    sample = {"time": 1.0, "busy": 0, "total": 0, "available": 1, "threads": [], "resident": []}
    log.write_text(json.dumps(sample) + "\n" + json.dumps({**sample, "time": 2.0}) + "\n")
    with pytest.raises(ValueError, match="does not start with the machine's line"):
        host_usage(log, 0.0, 3.0)


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
