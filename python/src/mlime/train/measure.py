"""What a training run did over a span of its own steps: rate, device, host.

A run measures itself while it trains. Its step records carry wall-clock
times; its profiled window writes a phase table to the metrics and,
optionally, a torch.profiler trace; and the kernel samples the GPUs
(`GpuSampler`, `gpu.csv`) and the trainer's threads (`CpuSampler`,
`cpu.jsonl`) beside it for the whole segment. :func:`measure` reads those
files back and reports the run's steady state -- the unprofiled steps after
its window and its last recompile -- so two cards, or two versions of the
code, compare on steps they actually trained.

The samplers import nothing beyond the standard library: the kernel starts
them before the training processes, from the package it has just staged.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

#: What `GpuSampler` asks nvidia-smi for, in the order its CSV holds them.
GPU_FIELDS = (
    "index",
    "timestamp",
    "utilization.gpu",
    "memory.used",
    "clocks.sm",
    "temperature.gpu",
    "power.draw",
    "clocks_throttle_reasons.active",
)

#: The processes whose threads `CpuSampler` follows: the trainers, the
#: launcher that spawns them, and the segmenter they read readings from.
TRAINER_COMMANDS = ("mlime", "torch.distributed.run", "ime-segment")


class GpuSampler:
    """`nvidia-smi -lms` running beside the training processes, into a CSV.

    The check is the driver's, not torch's: a step that leaves the card idle
    while the host collates shows up here even though no device code knows
    about it. 500 ms is fine enough to catch a step boundary and coarse
    enough to keep a twelve-hour log near 10 MB. The samples go to a file: a
    pipe nobody drains fills after a few minutes and stalls `nvidia-smi`.

    Beside utilisation it reads what can slow a busy card down: the SM
    clock, the temperature, the power draw and the driver's throttle-reason
    bits.
    """

    def __init__(self, log: Path, interval_ms: int = 500):
        self._log = log
        self._argv = [
            "nvidia-smi",
            f"--query-gpu={','.join(GPU_FIELDS)}",
            "--format=csv,noheader,nounits",
            "-lms",
            str(interval_ms),
        ]
        self._process: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> GpuSampler:
        with self._log.open("w", encoding="utf-8") as sink:
            self._process = subprocess.Popen(self._argv, stdout=sink, stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, *_: object) -> None:
        if self._process is None:
            raise RuntimeError("the GPU sampler was never started")
        if self._process.poll() is not None:
            raise RuntimeError(
                f"nvidia-smi exited {self._process.returncode} before the run did: "
                f"{' '.join(self._argv)}"
            )
        self._process.terminate()
        try:
            self._process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()


class CpuSampler:
    """The trainers' CPU time, per thread, sampled from `/proc` into JSON lines.

    A card idling because the thread that launches its work is out of CPU
    shows up here and nowhere in the device's own numbers. Each sample line
    holds the machine's busy and total jiffies and the cumulative user+system
    ticks of every thread of a `TRAINER_COMMANDS` process; a process's
    command is written once, the first time it is seen.
    """

    def __init__(self, log: Path, interval: float = 5.0):
        self._log = log
        self._interval = interval
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._commands: dict[int, str | None] = {}

    @staticmethod
    def _machine() -> tuple[int, int]:
        fields = [int(v) for v in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
        idle = fields[3] + fields[4]
        return sum(fields) - idle, sum(fields)

    @staticmethod
    def _command(pid: int) -> str:
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:  # the process ended between the listing and the read
            return "?"
        words = [w for w in raw.decode(errors="replace").split("\0") if w]
        return " ".join(Path(w).name if i == 0 else w for i, w in enumerate(words[:6]))

    def _followed(self, pid: int, sink: Any) -> bool:
        if pid not in self._commands:
            command = self._command(pid)
            followed = any(name in command for name in TRAINER_COMMANDS)
            self._commands[pid] = command if followed else None
            if followed:
                sink.write(json.dumps({"pid": pid, "command": command}) + "\n")
        return self._commands[pid] is not None

    def _threads(self, sink: Any) -> list[tuple[int, int, int]]:
        ticks = []
        for process in Path("/proc").glob("[0-9]*"):
            pid = int(process.name)
            if not self._followed(pid, sink):
                continue
            for task in process.glob("task/[0-9]*/stat"):
                try:
                    text = task.read_text()
                except OSError:  # the thread ended between the glob and the read
                    continue
                rest = text[text.rindex(")") + 2 :].split()
                ticks.append((pid, int(task.parent.name), int(rest[11]) + int(rest[12])))
        return ticks

    def _run(self) -> None:
        with self._log.open("w", encoding="utf-8") as sink:
            while not self._stopped.is_set():
                busy, total = self._machine()
                threads = self._threads(sink)
                sink.write(
                    json.dumps(
                        {"time": time.time(), "busy": busy, "total": total, "threads": threads}
                    )
                    + "\n"
                )
                sink.flush()
                self._stopped.wait(self._interval)

    def __enter__(self) -> CpuSampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stopped.set()
        self._thread.join()


def records(metrics: Path, event: str) -> list[dict[str, Any]]:
    """Every *event* record of a metrics file, in order."""
    lines = metrics.read_text(encoding="utf-8").splitlines()
    return [r for r in (json.loads(line) for line in lines) if r.get("event") == event]


def steady_steps(
    steps: list[dict[str, Any]], profile: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """The step records of the run's steady state.

    That is every record logged after the profiled window closed -- its
    steps are serialised by the profiler's marks -- and after the last
    record that paid for a recompile. The first record kept is the span's
    starting point, so the steps it covers are only the ones after it.
    """
    start = 0.0
    if profile is not None:
        window = profile["window"]
        if len(window) != 2:
            raise ValueError("the profile record's window never closed")
        start = window[1]
    compiled = [r["time"] for r in steps if r.get("recompiles", 0) > 0]
    if compiled:
        start = max(start, max(compiled))
    steady = [r for r in steps if r["time"] > start]
    if len(steady) < 2:
        raise ValueError(
            f"{len(steady)} step records after the window and the last recompile; a span needs two"
        )
    return steady


def rate(steady: list[dict[str, Any]]) -> dict[str, Any]:
    """Steps/s over the span, and what one rank's batches put through per second.

    Step records describe the logged step's first rank alone, and only every
    ``log_every``-th step is logged, so the per-rank figures are the logged
    steps' mean size times the step rate.
    """
    first, last = steady[0], steady[-1]
    seconds = last["time"] - first["time"]
    steps = last["step"] - first["step"]
    if seconds <= 0 or steps <= 0:
        raise ValueError(f"an empty span: {steps} steps in {seconds:.1f}s")
    per_second = steps / seconds
    covered = steady[1:]

    def mean(field: str) -> float:
        return sum(float(r[field]) for r in covered) / len(covered)

    return {
        "first_step": first["step"],
        "last_step": last["step"],
        "seconds": round(seconds, 1),
        "seconds_per_step": round(seconds / steps, 3),
        "steps_per_second": round(per_second, 4),
        "examples_per_second_per_rank": round(mean("examples") * per_second, 1),
        "padded_tokens_per_second_per_rank": round(
            (mean("tokens") + mean("context_tokens")) * per_second, 1
        ),
    }


def gpu_means(log: Path, since: float, until: float) -> dict[str, Any]:
    """Per GPU over the samples inside the span: means, peak temperature, throttling."""
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in log.read_text(encoding="utf-8").splitlines():
        fields = dict(zip(GPU_FIELDS, (f.strip() for f in line.split(",")), strict=False))
        if len(fields) != len(GPU_FIELDS):
            continue  # the line nvidia-smi was writing when it was stopped
        at = datetime.strptime(fields["timestamp"], "%Y/%m/%d %H:%M:%S.%f").timestamp()
        if since <= at <= until:
            samples[fields["index"]].append(fields)
    if not samples:
        raise ValueError(f"{log} holds no GPU sample inside the span")
    means = {}
    for index, rows in sorted(samples.items()):

        def mean(field: str, rows: list[dict[str, Any]] = rows) -> float:
            return round(sum(float(r[field]) for r in rows) / len(rows), 1)

        throttle = Counter(r["clocks_throttle_reasons.active"] for r in rows)
        means[index] = {
            "utilization": mean("utilization.gpu"),
            "memory_mib": mean("memory.used"),
            "sm_mhz": mean("clocks.sm"),
            "watts": mean("power.draw"),
            "max_celsius": max(int(r["temperature.gpu"]) for r in rows),
            "throttle_share": {k: round(v / len(rows), 2) for k, v in throttle.most_common()},
            "samples": len(rows),
        }
    return means


def cpu_usage(log: Path, since: float, until: float) -> dict[str, Any]:
    """Machine busy percent and each trainer process's cores over the span.

    A process's cores split into its main thread's and the rest's: the main
    thread is the one that launches device work.
    """
    commands = {}
    samples = []
    for line in log.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "command" in record:
            commands[record["pid"]] = record["command"]
        elif since <= record["time"] <= until:
            samples.append(record)
    if len(samples) < 2:
        raise ValueError(
            f"{log} holds {len(samples)} CPU samples inside the span; a usage needs two"
        )
    first, last = samples[0], samples[-1]
    seconds = last["time"] - first["time"]
    hertz = os.sysconf("SC_CLK_TCK")
    before = {(pid, tid): ticks for pid, tid, ticks in first["threads"]}
    cores: dict[int, dict[str, float]] = defaultdict(lambda: {"main": 0.0, "other": 0.0})
    for pid, tid, ticks in last["threads"]:
        used = (ticks - before.get((pid, tid), ticks)) / hertz / seconds
        cores[pid]["main" if tid == pid else "other"] += used
    busiest = sorted(cores.items(), key=lambda kv: -(kv[1]["main"] + kv[1]["other"]))
    return {
        "busy_percent": round(
            100 * (last["busy"] - first["busy"]) / (last["total"] - first["total"]), 1
        ),
        "cores": os.cpu_count(),
        "samples": len(samples),
        "processes": [
            {
                "pid": pid,
                "command": commands.get(pid, "?"),
                "main_cores": round(used["main"], 2),
                "other_cores": round(used["other"], 2),
            }
            for pid, used in busiest
            if used["main"] + used["other"] > 0.01
        ],
    }


#: Host calls that put work on a device: one kernel each, or one whole graph.
KERNEL_LAUNCHES = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernelEx")
GRAPH_LAUNCHES = ("cudaGraphLaunch", "cuGraphLaunch")


def trace_summary(path: Path) -> dict[str, Any]:
    """Where the device worked and idled in a trace, and what the host launched.

    The span runs from the first phase mark to the last. Device time is the
    union of kernel, copy and set intervals. Idle gaps are charged to the
    phase they fall in and to the innermost region annotated on the thread
    that marks the phases. Launches are the host's launch calls on that
    thread, counted per phase and per region: a captured search shows one
    graph launch where eager code shows hundreds of kernel launches.
    """
    events = json.loads(path.read_text(encoding="utf-8"))["traceEvents"]
    marks: list[tuple[float, str]] = sorted(
        (float(e["ts"]), str(e["name"])[len("phase.") :])
        for e in events
        if str(e.get("name", "")).startswith("phase.")
    )
    if len(marks) < 2:
        raise ValueError(f"{path} holds {len(marks)} phase marks; a span needs two")
    main = next(e["tid"] for e in events if str(e.get("name", "")).startswith("phase."))
    start, end = marks[0][0], marks[-1][0]
    phases = [(marks[i - 1][0], marks[i][0], marks[i][1]) for i in range(1, len(marks))]
    regions = [
        (e["ts"], e["ts"] + e["dur"], e["name"])
        for e in events
        if e.get("cat") == "user_annotation"
        and "dur" in e
        and e["tid"] == main
        and not e["name"].startswith("phase.")
    ]

    def phase_at(t: float) -> str:
        for a, b, name in phases:
            if a <= t < b:
                return name
        return "-"

    def region_at(t: float) -> str:
        inside: list[tuple[float, str]] = [(b - a, name) for a, b, name in regions if a <= t < b]
        return min(inside)[1] if inside else "-"

    device = sorted(
        (e["ts"], e["ts"] + e["dur"])
        for e in events
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e
    )
    busy = 0.0
    cursor = start
    idle: dict[tuple[str, str], float] = defaultdict(float)
    for a, b in device:
        if b <= start or a >= end:
            continue
        a, b = max(a, start), min(b, end)
        if a > cursor:
            middle = (cursor + a) / 2
            idle[(phase_at(middle), region_at(middle))] += a - cursor
        if b > cursor:
            busy += b - max(a, cursor)
            cursor = b
    if end > cursor:
        idle[(phase_at((cursor + end) / 2), "-")] += end - cursor

    launches: dict[str, Counter[str]] = {"phase": Counter(), "region": Counter()}
    graphs: dict[str, Counter[str]] = {"phase": Counter(), "region": Counter()}
    for e in events:
        name = e.get("name", "")
        if e.get("tid") != main or not start <= e.get("ts", -1) < end:
            continue
        table = launches if name in KERNEL_LAUNCHES else graphs if name in GRAPH_LAUNCHES else None
        if table is None:
            continue
        table["phase"][phase_at(e["ts"])] += 1
        table["region"][region_at(e["ts"])] += 1

    steps = sum(1 for _, name in marks if name == "optimizer")
    phase_seconds: dict[str, float] = defaultdict(float)
    for a, b, name in phases:
        phase_seconds[name] += (b - a) / 1e6
    return {
        "trace": path.name,
        "steps": steps,
        "span_seconds": round((end - start) / 1e6, 3),
        "device_busy_fraction": round(busy / (end - start), 3),
        "phase_seconds": {k: round(v, 3) for k, v in phase_seconds.items()},
        "kernel_launches_per_step": {
            kind: {k: round(v / steps, 1) for k, v in table.most_common()}
            for kind, table in launches.items()
        },
        "graph_launches_per_step": {
            kind: {k: round(v / steps, 1) for k, v in table.most_common()}
            for kind, table in graphs.items()
        },
        "idle_ms_per_step": [
            {"phase": p, "region": r, "ms": round(v / 1e3 / steps, 1)}
            for (p, r), v in sorted(idle.items(), key=lambda kv: -kv[1])[:12]
        ],
    }


def measure(
    run_dir: Path,
    gpu_log: Path | None = None,
    cpu_log: Path | None = None,
    trace: Path | None = None,
) -> dict[str, Any]:
    """The steady state of the run in *run_dir*, from the files the run left.

    Rates come from the step records; the GPU and CPU logs, when given, are
    cut to the same wall-clock span; the profile record's phase table and
    *trace*, when given, say where the profiled steps spent their time.
    """
    metrics = run_dir / "metrics.jsonl"
    steps = records(metrics, "step")
    profiles = records(metrics, "profile")
    profile = profiles[0] if profiles else None
    steady = steady_steps(steps, profile)
    since, until = steady[0]["time"], steady[-1]["time"]
    report: dict[str, Any] = {
        # The first step's clock includes the compile, when the run compiles.
        "first_step_seconds": steps[0]["seconds"],
        "intervals_in_span": sum(
            1
            for r in records(metrics, "interval")
            if steady[0]["step"] < r["step"] <= steady[-1]["step"]
        ),
        "rate": rate(steady),
    }
    if gpu_log is not None:
        report["gpu"] = gpu_means(gpu_log, since, until)
    if cpu_log is not None:
        report["cpu"] = cpu_usage(cpu_log, since, until)
    if profile is not None:
        report["profile"] = {k: profile[k] for k in profile if k not in ("event", "window")}
    if trace is not None:
        report["trace"] = trace_summary(trace)
    return report
