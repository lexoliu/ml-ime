"""``mlime-coreml``: convert a key-model bundle, verify it, package it, measure it.

``convert`` is the pipeline (convert, verify, package), as repo-search's. ``port`` runs the
float32 checks alone and works on any host; ``package`` re-packages models already converted;
``bench`` is the decision measurement on random-weight bundles of the real dimensions.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from mlime_coreml import spec
from mlime_coreml.bundle import MANIFEST, TOKENIZER, read_bundle
from mlime_coreml.port import check_padding, check_port
from mlime_coreml.spec import Family

log = logging.getLogger(__name__)


def _port(args: argparse.Namespace) -> None:
    bundle = read_bundle(args.bundle)
    checks = check_port(bundle) + check_padding(bundle)
    for check in checks:
        log.info("%s %s gap %.3e", "ok  " if check.ok else "FAIL", check.name, check.max_abs)
    if not all(check.ok for check in checks):
        sys.exit("the float32 port does not reproduce the reference")


def _convert(args: argparse.Namespace) -> None:
    from mlime_coreml.convert import convert_family
    from mlime_coreml.package import build_archive
    from mlime_coreml.verify import verify

    work_dir = args.work_dir.resolve()
    build_dir, models_dir = work_dir / "build", work_dir / "models"
    build_dir.mkdir(parents=True, exist_ok=True)
    bundle = read_bundle(args.bundle)
    converted = {
        family: convert_family(bundle, family, build_dir, models_dir) for family in spec.FAMILIES
    }
    verify(bundle, converted, build_dir, work_dir, require_neural_engine=not args.no_neural_engine)
    for family in spec.FAMILIES:
        shutil.rmtree(converted[family].package)
    if build_dir.exists():
        shutil.rmtree(build_dir)
    # `models/` is then what the archive unpacks to, less the report: a loader's directory.
    for name in (MANIFEST, TOKENIZER):
        shutil.copyfile(args.bundle / name, models_dir / name)
    if args.no_package:
        return
    archive, digest = build_archive(
        bundle.manifest, args.bundle, models_dir, work_dir / "verification.json", work_dir
    )
    log.info("wrote %s sha256=%s", archive, digest)


def _package(args: argparse.Namespace) -> None:
    from mlime_coreml.package import build_archive

    work_dir = args.work_dir.resolve()
    bundle = read_bundle(args.bundle)
    archive, digest = build_archive(
        bundle.manifest, args.bundle, work_dir / "models", work_dir / "verification.json", work_dir
    )
    log.info("wrote %s sha256=%s", archive, digest)


def _bench(args: argparse.Namespace) -> None:
    from mlime_coreml.measure import bench

    families: tuple[Family, ...] = tuple(args.family) if args.family else spec.FAMILIES
    bench(read_bundle(args.bundle), args.work_dir.resolve(), families)


def main() -> None:
    """The command line."""
    parser = argparse.ArgumentParser(prog="mlime-coreml")
    commands = parser.add_subparsers(dest="command", required=True)

    def with_bundle(name: str, help_: str) -> argparse.ArgumentParser:
        command = commands.add_parser(name, help=help_)
        command.add_argument("--bundle", required=True, type=Path, help="the bundle directory")
        return command

    port = with_bundle("port", "check the float32 port against the reference (any host)")
    port.set_defaults(run=_port)

    convert = with_bundle("convert", "convert, verify and package a bundle")
    convert.add_argument("--work-dir", required=True, type=Path)
    convert.add_argument(
        "--no-neural-engine",
        action="store_true",
        help="a host without a Neural Engine: skip the placement gate, run every numeric check",
    )
    convert.add_argument("--no-package", action="store_true")
    convert.set_defaults(run=_convert)

    package = with_bundle("package", "archive models already converted under --work-dir")
    package.add_argument("--work-dir", required=True, type=Path)
    package.set_defaults(run=_package)

    bench = with_bundle("bench", "convert each family and measure every function")
    bench.add_argument("--work-dir", required=True, type=Path)
    bench.add_argument("--family", action="append", choices=spec.FAMILIES)
    bench.set_defaults(run=_bench)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.run(args)
