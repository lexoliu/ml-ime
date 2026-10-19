"""The e2e kernel's pip installs are the lockfile's, package for package.

The kernel installs its `REQUIREMENTS` with `--no-deps` beside the Colab
image's torch, and the stager installs the same list before it imports the
Hub client, so the list must be exactly what `uv.lock` resolves: every
package pinned at the lock's version, and every Linux dependency of a pinned
package pinned too. A drift fails here instead of moving a tokenizer -- and
with it the batches, the census and `max_steps` -- between two legs.
"""

import ast
import importlib.util
import re
import tomllib
from pathlib import Path

from packaging.markers import Marker
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "kaggle" / "e2e" / "kernel.py"
LOCK = ROOT / "python" / "uv.lock"

#: The Colab image the kernel installs into: CPython 3.13 on x86_64 Linux
#: (the October image's paths read `python3.13`).
VM = {
    "sys_platform": "linux",
    "platform_system": "Linux",
    "platform_machine": "x86_64",
    "python_version": "3.13",
    "python_full_version": "3.13.0",
    "implementation_name": "cpython",
    "platform_python_implementation": "CPython",
    "os_name": "posix",
    "extra": "",
}
#: Wheel tags pip on that image installs without building anything.
INSTALLABLE = re.compile(
    r"-(cp313-cp313|cp3\d+-abi3|py3-none|py2\.py3-none)-([^-]*manylinux[^-]*x86_64[^-]*|any)\.whl$"
)


def pins() -> dict[str, str]:
    for node in ast.parse(KERNEL.read_text()).body:
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == [
            "REQUIREMENTS"
        ]:
            pinned: dict[str, str] = {}
            for requirement in ast.literal_eval(node.value):
                name, separator, version = requirement.partition("==")
                assert separator, f"{requirement} is not pinned to one version"
                pinned[canonicalize_name(name)] = version
            return pinned
    raise AssertionError(f"{KERNEL} holds no REQUIREMENTS")


def locked() -> dict[str, dict[str, object]]:
    with LOCK.open("rb") as handle:
        lock = tomllib.load(handle)
    return {canonicalize_name(package["name"]): package for package in lock["package"]}


def test_every_pin_is_the_version_the_lock_resolves() -> None:
    packages = locked()
    drifted = {
        name: (version, packages[name]["version"] if name in packages else None)
        for name, version in pins().items()
        if name not in packages or packages[name]["version"] != version
    }
    assert drifted == {}, "kernel pin vs uv.lock"


def test_every_linux_dependency_of_a_pin_is_pinned_and_torch_is_the_images() -> None:
    packages = locked()
    pinned = pins()
    missing = set()
    for name in pinned:
        for dependency in packages[name].get("dependencies", []):  # type: ignore[union-attr]
            marker = dependency.get("marker")
            if marker is not None and not Marker(marker).evaluate(VM):
                continue
            if canonicalize_name(dependency["name"]) not in pinned:
                missing.add(f"{dependency['name']} (of {name})")
    assert missing == set()
    assert "torch" not in pinned
    for root in ("transformers", "tokenizers", "polars", "huggingface-hub"):
        assert root in pinned


def test_every_pin_installs_from_a_wheel_on_the_vm() -> None:
    packages = locked()
    without = [
        name
        for name in pins()
        if not any(
            INSTALLABLE.search(wheel["url"])  # type: ignore[index]
            for wheel in packages[name].get("wheels", [])  # type: ignore[union-attr]
        )
    ]
    assert without == []


def test_the_stager_installs_the_kernels_pins() -> None:
    spec = importlib.util.spec_from_file_location("e2e_stage", ROOT / "colab" / "e2e-stage.py")
    assert spec is not None and spec.loader is not None
    stager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stager)
    assert {
        canonicalize_name(r.partition("==")[0]): r.partition("==")[2]
        for r in stager.requirements(KERNEL)
    } == pins()
