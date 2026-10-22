"""Publish the test fixtures' graphs and weights, and pin them.

Run from the repository root once `charlm_fixtures.py` or `routea_fixtures.py`
has rewritten a set's directory:

    uv run --project python python/scripts/publish_fixtures.py

`crates/ime-fixtures/fixtures.json` names each set's committed directory and
the public Hugging Face dataset repository. Every `.onnx` and `.weights` file
under those directories -- git ignores them -- goes up in one commit, at
`<set>/<path>`, and the manifest is rewritten with that commit and each file's
SHA-256: the pin the `ime-fixtures` build script fetches and verifies. Commit
the manifest with the text files the generator rewrote. Needs `hf auth login`.
"""

import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

from mlime.logging import log

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "crates/ime-fixtures/fixtures.json"

#: The suffixes that leave the tree; the build script skips the same ones when
#: it copies a set's committed files.
SUFFIXES = (".onnx", ".weights")


def main() -> None:
    """Upload every set's graphs and weights and rewrite the pin."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    operations = []
    for name, entry in manifest["sets"].items():
        committed = REPO / entry["committed"]
        files = sorted(path for path in committed.rglob("*") if path.suffix in SUFFIXES)
        if not files:
            raise SystemExit(f"{committed} holds no graphs or weights; run its generator first")
        entry["files"] = {
            path.relative_to(committed).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        }
        operations += [
            CommitOperationAdd(f"{name}/{path.relative_to(committed).as_posix()}", path)
            for path in files
        ]

    api = HfApi()
    api.create_repo(manifest["repository"], repo_type="dataset", private=False, exist_ok=True)
    commit = api.create_commit(
        manifest["repository"],
        operations,
        commit_message="Publish the test fixtures",
        repo_type="dataset",
    )
    manifest["revision"] = commit.oid
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log.info("published fixtures", repository=manifest["repository"], revision=commit.oid)


if __name__ == "__main__":
    main()
