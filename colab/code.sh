# Sourced by colab/e2e.sh: the code a Colab leg runs.
#
# A leg runs one commit of the public repository, which the VM fetches from
# GitHub with no token. `leg_commit REPO [REV]` names that commit -- REV, or
# REPO's HEAD -- and refuses a checkout with uncommitted changes (what runs
# would not be what is in front of you) and a commit no branch of origin
# holds (the VM could not fetch it).
#
# A ranking stage also reads segmentations from `ime-segment`, a Rust binary.
# `kernel_ranks KERNEL` says whether a stamped kernel is one, and
# `build_segmenter REPO COMMIT` builds the binary from that commit for the
# VM and prints its path. The target is static musl: it runs on the Colab
# image's glibc x86_64 Linux with no shared library to match, and rust-lld
# links it here without a cross toolchain (.cargo/config.toml).

segmenter_target=x86_64-unknown-linux-musl

leg_commit() {
  local repo=$1 rev=${2:-HEAD} sha
  if [ -n "$(git -C "$repo" status --porcelain)" ]; then
    echo "$repo has uncommitted changes; a leg runs a commit, so commit them first" >&2
    return 2
  fi
  if ! sha=$(git -C "$repo" rev-parse --verify --quiet "$rev^{commit}"); then
    echo "$rev names no commit in $repo" >&2
    return 2
  fi
  # The remote-tracking branches are only as new as the last fetch.
  git -C "$repo" fetch --quiet origin || return 2
  if [ -z "$(git -C "$repo" branch -r --list 'origin/*' --contains "$sha")" ]; then
    echo "$sha is on no branch of origin, so the VM cannot fetch it; push it first" >&2
    return 2
  fi
  echo "$sha"
}

kernel_ranks() {
  python3 - "$1" <<'EOF'
import ast, sys
tree = ast.parse(open(sys.argv[1]).read())
[value] = [
    node.value for node in tree.body
    if isinstance(node, ast.Assign) and [t.id for t in node.targets if isinstance(t, ast.Name)] == ["RANKING"]
]
sys.exit(0 if ast.literal_eval(value) is not None else 1)
EOF
}

build_segmenter() {
  local repo=$1 commit=$2 built
  if [ "$(git -C "$repo" rev-parse HEAD)" != "$commit" ]; then
    echo "ime-segment is built from the leg's commit; check out $commit to start a ranking leg" >&2
    return 2
  fi
  built=$(cd "$repo" && cargo build --release --locked -p ime-pinyin --bin ime-segment \
    --target "$segmenter_target" --message-format=json-render-diagnostics) || return 2
  python3 -c '
import json, sys
[path] = [
    record["executable"]
    for record in map(json.loads, sys.stdin)
    if record.get("reason") == "compiler-artifact" and record["target"]["name"] == "ime-segment"
]
print(path)' <<<"$built"
}
