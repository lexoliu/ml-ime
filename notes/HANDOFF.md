# Handoff — how to run this project without the previous operator

Written 2026-09-16 for whoever (person or model) picks the work up. Everything
here is self-contained; the previous operator's private memory is not needed.
Read this file, then `notes/route-a-v1.md`, `notes/v2-data-prep.md` and
`kaggle/README.md`.

## 0. One-page summary

- **Project**: a neural pinyin input method for macOS. An encoder-only,
  non-autoregressive model (a MacBERT-initialised fill tower plus a context
  tower attached through zero-initialised gated cross-attention) emits a
  per-position distribution over homophones, fused with a Kneser-Ney trigram in
  the beam search. v1 passed the kill gate (fused, context on: 71.2% sentence
  top-1 against the trigram's 55.1%); see `notes/route-a-v1.md`.
- **Running now**: v2, two epochs over all of run3 (41.28M segments),
  244,797 steps on Kaggle 2×T4, as a chain of kernels that resume each other.
  As of 2026-09-16 the run is at 191,350 steps (after segment 4).
- **Fully automatic from here**: launchd on this Mac mini runs
  `kaggle/chain.sh` hourly; it pushes the next segment when the quota allows,
  downloads every finished segment, and evaluates the segment that reaches
  244,797 steps into `results.md`.
- **What a person does**: read `data/route-a-v2/chain.log`, follow section 5
  when something breaks, and turn `results.md` into `notes/route-a-v2.md`
  (section 4).
- **Rules**: never push to `dev` or `main`; one issue per problem, one PR per
  issue targeting `dev`, squash-merged by the operator once CI is green;
  commits must be signed (configured on this machine). Documentation is
  written in English.

## 1. Machines, credentials, tools

- **This machine**: `lexos-mac-mini` (Apple M1, 8 GB). Repo at
  `~/Coding/ml-ime`, branch `dev`. Rust toolchain (cargo 1.98), `uv`, Kaggle
  CLI 2.2.4 (`~/.local/bin/kaggle`), `gh` 2.99 (`~/.local/bin/gh`, logged in
  as lexoliu with `repo`, `read:org`, `gist`, `admin:ssh_signing_key`).
- **Git identity/signing**: `user.name "Lexo Liu"`, `user.email me@lexo.cool`,
  SSH signing with `~/.ssh/id_ed25519` (registered on GitHub as a signing key).
  The repository ruleset requires signed commits; unsigned PRs show "BLOCKED".
- **Kaggle auth**: OAuth in `~/.kaggle/credentials.json`. The access token
  lapses roughly every 12–16 h; right after that every command fails with
  "Permission 'kernels.get' was denied" / "Authentication required". It
  refreshes itself within ~1 h. Retry later before doing anything else. If it
  never recovers: `kaggle auth login` (browser) or save an API token from
  https://www.kaggle.com/settings/api to `~/.kaggle/access_token`.
- **Kaggle quota**: `kaggle quota` prints GPU hours used/remaining and the
  reset time (Saturday 00:00 UTC = Friday 20:00 America/New_York). 30 h/week.
- **Python env**: `cd python && uv sync --extra train`. Rust: `cargo build
  --release -p ime-cli` (target dir `target/`, already warm).
- **Data on this machine** (gitignored `data/`): `run3_pool` (eval3 and the
  abbreviated/mixed twins, annotations), `run3/ngram.bin` (41M-line trigram,
  643 MiB, the baseline), `run3/samples`, `run3-v1/{samples,labels}`,
  `run3-rest/{samples,labels}` (all 404 rest shards labelled),
  `route-a-assets-v2/` (what the training kernel mounts), `route-a-v1/` (the
  v1 kernel output and score files), `route-a-v2/s<n>/` (every harvested
  segment). Nothing under `data/` is on GitHub.

## 2. Kaggle datasets and kernels

| dataset | holds |
|---|---|
| `lexoliu/mlime-src` | the `mlime` Python package from `dev` (must be versioned with `--dir-mode tar`, see §5) |
| `lexoliu/mlime-route-a-assets` | `char_pinyin.tsv`, `emittable.txt`, `syllables.txt`, `typed_spans.txt`, `eval3*.jsonl`, `lattice*.jsonl` |
| `lexoliu/mlime-run3-v1-samples`, `-labels` | the 10.03M v1 subset and its g2pW labels |
| `lexoliu/mlime-run3-rest-samples`, `-labels` | the 31.25M rest of run3 and its labels (404 shards each) |
| `lexoliu/mlime-g2pw-model` | g2pW ONNX (labelling only) |

Kernels: `lexoliu/mlime-route-a-v2-s<n>` is training segment n (script
`kaggle/route-a-v2/kernel.py`, metadata template beside it). Segment n mounts
segment n−1's output as a kernel source and resumes from `checkpoint-paused.pt`.
The first segment counted the exact number of batches (`batch-counts.json`) and
fixed `max_steps = 244797`; every later segment reads it from the previous
`run-config.json`. A segment trains on a wall budget (12 h session − 35 min
reserve; a segment that would reach `max_steps` but not also fit 3 h of scoring
pauses earlier) and writes `checkpoint-paused.pt`; the segment that reaches
`max_steps` writes `checkpoint-final.pt`, evaluates the six held-out shards, and
scores all three lattices with context on and off into
`scores-<lattice>-context-<on|off>.jsonl.gz`, with `"finished": true` in
`run-summary.json`.

## 3. The automatic chain (what runs without anyone)

`~/Library/LaunchAgents/cool.lexo.mlime-chain.plist` runs
`kaggle/chain.sh` every hour (`launchctl list | grep mlime-chain` to see it;
`launchctl unload/load <plist>` to stop/start). Each run, logged one line to
`data/route-a-v2/chain.log`:

1. finds the highest pushed segment n;
2. if it is RUNNING/QUEUED: waits; if ERROR: logs "a person has to look" and stops;
3. if COMPLETE and `data/route-a-v2/s<n>/harvested` is absent: downloads the
   output there (2–5 GB, includes the checkpoint), writes the `harvested` stamp
   once the whole download succeeded (a download that breaks off is redone next
   hour) and logs the summary;
4. if that summary says `"finished": true`: runs `kaggle/finish.sh
   data/route-a-v2/s<n>` (≈1.5 h of CPU) which writes `results.md` there, then
   stops pushing;
5. otherwise pushes segment n+1, but only when `kaggle quota` shows at least
   12 h left (a session started into less is killed before it pauses); a
   refused push (token lapsed) is retried next hour. A segment that ERRORed
   (typically killed by the quota) is re-pushed once a full session of quota is
   back, at most three times, after which the log says "a person has to look".

Expected timeline from 2026-09-16: s4 (4 h session, ends ~21:30 EDT Sep 16,
≈192k steps) → quota reset Fri Sep 18 20:00 EDT → s5 pushed within the hour,
pauses ≈241k after ~8.5 h (leaving the scoring reserve) → s6 pushed, finishes
the last ~4k steps, scores, `finished: true` ≈ Sat Sep 19 afternoon → harvested
and evaluated automatically by Sat evening.

## 4. What a person still does

1. **Watch** `tail data/route-a-v2/chain.log` once a day. "a person has to
   look" means a kernel errored: `kaggle kernels output lexoliu/mlime-route-a-v2-s<n>
   -p /tmp/x` and read `mlime-route-a-v2-s<n>.log` (JSON lines; the Python
   traceback is near the end).
2. **When `results.md` appears** in the finished segment's directory: write
   `notes/route-a-v2.md` in the style of `notes/route-a-v1.md` (training
   table from the segments' `run-summary.json` files, the results table, what
   the numbers say, next steps), via issue → PR → merge (§7). Compare against
   the v1 numbers in `notes/route-a-v1.md` and the trigram baselines in
   `notes/v2-data-prep.md`.
3. **If a segment must be re-run by hand**: `kaggle/README.md` §"Chaining a
   training run" has the exact push recipe; to spend a partial quota week, also
   `sed` `SESSION_SECONDS = 12 * 60 * 60` to the hours you have. Never push a
   new version of a kernel that is queued or running: it does not cancel the
   old one and both burn quota — `kaggle kernels delete -y <slug>` first.

## 5. Gotchas that already cost time (do not rediscover them)

- `kaggle datasets version -p DIR` skips subdirectories unless `--dir-mode tar`;
  `mlime-src` needs it. Verify with `kaggle datasets files <slug> --page-size 200`
  (paginated: follow "Next Page Token"; 200 per page).
- Both the samples and the labels datasets hold shards of the same name; find a
  mount by its parquet columns (`text` vs `syllables`), never by file name.
- Two forked GPU workers must not race to create the package symlink; the
  controller creates it once (already fixed in `kaggle/labels-v2`).
- `kaggle kernels status` for a never-pushed slug says "Cannot access kernel"
  or "404"; anything else that is not a status is an auth/API error.
- Running script kernels expose no logs; you only see them after completion.
- Squash-merging a base branch and deleting it CLOSES PRs stacked on it. Branch
  every PR from `dev`.
- The eval sets' dev/test split hashes the keystrokes (`EvalRecord::digest`), so
  the full/abbreviated/mixed test slices differ by ~20 records (issue #16);
  `--slice all` is the exact twin comparison.
- The v1 score files (`data/route-a-v1/scores-*.jsonl.gz`) were emitted against
  the pre-arbitration character table; the current binary refuses them ("does
  not describe this lattice"). v2 scores are emitted against
  `route-a-assets-v2` and match.

## 6. Evaluating by hand

```
# trigram only, test slice
target/release/ime-cli fused-eval --model data/run3/ngram.bin \
  --eval-set data/run3_pool/eval3.jsonl --emittable data/route-a-assets-v2/emittable.txt --slice test
# neural only
... --no-transition --scores data/route-a-v2/s6/scores-lattice-context-on.jsonl.gz
# fused: sweep on dev, report on test
... --slice dev --scores <scores> --weight 0.5 --weight 0.75 --weight 1 --weight 1.5 --weight 2
... --slice test --scores <scores> --weight <best>
```

`eval3-abbreviated.jsonl` / `eval3-mixed.jsonl` pair with
`scores-lattice-abbreviated-*` / `scores-lattice-mixed-*`. `kaggle/finish.sh
<segment dir>` does all of this and writes `results.md`.

## 7. Repository workflow

- Never push to `dev` or `main`. Branch from `dev`, commit (signed), `git push -u
  origin <branch>`, `gh pr create --base dev` with `Fixes #N`, wait for the four
  CI checks, `gh pr merge <n> --squash --delete-branch`, `git pull --ff-only`.
- Conventional commit messages; release-plz owns versions and changelog (do not
  hand-edit them). `main` is release-only and stays the owner's decision.
- Rust: `cargo clippy -p <crate> --all-targets -- -D warnings`, `cargo test -p
  <crate>`, `cargo fmt`. Python (from `python/`): `uv run ruff check src tests`,
  `uv run ruff format --check src tests`, `uv run mypy src`, `uv run pytest -q`.
  CI also checks that `crates/ime-pinyin/data` matches `mlime gen-pinyin-tables`.

## 8. After v2 — what the previous operator would do next

1. Write `notes/route-a-v2.md` from `results.md` (§4.2) and decide from the
   abbreviated/mixed numbers whether the NAR + trigram design carries
   abbreviations, which is the product-critical case.
2. Fix issue #16 (hash only text+context in `EvalRecord::digest`) so the three
   eval twins share one dev/test split; re-tune weights once.
3. The trigram is still load-bearing: neural-only top-8 barely exceeds top-1.
   An autoregressive rescorer or an iterative refinement pass over the NAR
   output is the next modelling step.
4. Inference on macOS (Core ML / ANE) was explicitly parked until the model was
   trained; `notes/inputmethodkit.md` and `notes/compute.md` hold what was known.
