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
- **Done**: v2, two epochs over all of run3 (41.28M segments), 244,797
  steps on Kaggle 2×T4 as a chain of seven kernels that resumed each other,
  finished 2026-09-19 (segment 6) and evaluated into
  `data/route-a-v2/s6/results.md`; the write-up is `notes/route-a-v2.md`.
- **Nothing runs unattended**: a training leg runs on Colab through
  `colab/e2e.sh` (§3) and is watched by its own `wait`.
- **What a person does**: follow section 5 when something breaks, and turn a
  finished run's evaluations into a note under `notes/` (section 4).
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

## 2. Datasets and kernels

Datasets live only on Hugging Face, as private dataset repos under `lexoliu`
named like the Kaggle slugs below (#136): the Colab staging downloads them at
about 150 MB/s, where Kaggle served its datasets at about 17 MB/s and kernel
outputs at about 12 MiB/s. Push a new version with
`hf upload lexoliu/<name> <dir> --repo-type dataset`; log in once with
`hf auth login` (a browser device flow, no personal access token). The
finished runs a stage starts from are repos too (`lexoliu/mlime-e2e-stage2-s0`:
`run-config.json`, `run-summary.json`, `run/checkpoint-final.pt`), and a Colab
leg's resume trio lives in the private bucket `lexoliu/mlime-resume` under
`<run>/step-<step>/`.

The code is not a dataset: a leg runs one commit of this public repository,
which the VM fetches from GitHub (`colab/e2e.sh start --commit`, default the
checkout's HEAD, refused when uncommitted or not on origin).

| dataset | holds |
|---|---|
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

## 3. Running a leg

`colab/e2e.sh <kernel dir> start` provisions a Colab runtime, stages the leg's
inputs and launches the kernel; `wait --deadline S` watches it and exits with
how the leg ended; `fetch` and `stop` follow (`kaggle/README.md` §Colab). The
kernel is stamped with `mlime e2e stamp`. Nothing outlives its controller: the
runtime's watchdog releases it ten minutes after the last probe.

## 4. What a person still does

1. **When a leg fails** (`wait` exits 12): `fetch` it and read `kernel.out`
   and `train.log`; the traceback is near the end.
2. **When a run finishes**: write its note under `notes/` in the style of
   `notes/route-a-v1.md` (training table from the legs' `run-summary.json`
   files, the results table, what
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
  a dataset laid out in directories needs it. Verify with `kaggle datasets files <slug> --page-size 200`
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
# the character LM as the transition, alone or fused with the trigram
... --lm data/char-lm-run/char-lm --scores <scores> --weight <best>
... --model data/run3/ngram.bin --lm data/char-lm-run/char-lm --lm-weight 1 --scores <scores> --weight <best>
```

`--lm <dir>` points at the `prefill.onnx` + `charlm.onnx` + `charlm.weights`
+ `charlm.json` set that `mlime export char-lm` writes (from a `mlime train char-lm`
checkpoint): the graphs' initializers live in the external weights
file the manifest's `weights` table maps name-by-name (one `charlm.weights`
shared by both graphs, or `charlm.weights` + `prefill.weights` when their
tensor names differ), so `ime-lm` maps it once and shares it across the
per-thread sessions. `prefill.onnx` reads the prelude once per record,
`charlm.onnx` advances the surviving beams a step at a time. Given alone it
replaces the trigram; given with `--model` it is added to the trigram at
`--lm-weight w` (default 1), and the neural `--weight` sweep then runs over
the pair. The decoder keeps one state per beam, so a run with `--lm` is
slower than the trigram by roughly the model's step cost times the beam.

`--dump <dir>` writes every record's beam as JSONL; `mlime eval rescore` reranks
such a dump through the `MLIME_LLM_*` endpoint (`notes/rescore-ceiling.md`), and
`mlime eval generate` asks the same endpoint to write each record's sentence from
the context and keystrokes alone, or with the beam's hypotheses as hints
(`--with-hypotheses`), and scores it against the beam (`notes/generate-ceiling.md`).

`mlime eval rime --dump <dump> --eval-set <set> --library /opt/homebrew/lib/librime.dylib
--data-dir data/rime/user` types the same records into RIME through librime's C API
(`ctypes`, nothing compiled) and reports its default sentence against the beam's;
the data directory is `rime/plum` `:preset` deployed with `rime_deployer --build`,
with `luna_pinyin_simp.custom.yaml` disabling the user dictionary
(`notes/commercial-baselines.md`).

`eval3-abbreviated.jsonl` / `eval3-mixed.jsonl` pair with
`scores-lattice-abbreviated-*` / `scores-lattice-mixed-*`.

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

1. `notes/route-a-v2.md` is written. Its verdict on abbreviations: the fused
   decoder is 3.5× the trigram on fully abbreviated sentences (24.3% vs 7.0%
   top-1) but not usable for them yet, and neural-only top-8 barely exceeds
   top-1 in every setting.
2. Issue #40 measured both answers to that. `notes/rescore-ceiling.md`: a
   GPT-6-class reranker over the beam's eight hypotheses gains 2.5 / 3.6 / 5.4
   points (full / abbreviated / mixed) and is capped by the oracle, which is
   30% on abbreviated input. `notes/char-lm-v1.md`: a 30M-parameter character
   LSTM inside the beam, added to the trigram, gains about 1.2 points on every
   typing style and leaves top-8 where it was. `notes/generate-ceiling.md`:
   128 candidates hold the abbreviated answer 33.5% of the time, and GPT-6
   Luna writing the sentence from the context and keystrokes ties the decoder
   on abbreviated input (23.9 vs 24.3) and reaches 36.7 / 49.2 / 84.7
   (abbreviated / mixed / full) when shown the beam's hypotheses. That is the
   target; the gap is the reader inside the lattice, not the search.
   `notes/char-lm-v2.md`: the 60M-parameter transformer reader (3.70
   nats/char, int8 export) takes the fused decoder to 28.26 / 38.54 / 78.16
   and the abbreviated oracle at 32 to 36.30; the beam's shape is unchanged.
3. Fix issue #16 (hash only text+context in `EvalRecord::digest`) so the three
   eval twins share one dev/test split; re-tune weights once.
4. Issue #51 is measured (`notes/commercial-baselines.md`): RIME through
   librime, and Apple Pinyin, Sogou and Baidu through `macos/ImeDrive` on a
   fresh macOS VM per twin (the twins share sentences and every engine learns
   what it commits, so a second twin on the same machine is contaminated).
   Test top-1 on full / abbreviated / mixed: Apple 60.10 / 11.64 / 18.77,
   Sogou 52.99 / 6.89 / 12.02, Baidu 43.23 / 6.93 / 12.36, RIME 39.63 / 4.75 /
   8.55, against the decoder's 78.16 / 28.26 / 38.54. The decoder leads every
   engine on every style; the target that remains is the hinted-generation
   ceiling of `notes/generate-ceiling.md`. Commercial Chinese input methods
   are installed on cloud VMs only, never on a machine of ours.
5. Inference on macOS (Core ML / ANE) was explicitly parked until the model was
   trained; `notes/inputmethodkit.md` and `notes/compute.md` hold what was known.
