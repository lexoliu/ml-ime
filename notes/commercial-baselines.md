# Commercial baselines on the eval3 twins (2026-09-25, GUI engines 2026-09-27)

Issue #51. Every number the project reports is relative to its own trigram
(`notes/route-a-v2.md`, `notes/char-lm-v1.md`) or to the GPT-6 ceiling
(`notes/generate-ceiling.md`). This note places the engines a Mac user can
install on the same test slices, with the same records and the same
definitions. Each engine gets the record's keystrokes and nothing else: the
sentence scored is what a user gets by accepting the default candidate
until the input is consumed. RIME is bound through its library; Apple
Pinyin, Sogou and Baidu are driven through the screen on a fresh macOS VM
per twin (issue #59).

## RIME, luna_pinyin_simp (issue #58)

`mlime eval rime` binds librime 1.17.0 (Homebrew) through its C API with
`ctypes`, deploys `rime/plum` `:preset` into `data/rime/user` with
`luna_pinyin_simp.custom.yaml` (`translator/enable_user_dict: false`, so
nothing is learnt between records; `menu/page_size: 10`), and types each
record into a fresh session. The first page of candidates is kept as well;
"first page exact" is the share of records whose expected sentence is one
of them. Characters are scored positionally when the committed sentence has
the expected length, by edit distance otherwise, as `mlime eval generate`
does.

| twin | records | RIME top-1 | first page exact | characters right | length mismatches | decoder (route A v2 + char LM) | trigram alone |
|---|---|---|---|---|---|---|---|
| full | 5,027 | **39.63%** | 39.63% | 82.51% | 24 | 77.50% | 55.10% |
| abbreviated | 5,050 | **4.75%** | 4.83% | 39.37% | 571 | 25.54% | 7.01% |
| mixed | 5,041 | **8.55%** | 8.59% | 50.85% | 474 | 35.39% | 16.19% |

Among the committed sentences of the expected length (5,003 / 4,479 /
4,567 records) the positional character accuracy is 82.66 / 41.44 / 52.68%.
The whole run takes about a minute per twin.

What the numbers say. RIME's default schema is a dictionary and a small
essay-based language model, with no corpus n-gram behind it and no view of
the context, and it shows: on full pinyin it converts four sentences in ten
where the run3 trigram alone converts five and a half and the decoder
nearly eight; on abbreviated input it accepts initials (`nkykk` →
你可以看看) but ranks them by frequency alone and gets one sentence in
twenty. It is the floor of the comparison, not the target; the target is
what Sogou, Baidu and Apple's engine do with a corpus model and a cloud
behind them.

## Apple Pinyin, Sogou, Baidu (issue #59)

These engines expose no API. `macos/ImeDrive` (Swift) types each record's
keystrokes through `CGEvent` into a text view of its own process with the
engine's input source selected, reads the candidate window through the
accessibility tree, accepts the default candidate until the composition is
consumed, and journals one JSON line per record; `mlime eval gui` scores
the journal with the same `evaluate` as the RIME row. Commercial Chinese
input methods are never installed on a machine of ours: every run was a
Devin macOS VM (macOS 26.5.2, Apple silicon), one twin per fresh VM.

One twin per fresh engine state is the protocol, and it was learnt the hard
way: the three twins are the same sentences typed three ways, and every
engine learns each sentence it commits into its personal dictionary, so a
twin typed after another on the same machine recalls the sentences it just
saw. Sogou's abbreviated twin typed after its full-pinyin twin scored 48.42%
(record 0, `zxmlnd`, came back as the exact sentence); on a fresh VM it
scores 6.89%. The harness records the engine's learned-dictionary state in
the journal's meta line so a report says what produced it.

Engine settings are the defaults after installation from the official
installer: Sogou 6.25.1 with cloud candidates and local learning on and no
account (its kernel store counted 184k cloud requests during the runs);
Baidu 6.0.3 (6.0.5 on the full twin) with its custom-drawn candidate panel,
which exposes no accessibility text, so its "first page exact" cannot be
read and its candidate window is detected by window-server geometry; Apple
Pinyin (SCIM/ChineseIM 104) as shipped. The VMs had internet. A record is
given 20 s; a record over budget is journaled as a failure and scored as a
miss (3 / 0 / 0 for Baidu's twins, 1 / 0 / 0 for Apple's, 3 / 0 / 0 for
Sogou's).

| twin | engine | top-1 | first page exact | characters right | length mismatches | s / record |
|---|---|---|---|---|---|---|
| full | Apple Pinyin | **60.10%** | 63.40% | 90.49% | 31 | 1.11 |
| full | Sogou | 52.99% | 56.93% | 85.09% | 203 | 1.06 |
| full | Baidu | 43.23% | | 81.43% | 203 | 1.06 |
| full | RIME | 39.63% | 39.63% | 82.51% | 24 | |
| abbreviated | Apple Pinyin | **11.64%** | 13.23% | 51.46% | 957 | 0.87 |
| abbreviated | Baidu | 6.93% | | 38.90% | 569 | 0.63 |
| abbreviated | Sogou | 6.89% | 9.82% | 37.25% | 801 | 0.72 |
| abbreviated | RIME | 4.75% | 4.83% | 39.37% | 571 | |
| mixed | Apple Pinyin | **18.77%** | 20.43% | 61.45% | 744 | 0.84 |
| mixed | Baidu | 12.36% | | 51.14% | 535 | 0.89 |
| mixed | Sogou | 12.02% | 16.23% | 50.29% | 635 | 0.79 |
| mixed | RIME | 8.55% | 8.59% | 50.85% | 474 | |

Against the decoder (`notes/char-lm-v2.md`, route A v2 + the transformer
character LM, context on, test slices): 78.16 / 28.26 / 38.54% top-1 and
95.87 / 67.90 / 76.23% characters on full / abbreviated / mixed; the trigram
alone 55.10 / 7.01 / 16.19%.

What the numbers say. On full pinyin the best commercial engine, Apple's,
converts six sentences in ten and the decoder nearly eight: the context on
screen, which no engine sees, is worth eighteen points there. On
abbreviated and mixed input every engine is far below the decoder (11.6 and
18.8% against 28.3 and 38.5%): Sogou and Baidu accept initials but rank
them by frequency and get one sentence in fourteen, about what RIME gets;
Apple's engine is the only one that does better than a lexicon lookup. The
gap the product has to close is therefore not to the commercial engines but
to the hinted-generation ceiling of `notes/generate-ceiling.md` (36.7 / 49.2
/ 84.7); against the engines a user can install today the decoder already
leads on all three typing styles, by 18 points on full pinyin and by 17 to
20 on the abbreviated styles, with the caveat that the engines were
measured without context and without a personal dictionary, which is the
comparison the product's first-day user faces.

## Reproduce

RIME:

```
brew install librime
git clone --depth 1 https://github.com/rime/plum.git data/rime/plum
rime_dir=data/rime/user bash data/rime/plum/rime-install :preset
printf 'patch:\n  translator/enable_user_dict: false\n  menu/page_size: 10\n' > data/rime/user/luna_pinyin_simp.custom.yaml
(cd data/rime/user && rime_deployer --add-schema luna_pinyin_simp && rime_deployer --build . . build)
mlime eval rime --dump data/rescore-v2/eval3-abbreviated/neural-w1.500-kn-trigram-test.jsonl \
  --eval-set data/run3_pool/eval3-abbreviated.jsonl \
  --library /opt/homebrew/lib/librime.dylib --data-dir data/rime/user --out abbreviated.json
```

The dump names the test slice's records (`fused-eval --dump`, as in
`notes/rescore-ceiling.md`); the full and mixed twins use their own dumps
and eval sets.

The GUI engines, on a fresh macOS VM per twin (`macos/ImeDrive/README.md`
has the install and enable steps per engine and the fresh-state rule):

```
macos/ImeDrive/build.sh
macos/ImeDrive/build/ImeDrive.app/Contents/MacOS/ime-drive --engine sogou \
  --eval-set data/run3_pool/eval3-abbreviated.jsonl --slice test --out data/gui/sogou/eval3-abbreviated.jsonl
mlime eval gui --engine sogou --results data/gui/sogou/eval3-abbreviated.jsonl \
  --eval-set data/run3_pool/eval3-abbreviated.jsonl --out sogou-eval3-abbreviated.json
```

The journal is resumable: started again with the same `--out` path the
driver skips the records already in it. The journals and reports behind the
table are under `data/gui/` (`<engine>-<twin>/`).
