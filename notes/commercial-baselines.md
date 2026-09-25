# Commercial baselines on the eval3 twins (2026-09-25)

Issue #51. Every number the project reports is relative to its own trigram
(`notes/route-a-v2.md`, `notes/char-lm-v1.md`) or to the GPT-6 ceiling
(`notes/generate-ceiling.md`). This note places the engines a Mac user can
install on the same test slices, with the same records and the same
definitions. Each engine gets the record's keystrokes and nothing else: the
sentence scored is what a user gets by accepting the default candidate
until the input is consumed.

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

These engines expose no API. The harness for them types through CGEvent
into a text view, reads the candidate window through the accessibility
tree and accepts the default until the input is consumed; it needs the
engines installed, their input sources enabled, and Accessibility granted
to the driver process, which are actions on the operator's Mac. Their rows
join this table when that is done.

## Reproduce

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
