# The knowledge slice (2026-10-21, #207)

## Verdict

The slice exists, and eval3 is too small for it. The test slices hold 518 / 512 / 522 records with an entity (full / abbreviated / mixed), about half the ~1,000 #191 set as the bar: a 2-point difference does not separate at z 1.96 at this size. **A new knowledge set (#191's K2) is needed.**

What the slice already shows:

- **The hosted generators lead, mostly outside the entities on full pinyin.** With context, GPT-6 Luna (hinted) converts 88.03% of the slice's full-pinyin records against the fused decoder's 80.12% and #132's 79.15% (47 gained, 6 lost against fused, z 5.63). At the entities themselves the gap is small: 98.07% exact against 96.67% and 95.97% (12 / 4, z 2.00). On abbreviated and mixed input the entity lead is real: 69.07% against 63.97% (z 3.26) and 80.59% against 75.56% (z 3.78).
- **swe-2-high leads further,** on its record % 5 subset (102–107 slice records a twin): 93.46 / 50.98 / 55.66% slice top-1 against the fused decoder's 79.44 / 26.47 / 39.62% (z 3.64 / 4.64 / 3.40), and #132's 75.70 / 24.51 / 36.79%. Its entity exact is 100.00 / 80.91 / 82.14% against 97.32 / 64.55 / 76.79% for fused (z 1.73 / 3.84 / 1.60). Luna on the same subset sits between them: 87.85 / 36.27 / 49.06%.
- **#132 is level with the fused decoder** over the whole slice, by records and by entities, with context on and off. The one exception is a gain at the entities on abbreviated input without context (50.97% against 47.98%, z 2.02).
- **The commercial engines know idioms by their initials, and the fused decoder does not.** Without context, Apple Pinyin spells 78.85% of the abbreviated twin's idioms and RIME 77.88%, against the fused decoder's 40.38% (entities z 5.35 and 5.46); on the mixed twin Apple's 80.00% against 51.82% (z 5.40). Apart from idioms the fused decoder leads every engine. By records on full pinyin, it leads by 9 to 14 z.

## The slice

`ime_eval::KnowledgeSlice`, beside the ambiguity slice and on the same machinery. A span of a record's expected text is an **entity** when:

1. it matches an entry of the knowledge lexicon, the entry's category kept;
2. the keystrokes typed for it admit another reading of the same length: some character of the span sits on a segment that also admits another character. The segments come from `SegmentLattice::cheapest_reading`, as for #150.

Matching is by string, over every start and length. **Where two matches nest, only the outer one is an entity** (北京大学 is one organisation, not an organisation holding a place). Where two overlap without nesting, both are entities, and their shared characters count once in the totals and once in each category that covers them.

The second condition removes nothing on eval3: 518 / 512 / 522 records hold a match before it and after it. The lexicon has 41,923 characters, so every syllable and every initial admits more than one. The condition stays as the definition's guard, and a unit test checks it against a lexicon where it does remove a span.

The metrics:

- **slice top-1**: sentence top-1 over the records holding an entity;
- **entity exact**: the top hypothesis has every character of the entity, positionally, as `ime-eval`'s character accuracy compares them;
- **characters**: character accuracy inside the entities;
- McNemar z against the fused decoder, by records and by entities (and by characters in the reports, where the characters of one entity are not independent).

Every number is per category as well as for the whole slice. The dev/test split is the usual hash at `--dev-share 0.0905`, and every table here is the test slice. No record of any slice is unspelled.

### The lexicon

`mlime lexicon knowledge` writes `data/lexicon/knowledge.tsv`, one `word<TAB>category` line per entry, 102,306 entries:

| category | sources | entries |
|---|---|---|
| person | Sogou 明星【官方推荐】, 歌手人名大全【官方推荐】, THUOCL_历史人名 | 18,241 |
| place | Sogou 全国县及县以上行政区划地名, 中国风景名胜, 世界所有国家及其首都 | 4,489 |
| organisation | Sogou 政府机关团体机构大全【官方推荐】, 中国高等院校（大学）大全【官方推荐】 | 36,817 |
| idiom | Sogou 成语俗语【官方推荐】, 三字成语【官方推荐】 | 40,687 |
| internet | 梗百科 (`data/gengbaike/`) | 2,072 |

Each Sogou list is the official recommendation of its category on pinyin.sogou.com, or the most downloaded one where the category has none. `mlime lexicon fetch` fetches them.

An entry counts when:

- it is **at least two Han characters** long. A single character is never knowledge: its keystrokes are one syllable, and what goes there is the language model's choice. Latin letters and digits cannot be typed as pinyin;
- the **ordinary vocabulary does not list it as an ordinary word.** The ordinary vocabulary is jieba's general dictionary (349k words, last changed 2014-02-07, pinned by commit and checksum). An entry jieba files under a common-word tag (noun, verb, adjective...) is read as that word in running text: the singer 苏醒 is the verb 苏醒, the band 面包 is bread. An entry jieba files as a proper name, idiom, set phrase or abbreviation (`nr nrt nrfg ns nt nz i l j`) counts. So does an entry jieba lacks: a 2014 dictionary lacks exactly the newer names and memes;
- it is **the first listing of its word**, in the table's order (names before phrases). This only decides the category of a word several lists hold.

A title's trailing disambiguator (`长城（电影）`) is removed before the rules apply.

**Left out, with the evidence.** The title lists that #191 named, zh.wikipedia and moegirl article titles, cannot be used. Neither can Sogou's film, TV and anime title lists, nor its mined new-word list. Their entries include ordinary words and whole ordinary phrases, because songs, films and articles are named with them. The rule above leaves those phrases in, since jieba lists no phrase. Over all 5,525 records of eval3:

| source | entries | records matched | after the ordinary-word rule | most frequent matches after it |
|---|---|---|---|---|
| zh.wikipedia titles (2026-10, simplified) | 1,751,442 | 4,877 | 2,382 | 了一, 我的, 是我, 美国, 给我, 都会 |
| moegirl titles (2025-10) | 80,112 | 2,741 | 1,047 | 中国, 美国, 啊啊啊啊啊啊啊啊啊啊啊啊, 给你, 东西 |
| Sogou 热门电影大全 | 60,347 | 2,225 | 600 | 我的, 美国, 不知道, 我要, 城市 |
| Sogou 网络流行新词 | 125,004 | 2,900 | 1,305 | 来了, 美国, 中国, 是你, 日本 |

jieba's tags cannot rescue them. It files 东西 and 太阳 as places and 谢谢, 阳光 and 小姐 as people, so keeping only the entries it tags as proper names keeps those words too. Any rule that separates 不知道 from 米哈游 is a language model's judgement, and a slice chosen by a language model measures that model. 哔哩网梗 (24 entries: 哈哈哈, 好的, 呵呵, 收到, 无语) is left out as interjections. The works category is therefore empty.

### Size

Test slices, under the fused decoder's dumps (every system covers the same records):

| twin | records | entities | characters | person | place | organisation | idiom | internet |
|---|---|---|---|---|---|---|---|---|
| full | 518 | 571 | 1,580 | 241 / 257 | 168 / 181 | 5 / 5 | 103 / 106 | 22 / 22 |
| abbreviated | 512 | 569 | 1,560 | 234 / 251 | 169 / 186 | 8 / 8 | 101 / 104 | 20 / 20 |
| mixed | 522 | 577 | 1,596 | 239 / 256 | 166 / 181 | 7 / 7 | 107 / 110 | 23 / 23 |

Per category: records / entities. Of 5,027 / 5,050 / 5,041 test records, about one in ten holds an entity.

### What the entities are

Full pinyin, test:

- **person** (233 distinct): 姚明, 周杰伦, 林俊杰, 刘亦菲, 赵本山, 朱棣, 唐玄宗, 白崇禧, 王宝强, 今川义元... A read of 60 at random found 6 homographs of ordinary words that jieba files as names (高飞, 张狂, 丁香, 武德, 上洛, 行一), and a few more common names that may not name anyone in their sentence (何平, 叶青).
- **place** (60 distinct): mostly countries, 中国 36 times and 美国 23 of 181. Every system spells them, so the category's z is mostly 0.
- **idiom** (103 distinct): 成语 and set phrases, 指鹿为马, 魑魅魍魉, 入木三分, 乱七八糟, 随时随地, plus a few common three-character phrases (东道主, 自己人) from 三字成语.
- **internet** (21 distinct): 心巴, 虾仁猪心, 啊对对对, 干净又卫生, 打工人, 整活, 老司机, 么么哒, 给力.
- **organisation** (5): 浙江大学, 湖北大学, 中央民族大学, 天河区政府, 北京奥组委.

## The systems

| system | dumps | context |
|---|---|---|
| fused | route A v2 + run3 trigram + int8 transformer reader, weight chosen on dev; the test dumps of `notes/ambiguity-slice.md` under `data/ambiguity-slice/fused/` | on and off |
| #132 | `data/e2e-v2-rank-g4/final/e2e-lattice*-context-{on,off}.jsonl` | on and off |
| Luna, hinted | GPT-6 Luna writing the sentence from the context, the keystrokes and the top-8 of the route A v2 + trigram beam (`notes/generate-ceiling.md`); `data/generate-v1/*-hints.answers.jsonl` | on |
| swe-2-high, hinted | the same prompt, on the records with `record % 5 == 0` of each test slice; `data/generate-swe2/` | on |
| Apple Pinyin, Sogou, Baidu | the `data/gui/` journals (`notes/commercial-baselines.md`) | none |
| RIME | `data/rime/results/` | none |

The hinted generators see the context, so their rows sit with context on. The engines and RIME see none, so they are compared with the fused decoder's context-off dumps.

## Results

Test slices. "gained / lost" counts what the system gets right and the fused decoder wrong, and the reverse. z is McNemar's statistic over the two.

### Context on

| style | system | slice top-1 | entity exact | characters | records: gained / lost, z | entities: gained / lost, z |
|---|---|---|---|---|---|---|
| full | fused | 80.12% | 96.67% | 98.04% | | |
| full | #132 | 79.15% | 95.97% | 96.90% | 24 / 29, −0.69 | 8 / 12, −0.89 |
| full | Luna, hinted | **88.03%** | **98.07%** | **98.86%** | 47 / 6, 5.63 | 12 / 4, 2.00 |
| abbreviated | fused | 25.59% | 63.97% | 69.62% | | |
| abbreviated | #132 | 26.37% | 66.08% | **71.15%** | 26 / 22, 0.58 | 35 / 23, 1.58 |
| abbreviated | Luna, hinted | **34.96%** | **69.07%** | 70.45% | 66 / 18, 5.24 | 54 / 25, 3.26 |
| mixed | fused | 37.36% | 75.56% | 76.82% | | |
| mixed | #132 | 36.59% | 75.74% | 78.13% | 24 / 28, −0.55 | 20 / 19, 0.16 |
| mixed | Luna, hinted | **48.47%** | **80.59%** | **79.95%** | 78 / 20, 5.86 | 44 / 15, 3.78 |

Luna's character accuracy on abbreviated input is below #132's although its entity exact is higher. 584 of its 5,050 abbreviated answers have another length than the wanted sentence (20 on full pinyin), and that shifts every later position.

### The swe-2-high subset, context on

Each slice narrowed to the records swe-2-high answered (`ime-knowledge --within`), so every row is on the same 107 / 102 / 106 records:

| style | system | slice top-1 | entity exact | characters | records: gained / lost, z | entities: gained / lost, z |
|---|---|---|---|---|---|---|
| full | fused | 79.44% | 97.32% | 98.15% | | |
| full | #132 | 75.70% | 93.75% | 95.38% | 4 / 8, −1.15 | 1 / 5, −1.63 |
| full | Luna, hinted | 87.85% | 98.21% | 99.08% | 12 / 3, 2.32 | 3 / 2, 0.45 |
| full | swe-2-high, hinted | **93.46%** | **100.00%** | **100.00%** | 16 / 1, 3.64 | 3 / 0, 1.73 |
| abbreviated | fused | 26.47% | 64.55% | 70.79% | | |
| abbreviated | #132 | 24.51% | 66.36% | 72.38% | 4 / 6, −0.63 | 6 / 4, 0.63 |
| abbreviated | Luna, hinted | 36.27% | 70.91% | 73.65% | 15 / 5, 2.24 | 11 / 4, 1.81 |
| abbreviated | swe-2-high, hinted | **50.98%** | **80.91%** | **83.49%** | 27 / 2, 4.64 | 20 / 2, 3.84 |
| mixed | fused | 39.62% | 76.79% | 77.09% | | |
| mixed | #132 | 36.79% | 74.11% | 75.54% | 3 / 6, −1.00 | 4 / 7, −0.90 |
| mixed | Luna, hinted | 49.06% | 76.79% | 76.78% | 17 / 7, 2.04 | 8 / 8, 0.00 |
| mixed | swe-2-high, hinted | **55.66%** | **82.14%** | **82.66%** | 21 / 4, 3.40 | 10 / 4, 1.60 |

### Context off

| style | system | slice top-1 | entity exact | characters | records: gained / lost, z | entities: gained / lost, z |
|---|---|---|---|---|---|---|
| full | fused | **75.68%** | **94.40%** | **96.58%** | | |
| full | #132 | 73.94% | 93.70% | 95.51% | 30 / 39, −1.08 | 11 / 15, −0.78 |
| full | Apple Pinyin | 54.83% | 92.12% | 95.13% | 20 / 128, −8.88 | 19 / 32, −1.82 |
| full | Sogou | 40.35% | 82.49% | 87.85% | 16 / 199, −12.48 | 10 / 78, −7.25 |
| full | Baidu | 33.20% | 82.14% | 88.23% | 10 / 230, −14.20 | 16 / 86, −6.93 |
| full | RIME | 36.29% | 82.49% | 88.67% | 13 / 217, −13.45 | 13 / 81, −7.01 |
| abbreviated | fused | 13.28% | 47.98% | 53.65% | | |
| abbreviated | #132 | **15.62%** | **50.97%** | **57.56%** | 27 / 15, 1.85 | 44 / 27, 2.02 |
| abbreviated | Apple Pinyin | 7.23% | 44.64% | 52.50% | 22 / 53, −3.58 | 77 / 96, −1.44 |
| abbreviated | Sogou | 3.52% | 31.11% | 41.03% | 10 / 60, −5.98 | 59 / 155, −6.56 |
| abbreviated | Baidu | 3.71% | 35.50% | 44.87% | 12 / 61, −5.74 | 63 / 134, −5.06 |
| abbreviated | RIME | 4.88% | 44.82% | 53.40% | 12 / 55, −5.25 | 76 / 94, −1.38 |
| mixed | fused | **30.08%** | **68.98%** | 70.24% | | |
| mixed | #132 | 27.97% | **68.98%** | **71.93%** | 24 / 35, −1.43 | 23 / 23, 0.00 |
| mixed | Apple Pinyin | 17.62% | 63.43% | 68.73% | 25 / 90, −6.06 | 41 / 73, −3.00 |
| mixed | Sogou | 8.43% | 49.74% | 55.95% | 11 / 124, −9.73 | 23 / 134, −8.86 |
| mixed | Baidu | 9.20% | 51.47% | 58.65% | 13 / 122, −9.38 | 30 / 131, −7.96 |
| mixed | RIME | 7.85% | 58.41% | 63.72% | 11 / 127, −9.87 | 44 / 105, −5.00 |

### Per category

Entity exact, with the entities z against the fused decoder in parentheses. Organisation (5–8 entities a twin) is in the reports and is left out here.

Context on:

| style | system | person | place | idiom | internet |
|---|---|---|---|---|---|
| full | fused | 95.72% | 98.90% | 94.34% | 100.00% |
| full | #132 | 95.33% (−0.33) | 98.90% (0.00) | 92.45% (−0.71) | 95.45% (−1.00) |
| full | Luna, hinted | 97.28% (1.26) | 98.90% (0.00) | 98.11% (2.00) | 100.00% (0.00) |
| abbreviated | fused | 59.36% | 76.88% | 52.88% | 45.00% |
| abbreviated | #132 | 64.54% (2.71) | 76.34% (−0.26) | 53.85% (0.26) | 45.00% (0.00) |
| abbreviated | Luna, hinted | 67.73% (3.28) | 80.65% (1.46) | 52.88% (0.00) | 50.00% (1.00) |
| mixed | fused | 75.00% | 88.95% | 58.18% | 56.52% |
| mixed | #132 | 76.17% (0.73) | 86.19% (−1.67) | 60.91% (0.90) | 56.52% (0.00) |
| mixed | Luna, hinted | 82.42% (3.53) | 90.61% (1.00) | 62.73% (1.21) | 65.22% (1.00) |

Context off:

| style | system | person | place | idiom | internet |
|---|---|---|---|---|---|
| full | fused | 92.22% | 97.79% | 92.45% | 100.00% |
| full | #132 | 92.22% (0.00) | 97.24% (−1.00) | 92.45% (0.00) | 86.36% (−1.73) |
| full | Apple Pinyin | 84.05% (−3.45) | 97.79% (0.00) | 100.00% (2.83) | 100.00% (0.00) |
| full | Sogou | 69.65% (−7.37) | 91.71% (−2.84) | 93.40% (0.30) | 100.00% (0.00) |
| full | Baidu | 71.60% (−6.57) | 88.95% (−3.27) | 92.45% (0.00) | 95.45% (−1.00) |
| full | RIME | 70.82% (−6.93) | 91.16% (−2.83) | 97.17% (1.89) | 72.73% (−2.45) |
| abbreviated | fused | 35.46% | 70.43% | 40.38% | 25.00% |
| abbreviated | #132 | 40.24% (2.35) | 70.43% (0.00) | 44.23% (0.82) | 35.00% (1.41) |
| abbreviated | Apple Pinyin | 20.32% (−4.99) | 56.45% (−3.61) | **78.85% (5.35)** | 45.00% (2.00) |
| abbreviated | Sogou | 16.73% (−5.74) | 33.33% (−7.48) | 58.65% (2.66) | 40.00% (1.13) |
| abbreviated | Baidu | 19.52% (−4.92) | 40.32% (−6.51) | 62.50% (3.43) | 50.00% (1.67) |
| abbreviated | RIME | 25.10% (−3.30) | 54.30% (−4.16) | **77.88% (5.46)** | 20.00% (−0.58) |
| mixed | fused | 64.84% | 86.74% | 51.82% | 56.52% |
| mixed | #132 | 65.62% (0.47) | 82.32% (−2.53) | 57.27% (1.50) | 56.52% (0.00) |
| mixed | Apple Pinyin | 49.22% (−5.77) | 74.59% (−4.31) | **80.00% (5.40)** | 47.83% (−0.82) |
| mixed | Sogou | 38.67% (−8.07) | 62.43% (−6.63) | 53.64% (0.34) | 43.48% (−1.00) |
| mixed | Baidu | 39.84% (−7.25) | 62.43% (−6.63) | 60.00% (1.62) | 43.48% (−1.13) |
| mixed | RIME | 45.31% (−6.06) | 72.93% (−4.35) | **71.82% (3.67)** | 17.39% (−2.71) |

## What the numbers say

- **Too small to decide #191.** On full pinyin the fused decoder misses only 19 of 571 entities, so no system can gain more than 19 there, about 3 points. Luna's 12 / 4 is just z 2.00. On abbreviated and mixed input the slice separates the LLM-scale gaps, 5 points and more, and nothing finer. K2's ~5,000 entity-dense records would hold about ten times as many entities.
- **The generators' full-pinyin lead is mostly context and grammar, not entities.** Luna gains 47 slice records and 12 entities. The sentences it fixes are mostly wrong elsewhere. On abbreviated and mixed input the entity gains are real for both generators, people above all (Luna 67.73% against 59.36%, z 3.28). On the same records, swe-2-high's gains are larger than Luna's: 20 / 2 against 11 / 4 on abbreviated input.
- **#132 knows what the fused decoder knows.** Every category is level with context on, except people on abbreviated input (64.54% against 59.36%, z 2.71). End-to-end training did not cost knowledge, and it did not add any.
- **Idioms by initials are a lexicon gap.** Typing `yffs`, the commercial engines and RIME look up 一帆风顺 whole in their dictionaries. The fused decoder reads character by character, and its trigram and reader give a four-character idiom too little mass to win from initials alone. Apple and RIME lead by 5.4–5.5 z on abbreviated idioms. With context the fused decoder still spells only 52.88% of them, which is below Apple's context-free 78.85%. This is a finding for the decoder, not for #191's teacher.
- **Elsewhere the engines know less than the fused decoder.** People and places show this on all three twins, without context on either side.

## Implementation

- **`ime-eval`.** The machinery both slices share is in `slice.rs`: reading a record's keystrokes back onto its text (`cheapest_reading`), which records a dump must cover, a record's answer, the paired statistic and the `--by-record` listing. `AmbiguitySlice` now sits on it unchanged in behaviour, and its tests pass unchanged. `knowledge.rs` holds `KnowledgeLexicon`, `KnowledgeSlice::select` / `within` / `measure`, `KnowledgeReport` and the comparison.
- **`ime-knowledge`** (`ime-eval`, feature `cli`). It rescores any dump against the eval set and the lexicon, compares later dumps with the first, narrows the slice with `--within <dump>` to the records a partial run answered, and lists records with `--by-record` with each entity as `span/category`, marked `!` where the top answer misspells it.
- **One row type.** Every system reaches the slice as `ime_eval::DumpRow` lines. `mlime eval dump-answers` turns a generator's answers file into one, over the records of the dump it was run on, and an unanswered record gets no hypothesis. `dump-gui` reads an ImeDrive journal through `gui.read_results`, and a failed record is unanswered. `dump-rime` reads `eval rime`'s report. A one-sentence system's hypothesis carries score 0.0.
- **`mlime lexicon knowledge`** writes the lexicon, and `mlime lexicon fetch` knows the new Sogou lists and their categories.

## Reproduce

```
mlime lexicon fetch
mlime lexicon knowledge                        # data/lexicon/knowledge.tsv
cargo build --release -p ime-eval --features cli --bin ime-knowledge
mlime eval dump-answers --dump data/rescore-v2/eval3/neural-w1.000-kn-trigram-test.jsonl \
  --eval-set data/run3_pool/eval3.jsonl --answers data/generate-v1/eval3-hints.answers.jsonl \
  --out data/knowledge-slice/dumps/full/luna.jsonl
mlime eval dump-gui --results data/gui/apple-eval3/apple/eval3.jsonl \
  --eval-set data/run3_pool/eval3.jsonl --out data/knowledge-slice/dumps/full/apple.jsonl
mlime eval dump-rime --report data/rime/results/eval3.json \
  --eval-set data/run3_pool/eval3.jsonl --out data/knowledge-slice/dumps/full/rime.jsonl
target/release/ime-knowledge --eval-set data/run3_pool/eval3.jsonl \
  --lexicon data/lexicon/knowledge.tsv --slice test \
  --dump data/ambiguity-slice/fused/full-context-on/dump/neural-w1.500-kn-trigram+char-lm-test.jsonl \
  --dump data/e2e-v2-rank-g4/final/e2e-lattice-context-on.jsonl \
  --dump data/knowledge-slice/dumps/full/luna.jsonl
```

The first dump is the baseline. Add `--within data/knowledge-slice/dumps/full/swe-2-high.jsonl` for the subset. `data/knowledge-slice/run.sh` is the whole sequence for every twin and system. The converted dumps, the reports and the `--by-record` listings are under `data/knowledge-slice/`. Scoring takes under a second a group, and nothing is decoded.

## Next

- **K2, the knowledge set** (#191). The size above is the case for it. Drawing its sentences by this lexicon, about one record in ten of real text holds an entity, so ~5,000 entity-dense records need ~50,000 candidate sentences from the held-out documents.
- **A works lexicon.** The works category stays empty until a list of titles exists that is not made of ordinary phrases. One candidate is titles the corpus itself writes inside 《》.
- **Idioms by initials in the decoder.** This belongs in an issue of its own: a phrase prior from the idiom lexicon, measured on this slice's idiom rows.
