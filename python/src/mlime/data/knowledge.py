"""The knowledge lexicon: the real-world names and phrases the knowledge slice of eval3 matches.

Issue #207. A position of eval3 measures world knowledge when it lies inside a
span naming a person, a place, an organisation, an idiom or an internet meme
(``ime_eval::KnowledgeSlice``). This module writes the lexicon that slice
reads, one ``word<TAB>category`` line per entry, from real lists only:

- Sogou's typed cell dictionaries (:data:`SLICE_DICTS`), fetched by
  ``mlime lexicon fetch``, each filed under one category;
- the 梗百科 (gengbaike.cn) entries crawled into ``data/gengbaike/``, filed
  under ``internet``.

An entry counts when it is at least :data:`MIN_LENGTH` Han characters long and
the ordinary vocabulary does not hold it as an ordinary word. The ordinary
vocabulary is jieba's general dictionary, pinned at :data:`JIEBA_URL`: an entry
it lists under a tag outside :data:`KNOWLEDGE_TAGS` (a noun, a verb, an
adjective...) is a homograph of an everyday word -- the singer 苏醒 is read as
the verb far more often than as the name -- and stays out. An entry jieba lacks
counts: its dictionary dates from 2014, so newer names and memes are exactly
what it lacks.

Title lists are not used. zh.wikipedia and moegirl article titles and Sogou's
film, TV and anime title lists name works with ordinary phrases (在一起,
不知道, 我们的), and Sogou's mined new-word list holds ordinary words; jieba
neither lists such phrases nor tags proper nouns reliably (it files 东西 as a
place and 谢谢 as a person), so no rule over it separates them from the names
they hold.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import httpx

from mlime.logging import log

from .scel import DICTS
from .shards import read_shards

#: jieba's general dictionary at the last commit that changed it (2014-02-07).
JIEBA_URL = (
    "https://raw.githubusercontent.com/fxsjy/jieba/"
    "dafc73425e618c1a0d562f84362b46bedf579bdb/jieba/dict.txt"
)
JIEBA_SHA256 = "7197c3211ddd98962b036cdf40324d1ea2bfaa12bd028e68faa70111a88e12a8"

#: The jieba tags that do not make an entry ordinary: person, place,
#: organisation and other proper names, idioms, set phrases and abbreviations.
KNOWLEDGE_TAGS = frozenset({"nr", "nrt", "nrfg", "ns", "nt", "nz", "i", "l", "j"})

#: The shortest entry that counts. A single character is never knowledge: the
#: keystrokes for it are a syllable, and every system's choice there is a
#: language-model choice.
MIN_LENGTH = 2

#: The Sogou dictionaries the slice matches, in the order an entry listed by
#: several takes its category from: names before phrases.
SLICE_DICTS = (
    "mingxing",
    "geshou-renming",
    "thuocl-lishi-renming",
    "xingzheng-quhua",
    "fengjing-mingsheng",
    "guojia-shoudu",
    "zhengfu-jigou",
    "gaodeng-yuanxiao",
    "chengyu-suyu",
    "sanzi-chengyu",
)

#: The category of the 梗百科 entries, which come after every Sogou list.
GENGBAIKE_CATEGORY = "internet"

_HAN = re.compile(r"^[㐀-䶿一-鿿]+$")
#: A trailing disambiguator, ``标题(作品)`` or ``标题（作品）``.
_DISAMBIGUATOR = re.compile(r"\s*[（(][^（）()]*[）)]\s*$")


@dataclass(frozen=True)
class Source:
    """One list of entries and the category every entry of it is filed under."""

    name: str
    category: str
    entries: tuple[str, ...]


def read_ordinary_vocabulary(path: Path) -> dict[str, str]:
    """jieba's dictionary as ``{word: tag}``; a ``word freq tag`` line each."""
    vocabulary: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split(" ")
        if len(fields) != 3:
            raise ValueError(f"{path}:{number} is not a `word freq tag` line: {line!r}")
        vocabulary[fields[0]] = fields[2]
    if not vocabulary:
        raise ValueError(f"{path} holds no words")
    return vocabulary


def fetch_ordinary_vocabulary(cache_dir: Path) -> Path:
    """Download jieba's pinned dictionary once and check it is the pinned file."""
    path = cache_dir / "jieba-dict.txt"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        log.info("downloading the ordinary vocabulary", url=JIEBA_URL)
        response = httpx.get(JIEBA_URL, follow_redirects=True, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != JIEBA_SHA256:
        raise ValueError(f"{path} has sha256 {digest}, not the pinned {JIEBA_SHA256}")
    return path


def entry(title: str) -> str:
    """A title as the slice matches it: its disambiguator and outer space removed."""
    return _DISAMBIGUATOR.sub("", title.strip())


def counts(word: str, ordinary: Mapping[str, str]) -> bool:
    """Whether *word* is a knowledge entry: long enough, all Han, and not an ordinary word."""
    if len(word) < MIN_LENGTH or not _HAN.match(word):
        return False
    tag = ordinary.get(word)
    return tag is None or tag in KNOWLEDGE_TAGS


def build(sources: Iterable[Source], ordinary: Mapping[str, str]) -> dict[str, str]:
    """Every counted entry with its category, the first source listing it deciding."""
    lexicon: dict[str, str] = {}
    for source in sources:
        kept = 0
        for title in source.entries:
            word = entry(title)
            if counts(word, ordinary) and word not in lexicon:
                lexicon[word] = source.category
                kept += 1
        log.info("knowledge source", source=source.name, entries=len(source.entries), new=kept)
    return lexicon


def sogou_sources(lexicon_dir: Path) -> list[Source]:
    """The Sogou dictionaries of :data:`SLICE_DICTS`, read from their fetched shards."""
    sources = []
    for slug in SLICE_DICTS:
        words = read_shards(lexicon_dir, slug).get_column("word").to_list()
        sources.append(Source(name=slug, category=DICTS[slug].category, entries=tuple(words)))
    return sources


def gengbaike_source(path: Path) -> Source:
    """The 梗百科 entries' titles."""
    titles = tuple(
        str(json.loads(line)["title"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return Source(name="gengbaike", category=GENGBAIKE_CATEGORY, entries=titles)


def write(lexicon: Mapping[str, str], out: Path) -> None:
    """One ``word<TAB>category`` line per entry, sorted by word."""
    out.write_text(
        "".join(f"{word}\t{lexicon[word]}\n" for word in sorted(lexicon)), encoding="utf-8"
    )
