"""Download and parse Sogou .scel cell dictionaries into word-pinyin lexicons.

A ``.scel`` file is a binary format that stores a pinyin table (syllable
inventory indexed by id) followed by word entries, each carrying the pinyin
indices that spell it. The parser here is strict: it validates the magic header,
requires every pinyin index to resolve, and raises on any structural violation.
The ~62 KB trailing section that some files carry (an internal Sogou index) is
ignored once the word section is fully consumed.

This module treats the data as a **lexicon** (word + pinyin pairs), not a
running-text corpus. It sits beside the corpus pipeline but is a distinct
artifact — words are already pinyin-annotated, so they skip the g2p stage.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import polars as pl

from mlime.logging import log

from .shards import ShardWriter

SCEL_MAGIC_PREFIX = bytes.fromhex("40150000")
PINYIN_TABLE_OFFSET = 0x1540

LEXICON_SCHEMA = pl.Schema(
    {
        "word": pl.String(),
        "pinyin": pl.List(pl.String()),
        "dict_id": pl.Int64(),
        "dict_name": pl.String(),
        "rank": pl.Int64(),
    }
)


@dataclass(frozen=True)
class ScelDict:
    """A known Sogou dictionary with its provenance and quality metadata."""

    id: int
    slug: str
    name: str
    quality: str

    @property
    def download_url(self) -> str:
        return f"https://pinyin.sogou.com/d/dict/download_cell.php?id={self.id}&name={self.slug}"

    @property
    def detail_url(self) -> str:
        return f"https://pinyin.sogou.com/dict/detail/index/{self.id}"


DICTS: dict[str, ScelDict] = {
    d.slug: d
    for d in (
        ScelDict(
            id=4,
            slug="wangluo-liuxing-xinci",
            name="网络流行新词",
            quality="premium — manually reviewed 2026-08-26, very high quality internet vocabulary",
        ),
        ScelDict(
            id=177287,
            slug="bilibili-wanggeng",
            name="哔哩网梗",
            quality="unreviewed — bilibili meme vocabulary, appears reasonable",
        ),
    )
}


@dataclass(frozen=True)
class LexiconEntry:
    """One word and its pinyin syllables from a scel file."""

    word: str
    pinyin: tuple[str, ...]


def parse_scel(data: bytes) -> list[LexiconEntry]:
    """Parse a .scel binary blob into lexicon entries.

    Raises on malformed data rather than silently truncating.
    """
    if len(data) < PINYIN_TABLE_OFFSET + 4:
        raise ValueError(f"file too small ({len(data)} bytes)")
    if data[:4] != SCEL_MAGIC_PREFIX:
        raise ValueError(f"bad scel magic: {data[:4].hex()}, expected {SCEL_MAGIC_PREFIX.hex()}")

    pinyin_map, pos = _parse_pinyin_table(data)
    entries, pos = _parse_word_entries(data, pos, pinyin_map)

    trailing = len(data) - pos
    if trailing > 0:
        log.debug("scel trailing bytes skipped", bytes=trailing)

    return entries


def _parse_pinyin_table(data: bytes) -> tuple[dict[int, str], int]:
    """Read the pinyin syllable table starting at PINYIN_TABLE_OFFSET."""
    pos = PINYIN_TABLE_OFFSET
    count = struct.unpack_from("<I", data, pos)[0]
    pos += 4

    pinyin_map: dict[int, str] = {}
    for _ in range(count):
        idx = struct.unpack_from("<H", data, pos)[0]
        pos += 2
        length = struct.unpack_from("<H", data, pos)[0]
        pos += 2
        syllable = data[pos : pos + length].decode("utf-16-le")
        pos += length
        if idx in pinyin_map:
            raise ValueError(f"duplicate pinyin index {idx}: {pinyin_map[idx]!r} vs {syllable!r}")
        pinyin_map[idx] = syllable

    log.debug("scel pinyin table parsed", count=len(pinyin_map))
    return pinyin_map, pos


def _parse_word_entries(
    data: bytes, pos: int, pinyin_map: dict[int, str]
) -> tuple[list[LexiconEntry], int]:
    """Read word entries until the structure no longer matches the expected format."""
    entries: list[LexiconEntry] = []
    while pos + 4 <= len(data):
        num_words = struct.unpack_from("<H", data, pos)[0]
        pinyin_bytes = struct.unpack_from("<H", data, pos + 2)[0]

        if pinyin_bytes == 0 or pinyin_bytes % 2 != 0 or num_words == 0:
            break

        needed = 4 + pinyin_bytes + num_words * 4
        if pos + needed > len(data):
            break

        pos += 4
        num_pinyins = pinyin_bytes // 2
        syllables: list[str] = []
        for _ in range(num_pinyins):
            idx = struct.unpack_from("<H", data, pos)[0]
            pos += 2
            if idx not in pinyin_map:
                raise ValueError(f"unknown pinyin index {idx} at offset 0x{pos - 2:X}")
            syllables.append(pinyin_map[idx])

        pinyin = tuple(syllables)

        for _ in range(num_words):
            word_len = struct.unpack_from("<H", data, pos)[0]
            pos += 2
            word = data[pos : pos + word_len].decode("utf-16-le")
            pos += word_len
            ext_len = struct.unpack_from("<H", data, pos)[0]
            pos += 2
            pos += ext_len
            entries.append(LexiconEntry(word=word, pinyin=pinyin))

    return entries, pos


def download_scel(dict_entry: ScelDict, cache_dir: Path) -> bytes:
    """Download a scel file, caching on disk. Returns the raw bytes."""
    cache_path = cache_dir / f"{dict_entry.slug}.scel"
    if cache_path.exists():
        log.info("scel cached", path=str(cache_path))
        return cache_path.read_bytes()

    cache_dir.mkdir(parents=True, exist_ok=True)
    log.info("downloading scel", url=dict_entry.download_url)
    client = httpx.Client(
        http1=True,
        follow_redirects=True,
        timeout=120,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
    )
    response = client.get(dict_entry.download_url)
    response.raise_for_status()

    if response.content[:4] != SCEL_MAGIC_PREFIX:
        raise ValueError(
            f"downloaded content is not a scel file (magic: {response.content[:4].hex()})"
        )

    cache_path.write_bytes(response.content)
    log.info("scel downloaded", path=str(cache_path), size=len(response.content))
    return response.content


def fetch_lexicon(
    slugs: tuple[str, ...],
    out_dir: Path,
    cache_dir: Path,
    entries_per_shard: int = 100_000,
) -> dict[str, int]:
    """Download, parse, and write lexicon shards for the requested dictionaries."""
    counts: dict[str, int] = {}
    for slug in slugs:
        dict_entry = DICTS[slug]
        data = download_scel(dict_entry, cache_dir)
        entries = parse_scel(data)

        with ShardWriter(out_dir, slug, LEXICON_SCHEMA, entries_per_shard) as writer:
            for rank, entry in enumerate(entries):
                writer.write(
                    {
                        "word": entry.word,
                        "pinyin": list(entry.pinyin),
                        "dict_id": dict_entry.id,
                        "dict_name": dict_entry.name,
                        "rank": rank,
                    }
                )

        counts[slug] = len(entries)
        log.info(
            "lexicon written",
            slug=slug,
            dict_name=dict_entry.name,
            entries=len(entries),
            quality=dict_entry.quality,
        )
    return counts


def read_lexicon(directory: Path, slug: str = "*") -> Iterator[LexiconEntry]:
    """Stream lexicon entries from parquet shards under *directory*."""
    from .shards import read_shards

    for row in read_shards(directory, slug).iter_rows(named=True):
        yield LexiconEntry(word=row["word"], pinyin=tuple(row["pinyin"]))
