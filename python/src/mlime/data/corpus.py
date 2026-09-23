"""Turn three upstream Chinese corpora into pinyin-conversion training samples.

The mix is deliberate. An input method is used to *write*, and what people write
splits roughly three ways: chat, news-register prose, and reference prose. Each
source contributes one of those registers:

``wiki``
    ``wikimedia/wikipedia`` config ``20231101.zh`` -- formal, encyclopaedic, and
    stored in whatever script its editors used, so it is the reason the
    normaliser converts to simplified rather than trusting the upstream text.
``dialogue``
    ``silver/lccc``, the LCCC cleaned Chinese conversation corpus. Closest to
    what someone actually types into a chat window, and the only source where a
    sample's context is a *preceding turn* rather than a preceding sentence. It
    still ships a loading script, which ``datasets`` 5 refuses to run, so it is
    read from the parquet branch the Hub auto-converts.
``news``
    ``SirlyDreamer/THUCNews``, the Sina news archive, already simplified.

Every sample carries an optional ``context`` field from the first day, because
the model conditions on the text already on the host's screen. During training
the field is dropped at random; here it is simply whatever preceded the target in
the same document.

The stage is split in two, and the line between them is the network. ``fetch``
writes the upstream text as it arrives, split only where upstream already split
it -- one part per article, one per dialogue turn. ``prepare`` does everything
else: normalising, sentence-splitting, filtering, deduplicating. Every rule worth
changing therefore lives on the side that can be re-run offline, so tightening a
filter or the normaliser costs a minute rather than another pass over the network.
"""

from __future__ import annotations

import hashlib
import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from mlime.locate import find_upwards
from mlime.logging import log

from .shards import ShardWriter, read_shards
from .text import HAN, Normalizer, han_ratio, split_sentences, strip_terminal_delimiter

#: Where ``ime-pinyin``'s character table lives inside the repository.
CHAR_TABLE = Path("crates") / "ime-pinyin" / "data" / "char_pinyin.tsv"

DOCUMENT_SCHEMA = pl.Schema(
    {
        "document_id": pl.String(),
        "source": pl.String(),
        "parts": pl.List(pl.String()),
    }
)

SAMPLE_SCHEMA = pl.Schema(
    {
        "id": pl.String(),
        "source": pl.String(),
        "text": pl.String(),
        "context": pl.String(),
    }
)


@dataclass(frozen=True)
class RawDocument:
    """One upstream document, untouched apart from being split where upstream split it."""

    document_id: str
    parts: tuple[str, ...]


@dataclass(frozen=True)
class Document:
    """One normalised document as an ordered run of segments.

    A segment is a sentence for prose and a turn for dialogue. ``joiner`` is what
    puts them back together when a run of them becomes a sample's context:
    sentences already carry their own terminal punctuation, turns do not.
    """

    document_id: str
    segments: tuple[str, ...]
    joiner: str = ""


@dataclass(frozen=True)
class Sample:
    """One training example: what to type, and what was on screen before it."""

    id: str
    source: str
    text: str
    context: str | None

    def row(self) -> dict[str, object]:
        """The sample as a ``SAMPLE_SCHEMA`` row."""
        return {"id": self.id, "source": self.source, "text": self.text, "context": self.context}

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Sample:
        """Rebuild a sample from a ``SAMPLE_SCHEMA`` parquet row."""
        context = row["context"]
        return cls(
            str(row["id"]),
            str(row["source"]),
            str(row["text"]),
            None if context is None else str(context),
        )

    @classmethod
    def read(cls, directory: Path, prefix: str = "*") -> Iterator[Sample]:
        """Stream the samples held in *directory*'s shards."""
        for row in read_shards(directory, prefix).iter_rows(named=True):
            yield cls.from_row(row)


def content_id(source: str, text: str, context: str | None) -> str:
    """A stable identifier derived from the sample's own content."""
    digest = hashlib.blake2b(
        "\x00".join((source, text, context or "")).encode("utf-8"), digest_size=12
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class HuggingFaceStream:
    """A streaming ``datasets`` load, checked against the fields the adapter reads."""

    path: str
    fields: frozenset[str]
    config: str | None = None
    revision: str | None = None
    data_dir: str | None = None

    def records(self, limit: int | None) -> Iterator[Mapping[str, Any]]:
        """Yield at most *limit* raw records, failing loudly on an unexpected shape."""
        from datasets import load_dataset

        dataset = load_dataset(
            self.path,
            self.config,
            split="train",
            streaming=True,
            revision=self.revision,
            data_dir=self.data_dir,
        )
        log.info("streaming", dataset=self.path, config=self.config, data_dir=self.data_dir)
        for index, record in enumerate(dataset):
            if limit is not None and index >= limit:
                return
            missing = self.fields - record.keys()
            if missing:
                raise KeyError(
                    f"{self.path} record {index} is missing {sorted(missing)}; "
                    f"it has {sorted(record.keys())}"
                )
            yield record


@dataclass(frozen=True)
class CorpusSource(ABC):
    """One upstream corpus, in two halves either side of the network.

    ``parts`` is the fetch half and does no interpretation. ``segments`` is the
    prepare half and holds every rule that might change.
    """

    name: str
    stream: HuggingFaceStream
    identifier_field: str | None

    @property
    def joiner(self) -> str:
        """What runs consecutive segments together when they become context."""
        return ""

    @abstractmethod
    def parts(self, record: Mapping[str, Any]) -> tuple[str, ...]:
        """The record's raw text, split only where upstream already split it."""

    @abstractmethod
    def segments(self, parts: Sequence[str], normalizer: Normalizer) -> tuple[str, ...]:
        """Normalise the parts into the units a sample can be built from."""

    def raw_documents(self, limit: int | None) -> Iterator[RawDocument]:
        """Stream upstream documents, skipping any that carry no text at all."""
        for record in self.stream.records(limit):
            parts = self.parts(record)
            if not any(parts):
                continue
            yield RawDocument(self._identify(record, parts), parts)

    def document(self, raw: RawDocument, normalizer: Normalizer) -> Document:
        """Normalise a fetched document into its segments."""
        return Document(raw.document_id, self.segments(raw.parts, normalizer), self.joiner)

    def _identify(self, record: Mapping[str, Any], parts: Sequence[str]) -> str:
        """Prefer the upstream identifier, falling back to the content's own hash."""
        if self.identifier_field:
            return str(record[self.identifier_field])
        return content_id(self.name, "\n".join(parts), None)


@dataclass(frozen=True)
class ProseSource(CorpusSource):
    """A source whose records are articles: one part, split into sentences.

    Wikipedia and THUCNews differ only in which fields hold the prose and whether
    the record carries a usable identifier, so they share one implementation.
    """

    text_fields: tuple[str, ...]

    def parts(self, record: Mapping[str, Any]) -> tuple[str, ...]:
        """The record's text fields, headline first, as a single part."""
        return ("\n".join(str(record[name]) for name in self.text_fields),)

    def segments(self, parts: Sequence[str], normalizer: Normalizer) -> tuple[str, ...]:
        """Normalise, then split into sentences."""
        return tuple(
            itertools.chain.from_iterable(split_sentences(normalizer(part)) for part in parts)
        )


@dataclass(frozen=True)
class DialogueSource(CorpusSource):
    """A source whose records are conversations: one segment per turn.

    Turns stay whole rather than being split into sentences, because a turn is
    the unit a person composes before pressing send -- which is exactly the unit
    an input method converts, and exactly what the preceding turns are context
    for.
    """

    turns_field: str

    @property
    def joiner(self) -> str:
        """Turns carry no terminal punctuation, so running them together would fuse them."""
        return "\n"

    def parts(self, record: Mapping[str, Any]) -> tuple[str, ...]:
        """One part per turn, in the order they were said."""
        turns = record[self.turns_field]
        if isinstance(turns, str):
            raise TypeError(f"{self.stream.path}.{self.turns_field} is a string, expected a list")
        return tuple(str(turn) for turn in turns)

    def segments(self, parts: Sequence[str], normalizer: Normalizer) -> tuple[str, ...]:
        """Normalise each turn, dropping any that emptied out."""
        return tuple(filter(None, (normalizer(part) for part in parts)))


WIKIPEDIA = ProseSource(
    name="wiki",
    stream=HuggingFaceStream(
        path="wikimedia/wikipedia",
        config="20231101.zh",
        fields=frozenset({"id", "text"}),
    ),
    identifier_field="id",
    text_fields=("text",),
)

DIALOGUE = DialogueSource(
    name="dialogue",
    stream=HuggingFaceStream(
        path="silver/lccc",
        revision="refs/convert/parquet",
        data_dir="large",
        fields=frozenset({"dialog"}),
    ),
    identifier_field=None,
    turns_field="dialog",
)

NEWS = ProseSource(
    name="news",
    stream=HuggingFaceStream(
        path="SirlyDreamer/THUCNews",
        fields=frozenset({"title", "text"}),
    ),
    identifier_field=None,
    text_fields=("title", "text"),
)

SOURCES: Mapping[str, CorpusSource] = {s.name: s for s in (WIKIPEDIA, DIALOGUE, NEWS)}


def default_char_table() -> Path | None:
    """Locate the repository's character table, if this is running from a checkout."""
    return find_upwards(CHAR_TABLE)


def load_reference_characters(path: Path) -> frozenset[str]:
    """The characters ``ime-pinyin`` can produce; a target may contain no other Han char."""
    if not path.is_file():
        raise FileNotFoundError(
            f"no character table at {path}; generate it with `mlime gen-pinyin-tables` "
            "or pass --char-table"
        )
    characters = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        character, separator, readings = line.partition("\t")
        if not separator or not readings or len(character) != 1:
            raise ValueError(f"{path}:{number} is not a `<char>\\t<readings>` row: {line!r}")
        characters.add(character)
    if not characters:
        raise ValueError(f"{path} is empty")
    return frozenset(characters)


@dataclass
class FilterCounts:
    """Why candidate targets were kept or dropped, for the run summary."""

    kept: int = 0
    too_short: int = 0
    too_long: int = 0
    not_chinese_enough: int = 0
    unknown_character: int = 0
    duplicate: int = 0

    @property
    def considered(self) -> int:
        """Every candidate the filter saw."""
        return (
            self.kept
            + self.too_short
            + self.too_long
            + self.not_chinese_enough
            + self.unknown_character
            + self.duplicate
        )


@dataclass
class SampleFilter:
    """Length, script, lexicon-coverage and duplicate gate on one source's targets.

    Holding the seen-hash set here is what makes deduplication per-source: two
    sources may legitimately contain the same sentence, and dropping the second
    would silently skew the mix.
    """

    known_characters: frozenset[str]
    min_characters: int = 4
    max_characters: int = 64
    min_han_ratio: float = 0.9
    counts: FilterCounts = field(default_factory=FilterCounts)
    _seen: set[bytes] = field(default_factory=set, repr=False)

    def accepts(self, text: str) -> bool:
        """Whether *text* is usable as a target, recording the verdict's reason."""
        if len(text) < self.min_characters:
            self.counts.too_short += 1
            return False
        if len(text) > self.max_characters:
            self.counts.too_long += 1
            return False
        if han_ratio(text) < self.min_han_ratio:
            self.counts.not_chinese_enough += 1
            return False
        if any(
            character not in self.known_characters for character in text if HAN.match(character)
        ):
            self.counts.unknown_character += 1
            return False
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
        if digest in self._seen:
            self.counts.duplicate += 1
            return False
        self._seen.add(digest)
        self.counts.kept += 1
        return True


def build_samples(
    document: Document,
    source: str,
    context_segments: int,
    max_context_characters: int,
    sample_filter: SampleFilter,
) -> Iterator[Sample]:
    """Yield one accepted sample per segment of *document*, with its preceding run as context."""
    for index, segment in enumerate(document.segments):
        text = strip_terminal_delimiter(segment).strip()
        if not sample_filter.accepts(text):
            continue
        preceding = document.segments[max(0, index - context_segments) : index]
        context = document.joiner.join(preceding)[-max_context_characters:] or None
        yield Sample(content_id(source, text, context), source, text, context)


def fetch(
    source: CorpusSource,
    out_dir: Path,
    limit: int | None = None,
    documents_per_shard: int = 20_000,
) -> int:
    """Stream *source* into document shards under *out_dir*. Returns the count."""
    with ShardWriter(out_dir, source.name, DOCUMENT_SCHEMA, documents_per_shard) as writer:
        for raw in source.raw_documents(limit):
            writer.write(
                {
                    "document_id": raw.document_id,
                    "source": source.name,
                    "parts": list(raw.parts),
                }
            )
    log.info("documents fetched", source=source.name, documents=writer.rows_written)
    return writer.rows_written


def prepare(
    raw_dir: Path,
    out_dir: Path,
    known_characters: frozenset[str],
    sources: tuple[str, ...],
    context_segments: int = 3,
    max_context_characters: int = 256,
    min_characters: int = 4,
    max_characters: int = 64,
    min_han_ratio: float = 0.9,
    limit: int | None = None,
    samples_per_shard: int = 100_000,
) -> dict[str, FilterCounts]:
    """Build filtered samples from fetched documents. Returns per-source filter counts."""
    normalizer = Normalizer()
    counts: dict[str, FilterCounts] = {}
    for name in sources:
        source = SOURCES[name]
        documents = read_shards(raw_dir, name)
        sample_filter = SampleFilter(
            known_characters,
            min_characters=min_characters,
            max_characters=max_characters,
            min_han_ratio=min_han_ratio,
        )
        samples = (
            sample
            for row in documents.iter_rows(named=True)
            for sample in build_samples(
                source.document(RawDocument(row["document_id"], tuple(row["parts"])), normalizer),
                name,
                context_segments,
                max_context_characters,
                sample_filter,
            )
        )
        with ShardWriter(out_dir, name, SAMPLE_SCHEMA, samples_per_shard) as writer:
            for sample in itertools.islice(samples, limit):
                writer.write(sample.row())
        counts[name] = sample_filter.counts
        log.info(
            "samples prepared",
            source=name,
            documents=documents.height,
            written=writer.rows_written,
            **asdict(sample_filter.counts),
        )
    return counts
