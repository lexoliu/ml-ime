"""The judge over FineWeb-2's mined sentences (#225): only natural usage of an entry is kept.

``ime-cli corpus mine`` finds sentences that hold one of #215's absent entries
and refuses the ones its deterministic gates can see are filler, but most of
what is left is still not anyone using the entry: a spinner's synonym swaps
read fluently sentence by sentence, and an entry's characters often meet only
across a word boundary. A model this small learns whatever it is shown, so a
mined record is kept only when it passes three rules, each run only on what
passed the one before:

1. **the word boundary** (:class:`BoundaryLexicon`): no occurrence of the entry
   may have its characters taken by a longer word there -- a word of the
   knowledge lexicon or of jieba's general dictionary that overlaps the entry
   without lying inside it;
2. **Jev** (:class:`Jev`): TypeSafe's System One must give the question
   :data:`QUESTION` a probability of at least :data:`JEV_THRESHOLD`;
3. **the review** (:class:`Review`): an OpenAI-compatible model answers the
   prompt in ``templates/judge_review_prompt.txt``, and must find the page's
   preceding text not spun and the sentence natural. A record on a spun page is
   dropped whole, even when the sentence itself reads well.

The question and the prompt were set on 120 hand-labelled mined records, which
showed the judges the last :data:`CONTEXT_CHARACTERS` characters of the context;
the judge shows them the same. On those records Jev at 0.7 followed by the review
kept 27, all real, of the 57 real ones. The threshold was then raised to 0.8 on
the run itself: among the first 1,188 reviews the review model rejected 33% of
the units Jev put in [0.7, 0.8), 17% of [0.8, 0.9) and 4% of 0.9 and above, and
the review serves one request at a time, so only units at 0.8 and above are
kept, every one of them reviewed. On the labels 0.8 followed by the review keeps
8, all real: the hand labels found 24 of the 25 records in [0.7, 0.8) real, and
the review kept 19 of them, none of them wrongly.

A unit's Jev verdict is its probability, and :data:`JEV_THRESHOLD` is applied
whenever a verdict is read (:func:`passes`), never stored: a log written under
another threshold is decided by the current one when a run resumes, and a unit
it dropped is dropped whatever later verdicts its log holds. A judged file
records the threshold it was finished under, and a run at another threshold
finishes it again from its log.

A record holding two entries is judged once per entry, and keeps the entries
that pass; it is kept when one does. Every verdict is appended to its mined
file's verdict log as it arrives, so a run stopped at any moment resumes with
only the unjudged records. Once every record of a mined file has its verdict,
the records it keeps are written to the samples directory under the shard's
own name, and the file's census is written last, marking it judged.
"""

from __future__ import annotations

import asyncio
import email.utils
import json
import random
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
import polars as pl
from jinja2 import Template
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mlime.locate import find_upwards
from mlime.logging import log

#: The mined source the judge reads, as the ingestion names it.
SOURCE = "fineweb2-mined"

#: TypeSafe's System One endpoint and the model it is asked with.
JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"

#: The least probability Jev must give :data:`QUESTION` for a record to go on.
JEV_THRESHOLD = 0.8

#: Requests in flight at once, to Jev and to the review model.
JEV_CONCURRENCY = 32
REVIEW_CONCURRENCY = 32

#: How much of the context's end the judges see, as the hand labels showed it.
CONTEXT_CHARACTERS = 160

#: The question Jev answers; its wording and criteria are the labelled ones.
QUESTION: dict[str, Any] = {
    "type": "noul",
    "instructions": (
        "The state holds a Chinese sentence mined from the web, the preceding text of its page, "
        "and an entry (an idiom or phrase) the sentence was matched on. Is the sentence natural "
        "Chinese written by a person that uses the entry, as that entry and with its own meaning?"
    ),
    "criteria": {
        "true": (
            "Fluent human writing (news, fiction, essays, forums, classical or religious text) "
            "where the entry is a real word of the sentence and is used with its meaning."
        ),
        "false": (
            "Machine-spun text whose words were swapped for odd synonyms or idioms (the page "
            "reads unnaturally), keyword or SEO spam, or the entry's characters only occur "
            "across word boundaries inside other words."
        ),
    },
}

#: The review prompt, a template over ``entry``, ``context`` and ``text``.
REVIEW_PROMPT = "templates/judge_review_prompt.txt"

#: Statuses that say "not now" rather than "never": rate limits, overload, gateway trouble.
TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504, 529})

#: Attempts a request gets before a transient failure fails the run.
MAX_ATTEMPTS = 8

#: The longest pause between attempts when the server names none.
MAX_BACKOFF_SECONDS = 60.0

#: Seconds a request may take: Jev answers in about one, the review model in tens.
JEV_TIMEOUT_SECONDS = 60.0
REVIEW_TIMEOUT_SECONDS = 180.0

#: How many judged records pass between two progress lines.
PROGRESS_EVERY = 1000

#: A mined file's entry rows, as mining writes them and the judge writes its kept ones.
ENTRY_ROW_SCHEMA = pl.Schema({"id": pl.String, "entry": pl.String})

Stage = Literal["boundary", "jev", "review"]
STAGES: tuple[Stage, ...] = ("boundary", "jev", "review")


class JudgeError(RuntimeError):
    """A failure the judge does not retry: the run stops and resumes once it is fixed."""


class JevSettings(BaseSettings):
    """TypeSafe's API key, from ``JEV_API_KEY``."""

    model_config = SettingsConfigDict(
        env_prefix="JEV_",
        env_file=find_upwards(".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = Field(description="Bearer token for TypeSafe's API")

    @classmethod
    def load(cls) -> JevSettings:
        """Read the key, raising a ``ValidationError`` when it is unset."""
        return cls()  # type: ignore[call-arg]


class ReviewSettings(BaseSettings):
    """The review model's endpoint, from the ``MLIME_JUDGE_LLM_*`` variables."""

    model_config = SettingsConfigDict(
        env_prefix="MLIME_JUDGE_LLM_",
        env_file=find_upwards(".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_url: str = Field(description="Root of the OpenAI-compatible API, ending in /v1")
    model: str = Field(description="Model id to review with")
    key_file: Path = Field(description="File holding the endpoint's bearer token")

    @classmethod
    def load(cls) -> ReviewSettings:
        """Read the settings, raising a ``ValidationError`` naming any that are unset."""
        return cls()  # type: ignore[call-arg]

    def api_key(self) -> str:
        """The bearer token, read from :attr:`key_file`."""
        key = self.key_file.expanduser().read_text(encoding="utf-8").strip()
        if not key:
            raise JudgeError(f"{self.key_file} holds no key")
        return key


# ---------------------------------------------------------------------------
# Rule 1: the word boundary.


class BoundaryLexicon:
    """The words whose overlap with an entry shows its characters belong to something else."""

    def __init__(self, words: Iterable[str]) -> None:
        self._words = frozenset(word for word in words if len(word) >= 2)
        if not self._words:
            raise ValueError("the boundary lexicon holds no word")
        lengths: dict[str, set[int]] = {}
        for word in self._words:
            lengths.setdefault(word[0], set()).add(len(word))
        self._lengths = {first: tuple(sorted(found)) for first, found in lengths.items()}
        self._longest = max(len(word) for word in self._words)

    def __len__(self) -> int:
        return len(self._words)

    @classmethod
    def load(cls, entry_table: Path, ordinary: Mapping[str, str]) -> BoundaryLexicon:
        """The knowledge lexicon of #215's entry table together with jieba's dictionary."""
        lines = entry_table.read_text(encoding="utf-8").splitlines()
        header = lines[0] if lines else ""
        if not header.startswith("entry\tcategory\toccurrences") or not header.endswith(
            "\tblocked"
        ):
            raise ValueError(f"{entry_table} is not #215's entry table: its header is {header!r}")
        knowledge = [line.split("\t", 1)[0] for line in lines[1:] if line]
        return cls([*knowledge, *ordinary])

    def crossing(self, text: str, start: int, end: int) -> list[str]:
        """The words of *text* that overlap ``text[start:end]`` without lying inside it."""
        found = []
        for left in range(max(0, start - self._longest + 1), end):
            for length in self._lengths.get(text[left], ()):
                right = left + length
                if right > len(text):
                    break
                if right <= start or (left >= start and right <= end):
                    continue
                if text[left:right] in self._words:
                    found.append(text[left:right])
        return found

    def false_match(self, text: str, entry: str) -> list[str] | None:
        """The crossing words when every occurrence of *entry* is crossed, else ``None``.

        An entry used once as itself is used, whatever its other occurrences do.
        """
        crossings: list[str] = []
        start = text.find(entry)
        if start < 0:
            raise JudgeError(f"the mined sentence {text!r} does not hold its entry {entry!r}")
        while start >= 0:
            words = self.crossing(text, start, start + len(entry))
            if not words:
                return None
            crossings.extend(words)
            start = text.find(entry, start + 1)
        return crossings


# ---------------------------------------------------------------------------
# What is judged, and what the judges said.


@dataclass(frozen=True)
class Unit:
    """One entry of one mined record: the thing every rule gives its verdict on."""

    ordinal: int
    id: str
    entry: str
    text: str
    context: str

    @property
    def key(self) -> tuple[str, str]:
        """The record and entry, which name the unit in the verdict log."""
        return (self.id, self.entry)


@dataclass(frozen=True)
class JevAnswer:
    """Jev's probability that the record is natural usage, and the model that gave it."""

    p: float
    model: str


@dataclass(frozen=True)
class ReviewAnswer:
    """The review model's two judgements."""

    context_spun: bool
    sentence_natural: bool


class Asker[T](Protocol):
    """A judge that answers about one unit."""

    async def ask(self, unit: Unit) -> T: ...


def verdict_line(
    unit: Unit, stage: Literal["boundary", "review"], keep: bool, **detail: object
) -> dict[str, object]:
    """One line of the verdict log for a rule whose verdict is a yes or a no."""
    return {"id": unit.id, "entry": unit.entry, "stage": stage, "keep": keep, **detail}


def jev_line(unit: Unit, answer: JevAnswer, seconds: float) -> dict[str, object]:
    """One line of the verdict log for Jev, whose verdict is its probability."""
    return {
        "id": unit.id,
        "entry": unit.entry,
        "stage": "jev",
        "p": answer.p,
        "model": answer.model,
        "seconds": seconds,
    }


def passes(line: Mapping[str, Any]) -> bool:
    """Whether a verdict lets its unit go on, under the threshold in force now.

    A Jev line is read by its probability alone. Lines written before the
    threshold moved out of the log also hold the ``keep`` it gave then, which
    is ignored: the threshold is the run's decision, not the log's.
    """
    if line["stage"] == "jev":
        return bool(line["p"] >= JEV_THRESHOLD)
    return bool(line["keep"])


def deciding(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A unit's verdicts up to the first that stops it, checked to be in rule order.

    Verdicts after one that stops the unit were given under an earlier, looser
    threshold and decide nothing now.
    """
    for position, line in enumerate(lines):
        if position >= len(STAGES) or line["stage"] != STAGES[position]:
            raise JudgeError(f"verdicts out of order for {line['id']} {line['entry']}: {lines}")
    for position, line in enumerate(lines):
        if not passes(line):
            return lines[: position + 1]
    return lines


def next_stage(lines: list[dict[str, Any]]) -> Stage | None:
    """The rule a unit faces next, given its verdicts so far; ``None`` once it is decided."""
    lines = deciding(lines)
    if not lines:
        return "boundary"
    if not passes(lines[-1]) or len(lines) == len(STAGES):
        return None
    return STAGES[len(lines)]


# ---------------------------------------------------------------------------
# The HTTP side.


def retry_after_seconds(response: httpx.Response, now: datetime | None = None) -> float | None:
    """The pause a ``Retry-After`` header asks for, in seconds or as an HTTP date."""
    value = response.headers.get("retry-after")
    if value is None:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError) as error:
        raise JudgeError(
            f"{response.request.url} sent an unreadable Retry-After {value!r}"
        ) from error
    return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())


@dataclass
class Poster:
    """POSTs JSON, retrying a rate limit or a transient failure with backoff.

    A ``Retry-After`` header is honoured; without one the pause doubles from a
    second up to :data:`MAX_BACKOFF_SECONDS`, jittered so the requests in flight
    do not come back together. Any other failure, and a transient one that
    outlasts :data:`MAX_ATTEMPTS`, raises :class:`JudgeError`.
    """

    client: httpx.AsyncClient
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    max_attempts: int = MAX_ATTEMPTS
    retries: int = 0

    async def post(self, url: str, body: Mapping[str, object], headers: Mapping[str, str]) -> Any:
        """The JSON the server answers *body* with."""
        for attempt in range(self.max_attempts):
            pause: float | None = None
            try:
                response = await self.client.post(url, json=body, headers=dict(headers))
            except httpx.TransportError as error:
                reason = f"{type(error).__name__}: {error}"
            else:
                if response.status_code == 200:
                    return response.json()
                if response.status_code not in TRANSIENT_STATUSES:
                    raise JudgeError(
                        f"{url} answered {response.status_code}: {response.text[:500]}"
                    )
                reason = f"status {response.status_code}"
                pause = retry_after_seconds(response)
            if attempt + 1 == self.max_attempts:
                raise JudgeError(
                    f"{url} still failing after {self.max_attempts} attempts: {reason}"
                )
            if pause is None:
                pause = min(MAX_BACKOFF_SECONDS, 2.0**attempt) * random.uniform(0.5, 1.0)
            self.retries += 1
            log.debug("retrying", url=url, reason=reason, pause=round(pause, 1))
            await self.sleep(pause)
        raise AssertionError("unreachable: the last attempt returns or raises")


@dataclass
class Jev:
    """Asks TypeSafe's System One :data:`QUESTION` about a unit."""

    poster: Poster
    api_key: str = field(repr=False)

    async def ask(self, unit: Unit) -> JevAnswer:
        """Jev's probability that *unit* is natural usage of its entry."""
        answer = await self.poster.post(
            JEV_URL,
            {
                "model": JEV_MODEL,
                "state": {
                    "entry": unit.entry,
                    "sentence": unit.text,
                    "preceding_text": unit.context,
                },
                "questions": {"natural_usage": QUESTION},
            },
            {"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            p = float(answer["answers"]["natural_usage"]["noul"])
            model = str(answer["model"])
        except (KeyError, TypeError, ValueError) as error:
            raise JudgeError(f"Jev answered {unit.key} with {answer!r}") from error
        if not 0.0 <= p <= 1.0:
            raise JudgeError(f"Jev answered {unit.key} with probability {p}")
        return JevAnswer(p=p, model=model)


def review_template() -> Template:
    """The review prompt, kept in a file so it can be diffed and reviewed."""
    source = resources.files("mlime").joinpath(REVIEW_PROMPT).read_text(encoding="utf-8")
    template: Template = Template(source)
    return template


def parse_review(reply: str) -> ReviewAnswer:
    """The two judgements in the review model's reply: the JSON object it holds."""
    try:
        verdict = json.loads(reply[reply.index("{") : reply.rindex("}") + 1])
    except ValueError as error:
        raise JudgeError(f"the review reply holds no JSON object: {reply!r}") from error
    spun = verdict.get("context_spun") if isinstance(verdict, dict) else None
    natural = verdict.get("sentence_natural") if isinstance(verdict, dict) else None
    if not isinstance(spun, bool) or not isinstance(natural, bool):
        raise JudgeError(f"the review reply is not the two booleans asked for: {reply!r}")
    return ReviewAnswer(context_spun=spun, sentence_natural=natural)


@dataclass
class Review:
    """Asks an OpenAI-compatible chat model the review prompt about a unit."""

    poster: Poster
    base_url: str
    model: str
    api_key: str = field(repr=False)
    template: Template = field(default_factory=review_template, repr=False)

    async def ask(self, unit: Unit) -> ReviewAnswer:
        """Whether *unit*'s page is spun and its sentence natural, as the model sees them."""
        prompt = self.template.render(entry=unit.entry, context=unit.context, text=unit.text)
        answer = await self.poster.post(
            self.base_url.rstrip("/") + "/chat/completions",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
            {"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            reply = str(answer["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as error:
            raise JudgeError(f"the review model answered {unit.key} with {answer!r}") from error
        return parse_review(reply)


# ---------------------------------------------------------------------------
# The mined source on disk.


@dataclass(frozen=True)
class MinedLayout:
    """Where the mining run left a source's state, and where the judge writes."""

    root: Path

    @property
    def state(self) -> Path:
        """The source's run state."""
        return self.root / "ingest" / SOURCE

    @property
    def records(self) -> Path:
        """The mining record of every finished upstream file, ``NNNN.json``."""
        return self.state / "files"

    @property
    def unjudged(self) -> Path:
        """The shards mining wrote, waiting for the judge."""
        return self.state / "unjudged"

    @property
    def entry_rows(self) -> Path:
        """Each written record's entries, ``NNNN.tsv`` of ``id<TAB>entry``."""
        return self.state / "entries"

    @property
    def mined_entries(self) -> Path:
        """Every target with the count mining kept of it."""
        return self.state / "entries.tsv"

    @property
    def judge(self) -> Path:
        """The verdict logs, the kept entry rows and the judged files' censuses."""
        return self.state / "judge"

    @property
    def samples(self) -> Path:
        """The samples every training stage reads."""
        return self.root / "samples"

    @property
    def judged_entries(self) -> Path:
        """Every target with what mining kept of it and what the judge kept."""
        return self.state / "judged-entries.tsv"

    @property
    def census(self) -> Path:
        """The judge's census summed over the judged files."""
        return self.judge / "census.json"

    def verdicts(self, ordinal: int) -> Path:
        """The verdict log of one mined file."""
        return self.judge / f"{ordinal:04d}.jsonl"

    def judged(self, ordinal: int) -> Path:
        """The census of one judged file, written last."""
        return self.judge / f"{ordinal:04d}.json"

    def kept_rows(self, ordinal: int) -> Path:
        """The entries the judge kept for each kept record of one file."""
        return self.judge / f"{ordinal:04d}-entries.tsv"

    def shards(self, ordinal: int) -> list[Path]:
        """The unjudged shards of one mined file."""
        return sorted(self.unjudged.glob(f"{SOURCE}-{ordinal:04d}???.parquet"))

    def mined_ordinals(self) -> list[int]:
        """The upstream files mining has finished."""
        return sorted(int(path.stem) for path in self.records.glob("*.json"))


@dataclass
class JudgeCensus:
    """What the judge counted over one mined file, or summed over many."""

    #: Records mining wrote.
    records: int = 0
    #: Entries of those records, each judged on its own.
    units: int = 0
    #: Units whose every occurrence lies across a longer word.
    boundary: int = 0
    #: Units Jev gave less than the threshold.
    jev: int = 0
    #: Units the review model found on a spun page.
    review_spun: int = 0
    #: Units the review model found unnatural on a page it did not find spun.
    review_unnatural: int = 0
    #: Units kept.
    kept_units: int = 0
    #: Records with a kept unit: what the samples hold.
    kept_records: int = 0

    def add(self, other: JudgeCensus) -> None:
        """Add another census into this one."""
        for name in (item.name for item in fields(self)):
            setattr(self, name, getattr(self, name) + getattr(other, name))


def load_units(layout: MinedLayout, ordinal: int) -> tuple[list[Unit], pl.DataFrame]:
    """One mined file's units, and its unjudged rows, checked against its mining record."""
    record = json.loads(layout.records.joinpath(f"{ordinal:04d}.json").read_text(encoding="utf-8"))
    written = int(record["census"]["written"])
    shards = layout.shards(ordinal)
    frame = (
        pl.concat([pl.read_parquet(path) for path in shards])
        if shards
        else pl.DataFrame(schema={"id": pl.String, "source": pl.String, "text": pl.String})
    )
    if frame.height != written:
        raise JudgeError(
            f"mined file {ordinal:04d} wrote {written} records, but {layout.unjudged} holds "
            f"{frame.height} of them; were its shards moved out of the unjudged directory?"
        )
    if written == 0:
        return [], frame
    if set(frame.get_column("source").unique().to_list()) != {SOURCE}:
        raise JudgeError(f"mined file {ordinal:04d} holds records of another source")
    rows = pl.read_csv(
        layout.entry_rows / f"{ordinal:04d}.tsv",
        separator="\t",
        has_header=False,
        schema=ENTRY_ROW_SCHEMA,
    )
    ids = set(frame.get_column("id").to_list())
    if set(rows.get_column("id").to_list()) != ids:
        raise JudgeError(f"mined file {ordinal:04d}'s entry rows and records name other records")
    joined = rows.join(frame, on="id", how="left", maintain_order="left")
    units = [
        Unit(
            ordinal=ordinal,
            id=row["id"],
            entry=row["entry"],
            text=row["text"],
            context=(row["context"] or "")[-CONTEXT_CHARACTERS:],
        )
        for row in joined.iter_rows(named=True)
    ]
    return units, frame


def read_verdicts(path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """A verdict log by unit; a last line cut short by a crash is dropped from the file."""
    if not path.exists():
        return {}
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        cut = raw.rfind(b"\n") + 1
        log.warning("dropping a verdict line cut short", path=str(path), bytes=len(raw) - cut)
        with path.open("r+b") as handle:
            handle.truncate(cut)
        raw = raw[:cut]
    verdicts: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for line in raw.decode("utf-8").splitlines():
        verdict = json.loads(line)
        verdicts.setdefault((verdict["id"], verdict["entry"]), []).append(verdict)
    return verdicts


def census_of(
    units: list[Unit], verdicts: Mapping[tuple[str, str], list[dict[str, Any]]]
) -> tuple[JudgeCensus, dict[str, list[str]]]:
    """A decided file's census, and the entries kept for each kept record."""
    census = JudgeCensus(records=len({unit.id for unit in units}), units=len(units))
    kept: dict[str, list[str]] = {}
    for unit in units:
        lines = verdicts[unit.key]
        if next_stage(lines) is not None:
            raise JudgeError(f"{unit.key} of file {unit.ordinal:04d} is not decided yet")
        last = deciding(lines)[-1]
        if passes(last):
            census.kept_units += 1
            kept.setdefault(unit.id, []).append(unit.entry)
        elif last["stage"] == "boundary":
            census.boundary += 1
        elif last["stage"] == "jev":
            census.jev += 1
        elif last["context_spun"]:
            census.review_spun += 1
        else:
            census.review_unnatural += 1
    census.kept_records = len(kept)
    return census, kept


def write_aside(path: Path, write: Callable[[Path], object]) -> None:
    """Write *path* through a sibling and rename it into place."""
    staged = path.with_name(path.name + ".part")
    write(staged)
    staged.replace(path)


def finish_file(
    layout: MinedLayout,
    ordinal: int,
    units: list[Unit],
    verdicts: Mapping[tuple[str, str], list[dict[str, Any]]],
) -> JudgeCensus:
    """Write a decided file's kept records to the samples, its kept entries, then its census."""
    census, kept = census_of(units, verdicts)
    layout.samples.mkdir(parents=True, exist_ok=True)
    keep = sorted(kept)
    for shard in layout.shards(ordinal):
        frame = pl.read_parquet(shard).filter(pl.col("id").is_in(keep))
        target = layout.samples / shard.name
        if frame.height:
            write_aside(target, frame.write_parquet)
        elif target.exists():
            target.unlink()
    kept_rows = pl.DataFrame(
        [(key, entry) for key in keep for entry in kept[key]], schema=ENTRY_ROW_SCHEMA, orient="row"
    )
    write_aside(
        layout.kept_rows(ordinal),
        lambda path: kept_rows.write_csv(path, separator="\t", include_header=False),
    )
    entries: dict[str, int] = {}
    for key in keep:
        for entry in kept[key]:
            entries[entry] = entries.get(entry, 0) + 1
    write_aside(
        layout.judged(ordinal),
        lambda path: path.write_text(
            json.dumps(
                {
                    "ordinal": ordinal,
                    "jev_threshold": JEV_THRESHOLD,
                    "census": asdict(census),
                    "entries": entries,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        ),
    )
    log.info("mined file judged", ordinal=ordinal, **asdict(census))
    return census


def write_summary(layout: MinedLayout) -> JudgeCensus:
    """Sum the judged files into the census and the per-entry table."""
    total = JudgeCensus()
    kept: dict[str, int] = {}
    judged = sorted(layout.judge.glob("[0-9][0-9][0-9][0-9].json"))
    for path in judged:
        record = json.loads(path.read_text(encoding="utf-8"))
        total.add(JudgeCensus(**record["census"]))
        for entry, count in record["entries"].items():
            kept[entry] = kept.get(entry, 0) + count
    mined = layout.mined_ordinals()
    write_aside(
        layout.census,
        lambda path: path.write_text(
            json.dumps(
                {"files_mined": len(mined), "files_judged": len(judged), "census": asdict(total)},
                indent=2,
            ),
            encoding="utf-8",
        ),
    )
    table = pl.read_csv(
        layout.mined_entries,
        separator="\t",
        schema_overrides={"entry": pl.String, "category": pl.String, "kept": pl.Int64},
    ).rename({"kept": "mined"})
    unknown = set(kept) - set(table.get_column("entry").to_list())
    if unknown:
        raise JudgeError(f"the judge kept entries mining does not list: {sorted(unknown)[:10]}")
    table = table.with_columns(
        pl.col("entry").replace_strict(kept, default=0, return_dtype=pl.Int64).alias("kept")
    )
    write_aside(layout.judged_entries, lambda path: table.write_csv(path, separator="\t"))
    return total


# ---------------------------------------------------------------------------
# The run.


@dataclass
class Progress:
    """Counts decided units and logs the rate now and then."""

    total: int
    started: float = field(default_factory=time.monotonic)
    decided: int = 0

    def tick(self) -> None:
        """One more unit decided."""
        self.decided += 1
        if self.decided % PROGRESS_EVERY == 0 or self.decided == self.total:
            elapsed = time.monotonic() - self.started
            log.info(
                "judging",
                decided=self.decided,
                of=self.total,
                per_second=round(self.decided / elapsed, 2) if elapsed else None,
            )


@dataclass
class FileState:
    """One mined file being judged: its units, their verdicts, and its open log."""

    ordinal: int
    units: list[Unit]
    verdicts: dict[tuple[str, str], list[dict[str, Any]]]
    log_path: Path
    open_units: int = 0
    finished: bool = False

    def record(self, line: dict[str, object]) -> None:
        """Append one verdict to the log and to memory, before anything acts on it."""
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
        self.verdicts.setdefault((str(line["id"]), str(line["entry"])), []).append(dict(line))


def finished_at_threshold(layout: MinedLayout, ordinal: int) -> bool:
    """Whether a mined file was judged under the threshold in force now."""
    path = layout.judged(ordinal)
    if not path.exists():
        return False
    threshold = json.loads(path.read_text(encoding="utf-8")).get("jev_threshold")
    if threshold == JEV_THRESHOLD:
        return True
    log.warning(
        "judging a file again under another Jev threshold",
        ordinal=ordinal,
        judged_at=threshold,
        threshold=JEV_THRESHOLD,
    )
    return False


async def judge_source(
    layout: MinedLayout,
    lexicon: BoundaryLexicon,
    jev: Asker[JevAnswer],
    review: Asker[ReviewAnswer],
    jev_concurrency: int = JEV_CONCURRENCY,
    review_concurrency: int = REVIEW_CONCURRENCY,
) -> JudgeCensus:
    """Judge every finished mined file not yet judged, resuming from the verdict logs."""
    if jev_concurrency < 1 or review_concurrency < 1:
        raise ValueError("each judge needs at least one request in flight")
    layout.judge.mkdir(parents=True, exist_ok=True)
    states: dict[int, FileState] = {}
    jev_queue: asyncio.Queue[Unit] = asyncio.Queue()
    review_queue: asyncio.Queue[Unit | None] = asyncio.Queue()
    for ordinal in layout.mined_ordinals():
        if finished_at_threshold(layout, ordinal):
            continue
        units, _ = load_units(layout, ordinal)
        if len({unit.key for unit in units}) != len(units):
            raise JudgeError(f"mined file {ordinal:04d} lists an entry twice for one record")
        state = FileState(
            ordinal, units, read_verdicts(layout.verdicts(ordinal)), layout.verdicts(ordinal)
        )
        states[ordinal] = state
        for unit in units:
            lines = state.verdicts.get(unit.key, [])
            stage = next_stage(lines)
            if stage == "boundary":
                crossing = lexicon.false_match(unit.text, unit.entry)
                if crossing is None:
                    state.record(verdict_line(unit, "boundary", True))
                    stage = "jev"
                else:
                    state.record(verdict_line(unit, "boundary", False, crossing=crossing))
                    stage = None
            if stage == "jev":
                jev_queue.put_nowait(unit)
            elif stage == "review":
                review_queue.put_nowait(unit)
            if stage is not None:
                state.open_units += 1
    progress = Progress(total=jev_queue.qsize() + review_queue.qsize())
    log.info(
        "judge starting",
        files=len(states),
        jev=jev_queue.qsize(),
        review=review_queue.qsize(),
        jev_concurrency=jev_concurrency,
        review_concurrency=review_concurrency,
    )

    async def decided(unit: Unit) -> None:
        progress.tick()
        state = states[unit.ordinal]
        state.open_units -= 1
        if state.open_units == 0:
            await asyncio.to_thread(finish_file, layout, state.ordinal, state.units, state.verdicts)
            state.finished = True

    async def jev_worker() -> None:
        while True:
            try:
                unit = jev_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            asked = time.monotonic()
            answer = await jev.ask(unit)
            line = jev_line(unit, answer, round(time.monotonic() - asked, 3))
            states[unit.ordinal].record(line)
            if passes(line):
                review_queue.put_nowait(unit)
            else:
                await decided(unit)

    async def review_worker() -> None:
        while (unit := await review_queue.get()) is not None:
            asked = time.monotonic()
            answer = await review.ask(unit)
            keep = answer.sentence_natural and not answer.context_spun
            states[unit.ordinal].record(
                verdict_line(
                    unit,
                    "review",
                    keep,
                    context_spun=answer.context_spun,
                    sentence_natural=answer.sentence_natural,
                    seconds=round(time.monotonic() - asked, 3),
                )
            )
            await decided(unit)

    async def jev_stage() -> None:
        async with asyncio.TaskGroup() as group:
            for _ in range(jev_concurrency):
                group.create_task(jev_worker())
        for _ in range(review_concurrency):
            review_queue.put_nowait(None)

    async with asyncio.TaskGroup() as group:
        group.create_task(jev_stage())
        for _ in range(review_concurrency):
            group.create_task(review_worker())

    # A file every unit of which was decided before this run (or by rule 1) finishes here.
    for state in states.values():
        if state.open_units == 0 and not state.finished:
            finish_file(layout, state.ordinal, state.units, state.verdicts)
    return write_summary(layout)


async def judge_with_endpoints(
    layout: MinedLayout,
    lexicon: BoundaryLexicon,
    jev_concurrency: int = JEV_CONCURRENCY,
    review_concurrency: int = REVIEW_CONCURRENCY,
) -> JudgeCensus:
    """:func:`judge_source` against the real endpoints, configured from the environment."""
    jev_settings = JevSettings.load()
    review_settings = ReviewSettings.load()
    jev_limits = httpx.Limits(max_connections=jev_concurrency)
    review_limits = httpx.Limits(max_connections=review_concurrency)
    async with (
        httpx.AsyncClient(timeout=JEV_TIMEOUT_SECONDS, limits=jev_limits) as jev_client,
        httpx.AsyncClient(timeout=REVIEW_TIMEOUT_SECONDS, limits=review_limits) as review_client,
    ):
        jev_poster = Poster(jev_client)
        review_poster = Poster(review_client)
        started = time.monotonic()
        census = await judge_source(
            layout,
            lexicon,
            Jev(jev_poster, jev_settings.api_key),
            Review(
                review_poster,
                review_settings.base_url,
                review_settings.model,
                review_settings.api_key(),
            ),
            jev_concurrency,
            review_concurrency,
        )
        log.info(
            "judge finished",
            seconds=round(time.monotonic() - started, 1),
            jev_retries=jev_poster.retries,
            review_retries=review_poster.retries,
            **asdict(census),
        )
    return census
