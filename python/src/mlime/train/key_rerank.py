"""Re-scoring a key decode's hypotheses by the sequence-level training score (S4, #278).

``mlime eval keys`` dumps each record's top hypotheses with the search's own
score. This reads such a dump back and scores the same sentences, and the
gold, by ``S(y)`` (:func:`~mlime.train.key_ranking.sentence_scores`), the score
:class:`~mlime.train.key_ranking.KeyRankingModel` trains: how often it puts
the gold first among what the search listed says whether the stage moved the
ranking, apart from what it did to the search.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from mlime.legality import MAX_ARC_KEYS, GoldArcs
from mlime.train.e2e_model import KeyRankingConfig
from mlime.train.key_eval import KeyDecoder, KeyRecord
from mlime.train.key_ranking import sentence_scores


@dataclass(frozen=True)
class RerankReport:
    """What re-scoring a key decode's hypotheses by ``S(y)`` does to its top-1 (S4, #278).

    ``records`` counts the dump's records; ``unreachable`` those whose gold
    sentence is not a legal reading of the keys (nothing scores it);
    ``listed`` those whose gold is among the dump's hypotheses (the top-k
    ceiling); ``searched`` those the search put first; ``reranked`` those
    ``S`` puts first among the hypotheses; and ``outscored`` those whose
    gold scores above every hypothesis the dump lists other than itself.
    """

    records: int
    unreachable: int
    listed: int
    searched: int
    reranked: int
    outscored: int

    def render(self) -> str:
        """The counts as the share of the dump's records."""
        lines = [f"records            {self.records}"]
        for name, count in (
            ("gold unreachable", self.unreachable),
            ("gold listed", self.listed),
            ("top-1, the search", self.searched),
            ("top-1, reranked by S", self.reranked),
            ("gold outscores the rest", self.outscored),
        ):
            lines.append(f"{name:<19}{count} ({count / max(self.records, 1):.2%})")
        return "\n".join(lines)


@torch.no_grad()
def rerank_dump(
    decoder: KeyDecoder,
    records: Sequence[KeyRecord],
    dump: Path,
    ranking: KeyRankingConfig,
) -> RerankReport:
    """Re-score the hypotheses of an ``eval keys`` *dump* by the training score ``S(y)``.

    *records* are the evaluation set's, indexed by the dump's ``record``. Each
    record's hypotheses and its gold sentence are scored teacher-forced over
    its keys with *ranking*'s ``beta`` and ``gamma`` -- the score
    :class:`KeyRankingModel` trains, so a model trained on it reads back here
    as the share of records it puts first -- with the arc costs of each
    sentence's own legal alignments (:meth:`~mlime.legality.Legality.gold_ids`).
    """
    model = decoder.model
    if model.training:
        raise ValueError("rerank a model in evaluation mode")
    index = {character: position for position, character in enumerate(decoder.lexicon.characters)}
    rows = {}
    # The kernel publishes its dumps gzipped; `eval keys` writes them plain.
    opened = gzip.open if dump.suffix == ".gz" else open
    with opened(dump, "rt", encoding="utf-8") as lines:
        for line in lines:
            row = json.loads(line)
            rows[int(row["record"])] = [str(each["text"]) for each in row["hypotheses"]]
    chosen = [records[number] for number in sorted(rows)]
    counts = dict.fromkeys(("unreachable", "listed", "searched", "reranked", "outscored"), 0)
    for group in decoder.groups(chosen):
        inputs = decoder.collator.inputs(
            [record.keys for record in group], [record.context for record in group]
        ).to(decoder.device)
        states = model.key_states(inputs)
        log_probs = model.ctc_log_probs(states.states)
        examples, width, columns = log_probs.shape
        flat = log_probs.reshape(-1, columns)
        key_index = torch.arange(examples * width, device=flat.device).view(examples, width)
        sentences: list[tuple[int, str, tuple[int, ...]]] = []
        arcs: list[GoldArcs | None] = []
        reachable = []
        for number, record in enumerate(group):
            listed = rows[record.record]
            texts = [record.text, *(text for text in listed if text != record.text)]
            gold = gold_arcs(decoder, record, record.text, index)
            reachable.append(gold is not None)
            counts["unreachable"] += gold is None
            counts["listed"] += record.text in listed
            counts["searched"] += bool(listed) and listed[0] == record.text
            for text in texts[1 if gold is None else 0 :]:
                ids = tuple(index[character] for character in text)
                sentences.append((number, text, ids))
                arcs.append(
                    gold if text == record.text else gold_arcs(decoder, record, text, index)
                )
        owner = torch.tensor([each[0] for each in sentences], device=flat.device)
        sequences, lengths = pad_sentences([each[2] for each in sentences], flat.device)
        cost = torch.full(
            (len(sentences), sequences.shape[1], width, MAX_ARC_KEYS),
            float("inf"),
            device=flat.device,
        )
        for item, arc in enumerate(arcs):
            if arc is None:
                raise ValueError(f"hypothesis {sentences[item][1]!r} is not a legal reading")
            dense = torch.from_numpy(arc.dense()).float()
            cost[item, : dense.shape[0], : dense.shape[1]] = dense.to(flat.device)
        keys = torch.tensor([len(record.keys) for record in group], device=flat.device)
        scores = sentence_scores(
            model,
            states,
            flat,
            key_index,
            keys,
            owner,
            sequences,
            lengths,
            cost,
            ranking.gamma,
            int(lengths.sum()),
        ).score(ranking.beta)
        position = 0
        for number, record in enumerate(group):
            texts = [text for who, text, _ in sentences if who == number]
            values = scores[position : position + len(texts)].tolist()
            position += len(texts)
            listed = rows[record.record]
            by_text = dict(zip(texts, values, strict=True))
            candidates = [text for text in listed if text in by_text]
            if candidates:
                counts["reranked"] += max(candidates, key=lambda text: by_text[text]) == record.text
            rest = [by_text[text] for text in candidates if text != record.text]
            if reachable[number] and (not rest or by_text[record.text] > max(rest)):
                counts["outscored"] += bool(listed)
    return RerankReport(records=len(chosen), **counts)


def gold_arcs(
    decoder: KeyDecoder, record: KeyRecord, text: str, index: dict[str, int]
) -> GoldArcs | None:
    """The legal alignments of *text* over *record*'s keys; ``None`` when it has none."""
    if not text or any(character not in index for character in text):
        return None
    return decoder.legality.gold_ids(record.keys, [index[each] for each in text], decoder.options)


def pad_sentences(
    sentences: Sequence[tuple[int, ...]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sentences as emission indices ``[I, M]`` (zero past the end) and their lengths ``[I]``."""
    width = max(map(len, sentences))
    padded = torch.zeros(len(sentences), width, dtype=torch.long)
    for row, ids in enumerate(sentences):
        padded[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return padded.to(device), torch.tensor([len(ids) for ids in sentences], device=device)
