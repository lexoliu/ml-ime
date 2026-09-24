"""The reranker's answer parsing and its evaluation arithmetic."""

import pytest

from mlime.rescore import Choice, DumpedRecord, Hypothesis, evaluate, parse_choice


def test_parse_choice_accepts_a_bare_number_in_range() -> None:
    assert parse_choice("3", 8) == 2
    assert parse_choice("答案：1", 8) == 0
    assert parse_choice(" 8\n", 8) == 7


@pytest.mark.parametrize("content", ["", "1 or 2", "0", "9", "第一个"])
def test_parse_choice_rejects_anything_but_one_number_in_range(content: str) -> None:
    with pytest.raises(ValueError, match="answer"):
        parse_choice(content, 8)


def _record(index: int, text: str, hypotheses: tuple[str, ...]) -> DumpedRecord:
    return DumpedRecord(
        index=index,
        text=text,
        pinyin="",
        context=None,
        hypotheses=tuple(Hypothesis(text=h, score=-float(i)) for i, h in enumerate(hypotheses)),
    )


def test_evaluate_counts_beam_reranked_oracle_and_unanswered() -> None:
    choices = [
        # beam right, model right
        Choice(record=_record(0, "你好", ("你好", "拟好")), picked=0),
        # beam wrong, model right
        Choice(record=_record(1, "再见", ("在见", "再见")), picked=1),
        # beam wrong, model wrong, answer present
        Choice(record=_record(2, "谢谢", ("写写", "谢谢")), picked=0),
        # answer absent, unanswered falls back to the beam
        Choice(record=_record(3, "晚安", ("完安", "万安")), picked=None, reason="timeout"),
    ]
    result = evaluate(choices)
    assert result.records == 4
    assert result.beam_top1 == 0.25
    assert result.reranked_top1 == 0.5
    assert result.oracle == 0.75
    assert result.unanswered == 1
