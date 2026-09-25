"""The generator's answer parsing and its evaluation arithmetic."""

import pytest

from mlime.generate import Generation, characters_right, evaluate, parse_sentence
from mlime.rescore import DumpedRecord, Hypothesis


def test_parse_sentence_takes_the_first_line_without_quotes_or_final_punctuation() -> None:
    assert parse_sentence("你可以看看") == "你可以看看"
    assert parse_sentence("「你可以看看。」\n（根据上文）") == "你可以看看"
    assert parse_sentence("\n  “二次元辣么美”！ \n") == "二次元辣么美"


@pytest.mark.parametrize("content", ["", "\n\n", "。", "你可以 看看", '""'])
def test_parse_sentence_rejects_empty_or_broken_answers(content: str) -> None:
    with pytest.raises(ValueError, match="answer"):
        parse_sentence(content)


def test_characters_right_is_the_length_less_the_edit_distance_floored_at_zero() -> None:
    assert characters_right("你可以看看", "你可以看看") == 5
    assert characters_right("你可以看看", "你可以看完") == 4
    assert characters_right("你可以看看", "你可以看") == 4
    assert characters_right("你好", "再见了吗") == 0


def _record(index: int, text: str, hypotheses: tuple[str, ...]) -> DumpedRecord:
    return DumpedRecord(
        index=index,
        text=text,
        pinyin="",
        context=None,
        hypotheses=tuple(Hypothesis(text=h, score=-float(i)) for i, h in enumerate(hypotheses)),
    )


def test_evaluate_counts_generated_beam_oracle_characters_and_gaps() -> None:
    generations = [
        # beam right, model right
        Generation(record=_record(0, "你好", ("你好", "拟好")), text="你好"),
        # beam wrong, answer in the beam, model right
        Generation(record=_record(1, "再见", ("在见", "再见")), text="再见"),
        # answer absent from the beam, model one character off
        Generation(record=_record(2, "谢谢你", ("写写你", "谢写你")), text="谢谢您"),
        # model wrote the wrong length
        Generation(record=_record(3, "晚安", ("完安", "万安")), text="晚安了"),
        # unanswered
        Generation(record=_record(4, "早", ("找", "早")), text=None, reason="timeout"),
    ]
    result = evaluate(generations)
    assert result.records == 5
    assert result.generated_top1 == pytest.approx(2 / 5)
    assert result.beam_top1 == pytest.approx(1 / 5)
    assert result.oracle == pytest.approx(3 / 5)
    assert result.characters == 10
    assert result.characters_right == pytest.approx((2 + 2 + 2 + 1 + 0) / 10)
    assert result.length_mismatches == 1
    assert result.unanswered == 1
