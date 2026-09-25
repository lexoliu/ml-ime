"""The RIME harness's evaluation arithmetic; the library itself is exercised by the command."""

import pytest

from mlime.rescore import DumpedRecord, Hypothesis
from mlime.rime import Typed, evaluate


def _record(index: int, text: str, hypotheses: tuple[str, ...]) -> DumpedRecord:
    return DumpedRecord(
        index=index,
        text=text,
        pinyin="",
        context=None,
        hypotheses=tuple(Hypothesis(text=h, score=-float(i)) for i, h in enumerate(hypotheses)),
    )


def test_evaluate_counts_rime_first_page_beam_characters_and_lengths() -> None:
    typed = [
        # both right, expected on the first page
        Typed(
            record=_record(0, "你好", ("你好", "拟好")), committed="你好", first_page=("你好", "你")
        ),
        # rime wrong by one character, expected not on the first page, beam right
        Typed(
            record=_record(1, "再见", ("再见", "在见")), committed="在见", first_page=("在见", "在")
        ),
        # rime committed a shorter sentence
        Typed(
            record=_record(2, "谢谢你", ("写写你",)),
            committed="谢谢",
            first_page=("谢谢", "谢谢你"),
        ),
    ]
    result = evaluate(typed)
    assert result.records == 3
    assert result.rime_top1 == pytest.approx(1 / 3)
    assert result.first_page_exact == pytest.approx(2 / 3)
    assert result.beam_top1 == pytest.approx(2 / 3)
    assert result.characters == 7
    assert result.characters_right == pytest.approx((2 + 1 + 2) / 7)
    assert result.length_mismatches == 1
