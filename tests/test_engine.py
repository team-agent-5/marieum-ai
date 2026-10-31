"""server/engine.py의 줄 해석(row_result) 단위 테스트. 모델 없이 돈다 (torch·transformers는 import에 필요).

  python -m pytest tests -q
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from server.engine import row_result  # noqa: E402

EOT = 50257
NOTIMESTAMPS = 50363
SPECIAL = {EOT, 50258, 50264, 50359, NOTIMESTAMPS}


def test_capped_row_padded_with_eot_is_not_ended():
    lp, ended = row_result([10, 11, 12, EOT, EOT, EOT], [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6], 3, EOT, SPECIAL)
    assert ended is False
    assert lp == [-0.1, -0.2, -0.3]


def test_eot_within_cap_is_natural_end():
    lp, ended = row_result([10, EOT, EOT, EOT, EOT, EOT], [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6], 5, EOT, SPECIAL)
    assert ended is True
    assert lp == [-0.1]


def test_special_tokens_have_no_logprob():
    lp, ended = row_result([10, NOTIMESTAMPS, 11, EOT], [-0.1, -0.9, -0.2, -0.3], 4, EOT, SPECIAL)
    assert ended is True
    assert lp == [-0.1, -0.2]
