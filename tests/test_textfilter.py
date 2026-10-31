"""server/textfilter.py 단위 테스트 (백엔드 피드백 ④⑤)."""

from server.textfilter import clean_text, syllables, too_fast


def test_clean_text_removes_broken_hangul():
    assert clean_text("반지\ufffd") == "반지"
    assert clean_text(" 물 \ufffd 주세요 ") == "물 주세요"


def test_too_fast_flags_long_sentence_from_short_sound():
    s = "아들과 병원에 가보니 하필 늑막염이었다"
    assert syllables(s) == 17
    assert too_fast(s, 1000, 10)  # 1초 말소리에 17음절
    assert not too_fast(s, 2500, 10)
    assert not too_fast("물 주세요", 100, 10)  # 말소리는 최소 1초로 계산 → 10음절 이하는 안 버림
    assert not too_fast(s, 100, 0)  # 0이면 끔
