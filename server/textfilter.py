"""전사 글자 후처리 (2026-11-01 백엔드 피드백 ④⑤).

- clean_text: 토큰이 한글 한 글자 중간에서 끝나면 디코딩에 U+FFFD가 남는다 → 지운다
- too_fast: 말소리 길이에 비해 글자가 너무 많으면 모델이 지어낸 문장으로 본다.
  이 모델은 잡음·짧은 소리에서 학습 문장 한 줄("코끼리는 귀가 아주 큽니다")을 통째로 낸다
"""

import re

_SPACES = re.compile(r"\s+")
_HANGUL = re.compile(r"[가-힣]")


def clean_text(text):
    return _SPACES.sub(" ", text.replace("\ufffd", "")).strip()


def syllables(text):
    return len(_HANGUL.findall(text))


def too_fast(text, speech_ms, max_per_s):
    """한글 음절 수 / 말소리 초 > max_per_s 이면 True. 말소리는 VAD가 말이라고 본 창만 센다.

    VAD가 작은 말소리를 덜 셀 수 있어 말소리는 최소 1초로 계산한다 (max_per_s=10이면 10음절 이하는 안 버린다)
    """
    if max_per_s <= 0:
        return False
    return syllables(text) > max_per_s * max(speech_ms, 1000) / 1000
