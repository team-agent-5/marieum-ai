"""server/stream.py 단위 테스트. GPU·모델 없이 가짜 VAD(에너지 기준)로 돈다.

  python -m pytest tests -q
"""

import numpy as np

from server.config import Settings
from server.stream import CONVERSATION, SR, UTTERANCE, Agreement, Segmenter

CFG = Settings()


def fake_vad(windows):
    return [1.0 if np.sqrt((w**2).mean()) > 0.01 else 0.0 for w in windows]


def tone(s):
    t = np.arange(int(s * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(s):
    return np.zeros(int(s * SR), np.float32)


def feed_frames(seg, audio, frame_ms=100):
    out, step = [], SR * frame_ms // 1000
    for i in range(0, len(audio), step):
        out += seg.feed(audio[i:i + step])
    return out


def test_utterance_one_final_trimmed():
    seg = Segmenter(UTTERANCE, fake_vad, CFG)
    closed = feed_frames(seg, np.concatenate([silence(1.0), tone(2.0), silence(1.0)]))
    assert closed == []
    last = seg.finish()
    assert last.has_speech and last.reason == "end" and last.seg_id == 0
    # 앞뒤 무음을 자르고 패딩 200ms씩만 남긴다
    assert abs(len(last.audio) / SR - 2.4) < 0.1
    assert seg.finish() is None


def test_utterance_silence_only_has_no_speech():
    seg = Segmenter(UTTERANCE, fake_vad, CFG)
    feed_frames(seg, silence(3.0))
    last = seg.finish()
    assert not last.has_speech and len(last.audio) == 0


def test_utterance_too_long_closes_session():
    seg = Segmenter(UTTERANCE, fake_vad, CFG)
    closed = feed_frames(seg, tone(10.5))
    assert len(closed) == 1 and closed[0].reason == "too_long"
    assert seg.done and seg.finish() is None


def test_conversation_splits_on_pause():
    seg = Segmenter(CONVERSATION, fake_vad, CFG)
    audio = np.concatenate([silence(0.5), tone(1.5), silence(2.8), tone(1.0), silence(0.3)])
    closed = feed_frames(seg, audio)
    assert [c.reason for c in closed] == ["silence"]
    assert closed[0].seg_id == 0 and closed[0].has_speech
    last = seg.finish()
    assert last.seg_id == 1 and last.has_speech and last.reason == "end"


def test_conversation_pause_inside_sentence_is_kept():
    # 1.75초보다 짧은 쉼은 문장을 끊지 않는다 (구음장애 발화는 중간 쉼이 길다)
    seg = Segmenter(CONVERSATION, fake_vad, CFG)
    closed = feed_frames(seg, np.concatenate([tone(1.0), silence(1.5), tone(1.0)]))
    assert closed == []
    assert seg.finish().seg_id == 0


def test_conversation_drops_short_noise():
    seg = Segmenter(CONVERSATION, fake_vad, CFG)
    closed = feed_frames(seg, np.concatenate([tone(0.06), silence(1.5)]))
    assert closed == []
    assert not seg.finish().has_speech


def test_conversation_max_segment_cut_keeps_rest():
    seg = Segmenter(CONVERSATION, fake_vad, Settings(max_segment_s=10.0))
    closed = feed_frames(seg, np.concatenate([tone(6.0), silence(0.3), tone(6.0)]))
    assert len(closed) == 1 and closed[0].reason == "max_length"
    # 마지막 5초 안의 쉼(6.0–6.3초)에서 잘라야 한다
    assert 5.9 < closed[0].duration_ms / 1000 < 6.4
    rest = seg.finish()
    assert rest.seg_id == 1 and rest.has_speech


def test_interim_cadence_and_snapshot():
    # 길이를 VAD 창(32ms) 배수로 맞춘다. 안 맞으면 말소리 꼬리가 다음 창에 섞여 "새 말소리"로 잡힌다
    seg = Segmenter(CONVERSATION, fake_vad, CFG)
    feed_frames(seg, silence(0.512))
    assert not seg.interim_ready()
    feed_frames(seg, tone(0.192))
    assert not seg.interim_ready()  # 말소리 300ms 미만
    feed_frames(seg, tone(0.384))
    assert seg.interim_ready()
    snap = seg.snapshot()
    assert snap.seg_id == 0 and 0.6 < len(snap.audio) / SR < 1.0
    assert not seg.interim_ready()
    feed_frames(seg, silence(0.608))
    assert not seg.interim_ready()  # 새 말소리가 없으면 다시 돌리지 않는다
    feed_frames(seg, tone(0.608))
    assert seg.interim_ready()


def test_conversation_default_max_segment_hard_cut():
    seg = Segmenter(CONVERSATION, fake_vad, CFG)  # 15초, 쉼 없음 → 그 자리에서 자른다
    closed = feed_frames(seg, tone(31.0))
    assert [c.reason for c in closed] == ["max_length", "max_length"]
    assert all(14.9 < c.duration_ms / 1000 < 15.1 for c in closed)
    assert seg.finish().has_speech


def test_agreement_commits_only_repeated_prefix():
    a = Agreement()
    assert a.update("미지근한") == ("", "미지근한")
    assert a.update("미지근한") == ("", "미지근한")  # 가설의 마지막 단어는 확정 안 함
    assert a.update("미지근한 물") == ("미지근한", "물")
    assert a.update("미지근한 물 주") == ("미지근한 물", "주")
    assert a.update("미지근한 물 주세요.") == ("미지근한 물", "주세요.")  # "주"는 "주세요."와 다르다
    # 앞부분이 뒤집혀도 확정분은 줄지 않는다
    assert a.update("미지그한 물 주세요")[0] == "미지근한 물"


def test_agreement_ignores_punctuation():
    a = Agreement()
    a.update("물 주세요.")
    assert a.update("물 주세요, 지금")[0] == "물 주세요,"


def test_word_delta():
    from server.stream import word_delta
    assert word_delta([], "물") == (["물"], False)
    assert word_delta(["물"], "물 좀") == (["좀"], False)
    assert word_delta(["물", "좀"], "물 좀") == ([], False)
    assert word_delta(["물", "좀"], "물 좀 주세요.") == (["주세요."], False)
    assert word_delta(["물", "좀"], "물, 좀 주세요") == (["주세요"], False)  # 구두점 무시
    assert word_delta(["믈"], "물 좀 주세요") == (["물", "좀", "주세요"], True)  # 앞에서 보낸 단어가 바뀜
