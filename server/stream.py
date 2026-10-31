"""스트리밍 세션의 구간 나누기와 interim 확정 규칙 (CALIBRATION.md §0.2, §0.3).

torch에 의존하지 않는다. VAD는 밖에서 주입한다: vad(windows[n, 512]) -> 말소리 확률[n].
테스트에서는 가짜 VAD를 넣는다 (tests/test_stream.py).
"""

import re
from collections import deque
from dataclasses import dataclass

import numpy as np

SR = 16000
WIN = 512  # Silero VAD 창 = 32ms

UTTERANCE = "utterance"  # 발화 1개 = 세션 1개, final 1회 (v4 기본)
CONVERSATION = "conversation"  # 실시간 대화: 쉼마다 구간 final, 세션은 계속
MODES = (UTTERANCE, CONVERSATION)


def _win(ms):
    return max(1, int(round(ms * SR / 1000 / WIN)))


def _ms(n_win):
    return int(n_win * WIN * 1000 // SR)


@dataclass
class Segment:
    seg_id: int
    audio: np.ndarray  # 앞뒤 무음을 자른 음성. 말소리가 없으면 빈 배열
    has_speech: bool
    duration_ms: int  # 자르기 전 구간 길이
    end_ms: int  # 세션 시작 기준 구간 끝 위치
    reason: str  # silence | max_length | end | too_long
    vad_speech_ms: int  # VAD 말소리 창 수 × 32ms. 구간 중간 쉼은 빠진다 (로그용. 상한 R 재측정에 쓴다)


@dataclass
class Snapshot:
    seg_id: int
    audio: np.ndarray
    audio_end_ms: int  # 세션 시작 기준, 이 음성이 끝나는 위치
    vad_speech_ms: int


class Segmenter:
    def __init__(self, mode, vad, cfg):
        assert mode in MODES
        self.mode, self.vad = mode, vad
        self.thr = cfg.vad_threshold
        self.pad_win = _win(cfg.speech_pad_ms)
        self.min_speech_win = _win(cfg.min_speech_ms)
        self.end_sil_win = _win(cfg.end_silence_ms)
        self.max_seg_win = _win(cfg.max_segment_s * 1000)
        self.step_win = _win(cfg.interim_step_ms)
        self.min_interim_win = _win(cfg.min_interim_speech_ms)
        limit_s = cfg.utterance_max_s if mode == UTTERANCE else cfg.session_max_s
        self.limit_win = _win(limit_s * 1000)
        self.cut_search_win = _win(5000)  # 최대 길이에 닿으면 마지막 5초 안의 가장 늦은 쉼에서 자른다

        self.carry = np.zeros(0, np.float32)
        self.total_win = 0
        self.seg_id = 0
        self.done = False
        self.preroll = deque(maxlen=self.pad_win)
        self._start_segment([], [], self.total_win) if mode == UTTERANCE else self._idle()

    # ---- 상태 -------------------------------------------------------------
    def _idle(self):
        self.in_seg = False
        self.windows, self.probs = [], []

    def _start_segment(self, windows, probs, start_win):
        self.in_seg = True
        self.windows, self.probs = list(windows), list(probs)
        self.seg_start_win = start_win
        self.speech_win = sum(p >= self.thr for p in self.probs)
        self.silence_run = 0
        for p in reversed(self.probs):
            if p >= self.thr:
                break
            self.silence_run += 1
        self.last_snap = 0
        self.speech_since_snap = self.speech_win > 0

    @property
    def elapsed_ms(self):
        return _ms(self.total_win)

    # ---- 입력 -------------------------------------------------------------
    def feed(self, samples):
        """16kHz float32 음성을 넣는다. 닫힌 구간 목록을 돌려준다."""
        windows = self.take_windows(samples)
        return self.push_windows(windows, self.vad(windows) if len(windows) else [])

    # 서버는 아래 둘을 나눠 부른다: take_windows + VAD는 작업 스레드에서, push_windows는 이벤트 루프에서.
    # 구간 상태(windows, probs 등)를 이벤트 루프에서만 바꿔야 interim snapshot과 겹치지 않는다.
    def take_windows(self, samples):
        buf = np.concatenate([self.carry, samples.astype(np.float32, copy=False)])
        n = len(buf) // WIN
        self.carry = buf[n * WIN:]
        return buf[: n * WIN].reshape(n, WIN)

    def push_windows(self, windows, probs):
        if self.done:
            return []
        out = []
        for w, p in zip(windows, probs):
            out += self._push(w, float(p))
            if self.done:
                break
        return out

    def _push(self, w, p):
        self.total_win += 1
        speech = p >= self.thr
        if not self.in_seg:  # 대화 모드에서 말 시작 전
            if speech:
                pre = list(self.preroll)
                self.preroll.clear()
                self._start_segment(pre + [w], [0.0] * len(pre) + [p], self.total_win - len(pre) - 1)
            else:
                self.preroll.append(w)
            return self._check_limit()

        self.windows.append(w)
        self.probs.append(p)
        if speech:
            self.speech_win += 1
            self.silence_run = 0
            self.speech_since_snap = True
        else:
            self.silence_run += 1

        if self.mode == CONVERSATION:
            if self.silence_run >= self.end_sil_win:
                if self.speech_win >= self.min_speech_win:
                    return [self._close("silence")] + self._check_limit()
                self._drop_noise()
            elif len(self.windows) >= self.max_seg_win:
                return [self._cut()] + self._check_limit()
        return self._check_limit()

    def _check_limit(self):
        if self.total_win < self.limit_win:
            return []
        self.done = True
        if self.in_seg:
            return [self._close("too_long")]
        return [self._empty("too_long")]

    def _drop_noise(self):
        # 말소리가 너무 짧은 구간(숨소리, 클릭음)은 구간을 열지 않은 것으로 친다
        self.preroll.extend(self.windows[-self.pad_win:])
        self._idle()

    # ---- 구간 닫기 ---------------------------------------------------------
    def _close(self, reason):
        idx = [i for i, p in enumerate(self.probs) if p >= self.thr]
        has_speech = len(idx) >= self.min_speech_win
        if has_speech:
            a = max(0, idx[0] - self.pad_win)
            b = min(len(self.windows), idx[-1] + 1 + self.pad_win)
            audio = np.concatenate(self.windows[a:b])
        else:
            audio = np.zeros(0, np.float32)
        seg = Segment(self.seg_id, audio, has_speech, _ms(len(self.windows)),
                      _ms(self.seg_start_win + len(self.windows)), reason, _ms(len(idx)))
        self.seg_id += 1
        if self.mode == CONVERSATION:
            self.preroll.clear()
            self.preroll.extend(self.windows[-self.pad_win:])
            self._idle()
        else:
            self.done = True
        return seg

    def _empty(self, reason):
        seg = Segment(self.seg_id, np.zeros(0, np.float32), False, 0, self.elapsed_ms, reason, 0)
        self.seg_id += 1
        return seg

    def _cut(self):
        """구간이 최대 길이에 닿았다. 마지막 5초 안의 가장 늦은 쉼에서 자르고 나머지는 다음 구간으로."""
        lo = max(1, len(self.windows) - self.cut_search_win)
        quiet = [i for i in range(lo, len(self.windows)) if self.probs[i] < self.thr]
        cut = quiet[-1] if quiet else len(self.windows)  # 쉼이 없으면 그 자리에서 자른다
        rest_w, rest_p = self.windows[cut:], self.probs[cut:]
        rest_start = self.seg_start_win + cut
        self.windows, self.probs = self.windows[:cut], self.probs[:cut]
        seg = self._close("max_length")
        self._start_segment(rest_w, rest_p, rest_start)
        return seg

    def finish(self):
        """end 수신. 마지막 구간을 닫는다 (말소리가 없어도 Segment를 돌려준다)."""
        if self.done:
            return None
        self.done = True
        if self.in_seg:
            if len(self.carry):  # 32ms보다 짧은 꼬리는 VAD 없이 무음으로 붙인다
                self.windows.append(self.carry)
                self.probs.append(0.0)
            return self._close("end")
        return self._empty("end")

    # ---- interim ---------------------------------------------------------
    def interim_ready(self):
        return (not self.done and self.in_seg and self.speech_since_snap
                and self.speech_win >= self.min_interim_win
                and len(self.windows) - self.last_snap >= self.step_win)

    def snapshot(self):
        self.last_snap = len(self.windows)
        self.speech_since_snap = False
        first = next(i for i, p in enumerate(self.probs) if p >= self.thr)
        a = max(0, first - self.pad_win)  # 발화 모드는 세션 시작부터 쌓이므로 앞 무음을 자른다
        return Snapshot(self.seg_id, np.concatenate(self.windows[a:]),
                        _ms(self.seg_start_win + len(self.windows)), _ms(self.speech_win))


_PUNCT = re.compile(r"[.,?!~·…]")


def _norm(word):
    return _PUNCT.sub("", word)


class Agreement:
    """LocalAgreement-2: 연속한 두 interim이 같게 낸 앞부분만 확정(stable)한다.

    - 확정은 늘기만 하고 줄지 않는다 (화면에서 확정 글자가 바뀌지 않게)
    - 현재 가설의 마지막 단어는 확정하지 않는다 ("화장" → "화장실에"처럼 말이 이어질 수 있음)
    - final은 이 규칙과 무관하게 구간 전체를 다시 전사한 결과다. 드물게 확정분과 다를 수 있다
    """

    def __init__(self):
        self.committed = []
        self.prev = None

    def update(self, text):
        words = text.split()
        n = len(self.committed)
        consistent = [_norm(w) for w in words[:n]] == [_norm(w) for w in self.committed]
        if self.prev is not None and consistent:
            lcp = 0
            for a, b in zip(self.prev, words):
                if _norm(a) != _norm(b):
                    break
                lcp += 1
            lcp = min(lcp, len(words) - 1)
            if lcp > n:
                self.committed = words[:lcp]
        self.prev = words
        n = len(self.committed)
        tail = words[n:] if len(words) > n else []
        return " ".join(self.committed), " ".join(tail)


def word_delta(sent, text):
    """이미 내려보낸 단어(sent) 뒤로 새로 붙은 단어만 돌려준다: (새 단어 목록, revised).

    text가 sent로 시작하지 않으면(앞에서 보낸 단어와 달라졌으면) text 전체와 revised=True.
    구두점은 무시하고 비교한다 ("주세요"와 "주세요."는 같다).
    """
    words = text.split()
    n = len(sent)
    if [_norm(w) for w in words[:n]] == [_norm(w) for w in sent]:
        return words[n:], False
    return words, True
