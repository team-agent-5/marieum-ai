"""/asr-stream 통합 테스트. 실제 uvicorn 서버를 띄우고 WebSocket으로 붙는다.

GPU·모델 없이 돈다: Whisper 엔진은 가짜(음성 길이로 정해진 단어를 돌려줌), VAD는 에너지 기준.
확인하는 것: 메시지 형식, 두 모드의 final 규칙, 에러 코드, 연결 유지, interim 확정분 규칙.

  python -m pytest tests -q
"""

import json
import socket
import threading
import time
import uuid

import numpy as np
import pytest
import uvicorn
from websockets.sync.client import connect

import server.app as app_mod

SR = 16000
FRAME = SR // 10  # 100ms


class FakeEngine:
    """0.5초마다 단어 하나. '단어1 단어2 ...' 처럼 길이에 따라 앞부분이 유지되는 가설을 낸다."""

    load_s = 0.0

    def transcribe(self, audios, max_new_tokens):
        out = []
        for a in audios:
            rms = float(np.sqrt((a**2).mean())) if len(a) else 0.0
            n = max(1, int(len(a) / SR * 2))
            silent = rms < 0.01
            out.append({"text": "" if silent else " ".join(f"단어{i}" for i in range(1, n + 1)),
                        "confidence": 0.9, "avg_logprob": -0.1,
                        "no_speech_prob": 0.9 if silent else 0.0,
                        "gen_tokens": n + 1, "ended_eot": True})
        time.sleep(0.01)
        return out, {"feat_ms": 1.0, "gpu_ms": 10.0, "batch": len(audios)}


class FakeVadModel:
    def reset_states(self):
        pass


def energy_vad(_model):
    return lambda windows: [1.0 if np.sqrt((w**2).mean()) > 0.01 else 0.0 for w in windows]


def fake_load():
    app_mod.state["gpu"] = {"name": "fake", "memory_free_mb": 0, "memory_total_mb": 0}
    return FakeEngine()


@pytest.fixture(scope="module")
def url():
    saved = (app_mod.load_model, app_mod.make_vad, app_mod.vads.acquire)
    app_mod.load_model = fake_load
    app_mod.make_vad = energy_vad
    app_mod.vads.acquire = lambda: FakeVadModel()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app_mod.app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started and app_mod.state["status"] == "ok":
            break
        time.sleep(0.05)
    assert app_mod.state["status"] == "ok"
    yield f"ws://127.0.0.1:{port}/asr-stream"
    server.should_exit = True
    th.join(timeout=5)
    app_mod.load_model, app_mod.make_vad, app_mod.vads.acquire = saved


def pcm(*parts):
    """("tone", 초) / ("sil", 초) 조각을 PCM16 바이트로."""
    xs = []
    for kind, sec in parts:
        n = int(sec * SR)
        x = 0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / SR) if kind == "tone" else np.zeros(n)
        xs.append(x)
    return (np.concatenate(xs) * 32767).astype("<i2").tobytes()


def start(ws, mode="utterance", **kw):
    trace = kw.pop("trace_id", str(uuid.uuid4()))
    msg = {"type": "start", "trace_id": trace, "patient_id": "test", "sample_rate": SR,
           "encoding": "pcm16", "language": "ko", "mode": mode, **kw}
    ws.send(json.dumps(msg))
    return trace


def send_audio(ws, data):
    for i in range(0, len(data), FRAME * 2):
        ws.send(data[i:i + FRAME * 2])


def recv_until_last(ws, timeout=10):
    msgs = []
    end = time.time() + timeout
    while time.time() < end:
        m = json.loads(ws.recv(timeout=end - time.time()))
        msgs.append(m)
        if m["type"] == "transcript_final" and m["is_last"]:
            return msgs
    raise AssertionError(f"is_last final 없음: {msgs}")


def test_healthz_and_livez(url):
    import urllib.request
    base = url.replace("ws://", "http://").replace("/asr-stream", "")
    h = json.load(urllib.request.urlopen(base + "/healthz"))
    assert h["status"] == "ok" and h["models"] == ["asr"] and "asr" in h["model_version"]
    assert json.load(urllib.request.urlopen(base + "/livez"))["status"] == "alive"


def test_utterance_interims_then_one_final(url):
    with connect(url) as ws:
        trace = start(ws)
        send_audio(ws, pcm(("sil", 0.5), ("tone", 2.5), ("sil", 0.5)))
        time.sleep(0.5)  # interim이 나갈 시간
        ws.send(json.dumps({"type": "end", "trace_id": trace}))
        msgs = recv_until_last(ws)
    interims = [m for m in msgs if m["type"] == "transcript_interim"]
    finals = [m for m in msgs if m["type"] == "transcript_final"]
    assert len(finals) == 1 and finals[0]["segment_id"] == 0 and finals[0]["text"]
    assert all(m["trace_id"] == trace for m in msgs)
    f = finals[0]
    for key in ("confidence", "duration_ms", "model_version", "no_speech_prob"):
        assert key in f
    assert interims, "말하는 도중 interim이 와야 한다"
    for m in interims:
        assert m["is_final"] is False and m["segment_id"] == 0
        assert m["text"] == " ".join(t for t in (m["stable_text"], m["unstable_text"]) if t)
    # 확정분은 줄지 않는다
    stables = [m["stable_text"] for m in interims]
    assert all(b.startswith(a) for a, b in zip(stables, stables[1:]))


def test_utterance_silence_skips_whisper(url):
    with connect(url) as ws:
        trace = start(ws)
        send_audio(ws, pcm(("sil", 2.0)))
        ws.send(json.dumps({"type": "end", "trace_id": trace}))
        msgs = recv_until_last(ws)
    assert [m["type"] for m in msgs] == ["transcript_final"]
    f = msgs[0]
    assert f["text"] == "" and f["confidence"] == 0.0 and f["no_speech_prob"] == 1.0


def test_conversation_final_per_pause(url):
    with connect(url) as ws:
        trace = start(ws, mode="conversation")
        send_audio(ws, pcm(("tone", 1.0), ("sil", 1.2), ("tone", 1.0), ("sil", 0.2)))
        time.sleep(0.3)
        ws.send(json.dumps({"type": "end", "trace_id": trace}))
        msgs = recv_until_last(ws)
    finals = [m for m in msgs if m["type"] == "transcript_final"]
    assert [(f["segment_id"], f["is_last"]) for f in finals] == [(0, False), (1, True)]
    assert all(f["text"] for f in finals)
    # 구간 final 뒤에 그 구간 interim은 오지 않는다
    seen_final = set()
    for m in msgs:
        if m["type"] == "transcript_final":
            seen_final.add(m["segment_id"])
        elif m["type"] == "transcript_interim":
            assert m["segment_id"] not in seen_final


def test_utterance_too_long(url):
    with connect(url) as ws:
        trace = start(ws)
        send_audio(ws, pcm(("tone", 10.6)))
        msgs = recv_until_last(ws)
        types = [m["type"] for m in msgs if m["type"] != "transcript_interim"]
        assert types == ["error", "transcript_final"]
        assert msgs[[m["type"] for m in msgs].index("error")]["code"] == "AUDIO_TOO_LONG"
        # 자동 종료 뒤 남은 오디오와 늦은 end는 조용히 버린다
        send_audio(ws, pcm(("tone", 0.3)))
        ws.send(json.dumps({"type": "end", "trace_id": trace}))
        ws.send(json.dumps({"type": "ping", "trace_id": "k1"}))
        assert json.loads(ws.recv(timeout=5)) == {"type": "pong", "trace_id": "k1"}


@pytest.mark.parametrize("override,code", [
    ({"encoding": "opus"}, "INVALID_FORMAT"),
    ({"sample_rate": 96000}, "INVALID_FORMAT"),
    ({"mode": "dictation"}, "INVALID_FORMAT"),
    ({"language": "en"}, "INVALID_FORMAT"),
    ({"trace_id": "bad id!"}, "INVALID_FORMAT"),
])
def test_invalid_start_keeps_connection(url, override, code):
    with connect(url) as ws:
        start(ws, **override)
        err = json.loads(ws.recv(timeout=5))
        assert err["type"] == "error" and err["code"] == code
        # 연결은 그대로라 다음 세션을 바로 쓸 수 있다
        trace = start(ws)
        send_audio(ws, pcm(("tone", 0.6)))
        ws.send(json.dumps({"type": "end", "trace_id": trace}))
        assert recv_until_last(ws)[-1]["trace_id"] == trace


def test_no_session(url):
    with connect(url) as ws:
        send_audio(ws, pcm(("tone", 0.3)))  # 프레임 3개 → NO_SESSION은 한 번만
        ws.send(json.dumps({"type": "end", "trace_id": "never-started"}))
        a = json.loads(ws.recv(timeout=5))
        b = json.loads(ws.recv(timeout=5))
        assert (a["code"], a["trace_id"]) == ("NO_SESSION", "unbound")
        assert (b["code"], b["trace_id"]) == ("NO_SESSION", "never-started")
        ws.send(json.dumps({"type": "ping", "trace_id": "k2"}))
        assert json.loads(ws.recv(timeout=5))["type"] == "pong"


def test_sessions_on_separate_connections_do_not_mix(url):
    # 권장 방식: 세션마다 연결. 동시에 흘려도 서로 섞이지 않는다
    with connect(url) as a, connect(url) as b:
        ta, tb = start(a), start(b)
        da, db = pcm(("tone", 1.0)), pcm(("tone", 3.0))
        for i in range(0, max(len(da), len(db)), FRAME * 2):
            if i < len(da):
                a.send(da[i:i + FRAME * 2])
            if i < len(db):
                b.send(db[i:i + FRAME * 2])
        a.send(json.dumps({"type": "end", "trace_id": ta}))
        b.send(json.dumps({"type": "end", "trace_id": tb}))
        fa, fb = recv_until_last(a)[-1], recv_until_last(b)[-1]
    assert fa["trace_id"] == ta and fb["trace_id"] == tb
    assert len(fb["text"].split()) > len(fa["text"].split())
