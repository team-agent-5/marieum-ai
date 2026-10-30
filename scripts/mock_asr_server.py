"""Mock ASR 서버 (ml-server-protocol.md §8 + CALIBRATION.md §0.3 필드).

GPU 없이 실서버와 같은 메시지를 보낸다. 백엔드·앱 선행 개발용.
- 같은 입력엔 같은 샘플 (구간 첫 프레임의 바이트 합으로 고른다)
- interim: 오디오 0.5초, 1.0초 받았을 때 (실서버처럼 말하는 도중에 온다)
- mode=conversation: 오디오 3초마다 구간 final(is_last=false), end에서 is_last=true

  python scripts/mock_asr_server.py            # ws://0.0.0.0:9001/asr-stream
"""

import asyncio
import json

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()
MODEL_VERSION = "mock-ws-v2"
BYTES_PER_S = 16000 * 2

SAMPLES = [
    {"final": "미지근한 물 주세요", "interim": ["물", "미지근한 물"], "conf": 0.84},
    {"final": "화장실에 가고 싶어요", "interim": ["화장", "화장실에 가고"], "conf": 0.88},
    {"final": "숨이 좀 가빠요", "interim": ["숨", "숨이 가"], "conf": 0.71},
    {"final": "다리가 많이 아파요", "interim": ["다리", "다리가 아"], "conf": 0.76},
    {"final": "조금 추워요 담요 주세요", "interim": ["추", "조금 추워"], "conf": 0.80},
    {"final": "저녁 약 아직 안 먹었어요", "interim": ["약", "저녁 약"], "conf": 0.78},
    {"final": "잠이 안 와요", "interim": ["잠", "잠이 안"], "conf": 0.82},
    {"final": "보호자에게 전화해 주세요", "interim": ["전화", "보호자에게"], "conf": 0.86},
    {"final": "목이 많이 말라요", "interim": ["목", "목이 말"], "conf": 0.74},
    {"final": "", "interim": ["", ""], "conf": 0.05},
]


def split_stable(prev, text):
    # 실서버처럼 앞 interim과 겹치는 앞 단어만 stable로 (마지막 단어 제외)
    a, b = prev.split(), text.split()
    n = 0
    while n < min(len(a), len(b) - 1) and a[n] == b[n]:
        n += 1
    return " ".join(b[:n]), " ".join(b[n:])


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "models": ["asr-mock"],
            "gpu": {"name": "mock", "memory_free_mb": 0, "memory_total_mb": 0},
            "model_version": {"asr": MODEL_VERSION}, "uptime_sec": 0}


@app.get("/livez")
async def livez():
    return {"status": "alive"}


class MockSession:
    def __init__(self, trace_id, mode):
        self.trace_id, self.mode = trace_id, mode
        self.bytes = 0  # 세션 전체
        self.seg_bytes = 0  # 현재 구간
        self.pick = None  # 구간 첫 프레임 바이트로 샘플을 고르고 구간 끝까지 유지
        self.seg_id = 0
        self.sent_interim = 0
        self.prev = ""

    def sample(self):
        return SAMPLES[(self.pick or 0) % len(SAMPLES)]

    def final_msg(self, is_last):
        s = self.sample() if self.seg_bytes else SAMPLES[-1]
        msg = {"type": "transcript_final", "trace_id": self.trace_id, "text": s["final"],
               "confidence": s["conf"], "duration_ms": self.seg_bytes * 1000 // BYTES_PER_S,
               "model_version": MODEL_VERSION, "no_speech_prob": 0.9 if not s["final"] else 0.03,
               "segment_id": self.seg_id, "is_last": is_last}
        self.seg_id += 1
        self.seg_bytes = self.sent_interim = 0
        self.pick = None
        self.prev = ""
        return msg


@app.websocket("/asr-stream")
async def asr_stream(ws: WebSocket):
    await ws.accept()
    sessions, current = {}, None
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("bytes") is not None:
                s = current
                if s is None:
                    continue
                data = msg["bytes"]
                s.bytes += len(data)
                s.seg_bytes += len(data)
                if s.pick is None:
                    s.pick = sum(data)
                due = [0.5, 1.0]
                if s.sent_interim < 2 and s.seg_bytes >= due[s.sent_interim] * BYTES_PER_S:
                    smp = s.sample()
                    text = smp["interim"][s.sent_interim]
                    stable, unstable = split_stable(s.prev, text)
                    s.prev = text
                    s.sent_interim += 1
                    await asyncio.sleep(0.15)
                    await ws.send_json({
                        "type": "transcript_interim", "trace_id": s.trace_id, "text": text,
                        "confidence": max(smp["conf"] - 0.3 + 0.1 * s.sent_interim, 0.05),
                        "is_final": False, "segment_id": s.seg_id, "stable_text": stable,
                        "unstable_text": unstable, "audio_end_ms": s.bytes * 1000 // BYTES_PER_S})
                if s.mode == "conversation" and s.seg_bytes >= 3 * BYTES_PER_S:
                    await ws.send_json(s.final_msg(is_last=False))
                continue
            p = json.loads(msg.get("text") or "{}")
            kind, trace = p.get("type"), p.get("trace_id", "unbound")
            if kind == "ping":
                await ws.send_json({"type": "pong", "trace_id": trace})
            elif kind == "start":
                current = sessions[trace] = MockSession(trace, p.get("mode", "utterance"))
            elif kind == "end":
                s = sessions.pop(trace, None)
                if s is None:
                    await ws.send_json({"type": "error", "trace_id": trace, "code": "NO_SESSION",
                                        "message": "시작하지 않은 세션입니다."})
                    continue
                if current is s:
                    current = None
                await asyncio.sleep(0.3)
                await ws.send_json(s.final_msg(is_last=True))
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9001)
