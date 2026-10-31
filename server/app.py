"""ML 서버: WebSocket /asr-stream + /healthz + /livez (CALIBRATION.md §0).

실행 (워커 1개 고정, §8):
  ML_MODEL_DIR=models/ASR_Whisper_Disease_General \
  uvicorn server.app:app --host 0.0.0.0 --port 8001 --workers 1
"""

import asyncio
import json
import os
import re
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .config import settings as S
from .stream import CONVERSATION, MODES, SR, UTTERANCE, Agreement, Segmenter

ENDPOINT = "/asr-stream"
TRACE_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
MIN_INTERIM_GAP_S = 0.3
MAX_TEXT = 1000

state = {"status": "loading", "started": time.monotonic(), "engine": None, "error": None,
         "gpu": {}, "scheduler": None}
vad_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vad")


def log(event, trace_id="unbound", **kw):
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
           "trace_id": trace_id, "endpoint": ENDPOINT, "event": event,
           "model_version": S.model_version, **kw}
    print(json.dumps(rec, ensure_ascii=False), flush=True)


# ---- 모델과 VAD -----------------------------------------------------------
class VadModels:
    """Silero VAD는 앞 소리를 내부에 기억하므로 세션마다 따로 쓴다. 다 쓴 모델은 초기화해서 재사용."""

    def __init__(self):
        self.free = []

    def acquire(self):
        if self.free:
            return self.free.pop()
        from silero_vad import load_silero_vad
        return load_silero_vad()

    def release(self, m):
        m.reset_states()
        self.free.append(m)


vads = VadModels()


def make_vad(model):
    import torch

    def vad(windows):
        with torch.no_grad():
            return [float(model(torch.from_numpy(np.ascontiguousarray(w)), SR)) for w in windows]
    return vad


def load_model():
    try:
        from .engine import AsrEngine
        import torch
        eng = AsrEngine(S.model_dir, S.dtype)
        # 워밍업: 첫 추론은 CUDA 초기화로 약 0.86초 걸린다 (2026-10-31 실측). interim 목표가 300ms라
        # 준비 완료(/healthz 200) 전에 batch 1–4 크기로 한 번씩 돌려 둔다. #34(워밍업 안 함)를 바꾼 것
        t = time.perf_counter()
        for b in range(1, S.batch_max + 1):
            eng.transcribe([np.zeros(SR, np.float32)] * b, 8)
        log("warmup_done", warmup_s=round(time.perf_counter() - t, 2))
        # VAD 모델 미리 4개 (세션 시작 지연 줄이기). 한 번에 만들고 넣어야 4개가 된다
        vads.free.extend([vads.acquire() for _ in range(4)])
        state["engine"] = eng
        free, total = torch.cuda.mem_get_info()
        state["gpu"] = {"name": torch.cuda.get_device_name(0), "memory_free_mb": free // 2**20,
                        "memory_total_mb": total // 2**20}
        log("model_loaded", load_s=round(eng.load_s, 1), dtype=S.dtype)
        return eng
    except Exception as e:  # noqa: BLE001
        state["status"], state["error"] = "error", repr(e)
        log("model_load_failed", status="error", message=repr(e))
        raise


async def gpu_info_loop():
    while True:
        await asyncio.sleep(10)
        try:
            import torch
            free, total = torch.cuda.mem_get_info()
            state["gpu"].update(memory_free_mb=free // 2**20, memory_total_mb=total // 2**20)
        except Exception:  # noqa: BLE001
            pass


@asynccontextmanager
async def lifespan(app):
    from .scheduler import GpuScheduler

    async def boot():
        loop = asyncio.get_running_loop()
        try:
            eng = await loop.run_in_executor(None, load_model)
        except Exception:  # noqa: BLE001
            return
        sch = GpuScheduler(eng, S.batch_max, S.interim_max_new_tokens, S.final_max_new_tokens)
        sch.start()
        state["scheduler"] = sch
        state["status"] = "ok"

    tasks = [asyncio.create_task(boot()), asyncio.create_task(gpu_info_loop())]
    yield
    for t in tasks:
        t.cancel()
    if state["scheduler"]:
        await state["scheduler"].stop()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    up = int(time.monotonic() - state["started"])
    if state["status"] != "ok":
        body = {"status": state["status"], "models": [], "uptime_sec": up,
                "message": state["error"] or "모델 로딩 중 (예상 20s)"}
        return JSONResponse(body, status_code=503)
    sch = state["scheduler"]
    return {"status": "ok", "models": ["asr"], "gpu": state["gpu"],
            "model_version": {"asr": S.model_version}, "uptime_sec": up,
            "sessions": len(Session.live), "gpu_pending": sch.pending()}


@app.get("/livez")
async def livez():
    return {"status": "alive"}


# ---- 세션 ---------------------------------------------------------------
class Session:
    live = {}  # key -> Session (healthz 집계용)

    def __init__(self, conn, trace_id, patient_id, mode, sr, vad_model):
        self.conn, self.trace_id, self.patient_id, self.mode, self.sr = conn, trace_id, patient_id, mode, sr
        self.key = f"{id(conn)}:{trace_id}"
        self.vad_model = vad_model
        self.seg = Segmenter(mode, make_vad(vad_model), S)
        self.agree = Agreement()
        self.agree_seg = 0
        self.odd = b""  # 홀수 바이트 프레임의 남은 1바이트
        self.bytes_total = 0
        self.t_start = time.perf_counter()
        self.frame_times = []  # (세션 기준 오디오 끝 ms, 도착 시각) 지연 측정용
        self.interim_inflight = False
        self.interim_count = 0
        self.last_interim_sent = 0.0
        self.gap_timer = None
        self.last_final = None  # final은 순서대로 보낸다
        self.finalized_seg = -1
        self.closed = False
        self.debug_audio = [] if S.debug_save else None
        Session.live[self.key] = self

    # 오디오 -----------------------------------------------------------------
    def decode(self, data):
        data = self.odd + data
        self.odd = data[len(data) // 2 * 2:]
        x = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
        if self.sr != SR and len(x):
            import torch
            import torchaudio
            x = torchaudio.functional.resample(torch.from_numpy(x), self.sr, SR).numpy()
        return x

    async def on_audio(self, data):
        if self.closed:
            return
        self.bytes_total += len(data)
        loop = asyncio.get_running_loop()

        def work():  # 작업 스레드: 디코딩과 VAD만. 구간 상태는 아래에서 이벤트 루프가 바꾼다
            x = self.decode(data)
            if self.debug_audio is not None:
                self.debug_audio.append(x)
            windows = self.seg.take_windows(x)
            return windows, (self.seg.vad(windows) if len(windows) else [])

        windows, probs = await loop.run_in_executor(vad_pool, work)
        if self.closed:
            return
        segments = self.seg.push_windows(windows, probs)
        self.frame_times.append((self.seg.elapsed_ms, time.perf_counter()))
        if len(self.frame_times) > 400:
            del self.frame_times[:200]
        for s in segments:
            last = s.reason == "too_long"
            if last:
                self.conn.mute_orphan_audio = True
                limit = S.utterance_max_s if self.mode == UTTERANCE else S.session_max_s
                await self.conn.send(self, {"type": "error", "code": "AUDIO_TOO_LONG",
                                            "message": f"녹음은 {limit:g}초를 초과할 수 없습니다."})
            self.final(s, is_last=last)
        self.maybe_interim()

    def arrival(self, audio_end_ms):
        for end_ms, t in reversed(self.frame_times):
            if end_ms <= audio_end_ms + 1:
                return t
        return self.t_start

    # interim ---------------------------------------------------------------
    def maybe_interim(self):
        if self.closed or self.interim_inflight or not self.seg.interim_ready():
            return
        wait = MIN_INTERIM_GAP_S - (time.perf_counter() - self.last_interim_sent)
        if wait > 0:  # 프로토콜 §7.1: interim은 최소 300ms 간격 (GPU가 밀렸다 풀리면 몰려 나갈 수 있음)
            if not self.gap_timer:
                def fire():
                    self.gap_timer = None
                    self.maybe_interim()
                self.gap_timer = asyncio.get_running_loop().call_later(wait, fire)
            return
        snap = self.seg.snapshot()
        self.interim_inflight = True
        fut = state["scheduler"].submit_interim(self.key, snap.audio)
        asyncio.create_task(self._interim(snap, fut))

    async def _interim(self, snap, fut):
        try:
            r = await fut
        except asyncio.CancelledError:
            r = None
        except Exception as e:  # noqa: BLE001
            log("interim_failed", self.trace_id, status="error", message=repr(e))
            r = None
        self.interim_inflight = False
        if r is not None and not self.closed and snap.seg_id > self.finalized_seg:
            if snap.seg_id != self.agree_seg:
                self.agree, self.agree_seg = Agreement(), snap.seg_id
            text = "" if self.silenced(r) else r["text"][:MAX_TEXT]
            stable, unstable = self.agree.update(text)
            self.interim_count += 1
            self.last_interim_sent = time.perf_counter()
            latency = (time.perf_counter() - self.arrival(snap.audio_end_ms)) * 1000
            await self.conn.send(self, {
                "type": "transcript_interim", "text": " ".join(t for t in (stable, unstable) if t),
                "confidence": r["confidence"], "is_final": False, "segment_id": snap.seg_id,
                "stable_text": stable, "unstable_text": unstable, "audio_end_ms": snap.audio_end_ms})
            log("transcript_interim", self.trace_id, status="ok", latency_ms=round(latency),
                confidence=r["confidence"], segment_id=snap.seg_id, interim_index=self.interim_count,
                audio_ms=len(snap.audio) * 1000 // SR, queue_ms=round(r["queue_ms"]),
                gpu_ms=round(r["gpu_ms"]), batch=r["batch"], batch_kinds=r["kinds"],
                gen_tokens=r["gen_tokens"], no_speech_prob=r["no_speech_prob"])
        self.maybe_interim()

    @staticmethod
    def silenced(r):
        # 무음 토큰 확률이 높으면 무음으로 본다 (config.py 주석 참고)
        return r["no_speech_prob"] > S.no_speech_threshold and r["avg_logprob"] <= S.logprob_threshold

    # final -----------------------------------------------------------------
    def final(self, seg, is_last):
        t_close = time.perf_counter()
        sch = state["scheduler"]
        sch.cancel_interim(self.key)
        fut = sch.submit_final(self.key, seg.audio) if seg.has_speech else None
        prev = self.last_final
        self.last_final = asyncio.create_task(self._final(seg, fut, prev, is_last, t_close))
        if is_last:
            self.closed = True

    async def _final(self, seg, fut, prev, is_last, t_close):
        r = None
        if fut is not None:
            try:
                r = await fut
            except Exception as e:  # noqa: BLE001
                log("final_failed", self.trace_id, status="error", message=repr(e))
                if prev:
                    await asyncio.shield(prev)
                await self.conn.send(self, {"type": "error", "code": "INTERNAL",
                                            "message": "전사 중 서버 오류가 발생했습니다."})
                if is_last:
                    self.cleanup()
                return
        if prev:
            await asyncio.shield(prev)  # 앞 구간 final을 먼저 보낸다
        self.finalized_seg = max(self.finalized_seg, seg.seg_id)
        if r is None:  # VAD가 말소리를 못 찾음 → Whisper 생략 (§6)
            text, conf, nsp = "", 0.0, 1.0
        else:
            text = "" if self.silenced(r) else r["text"][:MAX_TEXT]
            conf, nsp = (r["confidence"] if text else 0.0), r["no_speech_prob"]
        await self.conn.send(self, {
            "type": "transcript_final", "text": text, "confidence": conf,
            "duration_ms": seg.duration_ms, "model_version": S.model_version, "no_speech_prob": nsp,
            "segment_id": seg.seg_id, "is_last": is_last})
        extra = {} if r is None else {"queue_ms": round(r["queue_ms"]), "gpu_ms": round(r["gpu_ms"]),
                                       "batch": r["batch"], "batch_kinds": r["kinds"],
                                       "gen_tokens": r["gen_tokens"], "ended_eot": r["ended_eot"]}
        log("transcript_final", self.trace_id, status="ok",
            latency_ms=round((time.perf_counter() - t_close) * 1000), confidence=conf,
            segment_id=seg.seg_id, reason=seg.reason, is_last=is_last, vad_skip=r is None,
            no_speech_prob=nsp, duration_ms=seg.duration_ms, speech_ms=len(seg.audio) * 1000 // SR,
            **extra)
        if is_last:
            self.cleanup()

    def end(self):
        if self.closed:
            return
        seg = self.seg.finish()
        if seg is not None:
            self.final(seg, is_last=True)

    def cleanup(self):
        self.closed = True
        if self.gap_timer:
            self.gap_timer.cancel()
        if Session.live.pop(self.key, None) is None:
            return
        state["scheduler"].cancel_interim(self.key)
        self.conn.forget(self)
        vads.release(self.vad_model)
        log("session_closed", self.trace_id, status="ok", patient_id=self.patient_id, mode=self.mode,
            audio_bytes_total=self.bytes_total, interim_count=self.interim_count,
            vad_segments=self.seg.seg_id,
            session_duration_ms=round((time.perf_counter() - self.t_start) * 1000))
        if self.debug_audio:
            save_debug(self.trace_id, np.concatenate(self.debug_audio))

    def abort(self):
        """연결이 끊겼다. 진행 중 작업을 버린다."""
        self.closed = True
        if self.last_final:
            self.last_final.cancel()
        if Session.live.get(self.key) is self:
            self.cleanup()


def save_debug(trace_id, audio):
    os.makedirs(S.debug_dir, exist_ok=True)
    now = time.time()
    for f in os.listdir(S.debug_dir):
        p = os.path.join(S.debug_dir, f)
        if now - os.path.getmtime(p) > S.debug_ttl_s:
            os.remove(p)
    with wave.open(os.path.join(S.debug_dir, f"{trace_id}.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


# ---- 연결 ---------------------------------------------------------------
class Conn:
    def __init__(self, ws):
        self.ws = ws
        self.lock = asyncio.Lock()
        self.sessions = {}  # trace_id -> Session
        self.current = None  # 바이너리 프레임이 들어갈 세션 (가장 최근 start, 프로토콜 §5.1.2)
        self.recent_closed = []  # 자동 종료된 세션의 늦은 end를 조용히 무시
        self.warned_no_session = False  # NO_SESSION은 start 사이에 한 번만 보낸다
        self.mute_orphan_audio = False  # 길이 초과로 자동 종료된 뒤 이어 오는 프레임은 에러 없이 버린다
        self.open = True

    async def send(self, session, msg):
        trace = session.trace_id if session else msg.pop("trace_id", "unbound")
        if not self.open:
            return
        try:
            async with self.lock:
                await self.ws.send_text(json.dumps({**msg, "trace_id": trace}, ensure_ascii=False))
        except Exception:  # noqa: BLE001 - 끊긴 연결
            self.open = False

    async def error(self, trace_id, code, message):
        log("error", trace_id, status="error", code=code, message=message)
        await self.send(None, {"type": "error", "trace_id": trace_id, "code": code, "message": message})

    def forget(self, s):
        if self.sessions.get(s.trace_id) is s:
            del self.sessions[s.trace_id]
        if self.current is s:
            self.current = None
        self.recent_closed = (self.recent_closed + [s.trace_id])[-32:]


@app.websocket(ENDPOINT)
async def asr_stream(ws: WebSocket):
    await ws.accept()
    conn = Conn(ws)
    if state["status"] != "ok":
        await conn.error("unbound", "MODEL_UNAVAILABLE", "모델 로딩 중입니다. 잠시 후 다시 연결하세요.")
        await ws.close(1013)
        return
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                s = conn.current
                if s is None:
                    if not conn.warned_no_session and not conn.mute_orphan_audio:
                        conn.warned_no_session = True
                        await conn.error("unbound", "NO_SESSION", "start 없이 오디오를 받았습니다.")
                    continue
                await s.on_audio(msg["bytes"])
                continue
            await handle_text(conn, msg.get("text") or "")
    except WebSocketDisconnect:
        pass
    finally:
        conn.open = False
        for s in list(conn.sessions.values()):
            s.abort()


async def handle_text(conn, raw):
    try:
        p = json.loads(raw)
        kind, trace = p.get("type"), p.get("trace_id")
    except (ValueError, AttributeError):
        await conn.error("unbound", "INVALID_FORMAT", "JSON 메시지가 아닙니다.")
        return
    if kind == "ping":
        await conn.send(None, {"type": "pong", "trace_id": trace or "unbound"})
        return
    if not isinstance(trace, str) or not TRACE_RE.match(trace):
        await conn.error("unbound", "INVALID_FORMAT", "trace_id가 없거나 형식이 맞지 않습니다.")
        return
    if kind == "start":
        await start_session(conn, p, trace)
    elif kind == "end":
        s = conn.sessions.get(trace)
        if s is None:
            if trace not in conn.recent_closed:
                await conn.error(trace, "NO_SESSION", "시작하지 않은 세션입니다.")
            return
        if conn.current is s:
            conn.current = None
        s.end()
    else:
        await conn.error(trace, "INVALID_FORMAT", f"알 수 없는 메시지 type: {kind!r}")


async def start_session(conn, p, trace):
    # 세션 하나의 형식 오류는 error만 보내고 연결은 유지한다 (같은 연결의 다른 세션 보호, §0.6)
    enc, sr, mode = p.get("encoding", "pcm16"), p.get("sample_rate"), p.get("mode", UTTERANCE)
    if enc != "pcm16":
        return await conn.error(trace, "INVALID_FORMAT", "encoding은 pcm16만 지원합니다.")
    if not isinstance(sr, int) or not 8000 <= sr <= 48000:
        return await conn.error(trace, "INVALID_FORMAT", "sample_rate는 8000–48000 정수여야 합니다.")
    if mode not in MODES:
        return await conn.error(trace, "INVALID_FORMAT", f"mode는 {MODES} 중 하나여야 합니다.")
    if p.get("language", "ko") != "ko":
        return await conn.error(trace, "INVALID_FORMAT", "language는 ko만 지원합니다.")
    if trace in conn.sessions:
        return await conn.error(trace, "INVALID_FORMAT", "이미 진행 중인 trace_id입니다.")
    vad_model = await asyncio.get_running_loop().run_in_executor(vad_pool, vads.acquire)
    s = Session(conn, trace, str(p.get("patient_id", ""))[:64], mode, sr, vad_model)
    conn.sessions[trace] = s
    conn.current = s
    conn.warned_no_session = conn.mute_orphan_audio = False
    log("start", trace, status="ok", patient_id=s.patient_id, mode=mode, sample_rate=sr)
