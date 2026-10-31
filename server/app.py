"""ML 서버: WebSocket /asr-stream + POST /asr + /healthz + /livez (CALIBRATION.md §0).

실행 (워커 1개 고정, §8):
  ML_MODEL_DIR=models/ASR_Whisper_Disease_General \
  uvicorn server.app:app --host 0.0.0.0 --port 8001 --workers 1
"""

import asyncio
import json
import os
import io
import re
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .config import settings as S
from .stream import CONVERSATION, MODES, SR, UTTERANCE, Agreement, Segmenter, word_delta

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
            eng.transcribe([np.zeros(SR, np.float32)] * b, [8] * b)
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
        sch = GpuScheduler(eng, S.batch_max, S.interim_max_new_tokens, S.final_max_new_tokens,
                           S.token_cap_rate, S.token_cap_floor)
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
        self.sent_words = {}  # 구간(segment_id)별로 stable_delta로 이미 내려보낸 단어
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
        fut = state["scheduler"].submit_interim(self.key, snap.audio, len(snap.audio) * 1000 // SR)
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
            delta, revised = self.take_delta(snap.seg_id, stable)
            self.interim_count += 1
            self.last_interim_sent = time.perf_counter()
            latency = (time.perf_counter() - self.arrival(snap.audio_end_ms)) * 1000
            await self.conn.send(self, {
                "type": "transcript_interim", "text": " ".join(t for t in (stable, unstable) if t),
                "confidence": r["confidence"], "is_final": False, "segment_id": snap.seg_id,
                "stable_text": stable, "unstable_text": unstable, "audio_end_ms": snap.audio_end_ms,
                "stable_delta": delta, "stable_revised": revised})
            log("transcript_interim", self.trace_id, status="ok", latency_ms=round(latency),
                confidence=r["confidence"], segment_id=snap.seg_id, interim_index=self.interim_count,
                audio_ms=len(snap.audio) * 1000 // SR, queue_ms=round(r["queue_ms"]),
                gpu_ms=round(r["gpu_ms"]), batch=r["batch"], batch_kinds=r["kinds"],
                gen_tokens=r["gen_tokens"], no_speech_prob=r["no_speech_prob"], token_cap=r["token_cap"],
                ended_eot=r["ended_eot"], vad_speech_ms=snap.vad_speech_ms)
        self.maybe_interim()

    def take_delta(self, seg_id, text):
        """확정된 단어 중 아직 안 내려보낸 것만 (단어 단위 스트리밍용). 보낸 단어로 기록한다."""
        sent = self.sent_words.setdefault(seg_id, [])
        words, revised = word_delta(sent, text)
        self.sent_words[seg_id] = text.split() if revised else sent + words
        return " ".join(words), revised

    @staticmethod
    def silenced(r):
        # 무음 토큰 확률이 높으면 무음으로 본다 (config.py 주석 참고)
        return r["no_speech_prob"] > S.no_speech_threshold and r["avg_logprob"] <= S.logprob_threshold

    # final -----------------------------------------------------------------
    def final(self, seg, is_last):
        t_close = time.perf_counter()
        sch = state["scheduler"]
        sch.cancel_interim(self.key)
        fut = sch.submit_final(self.key, seg.audio, len(seg.audio) * 1000 // SR) if seg.has_speech else None
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
        delta, revised = self.take_delta(seg.seg_id, text)
        self.sent_words.pop(seg.seg_id, None)
        await self.conn.send(self, {
            "type": "transcript_final", "text": text, "confidence": conf,
            "duration_ms": seg.duration_ms, "model_version": S.model_version, "no_speech_prob": nsp,
            "segment_id": seg.seg_id, "is_last": is_last,
            "stable_delta": delta, "stable_revised": revised})
        extra = {} if r is None else {"queue_ms": round(r["queue_ms"]), "gpu_ms": round(r["gpu_ms"]),
                                       "batch": r["batch"], "batch_kinds": r["kinds"],
                                       "gen_tokens": r["gen_tokens"], "ended_eot": r["ended_eot"],
                                       "token_cap": r["token_cap"]}
        log("transcript_final", self.trace_id, status="ok",
            latency_ms=round((time.perf_counter() - t_close) * 1000), confidence=conf,
            segment_id=seg.seg_id, reason=seg.reason, is_last=is_last, vad_skip=r is None,
            no_speech_prob=nsp, duration_ms=seg.duration_ms, speech_ms=len(seg.audio) * 1000 // SR,
            vad_speech_ms=seg.vad_speech_ms, **extra)
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


# ---- POST /asr: 녹음 파일 한 번에 전사 --------------------------------------
class AsrInputError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def decode_upload(body, sample_rate):
    """WAV(PCM 8/16/24/32비트, 채널 수 무관) 또는 PCM16 LE 모노(sample_rate 쿼리 필요) → 16kHz float32 모노."""
    if body[:4] == b"RIFF":
        try:
            with wave.open(io.BytesIO(body)) as w:
                sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
                raw = w.readframes(w.getnframes())
        except (wave.Error, EOFError) as e:
            raise AsrInputError(415, "INVALID_FORMAT", f"PCM WAV만 지원합니다 ({e}).") from None
        if width == 3:  # 24비트 → 32비트로 늘려서 읽는다
            b = np.frombuffer(raw[: len(raw) // 3 * 3], np.uint8).reshape(-1, 3)
            raw = np.concatenate([np.zeros((len(b), 1), np.uint8), b], axis=1).tobytes()
            width = 4
        dtype = {1: np.uint8, 2: "<i2", 4: "<i4"}.get(width)
        if dtype is None:
            raise AsrInputError(415, "INVALID_FORMAT", f"지원하지 않는 WAV 샘플 크기: {width}바이트")
        x = np.frombuffer(raw[: len(raw) // (width * ch) * width * ch], dtype).astype(np.float32)
        x = (x - 128) / 128 if width == 1 else x / float(2 ** (8 * width - 1))
        x = x.reshape(-1, ch).mean(axis=1) if ch > 1 else x
    elif sample_rate is not None:
        sr = sample_rate
        x = np.frombuffer(body[: len(body) // 2 * 2], "<i2").astype(np.float32) / 32768.0
    else:
        raise AsrInputError(415, "INVALID_FORMAT",
                            "WAV 파일을 보내거나, PCM16 LE 모노 바이트와 ?sample_rate=를 함께 보내세요.")
    if not 8000 <= sr <= 48000:
        raise AsrInputError(400, "INVALID_FORMAT", "sample_rate는 8000–48000이어야 합니다.")
    if len(x) / sr > S.asr_max_s:
        raise AsrInputError(413, "AUDIO_TOO_LONG", f"음성은 {S.asr_max_s:g}초까지 받습니다.")
    if sr != SR and len(x):
        import torch
        import torchaudio
        x = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(x)), sr, SR).numpy()
    return x


def split_speech(x, vad_model):
    """대화 모드와 같은 기준(끝 무음, 구간 상한)으로 말소리 구간을 나눈다. 말소리 없는 구간은 버린다."""
    seg = Segmenter(CONVERSATION, make_vad(vad_model), S)
    segs = seg.feed(x)
    last = seg.finish()
    return [g for g in segs + ([last] if last else []) if g.has_speech]


def asr_error(status, code, message, trace):
    log("error", trace, endpoint="/asr", status="error", code=code, message=message)
    return JSONResponse({"code": code, "message": message, "trace_id": trace}, status_code=status)


@app.post("/asr")
async def asr(request: Request, sample_rate: int | None = None, trace_id: str | None = None):
    """녹음 파일을 한 번에 전사한다. 본문은 WAV 그대로(audio/wav) 또는 PCM16 LE 모노 + ?sample_rate=."""
    trace = trace_id or request.headers.get("x-trace-id") or str(uuid.uuid4())
    if not TRACE_RE.fullmatch(trace):
        return asr_error(400, "INVALID_FORMAT", "trace_id는 영문·숫자·_- 64자 이내여야 합니다.", "unbound")
    if state["status"] != "ok":
        return asr_error(503, "MODEL_UNAVAILABLE", "모델 로딩 중입니다. 잠시 후 다시 시도하세요.", trace)
    max_bytes = int(S.asr_max_s * 48000 * 2 * 4) + 1024  # 48kHz 스테레오 32비트 WAV 기준
    if int(request.headers.get("content-length") or 0) > max_bytes:
        return asr_error(413, "AUDIO_TOO_LONG", f"음성은 {S.asr_max_s:g}초까지 받습니다.", trace)
    body = await request.body()
    if not body:
        return asr_error(400, "INVALID_FORMAT", "본문에 음성이 없습니다.", trace)
    t0 = time.perf_counter()
    loop = asyncio.get_running_loop()
    try:
        x = await loop.run_in_executor(None, decode_upload, body, sample_rate)
    except AsrInputError as e:
        return asr_error(e.status, e.code, e.message, trace)

    try:
        vad_model = await loop.run_in_executor(vad_pool, vads.acquire)
        try:
            # 기본 스레드풀에서 돌린다: 긴 파일의 VAD가 스트리밍 세션용 vad_pool을 막지 않게
            segs = await loop.run_in_executor(None, split_speech, x, vad_model)
        finally:
            vads.release(vad_model)
        sch = state["scheduler"]
        results = await asyncio.gather(*(sch.submit_final(f"http:{trace}:{g.seg_id}", g.audio, len(g.audio) * 1000 // SR)
                                         for g in segs))
    except Exception as e:  # noqa: BLE001
        log("asr_failed", trace, endpoint="/asr", status="error", message=repr(e))
        return asr_error(500, "INTERNAL", "전사 중 서버 오류가 발생했습니다.", trace)

    out = []
    for g, r in zip(segs, results):
        log("transcript_file_segment", trace, endpoint="/asr", status="ok", segment_id=g.seg_id,
            gen_tokens=r["gen_tokens"], ended_eot=r["ended_eot"], token_cap=r["token_cap"],
            speech_ms=len(g.audio) * 1000 // SR, vad_speech_ms=g.vad_speech_ms)
        text = "" if Session.silenced(r) else r["text"][:MAX_TEXT]
        if text:
            out.append({"text": text, "confidence": r["confidence"],
                        "start_ms": g.end_ms - g.duration_ms, "end_ms": g.end_ms})
    total = sum(o["end_ms"] - o["start_ms"] for o in out)
    conf = round(sum(o["confidence"] * (o["end_ms"] - o["start_ms"]) for o in out) / total, 4) if total else 0.0
    body = {"trace_id": trace, "text": " ".join(o["text"] for o in out)[:MAX_TEXT], "confidence": conf,
            "duration_ms": len(x) * 1000 // SR, "model_version": S.model_version, "segments": out}
    log("transcript_file", trace, endpoint="/asr", status="ok",
        latency_ms=round((time.perf_counter() - t0) * 1000), confidence=conf,
        duration_ms=body["duration_ms"], segments=len(segs), text_segments=len(out))
    return body
