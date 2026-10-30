"""/asr-stream 테스트 클라이언트 (CALIBRATION.md §0.7).

WAV를 100ms 프레임(PCM16 16kHz)으로 실제 속도에 맞춰 보내고, interim/final 도착 시각을 잰다.
세션마다 연결을 따로 연다 (§0.6).

  # 발화 모드, 조각 4개를 동시에
  python scripts/stream_client.py --wav data/chunks/chunk_0{1,2,3,4}.wav --sessions 4
  # 대화 모드, sample.wav 앞 60초를 한 세션으로
  python scripts/stream_client.py --mode conversation --wav data/sample.wav --max-seconds 60 --show
"""

import argparse
import asyncio
import json
import math
import time
import uuid
import wave

import numpy as np
import websockets

FRAME_MS = 100
SR = 16000


def load_pcm16(path, offset_s, max_s):
    with wave.open(path, "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        w.setpos(int(offset_s * sr))
        n = w.getnframes() - int(offset_s * sr)
        if max_s:
            n = min(n, int(max_s * sr))
        x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        import torch
        import torchaudio
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def pct(a, q):
    return round(float(np.percentile(a, q))) if len(a) else None


async def run_session(idx, url, pcm, mode, delay, show, speed):
    await asyncio.sleep(delay)
    trace = str(uuid.uuid4())
    step = SR * 2 * FRAME_MS // 1000
    frames = [pcm[i:i + step] for i in range(0, len(pcm), step)]
    send_t = []
    events = []
    async with websockets.connect(url, max_size=2**20) as ws:
        await ws.send(json.dumps({"type": "start", "trace_id": trace, "patient_id": f"load-{idx}",
                                  "sample_rate": SR, "encoding": "pcm16", "language": "ko",
                                  "mode": mode}))
        t0 = time.perf_counter()
        end_t = None

        async def sender():
            nonlocal end_t
            for k, f in enumerate(frames):
                target = t0 + k * FRAME_MS / 1000 / speed
                await asyncio.sleep(max(0, target - time.perf_counter()))
                send_t.append(time.perf_counter())
                await ws.send(f)
            await asyncio.sleep(max(0, t0 + len(frames) * FRAME_MS / 1000 / speed - time.perf_counter()))
            end_t = time.perf_counter()
            await ws.send(json.dumps({"type": "end", "trace_id": trace}))

        snd = asyncio.create_task(sender())
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                events.append((time.perf_counter(), {"type": "client_timeout"}))
                break
            m = json.loads(raw)
            now = time.perf_counter()
            events.append((now, m))
            if show:
                kind = m["type"].replace("transcript_", "")
                body = (f"[{m.get('stable_text', '')}|{m.get('unstable_text', '')}]"
                        if kind == "interim" else repr(m.get("text", m.get("message"))))
                print(f"  s{idx} {now - t0:6.2f}s {kind:8s} seg={m.get('segment_id', '-')} "
                      f"conf={m.get('confidence', '-')} {body}", flush=True)
            if m["type"] == "transcript_final" and m.get("is_last", True):
                break
        await snd

    interim_lat, final_lat, finals, errors = [], [], [], []
    for t, m in events:
        if m["type"] == "transcript_interim":
            k = min(len(send_t) - 1, max(0, math.ceil(m["audio_end_ms"] / FRAME_MS / speed) - 1))
            interim_lat.append((t - send_t[k]) * 1000)
        elif m["type"] == "transcript_final":
            finals.append(m)
            if m.get("is_last", True) and end_t:
                final_lat.append((t - end_t) * 1000)
        elif m["type"] in ("error", "client_timeout"):
            errors.append(m)
    return {"idx": idx, "trace_id": trace, "audio_s": len(pcm) / 2 / SR, "interim_ms": interim_lat,
            "final_ms": final_lat, "finals": finals, "errors": errors}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8001/asr-stream")
    ap.add_argument("--wav", nargs="+", required=True)
    ap.add_argument("--mode", default="utterance", choices=["utterance", "conversation"])
    ap.add_argument("--sessions", type=int, default=1, help="동시 세션 수 (WAV를 돌아가며 씀)")
    ap.add_argument("--rounds", type=int, default=1, help="동시 세션 묶음을 몇 번 반복할지")
    ap.add_argument("--stagger", type=float, default=0.13, help="세션 시작 간격(초). 0이면 동시에")
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--offset-step", type=float, default=0.0, help="세션마다 offset을 이만큼 더 민다")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = 실제 속도")
    ap.add_argument("--show", action="store_true", help="메시지마다 출력")
    ap.add_argument("--out", help="결과 JSON 저장 경로")
    args = ap.parse_args()

    n_pcm = max(len(args.wav), args.sessions if args.offset_step else 1)
    pcms = [load_pcm16(args.wav[i % len(args.wav)], args.offset + i * args.offset_step, args.max_seconds)
            for i in range(n_pcm)]
    results = []
    for r in range(args.rounds):
        jobs = [run_session(r * args.sessions + i, args.url, pcms[(r * args.sessions + i) % len(pcms)],
                            args.mode, i * args.stagger, args.show, args.speed)
                for i in range(args.sessions)]
        results += await asyncio.gather(*jobs)

    il = [x for r in results for x in r["interim_ms"]]
    fl = [x for r in results for x in r["final_ms"]]
    per = [len(r["interim_ms"]) / max(r["audio_s"], 1e-6) for r in results]
    print(f"\n[sessions] {len(results)} (동시 {args.sessions} × {args.rounds}회, mode={args.mode})")
    print(f"[interim] 개수 {len(il)}, 초당 {np.mean(per):.2f}개/세션, 지연 ms p50={pct(il, 50)} "
          f"p95={pct(il, 95)} max={pct(il, 100)}")
    print(f"[final]   마지막 final 지연(end 후) ms p50={pct(fl, 50)} p95={pct(fl, 95)} max={pct(fl, 100)}")
    print(f"[errors]  {sum(len(r['errors']) for r in results)}",
          [e.get("code", e["type"]) for r in results for e in r["errors"]][:10])
    for r in results[: min(len(results), 8)]:
        texts = [f["text"] for f in r["finals"] if f["text"]]
        print(f"  s{r['idx']} {r['audio_s']:.1f}s finals={len(r['finals'])} {' / '.join(texts)[:150]}")
    if args.out:
        json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
