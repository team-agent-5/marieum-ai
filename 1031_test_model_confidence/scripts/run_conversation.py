"""부분(part)들을 conversation 모드로 실험 서버에 흘려 보내고 받은 메시지를 전부 저장한다 (TEST_PLAN.md 5.2).

- 부분 1개 = 연결 1개 = 세션 1개. 100ms 프레임을 실제 속도로, 동시 N세션 (긴 부분부터)
- trace_id: exp1031-<화자>-p<부분>-<실행ID>. 서버 로그와 합칠 때 쓴다
- 녹음 안에 30~50초 쉼이 있어서 메시지가 한동안 안 와도 끊지 않는다. end 후 --end-timeout초 안에
  is_last final이 없으면 타임아웃으로 기록한다
- 결과: <out>/<화자>_pNN.jsonl (메시지마다 한 줄, 받는 즉시 기록). 끝까지 받은 부분만 .jsonl로 바뀌고,
  중간에 멈추면 .partial.jsonl로 남는다. 다시 실행하면 끝난 부분은 건너뛴다
- 30초마다 운영 8001과 실험 서버의 /healthz를 <out>/ops.jsonl에 기록한다 (운영 트래픽과 겹쳤는지)

GPU 서버에서 (~/exp-1031):
  ~/marieum-ml/.venv/bin/python scripts/run_conversation.py --url ws://127.0.0.1:8002/asr-stream \
      --parts parts.json --out results --sessions 4
로컬 Mock으로 먼저 확인 (빠르게):
  python scripts/run_conversation.py --url ws://127.0.0.1:9001/asr-stream --parts data/parts.json \
      --out data/mock_results --only KJJ KSW --speed 20
"""

import argparse
import asyncio
import json
import math
import os
import time
import urllib.error
import urllib.request
import wave
from datetime import datetime, timezone

import numpy as np
import websockets

SR = 16000
FRAME_MS = 100
STEP = SR * 2 * FRAME_MS // 1000  # 6,400바이트


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def load_pcm(path):
    with wave.open(path, "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SR, 1, 2), f"16kHz 모노 16-bit 아님: {path}"
        return w.readframes(w.getnframes())


def http_json(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:  # noqa: BLE001 - 연결 안 됨 등
        return None, repr(e)


def health_url(ws_url):
    base = ws_url.replace("wss://", "https://").replace("ws://", "http://")
    return base.rsplit("/", 1)[0] + "/healthz"


def pct(a, q):
    return round(float(np.percentile(a, q))) if len(a) else None


async def ops_logger(path, urls, interval, stop):
    """운영 8001과 실험 서버의 상태를 주기적으로 남긴다."""
    with open(path, "a", encoding="utf-8") as f:
        while not stop.is_set():
            rec = {"ts": now_iso()}
            for name, url in urls.items():
                code, body = await asyncio.to_thread(http_json, url)
                if isinstance(body, dict):
                    rec[name] = {"http": code, "status": body.get("status"), "sessions": body.get("sessions"),
                                 "gpu_pending": body.get("gpu_pending"),
                                 "gpu_free_mb": (body.get("gpu") or {}).get("memory_free_mb")}
                else:
                    rec[name] = {"http": code, "error": body}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            try:
                await asyncio.wait_for(stop.wait(), interval)
            except asyncio.TimeoutError:
                pass


async def run_part(part, base, args, run_id):
    spk, no = part["speaker"], part["part"]
    name = f"{spk}_p{no:02d}"
    trace = f"exp1031-{spk}-p{no:02d}-{run_id}"
    pcm = load_pcm(os.path.join(base, part["wav"]))
    frames = [pcm[i:i + STEP] for i in range(0, len(pcm), STEP)]
    partial = os.path.join(args.out, f"{name}.partial.jsonl")
    done_path = os.path.join(args.out, f"{name}.jsonl")

    send_t = []  # 프레임 k를 보낸 시각 (t0 기준 초)
    stats = {"interim": 0, "final": 0, "error": 0, "interim_ms": []}
    status = "ok"
    end_sent = None
    with open(partial, "w", encoding="utf-8") as f:
        def write(rec):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

        write({"type": "_meta", "trace_id": trace, "run_id": run_id, "url": args.url, "speed": args.speed,
               "started_at": now_iso(), "frames": len(frames), "audio_s": round(len(pcm) / 2 / SR, 2), **part})
        try:
            async with websockets.connect(args.url, max_size=2**20, open_timeout=10,
                                          ping_interval=20, ping_timeout=60) as ws:
                await ws.send(json.dumps({"type": "start", "trace_id": trace, "patient_id": "exp1031",
                                          "sample_rate": SR, "encoding": "pcm16", "language": "ko",
                                          "mode": "conversation"}))
                t0 = time.perf_counter()

                async def sender():
                    nonlocal end_sent
                    for k, fr in enumerate(frames):
                        target = t0 + k * FRAME_MS / 1000 / args.speed
                        await asyncio.sleep(max(0, target - time.perf_counter()))
                        send_t.append(time.perf_counter() - t0)
                        await ws.send(fr)
                    # 마지막 프레임이 실제로 끝나는 시각까지 기다린 뒤 end
                    await asyncio.sleep(max(0, t0 + len(frames) * FRAME_MS / 1000 / args.speed - time.perf_counter()))
                    end_sent = time.perf_counter() - t0
                    await ws.send(json.dumps({"type": "end", "trace_id": trace}))
                    write({"type": "_end_sent", "t": round(end_sent, 3)})

                snd = asyncio.create_task(sender())
                recv = None
                try:
                    while True:
                        if recv is None:
                            recv = asyncio.ensure_future(ws.recv())
                        if not snd.done():  # 보내는 중: 긴 쉼이 있으므로 기다림 제한 없음
                            done, _ = await asyncio.wait({recv, snd}, return_when=asyncio.FIRST_COMPLETED)
                            if recv not in done:
                                snd.result()  # 보내다 실패했으면 여기서 예외
                                continue
                        else:  # end 후: --end-timeout초까지
                            left = max(0.1, end_sent + args.end_timeout - (time.perf_counter() - t0))
                            done, _ = await asyncio.wait({recv}, timeout=left)
                            if recv not in done:
                                status = "end_timeout"
                                write({"type": "_timeout", "t": round(time.perf_counter() - t0, 3)})
                                break
                        t = time.perf_counter() - t0
                        m = json.loads(recv.result())
                        recv = None
                        rec = {"t": round(t, 3), **m}
                        kind = m.get("type")
                        if kind == "transcript_interim":
                            k = min(len(send_t) - 1, max(0, math.ceil(m["audio_end_ms"] / FRAME_MS) - 1))
                            if k >= 0:
                                rec["lat_ms"] = round((t - send_t[k]) * 1000)
                                stats["interim_ms"].append(rec["lat_ms"])
                            stats["interim"] += 1
                        elif kind == "transcript_final":
                            stats["final"] += 1
                        elif kind == "error":
                            stats["error"] += 1
                        write(rec)
                        if kind == "transcript_final" and m.get("is_last"):
                            break
                finally:
                    for task in (snd, recv):
                        if task is not None and not task.done():
                            task.cancel()
        except Exception as e:  # noqa: BLE001 - 연결 실패, 끊김 등
            status = "failed"
            write({"type": "_exception", "message": repr(e)})
        write({"type": "_summary", "status": status, "finished_at": now_iso(), "interim": stats["interim"],
               "final": stats["final"], "error": stats["error"], "interim_p50_ms": pct(stats["interim_ms"], 50),
               "interim_p95_ms": pct(stats["interim_ms"], 95)})
    if status == "ok":
        os.replace(partial, done_path)
    return name, status, stats


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8002/asr-stream")
    ap.add_argument("--parts", default="parts.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--sessions", type=int, default=4, help="동시 세션 수")
    ap.add_argument("--only", nargs="+", help="이 화자만")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = 실제 속도. Mock 시험용으로만 올린다")
    ap.add_argument("--stagger", type=float, default=0.13, help="처음 세션들의 시작 간격(초)")
    ap.add_argument("--end-timeout", type=float, default=30.0, help="end 후 마지막 final을 기다리는 시간(초)")
    ap.add_argument("--ops-url", default="http://127.0.0.1:8001/healthz", help="운영 서버 healthz")
    ap.add_argument("--ops-interval", type=float, default=30.0)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(args.parts))
    parts = json.load(open(args.parts, encoding="utf-8"))
    if args.only:
        parts = [p for p in parts if p["speaker"] in args.only]
    os.makedirs(args.out, exist_ok=True)
    todo = [p for p in parts
            if not os.path.exists(os.path.join(args.out, f"{p['speaker']}_p{p['part']:02d}.jsonl"))]
    skipped = len(parts) - len(todo)
    todo.sort(key=lambda p: -p["duration_s"])  # 긴 부분부터 (전체 시간을 줄인다)

    exp_health = health_url(args.url)
    code, body = http_json(exp_health)
    if code != 200:
        raise SystemExit(f"실험 서버가 준비되지 않았다: {exp_health} → {code} {body}")
    if not todo:
        print(f"할 부분이 없다 (끝난 부분 {skipped}개)")
        return

    run_id = datetime.now().strftime("%m%d%H%M%S")
    total_s = sum(p["duration_s"] for p in todo)
    print(f"[run {run_id}] 부분 {len(todo)}개 (건너뜀 {skipped}), 오디오 {total_s / 60:.1f}분, "
          f"동시 {args.sessions}, 속도 x{args.speed:g} → 예상 약 {total_s / args.sessions / args.speed / 60:.0f}분+",
          flush=True)
    with open(os.path.join(args.out, "runs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": run_id, "started_at": now_iso(), "url": args.url, "sessions": args.sessions,
                            "speed": args.speed, "parts": [f"{p['speaker']}_p{p['part']:02d}" for p in todo]},
                           ensure_ascii=False) + "\n")

    stop = asyncio.Event()
    ops = asyncio.create_task(ops_logger(os.path.join(args.out, "ops.jsonl"),
                                         {"ops_8001": args.ops_url, "exp": exp_health}, args.ops_interval, stop))
    queue = asyncio.Queue()
    for p in todo:
        queue.put_nowait(p)
    t_start = time.perf_counter()
    results, done_s = [], [0.0]

    async def worker(i):
        await asyncio.sleep(i * args.stagger)
        while not queue.empty():
            p = queue.get_nowait()
            name, status, st = await run_part(p, base, args, run_id)
            done_s[0] += p["duration_s"]
            results.append((name, status))
            el = (time.perf_counter() - t_start) / 60
            print(f"  {el:6.1f}분 {name} {status:11s} interim {st['interim']:4d} (p50 {pct(st['interim_ms'], 50)}ms)"
                  f" final {st['final']:3d} error {st['error']}  [{len(results)}/{len(todo)}]", flush=True)

    try:
        await asyncio.gather(*(worker(i) for i in range(min(args.sessions, len(todo)))))
    finally:
        stop.set()
        await ops

    bad = [n for n, s in results if s != "ok"]
    print(f"[run {run_id}] 끝: {(time.perf_counter() - t_start) / 60:.1f}분, 성공 {len(results) - len(bad)}, "
          f"실패 {len(bad)} {bad if bad else ''}")
    if bad:
        print("실패한 부분은 .partial.jsonl로 남았다. 같은 명령을 다시 실행하면 그 부분만 다시 보낸다")


if __name__ == "__main__":
    asyncio.run(main())
