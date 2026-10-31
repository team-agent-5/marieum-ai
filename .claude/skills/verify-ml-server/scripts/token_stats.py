"""서버 로그(JSON 한 줄씩)에서 생성 토큰·상한·지연을 집계한다. 전사 텍스트는 읽지도 출력하지도 않는다.

  python token_stats.py <RUN>/server.log
  python token_stats.py <RUN>/server.log --trace <client_x.json의 trace_id> ...     # 특정 세션만
  python token_stats.py server.log --simulate 6 --simulate 11                       # 비례 상한을 넣었다면?
  sudo journalctl -u marieum-ml --since today -o cat > ops.log; python token_stats.py ops.log

읽는 필드 (server/app.py log):
  transcript_final:   gen_tokens, ended_eot, speech_ms(앞뒤 무음 자른 길이, 구간 중간 쉼 포함), reason, latency_ms
  transcript_interim: gen_tokens, audio_ms(스냅샷 길이), latency_ms
  token_cap (있으면): 그 작업에 적용된 상한. 상한 기능을 넣은 뒤의 로그에서 gen_tokens <= token_cap을 검사한다

--simulate R: cap = min(고정 상한, ceil(초 × R) + floor). final은 speech_ms, interim은 audio_ms로 계산한다.
  "정상 잘림" = 끝 토큰으로 끝난(ended_eot) 결과인데 gen_tokens > cap → 그 R이면 정상 문장이 잘렸을 것
종료 코드: token_cap 위반(gen_tokens > token_cap)이 있으면 1
"""

import argparse
import json
import math
import sys
from collections import Counter


def load(path, traces):
    recs = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            i = line.find("{")
            if i < 0:
                continue
            try:
                r = json.loads(line[i:])
            except ValueError:
                continue
            if not isinstance(r, dict) or "event" not in r:
                continue
            if traces and r.get("trace_id") not in traces:
                continue
            recs.append(r)
    return recs


def pct(a, q):
    if not a:
        return None
    a = sorted(a)
    return a[min(len(a) - 1, int(q / 100 * len(a)))]


def fmt(v, nd=0):
    return "-" if v is None else (f"{v:.{nd}f}" if nd else f"{round(v)}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949)에서도 한글이 깨지지 않게
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--trace", action="append", default=[], help="이 trace_id만 (여러 번 가능)")
    ap.add_argument("--simulate", type=float, action="append", default=[], help="초당 토큰 R (여러 번 가능)")
    ap.add_argument("--floor", type=int, default=8)
    ap.add_argument("--interim-max", type=int, default=96)
    ap.add_argument("--final-max", type=int, default=128)
    ap.add_argument("--min-speech-ms", type=int, default=1000, help="초당 토큰 통계에 넣을 최소 길이")
    args = ap.parse_args()

    recs = load(args.log, set(args.trace))
    ev = Counter(r["event"] for r in recs)
    fin = [r for r in recs if r["event"] == "transcript_final" and "gen_tokens" in r]
    itm = [r for r in recs if r["event"] == "transcript_interim" and "gen_tokens" in r]
    print(f"[로그] {args.log}  레코드 {len(recs)}  이벤트 {dict(ev)}")
    if not fin and not itm:
        print("gen_tokens가 있는 interim/final이 없다")
        return 0

    has_cap = any("token_cap" in r for r in fin + itm)
    viol = [r for r in fin + itm if "token_cap" in r and r["gen_tokens"] > r["token_cap"]]

    # final
    eot = [r for r in fin if r.get("ended_eot", True)]
    run = [r for r in fin if not r.get("ended_eot", True)]
    print(f"\n[final] {len(fin)}개  끝 토큰으로 끝남 {len(eot)}  상한에 닿음(ended_eot=false) {len(run)}")
    for r in run:
        cap = r.get("token_cap", "-")
        print(f"  상한 닿음 trace={r['trace_id']} seg={r.get('segment_id')} gen={r['gen_tokens']} cap={cap} "
              f"speech_ms={r.get('speech_ms')} reason={r.get('reason')} conf={r.get('confidence')}")
    rate = [r["gen_tokens"] / (r["speech_ms"] / 1000) for r in eot if r.get("speech_ms", 0) >= args.min_speech_ms]
    print(f"  초당 토큰(끝 토큰으로 끝남, speech_ms≥{args.min_speech_ms}): n={len(rate)} p50={fmt(pct(rate, 50), 2)} "
          f"p95={fmt(pct(rate, 95), 2)} p99={fmt(pct(rate, 99), 2)} max={fmt(pct(rate, 100), 2)}")
    if eot:
        top = max(eot, key=lambda r: r["gen_tokens"])
        print(f"  끝 토큰으로 끝난 최대 gen_tokens={top['gen_tokens']} (speech_ms={top.get('speech_ms')})  "
              f"최대 speech_ms={max(r.get('speech_ms', 0) for r in fin)}")
    lat = [r["latency_ms"] for r in fin if "latency_ms" in r]
    print(f"  지연 ms p50={fmt(pct(lat, 50))} p95={fmt(pct(lat, 95))} max={fmt(pct(lat, 100))}")

    # interim
    def interim_capped(r):
        return r["gen_tokens"] >= r.get("token_cap", args.interim_max)
    ic = [r for r in itm if interim_capped(r)]
    print(f"\n[interim] {len(itm)}개  상한에 닿음 {len(ic)} (gen_tokens ≥ {'token_cap' if has_cap else args.interim_max})")
    for r in ic[:10]:
        print(f"  상한 닿음 trace={r['trace_id']} seg={r.get('segment_id')} gen={r['gen_tokens']} "
              f"cap={r.get('token_cap', '-')} audio_ms={r.get('audio_ms')}")
    if len(ic) > 10:
        print(f"  … 외 {len(ic) - 10}개")
    lat = [r["latency_ms"] for r in itm if "latency_ms" in r]
    gpu = [r["gpu_ms"] for r in itm if "gpu_ms" in r]
    print(f"  지연 ms p50={fmt(pct(lat, 50))} p95={fmt(pct(lat, 95))} max={fmt(pct(lat, 100))}  "
          f"gpu_ms p50={fmt(pct(gpu, 50))} max={fmt(pct(gpu, 100))}")

    # 시뮬레이션
    for R in args.simulate:
        def cap(sec, hard):
            return min(hard, math.ceil(sec * R) + args.floor)
        cut_f = [r for r in eot if r["gen_tokens"] > cap(r.get("speech_ms", 0) / 1000, args.final_max)]
        save_f = sum(max(0, r["gen_tokens"] - cap(r.get("speech_ms", 0) / 1000, args.final_max)) for r in run)
        ok_i = [r for r in itm if not interim_capped(r)]
        cut_i = [r for r in ok_i if r["gen_tokens"] > cap(r.get("audio_ms", 0) / 1000, args.interim_max)]
        save_i = sum(max(0, r["gen_tokens"] - cap(r.get("audio_ms", 0) / 1000, args.interim_max)) for r in ic)
        print(f"\n[시뮬레이션 R={R:g}, floor={args.floor}] cap = min(고정, ceil(초×{R:g})+{args.floor})")
        print(f"  final 정상 잘림 {len(cut_f)}/{len(eot)} ({100 * len(cut_f) / max(1, len(eot)):.2f}%)  "
              f"폭주 {len(run)}개에서 줄어드는 토큰 {save_f}")
        for r in sorted(cut_f, key=lambda r: -r["gen_tokens"])[:5]:
            print(f"    잘릴 final gen={r['gen_tokens']} speech_ms={r.get('speech_ms')} "
                  f"cap={cap(r.get('speech_ms', 0) / 1000, args.final_max)}")
        print(f"  interim 정상 잘림 {len(cut_i)}/{len(ok_i)} ({100 * len(cut_i) / max(1, len(ok_i)):.2f}%)  "
              f"상한 닿은 {len(ic)}개에서 줄어드는 토큰 {save_i}")

    if has_cap:
        print(f"\n[token_cap 검사] 위반(gen_tokens > token_cap) {len(viol)}개")
        for r in viol[:10]:
            print(f"  위반 {r['event']} trace={r['trace_id']} seg={r.get('segment_id')} "
                  f"gen={r['gen_tokens']} cap={r['token_cap']}")
    else:
        print("\n[token_cap 검사] 로그에 token_cap 필드 없음 (상한 기능 이전 코드이거나 로그를 안 남김)")
    return 1 if viol else 0


if __name__ == "__main__":
    sys.exit(main())
