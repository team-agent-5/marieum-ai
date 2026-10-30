"""server/engine.py 검증 (CALIBRATION.md §0.5).

1. 엔진(인코더 1회 + encoder_outputs로 generate, batch)이 노트북 호출과 같은 전사를 내는지
2. batch 1 / batch 4 결과가 같은지, 걸리는 시간
3. no_speech_prob: 말소리 조각 vs 무음 vs 잡음
4. --dtype fp16이면 fp32 노트북 결과와 몇 건이 다른지

  python scripts/verify_engine.py --model models/ASR_Whisper_Disease_General --chunks data/chunks
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.smoke_test import load_wav  # noqa: E402
from server.engine import AsrEngine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--dtype", default="fp32")
    args = ap.parse_args()

    eng = AsrEngine(args.model, args.dtype)
    print(f"[load] {eng.load_s:.1f}s dtype={args.dtype} no_speech_id={eng.no_speech_id} "
          f"({eng.processor.tokenizer.convert_ids_to_tokens(eng.no_speech_id)})")
    manifest = json.load(open(os.path.join(args.chunks, "manifest.json")))
    audios = [load_wav(os.path.join(args.chunks, m["file"]))[0].numpy() for m in manifest]

    # 기준: 노트북 호출 (fp32, 1건씩)
    ref_model = eng.model if args.dtype == "fp32" else None
    if ref_model is None:
        from transformers import WhisperForConditionalGeneration
        ref_model = WhisperForConditionalGeneration.from_pretrained(args.model).to("cuda").eval()
    ref = []
    with torch.inference_mode():
        for a in audios:
            f = eng.processor(a, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")
            ids = ref_model.generate(f, forced_decoder_ids=eng.fdi)
            ref.append(eng.processor.batch_decode(ids, skip_special_tokens=True)[0].strip())

    eng.transcribe([audios[0]], 128)  # 첫 호출 제외
    b1, t1 = [], []
    for a in audios:
        r, tm = eng.transcribe([a], 128)
        b1.append(r[0])
        t1.append(tm["gpu_ms"])
    b4, t4 = [], []
    for i in range(0, len(audios), 4):
        r, tm = eng.transcribe(audios[i:i + 4], 128)
        b4 += r
        t4.append(tm["gpu_ms"])

    diff1 = [i for i in range(len(audios)) if b1[i]["text"] != ref[i]]
    diff4 = [i for i in range(len(audios)) if b4[i]["text"] != b1[i]["text"]]
    print(f"[same] 엔진 batch1 vs 노트북: {len(audios) - len(diff1)}/{len(audios)} 같음")
    for i in diff1[:10]:
        print(f"   다름 {manifest[i]['file']}: 노트북={ref[i]!r} 엔진={b1[i]['text']!r}")
    print(f"[same] batch4 vs batch1: {len(audios) - len(diff4)}/{len(audios)} 같음")
    for i in diff4[:10]:
        print(f"   다름 {manifest[i]['file']}: b1={b1[i]['text']!r} b4={b4[i]['text']!r}")
    cd = [abs(b4[i]["confidence"] - b1[i]["confidence"]) for i in range(len(audios))]
    print(f"[conf] batch4-batch1 차이 최대 {max(cd):.4f}")
    print(f"[time] batch1 gpu_ms 중앙값 {np.median(t1):.0f} (1건당) / batch4 gpu_ms 중앙값 "
          f"{np.median(t4):.0f} (4건 묶음, 1건당 {np.median(t4) / 4:.0f})")

    nsp = [r["no_speech_prob"] for r in b1]
    print(f"[no_speech] 말소리 조각 중앙값 {np.median(nsp):.4f} 최대 {max(nsp):.4f}")
    rng = np.random.default_rng(0)
    probes = {"무음 1초": np.zeros(16000, np.float32), "무음 3초": np.zeros(48000, np.float32),
              "잡음 1초 (-40dBFS)": (rng.standard_normal(16000) * 0.01).astype(np.float32),
              "잡음 1초 (-26dBFS)": (rng.standard_normal(16000) * 0.05).astype(np.float32),
              "말소리 앞 0.5초": audios[0][16000:24000]}
    r, _ = eng.transcribe(list(probes.values()), 128)
    for name, x in zip(probes, r):
        print(f"   {name}: no_speech={x['no_speech_prob']:.4f} avg_logprob={x['avg_logprob']:.2f} "
              f"conf={x['confidence']:.3f} text={x['text']!r}")
    print(f"[gpu] max_alloc={torch.cuda.max_memory_allocated() / 2**20:.0f}MiB")


if __name__ == "__main__":
    t = time.perf_counter()
    main()
    print(f"[done] {time.perf_counter() - t:.0f}s")
