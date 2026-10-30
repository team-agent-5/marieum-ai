"""조각 전체 전사와 긴 발화 시간 측정 (CALIBRATION.md §9, §10).

1. data/chunks의 조각을 서버와 같은 옵션으로 전사한다 (VAD 없음, 앞뒤 1초 무음 포함)
   - confidence 분포, 1건당 feat/gen 시간과 생성 토큰 수
   - 평서문·의문문 짝 확인: 조각 2k-1과 2k의 전사가 (문장부호 빼고) 같은지
2. 말소리 구간을 이어 붙여 약 10초, 약 19.5초 꽉 찬 음성을 만들고 gen 시간을 잰다
   (§9 1건당 GPU 예산 625ms 확인용)

사용법:
  python scripts/run_chunks.py --model models/ASR_Whisper_Disease_General \
      --sample data/sample.wav --chunks data/chunks --out data/chunks_result.json
"""

import argparse
import json
import os
import re
import time
import wave

import numpy as np
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from smoke_test import load_wav

MAX_NEW_TOKENS = 128
PUNCT = re.compile(r"[+*()?!,.~\-'\s]")  # 학습 전처리와 같은 문장부호 + 공백


def norm(t):
    return PUNCT.sub("", t)


class Asr:
    def __init__(self, model_dir):
        self.dev = torch.device("cuda")
        self.processor = WhisperProcessor.from_pretrained(model_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(self.dev).eval()
        self.fdi = self.processor.get_decoder_prompt_ids(language="ko", task="transcribe")
        self.special = set(self.processor.tokenizer.all_special_ids)
        self.eot = self.processor.tokenizer.eos_token_id

    def __call__(self, audio16k, max_new_tokens=MAX_NEW_TOKENS, max_time=2.5):
        t = time.perf_counter()
        feats = self.processor(audio16k.numpy(), sampling_rate=16000,
                               return_tensors="pt").input_features.to(self.dev)
        torch.cuda.synchronize()
        feat_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(
                feats, forced_decoder_ids=self.fdi, return_dict_in_generate=True,
                output_scores=True, max_new_tokens=max_new_tokens, max_time=max_time)
        torch.cuda.synchronize()
        gen_ms = (time.perf_counter() - t) * 1000
        seq = out.sequences[0].tolist()
        gen = seq[-len(out.scores):]
        probs = [torch.softmax(out.scores[i][0].float(), -1)[tid].item()
                 for i, tid in enumerate(gen) if tid not in self.special]
        return {
            "text": self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0],
            "confidence": round(float(np.mean(probs)), 4) if probs else 0.0,
            "gen_tokens": len(gen), "ended_eot": gen[-1] == self.eot,
            "feat_ms": round(feat_ms, 1), "gen_ms": round(gen_ms, 1),
        }


def pct(a, q):
    return round(float(np.percentile(a, q)), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    asr = Asr(args.model)
    manifest = json.load(open(os.path.join(args.chunks, "manifest.json")))

    # 첫 요청은 통계에서 뺀다 (§9)
    first, _, _ = load_wav(os.path.join(args.chunks, manifest[0]["file"]))
    warm = asr(first)
    print(f"[first] feat={warm['feat_ms']}ms gen={warm['gen_ms']}ms (통계 제외)")

    rows = []
    for m in manifest:
        audio, sr, dur = load_wav(os.path.join(args.chunks, m["file"]))
        r = asr(audio)
        r.update(file=m["file"], speech_s=m["speech_s"], chunk_s=round(dur, 2))
        rows.append(r)
        print(f"{m['file']} {dur:5.2f}s conf={r['confidence']:.3f} tok={r['gen_tokens']:3d} "
              f"gen={r['gen_ms']:6.1f}ms eot={r['ended_eot']}  {r['text']}")

    conf = np.array([r["confidence"] for r in rows])
    gen_ms = np.array([r["gen_ms"] for r in rows])
    feat_ms = np.array([r["feat_ms"] for r in rows])
    print("\n[confidence]", {"min": pct(conf, 0), "p10": pct(conf, 10), "median": pct(conf, 50),
                             "mean": round(float(conf.mean()), 4), "max": pct(conf, 100)})
    print("[gen_ms]", {"median": pct(gen_ms, 50), "p95": pct(gen_ms, 95), "max": pct(gen_ms, 100)})
    print("[feat_ms]", {"median": pct(feat_ms, 50), "max": pct(feat_ms, 100)})
    print("[empty]", sum(1 for r in rows if not r["text"].strip()),
          "[not_eot]", sum(1 for r in rows if not r["ended_eot"]))

    pairs = []
    for i in range(0, len(rows) - 1, 2):
        a, b = rows[i], rows[i + 1]
        pairs.append({"a": a["file"], "b": b["file"], "same": norm(a["text"]) == norm(b["text"]),
                      "text_a": a["text"], "text_b": b["text"]})
    print(f"[pairs] 짝 전사 일치 {sum(p['same'] for p in pairs)}/{len(pairs)}")
    for p in pairs:
        if not p["same"]:
            print(f"   다름: {p['a']} {p['text_a']!r} / {p['b']} {p['text_b']!r}")

    # 긴 발화: 말소리 구간을 0.3초 무음으로 이어 붙인다
    with wave.open(args.sample, "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    gap = np.zeros(int(0.3 * sr), dtype="<i2")
    long_runs = []
    for target in (10.0, 19.5):
        parts, total, used = [], 0.0, []
        for m in manifest:
            seg = pcm[int(m["speech_start_s"] * sr): int(m["speech_end_s"] * sr)]
            if total + len(seg) / sr > target:
                break
            parts += [seg, gap]
            total += len(seg) / sr + 0.3
            used.append(m["file"])
        x = torch.from_numpy(np.concatenate(parts).astype(np.float32) / 32768.0)[None]
        x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=16000)[0]
        # 실제 비용을 보려고 max_time 없이, 토큰 상한은 넉넉히
        res = [asr(x, max_new_tokens=256, max_time=None) for _ in range(3)]
        r = res[-1]
        med_gen = float(np.median([q["gen_ms"] for q in res]))
        print(f"\n[long {target}s] 실제 {total:.2f}s, 조각 {len(used)}개, tok={r['gen_tokens']}, "
              f"eot={r['ended_eot']}, feat={r['feat_ms']}ms, gen 중앙값={med_gen:.0f}ms "
              f"(3회 {[q['gen_ms'] for q in res]}), ms/token={med_gen / r['gen_tokens']:.1f}")
        print(f"   {r['text']}")
        long_runs.append({"target_s": target, "actual_s": round(total, 2), "chunks": used,
                          "gen_ms_runs": [q["gen_ms"] for q in res], **r})

    json.dump({"rows": rows, "pairs": pairs, "long": long_runs, "first": warm},
              open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
