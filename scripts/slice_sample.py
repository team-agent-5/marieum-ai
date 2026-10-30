"""sample.wav를 발화 단위 조각으로 자른다 (CALIBRATION.md §10).

- 학습 전처리처럼 에너지 기준으로 말소리를 찾고, 3초 이상 무음에서 자른다
- 조각마다 앞뒤에 무음 1초씩 붙인다 (서버 VAD가 앞뒤를 자르는지 확인용)
- 원본 샘플레이트(44.1kHz) 그대로 저장한다

사용법:
  python scripts/slice_sample.py data/sample.wav data/chunks
"""

import argparse
import json
import os
import wave

import numpy as np

FRAME_MS = 20
SPLIT_SILENCE_S = 3.0  # 이 이상 무음이면 자른다
MIN_SPEECH_S = 0.2  # 이보다 짧은 덩어리는 잡음으로 버린다
PAD_S = 1.0
THRESH_DB_ABOVE_FLOOR = 15.0  # 잡음 바닥 + 15dB 이상이면 말소리


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out_dir")
    args = ap.parse_args()

    with wave.open(args.src, "rb") as w:
        params = w.getparams()
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        assert width == 2 and ch == 1, "모노 16-bit만 가정"
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")

    hop = sr * FRAME_MS // 1000
    n_frames = len(pcm) // hop
    frames = pcm[: n_frames * hop].astype(np.float32).reshape(n_frames, hop) / 32768.0
    rms_db = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-10)
    floor = np.percentile(rms_db, 10)
    thr = floor + THRESH_DB_ABOVE_FLOOR
    speech = rms_db > thr
    print(f"[energy] floor={floor:.1f}dBFS, thr={thr:.1f}dBFS, peak={rms_db.max():.1f}dBFS")

    # 말소리 프레임 → 구간, 3초 미만 무음은 이어 붙인다
    segs, start = [], None
    for i, s in enumerate(speech):
        if s and start is None:
            start = i
        elif not s and start is not None:
            segs.append([start, i])
            start = None
    if start is not None:
        segs.append([start, n_frames])
    gap_frames = int(SPLIT_SILENCE_S * 1000 / FRAME_MS)
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < gap_frames:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    min_frames = int(MIN_SPEECH_S * 1000 / FRAME_MS)
    merged = [s for s in merged if s[1] - s[0] >= min_frames]

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []
    for k, (a, b) in enumerate(merged, 1):
        s0, s1 = a * hop, b * hop
        p0, p1 = max(0, s0 - int(PAD_S * sr)), min(len(pcm), s1 + int(PAD_S * sr))
        name = f"chunk_{k:02d}.wav"
        with wave.open(os.path.join(args.out_dir, name), "wb") as o:
            o.setparams(params)
            o.writeframes(pcm[p0:p1].tobytes())
        manifest.append({
            "file": name, "speech_start_s": round(s0 / sr, 3), "speech_end_s": round(s1 / sr, 3),
            "speech_s": round((s1 - s0) / sr, 3), "chunk_s": round((p1 - p0) / sr, 3),
        })
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    sp = np.array([m["speech_s"] for m in manifest])
    gaps = np.array([manifest[i + 1]["speech_start_s"] - manifest[i]["speech_end_s"]
                     for i in range(len(manifest) - 1)])
    print(f"[chunks] {len(manifest)}개 (목표 60)")
    print(f"[speech_s] min={sp.min():.2f}, median={np.median(sp):.2f}, max={sp.max():.2f}")
    if len(gaps):
        print(f"[gap_s] min={gaps.min():.2f}, median={np.median(gaps):.2f}, max={gaps.max():.2f}")
    for m in manifest:
        print(f"  {m['file']}  {m['speech_start_s']:8.2f}–{m['speech_end_s']:8.2f}s  speech={m['speech_s']:.2f}s")


if __name__ == "__main__":
    main()
