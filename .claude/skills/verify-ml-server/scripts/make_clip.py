"""WAV 일부를 16kHz 모노 PCM16 WAV로 잘라 낸다 (POST /asr, 발화 모드용). 또는 합성 음(톤/무음)을 만든다.

  python make_clip.py ~/exp-1031/audio16k/LSY_p01.wav 232 20 <RUN>/clip_LSY.wav
  python make_clip.py --synthetic "tone:2,sil:2.2,tone:3,sil:1" <RUN>/tone.wav    # 가짜 엔진 점검용

원본이 16kHz가 아니면 거부한다 (audio16k/는 이미 16kHz). 잘라 낸 파일은 RUN 폴더에만 두고 저장소에 넣지 않는다.
"""

import argparse
import sys
import wave

import numpy as np

SR = 16000


def write(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949)에서도 한글이 깨지지 않게
    ap = argparse.ArgumentParser()
    ap.add_argument("args", nargs="+", help="SRC OFFSET_S SECONDS OUT  또는  --synthetic PATTERN OUT")
    ap.add_argument("--synthetic", help='"tone:초,sil:초,..."')
    a = ap.parse_args()
    if a.synthetic:
        out = a.args[0]
        parts = []
        for item in a.synthetic.split(","):
            kind, sec = item.split(":")
            n = int(float(sec) * SR)
            parts.append(0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / SR) if kind == "tone" else np.zeros(n))
        write(out, np.concatenate(parts))
        print(f"{out} {sum(len(p) for p in parts) / SR:.2f}s (합성)")
        return 0
    src, off, sec, out = a.args[0], float(a.args[1]), float(a.args[2]), a.args[3]
    with wave.open(src, "rb") as w:
        if w.getframerate() != SR or w.getsampwidth() != 2:
            print(f"{src}: 16kHz PCM16만 받는다 ({w.getframerate()}Hz, {w.getsampwidth() * 8}bit)", file=sys.stderr)
            return 2
        ch = w.getnchannels()
        w.setpos(min(int(off * SR), w.getnframes()))
        raw = w.readframes(int(sec * SR))
    x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    x = x.reshape(-1, ch).mean(axis=1) if ch > 1 else x
    write(out, x)
    print(f"{out} {len(x) / SR:.2f}s ({src} {off:g}s~)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
