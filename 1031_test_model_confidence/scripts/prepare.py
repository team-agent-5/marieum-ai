"""환자 녹음을 conversation 모드 세션용 부분(part)으로 바꾼다 (TEST_PLAN.md 5.1).

- 16kHz 모노 PCM16으로 바꾼다 (스테레오는 평균, 32-bit float도 읽음)
- 세션 최대 600초라서 590초 이하 부분으로 나눈다. 540~590초 사이에서 가장 긴 쉼의 가운데에서 자른다
- BHN, HHS는 첫 부분만 만든다
- 출력
  - data/audio16k/<화자>_pNN.wav  (GPU 서버로 올림)
  - data/parts.json               (부분 목록, 정답 없음. GPU 서버로 올림)
  - data/refs.json                (화자별 대본과 메타. 로컬에만 둔다)

사용법 (1031_test_model_confidence 폴더에서):
  python scripts/prepare.py
  python scripts/prepare.py --only OTK KJJ   # 일부 화자만
"""

import argparse
import glob
import json
import math
import os
import struct
import wave

import numpy as np
from scipy.signal import resample_poly

SR = 16000
FRAME_MS = 20
MAX_PART_S = 590.0  # 서버 세션 상한 600초보다 조금 짧게
CUT_WINDOW_S = 50.0  # 540~590초에서 자를 곳을 찾는다
THRESH_DB_ABOVE_FLOOR = 15.0  # slice_sample.py와 같음: 잡음 바닥 + 15dB 넘으면 말소리
BLOCK_S = 60  # 큰 파일을 이만큼씩 읽는다

# 화자: (점수, 묶음, 태그, 학습 여부, 정렬 방식, 첫 부분만)
SPEAKERS = {
    "LOS": (0, "easy", [], False, "global", False),
    "KJJ": (1, "easy", ["음절끊김"], False, "global", False),
    "CJK": (1, "easy", [], False, "global", False),
    "JJH": (2, "easy", ["음절끊김"], False, "global", False),
    "LGS": (2, "easy", ["단조억양"], False, "global", False),
    "PAS": (2, "easy", ["음절끊김"], False, "global", False),
    "AJH": (2, "easy", ["음절끊김"], True, "global", False),
    "LUA": (2, "easy", ["음절끊김"], False, "global", False),
    "HHS": (5, "hard", [], False, "ref_prefix", True),
    "HJY": (5, "hard", [], False, "global", False),
    "KSW": (5, "hard", [], False, "global", False),
    "KMH": (4, "hard", [], False, "global", False),
    "LJG": (4, "hard", [], False, "ref_inside", False),
    "LSA": (3, "hard", [], False, "ref_inside", False),
    "LSY": (3, "hard", [], False, "global", False),
    "OTK": (3, "hard", [], False, "global", False),
    "BDS": (3, "hard", [], True, "global", False),
    "BHL": (3, "hard", [], True, "global", False),
    "BJS": (3, "hard", [], True, "global", False),
    "BHN": (3, "hard", ["ㅅ→th"], True, "ref_prefix", True),
    "HMS": (3, "hard", [], False, "global", False),
    "KJM": (3, "hard", [], False, "global", False),
}


# ---- WAV 읽기 (PCM16/24/32, float32. 큰 파일은 memmap) -----------------------
class Wav:
    def __init__(self, path):
        self.path = path
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(12)
            assert head[:4] == b"RIFF" and head[8:12] == b"WAVE", f"WAV 아님: {path}"
            pos, fmt = 12, None
            while pos + 8 <= size:
                f.seek(pos)
                cid, n = struct.unpack("<4sI", f.read(8))
                if cid == b"fmt ":
                    raw = f.read(n)
                    tag, self.ch, self.sr, _, self.block, self.bits = struct.unpack("<HHIIHH", raw[:16])
                    if tag == 0xFFFE:  # WAVE_FORMAT_EXTENSIBLE: 실제 형식은 subformat 앞 2바이트
                        tag = struct.unpack("<H", raw[24:26])[0]
                    fmt = tag
                elif cid == b"data":
                    self.offset = pos + 8
                    self.n = (min(n, size - self.offset)) // self.block
                    break
                pos += 8 + n + (n & 1)
        assert fmt in (1, 3), f"지원하지 않는 형식 {fmt}: {path}"
        self.float = fmt == 3
        assert (self.float and self.bits == 32) or (not self.float and self.bits in (16, 24, 32)), \
            f"지원하지 않는 비트 {self.bits}: {path}"
        self.duration_s = self.n / self.sr

    def read(self, start, stop):
        """[start, stop) 프레임을 모노 float32(-1~1)로."""
        start, stop = max(0, start), min(self.n, stop)
        if stop <= start:
            return np.zeros(0, np.float32)
        if self.bits == 24:
            with open(self.path, "rb") as f:
                f.seek(self.offset + start * self.block)
                b = np.frombuffer(f.read((stop - start) * self.block), np.uint8).reshape(-1, 3)
            v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
            x = (np.where(v >= 1 << 23, v - (1 << 24), v) / float(1 << 23)).astype(np.float32)
        else:
            dt = {(True, 32): "<f4", (False, 16): "<i2", (False, 32): "<i4"}[(self.float, self.bits)]
            m = np.memmap(self.path, dtype=dt, mode="r", offset=self.offset, shape=(self.n * self.ch,))
            x = np.asarray(m[start * self.ch: stop * self.ch], dtype=np.float32)
            if not self.float:
                x /= float(1 << (self.bits - 1))
        return x.reshape(-1, self.ch).mean(axis=1)


def frame_db(wav, limit):
    """20ms 프레임별 RMS(dBFS). 메모리를 아끼려고 BLOCK_S씩 읽는다."""
    hop = wav.sr * FRAME_MS // 1000
    n_frames = limit // hop
    out = np.empty(n_frames, np.float32)
    per_block = BLOCK_S * 1000 // FRAME_MS
    for i in range(0, n_frames, per_block):
        j = min(n_frames, i + per_block)
        x = wav.read(i * hop, j * hop).reshape(j - i, hop)
        out[i:j] = 10 * np.log10((x.astype(np.float64) ** 2).mean(axis=1) + 1e-10)
    return out


def plan_cuts(db, first_only):
    """부분 경계(초) 목록과 각 경계에서 자른 쉼 길이(초)."""
    floor = float(np.percentile(db, 10))
    silent = db <= floor + THRESH_DB_ABOVE_FLOOR
    fps = 1000 // FRAME_MS
    total_s = len(db) / fps
    bounds, pauses, start = [0.0], [], 0.0
    while total_s - start > MAX_PART_S:
        a, b = int((start + MAX_PART_S - CUT_WINDOW_S) * fps), int((start + MAX_PART_S) * fps)
        best_len, best_mid = 0, b  # 쉼이 없으면 590초에서 자른다
        k = a
        while k < b:
            if silent[k]:
                e = k
                while e < b and silent[e]:
                    e += 1
                if e - k > best_len:
                    best_len, best_mid = e - k, (k + e) // 2
                k = e
            else:
                k += 1
        start = best_mid / fps
        bounds.append(start)
        pauses.append(best_len / fps)
        if first_only:
            return bounds, pauses, floor
    if not first_only or len(bounds) == 1:
        bounds.append(total_s)
    return bounds, pauses, floor


def to_16k(x, sr):
    if sr != SR:
        g = math.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return (np.clip(x, -1, 1) * 32767).astype("<i2")


def write_wav(path, pcm):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def parse_name(stem, spk):
    """ID-02-25-N-OTK-02-01-M-30-SU → 질환 25, 검사 02-01."""
    t = stem.split("-")
    i = t.index(spk)
    j = next(k for k in range(i + 1, len(t)) if t[k] in ("M", "F"))
    return t[2], "-".join(t[i + 1:j])


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(here, "raw"))
    ap.add_argument("--out", default=os.path.join(here, "data"))
    ap.add_argument("--only", nargs="+", help="이 화자만 (기본: 22명 전부)")
    args = ap.parse_args()

    audio_dir = os.path.join(args.out, "audio16k")
    os.makedirs(audio_dir, exist_ok=True)
    speakers = args.only or list(SPEAKERS)
    for spk in speakers:  # 다시 만들 화자의 옛 부분 파일은 지운다
        for p in glob.glob(os.path.join(audio_dir, f"{spk}_p*.wav")):
            os.remove(p)

    parts_path, refs_path = os.path.join(args.out, "parts.json"), os.path.join(args.out, "refs.json")
    parts, refs = [], {}
    if args.only and os.path.exists(parts_path):  # 일부만 다시 만들 때는 나머지를 유지
        parts = [p for p in json.load(open(parts_path, encoding="utf-8")) if p["speaker"] not in speakers]
        refs = {k: v for k, v in json.load(open(refs_path, encoding="utf-8")).items() if k not in speakers}

    print(f"{'화자':4s} {'원본':>13s} {'길이':>7s} {'사용':>7s} 부분  자른 쉼(초)        대본 글자")
    for spk in speakers:
        score, group, tags, trained, align, first_only = SPEAKERS[spk]
        found = glob.glob(os.path.join(args.raw, "원본데이터", "*", f"*-{spk}-*.wav"))
        assert len(found) == 1, f"{spk}: wav {len(found)}개"
        src = found[0]
        stem = os.path.splitext(os.path.basename(src))[0]
        disease, test = parse_name(stem, spk)
        label = glob.glob(os.path.join(args.raw, "라벨링데이터", "*", stem + ".json"))
        assert len(label) == 1, f"{spk}: 라벨 {len(label)}개"
        transcript = json.load(open(label[0], encoding="utf-8-sig"))["Transcript"]

        wav = Wav(src)
        limit = min(wav.n, int((MAX_PART_S + 1) * wav.sr)) if first_only else wav.n
        db = frame_db(wav, limit)
        bounds, pauses, floor = plan_cuts(db, first_only)

        for k in range(len(bounds) - 1):
            s0, s1 = bounds[k], bounds[k + 1]
            pcm = to_16k(wav.read(int(s0 * wav.sr), int(s1 * wav.sr)), wav.sr)
            name = f"{spk}_p{k + 1:02d}.wav"
            write_wav(os.path.join(audio_dir, name), pcm)
            parts.append({"speaker": spk, "part": k + 1, "wav": f"audio16k/{name}", "src": os.path.basename(src),
                          "src_start_s": round(s0, 2), "src_end_s": round(s1, 2),
                          "duration_s": round(len(pcm) / SR, 2)})

        refs[spk] = {"src": os.path.basename(src), "disease": disease, "test": test, "trained": trained,
                     "score": score, "group": group, "tags": tags, "align": align,
                     "src_duration_s": round(wav.duration_s, 1), "used_s": round(bounds[-1], 1),
                     "src_format": f"{wav.sr}Hz ch{wav.ch} {wav.bits}bit{' float' if wav.float else ''}",
                     "noise_floor_db": round(floor, 1), "transcript": transcript}
        print(f"{spk:4s} {wav.sr:>6d}Hz ch{wav.ch} {wav.duration_s / 60:6.1f}m {bounds[-1] / 60:6.1f}m "
              f"{len(bounds) - 1:3d}  {', '.join(f'{p:.1f}' for p in pauses) or '-':18s} {len(transcript):6d}")

    order = {s: i for i, s in enumerate(SPEAKERS)}
    parts.sort(key=lambda p: (order[p["speaker"]], p["part"]))
    with open(parts_path, "w", encoding="utf-8") as f:
        json.dump(parts, f, ensure_ascii=False, indent=1)
    with open(refs_path, "w", encoding="utf-8") as f:
        json.dump({k: refs[k] for k in sorted(refs, key=order.get)}, f, ensure_ascii=False, indent=1)

    total = sum(p["duration_s"] for p in parts)
    longest = max(p["duration_s"] for p in parts)
    print(f"\n부분 {len(parts)}개, 합계 {total / 60:.1f}분, 가장 긴 부분 {longest:.1f}초")
    print(f"동시 4세션 실제 속도 예상: 약 {total / 4 / 60:.0f}분 (부분 길이가 고르지 않으면 더 걸림)")
    print(f"→ {parts_path}\n→ {refs_path}\n→ {audio_dir}")


if __name__ == "__main__":
    main()
