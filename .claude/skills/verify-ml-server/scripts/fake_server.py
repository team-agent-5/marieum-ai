"""GPU·모델 없는 PC에서 하네스(launch/doctor/drive/cleanup)만 점검하는 가짜 서버.

실제 server.app을 그대로 쓰고 엔진과 VAD만 바꾼다 (tests/test_app.py와 같은 방식).
- 엔진: 0.5초마다 단어 하나, gen_tokens = 단어 수 + 1. max_new_tokens(정수 또는 줄별 목록)를 넘으면 자르고 ended_eot=False
- VAD: 에너지 기준
모델 출력(환각, 실제 토큰 수, 지연)은 증명하지 못한다. 토큰 상한 검증은 GPU 서버에서 한다.

  REPO=<저장소> python fake_server.py --port 18002     # 보통은 FAKE=1 launch.sh 로 부른다
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get("REPO") or os.getcwd())

import uvicorn  # noqa: E402

import server.app as app_mod  # noqa: E402

SR = 16000


class FakeEngine:
    load_s = 0.0

    def transcribe(self, audios, max_new_tokens, *args, **kwargs):
        caps = max_new_tokens if isinstance(max_new_tokens, (list, tuple)) else [max_new_tokens] * len(audios)
        out = []
        for a, cap in zip(audios, caps):
            rms = float(np.sqrt((a**2).mean())) if len(a) else 0.0
            silent = rms < 0.01
            n = max(1, int(len(a) / SR * 2))
            gen = min(n + 1, int(cap))  # 단어 n개 + 끝 토큰. 상한에 닿으면 끝 토큰 없이 단어만
            words = n if gen == n + 1 else gen
            out.append({"text": "" if silent else " ".join(f"단어{i}" for i in range(1, words + 1)),
                        "confidence": 0.9, "avg_logprob": -0.1,
                        "no_speech_prob": 0.9 if silent else 0.0,
                        "gen_tokens": gen, "ended_eot": gen == n + 1})
        time.sleep(0.01)
        return out, {"feat_ms": 1.0, "gpu_ms": 10.0, "batch": len(audios)}


class FakeVadModel:
    def reset_states(self):
        pass


def energy_vad(_model):
    return lambda windows: [1.0 if np.sqrt((w**2).mean()) > 0.01 else 0.0 for w in windows]


def fake_load():
    app_mod.state["gpu"] = {"name": "fake", "memory_free_mb": 0, "memory_total_mb": 0}
    app_mod.log("model_loaded", load_s=0.0, dtype="fake")
    return FakeEngine()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    app_mod.load_model = fake_load
    app_mod.make_vad = energy_vad
    app_mod.vads.acquire = lambda: FakeVadModel()
    uvicorn.run(app_mod.app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
