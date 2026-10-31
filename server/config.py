"""환경변수 설정 (CALIBRATION.md §0.4). 기본값은 2026-10-31 합의값."""

import os
from dataclasses import dataclass


def _f(name, default):
    return float(os.environ.get(name, default))


def _i(name, default):
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    model_dir: str = os.environ.get("ML_MODEL_DIR", "models/ASR_Whisper_Disease_General")
    model_version: str = os.environ.get("ML_MODEL_VERSION", "yoona-J/ASR_Whisper_Disease_General")
    # fp16: 조각 60개 전사가 fp32 노트북과 60/60 같고, 4건 묶음 GPU 시간 702ms → 248ms (2026-10-31)
    dtype: str = os.environ.get("ML_DTYPE", "fp16")  # fp16 | fp32

    # GPU 작업자
    batch_max: int = _i("ML_BATCH_MAX", 4)
    interim_max_new_tokens: int = _i("ML_INTERIM_MAX_NEW_TOKENS", 96)
    final_max_new_tokens: int = _i("ML_FINAL_MAX_NEW_TOKENS", 128)
    # 말소리 길이 비례 상한 (MAX_NEW_TOKENS.md). 14는 2026-10-31·11-01 GPU 로그에서 정상 final을 하나도 안 자른 값. 0이면 끔
    token_cap_rate: float = _f("ML_TOKEN_CAP_RATE", 14.0)
    token_cap_floor: int = _i("ML_TOKEN_CAP_FLOOR", 8)

    # interim
    interim_step_ms: int = _i("ML_INTERIM_STEP_MS", 500)
    min_interim_speech_ms: int = _i("ML_MIN_INTERIM_SPEECH_MS", 300)

    # VAD (§6 값 유지 + 문장 끊기)
    vad_threshold: float = _f("ML_VAD_THRESHOLD", 0.5)
    speech_pad_ms: int = _i("ML_SPEECH_PAD_MS", 200)
    min_speech_ms: int = _i("ML_MIN_SPEECH_MS", 150)
    end_silence_ms: int = _i("ML_END_SILENCE_MS", 1750)

    # 길이
    utterance_max_s: float = _f("ML_UTTERANCE_MAX_S", 10.0)
    # 대화 모드 구간 최대 길이. 쉼 없이 말하는 4세션에서 10초면 interim p95 851ms, 6초면 543ms (2026-10-31).
    # 문장이 너무 일찍 끊겨 끝 무음 1.75초와 함께 15초로 늘림 (2026-11-01). Whisper 창 30초 안.
    # 18초는 4세션 연속 발화에서 interim p95 1353ms라 줄임
    max_segment_s: float = _f("ML_MAX_SEGMENT_S", 15.0)
    session_max_s: float = _f("ML_SESSION_MAX_S", 600.0)
    # POST /asr 파일 하나의 최대 길이. 긴 파일은 대화 모드처럼 쉼에서 나눠 각각 전사한다
    asr_max_s: float = _f("ML_ASR_MAX_S", 60.0)

    # Whisper 무음 판정: no_speech_prob > 0.6 이고 avg_logprob <= logprob_threshold 이면 빈 전사.
    # openai/whisper 기본은 -1.0이지만 이 모델은 무음에서 확신 있게 지어낸다
    # (무음 1초: no_speech 0.69, avg_logprob -0.54). 말소리 조각 60개의 no_speech는 최대 0.0000이라
    # no_speech만으로 거른다 (0.0 = logprob 조건 끔). verify_engine.py 2026-10-31
    no_speech_threshold: float = _f("ML_NO_SPEECH_THRESHOLD", 0.6)
    logprob_threshold: float = _f("ML_LOGPROB_THRESHOLD", 0.0)

    debug_save: bool = os.environ.get("ML_DEBUG_SAVE", "0") == "1"
    debug_dir: str = os.environ.get("ML_DEBUG_DIR", "/tmp/asr-debug")
    debug_ttl_s: int = _i("ML_DEBUG_TTL_S", 1800)


settings = Settings()
