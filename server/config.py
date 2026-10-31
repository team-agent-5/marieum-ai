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

    # interim
    interim_step_ms: int = _i("ML_INTERIM_STEP_MS", 500)
    # 잡음 몇 조각에 interim이 나가 지어낸 문장이 보이는 것을 줄인다 (2026-11-01, 이전 300)
    min_interim_speech_ms: int = _i("ML_MIN_INTERIM_SPEECH_MS", 500)

    # VAD (§6 값 유지 + 문장 끊기)
    vad_threshold: float = _f("ML_VAD_THRESHOLD", 0.5)
    speech_pad_ms: int = _i("ML_SPEECH_PAD_MS", 200)
    min_speech_ms: int = _i("ML_MIN_SPEECH_MS", 150)
    end_silence_ms: int = _i("ML_END_SILENCE_MS", 800)

    # 길이
    utterance_max_s: float = _f("ML_UTTERANCE_MAX_S", 10.0)
    # 대화 모드 구간 최대 길이. 쉼 없이 말하는 4세션에서 10초면 interim p95 851ms, 6초면 543ms (2026-10-31)
    max_segment_s: float = _f("ML_MAX_SEGMENT_S", 6.0)
    session_max_s: float = _f("ML_SESSION_MAX_S", 600.0)

    # Whisper 무음 판정: no_speech_prob > 0.6 이고 avg_logprob <= logprob_threshold 이면 빈 전사.
    # openai/whisper 기본은 -1.0이지만 이 모델은 무음에서 확신 있게 지어낸다
    # (무음 1초: no_speech 0.69, avg_logprob -0.54). 말소리 조각 60개의 no_speech는 최대 0.0000이라
    # no_speech만으로 거른다 (0.0 = logprob 조건 끔). verify_engine.py 2026-10-31
    no_speech_threshold: float = _f("ML_NO_SPEECH_THRESHOLD", 0.6)
    # 말소리 1초당 한글 음절이 이보다 많으면 지어낸 문장으로 보고 빈 글자로 바꾼다. 0이면 끔(기본).
    # VAD가 말소리를 덜 세면 정상 발화도 지워지는 반례가 있어(2026-11-01 검증), 로그의 syllables와
    # vad_speech_ms로 환자 음성 오탐률을 잰 뒤에 켠다
    max_syllables_per_s: float = _f("ML_MAX_SYLLABLES_PER_S", 0.0)
    logprob_threshold: float = _f("ML_LOGPROB_THRESHOLD", 0.0)

    debug_save: bool = os.environ.get("ML_DEBUG_SAVE", "0") == "1"
    debug_dir: str = os.environ.get("ML_DEBUG_DIR", "/tmp/asr-debug")
    debug_ttl_s: int = _i("ML_DEBUG_TTL_S", 1800)


settings = Settings()
