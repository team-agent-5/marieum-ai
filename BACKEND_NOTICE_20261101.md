# ML 서버 변경 안내 (2026-11-01)

운영 ML 서버(`10.1.2.11:8001`)에 아래 변경을 적용했습니다. 기존 메시지 형식은 그대로라서 **지금 백엔드 코드도 그대로 동작합니다.** 필드와 엔드포인트가 추가된 것이니, 필요한 부분만 반영해 주세요.

## 요약

| # | 변경 | 백엔드 영향 |
|---|---|---|
| 1 | 실시간(`conversation`) 모드의 문장 끝 무음 0.8초 → **1.75초**, 구간 상한 6초 → **15초** | 문장이 덜 끊깁니다. final은 말이 끝나고 약 1.75초 뒤에 옵니다. |
| 2 | interim과 final에 **`stable_delta`, `stable_revised`** 추가 | 확정된 단어만 새로 받을 수 있습니다 (선택). |
| 3 | 녹음 파일 전사 **`POST /asr`** 추가 | `/phrase/transcribe`의 가짜 응답을 실제 인식 결과로 바꿀 수 있습니다. |

---

## 1. 실시간 모드 문장 끊기 기준 변경

| 항목 | 이전 | 지금 |
|---|---|---|
| 문장 끝으로 보는 무음 길이 (`ML_END_SILENCE_MS`) | 800ms | **1750ms** |
| 한 구간 최대 길이 (`ML_MAX_SEGMENT_S`) | 6초 | **15초** |

- **`conversation` 모드에만 적용됩니다.** `utterance` 모드는 지금처럼 백엔드가 `end`를 보낼 때 끝나고, 최대 10초입니다.
- 1.75초보다 짧은 쉼은 한 문장으로 이어집니다. 쉼 없이 15초를 넘게 말하면 15초 근처의 쉼에서 잘립니다.
- final은 말이 끝나고 약 1.75초 + GPU 처리 0.1–0.4초 뒤에 옵니다.
- 측정 결과:
  - 동시 4세션이 쉼 없이 계속 말하는 최악 조건에서 interim 지연은 p50 493ms, p95 1132ms였습니다. 이전 6초 상한 때 p95는 543ms였습니다.
  - 짧은 문장 위주로 쓰면 이보다 낮습니다.

### ⚠️ `conversation` 모드를 쓴다면 꼭 확인해 주세요
- `conversation`은 **한 세션에 final이 여러 개** 옵니다. 문장마다 `segment_id`가 0, 1, 2…로 붙습니다.
- 지금 `ml_proxy.py`의 `_recv_loop`는 **첫 `transcript_final`에서 수신을 끝냅니다.** 이대로면 두 번째 문장부터 버려집니다.
- 이렇게 바꿔 주세요: **`is_last: true`인 final이 올 때까지 계속 받기.** 백엔드가 `end`를 보내면 마지막 final에 `is_last: true`가 붙습니다.
- `utterance` 모드만 쓴다면 final이 하나라서 지금 코드 그대로 괜찮습니다.

---

## 2. 확정된 단어만 내려주기: `stable_delta` / `stable_revised`

`transcript_interim`과 `transcript_final`에 필드 두 개가 추가됐습니다. 두 모드 모두에 붙습니다.

| 필드 | 타입 | 뜻 |
|---|---|---|
| `stable_delta` | string | 이번에 **새로 확정된 단어만** (공백으로 구분). 없으면 `""` |
| `stable_revised` | bool | 앞에서 보낸 확정 단어가 바뀌었는지. `true`면 `stable_delta`에 그 구간의 문장 **전체**가 들어 있음 |

예: "물 좀 주세요"
```json
{"type":"transcript_interim","segment_id":0,"text":"물 좀","stable_text":"물","unstable_text":"좀","stable_delta":"물","stable_revised":false, ...}
{"type":"transcript_interim","segment_id":0,"text":"물 좀 주세","stable_text":"물 좀","unstable_text":"주세","stable_delta":"좀","stable_revised":false, ...}
{"type":"transcript_final","segment_id":0,"text":"물 좀 주세요","stable_delta":"주세요","stable_revised":false,"is_last":true, ...}
```

쓰는 법:
1. `segment_id`별로 `stable_delta`를 이어 붙입니다.
2. `stable_revised: true`가 오면 그 구간에 붙여 둔 단어를 버리고, `stable_delta`로 **교체**합니다.
3. final이 오면 그 구간 문장은 final의 `text`로 확정합니다. 1–2번대로 이어 붙인 결과는 final `text`와 같습니다.

- 기존 필드(`text`, `stable_text`, `unstable_text`)는 그대로입니다. 안 쓸 거면 무시해도 됩니다.
- 구간이 길면 확정 단어가 몇 초 동안 안 나오다가 final에서 한꺼번에 오기도 합니다. 그때 `stable_revised: true`일 수 있습니다.

---

## 3. 녹음 파일 전사: `POST /asr`

녹음 파일 하나를 보내면 한 번에 전사합니다.

### 요청
```bash
# WAV 파일을 본문에 그대로
curl -X POST -H "Content-Type: audio/wav" -H "X-Trace-Id: abc123" \
     --data-binary @rec.wav http://10.1.2.11:8001/asr

# PCM16 LE 모노 바이트는 sample_rate를 같이
curl -X POST --data-binary @rec.pcm "http://10.1.2.11:8001/asr?sample_rate=16000&trace_id=abc123"
```
```python
# httpx 예시
r = await client.post("http://10.1.2.11:8001/asr", content=wav_bytes,
                      headers={"Content-Type": "audio/wav", "X-Trace-Id": trace_id}, timeout=15)
```

| 항목 | 내용 |
|---|---|
| 본문 | **WAV 파일 바이트 그대로**, 또는 PCM16 LE 모노 바이트 + `?sample_rate=` |
| WAV 형식 | PCM 8/16/24/32비트, 모노·스테레오, 8–48kHz. 서버가 16kHz 모노로 바꿉니다. |
| 최대 길이 | **60초** |
| trace_id | `X-Trace-Id` 헤더나 `?trace_id=` 중 하나 (영문·숫자·`_-`, 64자 이내). 없으면 서버가 만듭니다. |
| 언어 | 한국어만 |

**아직 안 되는 것:**
- **multipart/form-data 업로드는 안 됩니다.** 파일 바이트를 본문에 그대로 보내 주세요.
- MP3, M4A, WebM 같은 압축 포맷도 안 됩니다. 필요하면 말씀해 주세요.

### 응답 (200)
```json
{
  "trace_id": "abc123",
  "text": "문 열어",
  "confidence": 0.9951,
  "duration_ms": 2700,
  "model_version": "yoona-J/ASR_Whisper_Disease_General",
  "segments": [
    {"text": "문 열어", "confidence": 0.9951, "start_ms": 832, "end_ms": 2720}
  ]
}
```
- 긴 파일은 실시간 모드와 같은 기준(무음 1.75초, 최대 15초)으로 나눠 전사합니다. `segments`에 구간별 결과가 있습니다.
- `text`는 구간 텍스트를 공백으로 이은 것입니다. `confidence`는 구간 길이로 가중 평균한 값입니다.
- **말소리가 없으면** `text: ""`, `segments: []`, `confidence: 0.0`입니다. LLM이나 TTS로 넘기지 말고 "다시 말씀해 주세요" 같은 처리를 부탁드립니다.
- 처리 시간 실측: 2.7초 파일은 0.18초, 55초 파일은 1.25초.

### 에러
본문은 `{"code": "...", "message": "...", "trace_id": "..."}`입니다.

| 상태 | code | 원인 |
|---|---|---|
| 400 | `INVALID_FORMAT` | 본문이 비어 있음, sample_rate가 범위 밖, trace_id 형식 오류 |
| 413 | `AUDIO_TOO_LONG` | 60초 초과 |
| 415 | `INVALID_FORMAT` | WAV가 아니고 `sample_rate`도 없음 (압축 포맷 등) |
| 503 | `MODEL_UNAVAILABLE` | 서버 재시작 직후 모델 로딩 중 (약 10초). 잠시 뒤 다시 시도 |
| 500 | `INTERNAL` | 서버 오류 |

---

## 공통으로 확인 부탁드리는 것

1. **빈 final (`text: ""`) 처리**
   - 말소리 없이 `end`를 보내면 빈 final이 옵니다.
   - LLM이나 TTS로 넘기지 말고 재시도를 안내해 주세요.
2. **에러 코드 처리**
   - `INVALID_FORMAT`, `AUDIO_TOO_LONG` 말고도 `NO_SESSION`, `MODEL_UNAVAILABLE`, `INTERNAL`이 올 수 있습니다.
   - 모르는 코드가 오면 세션을 정리하고 재시도하게 해 주세요.
3. **서버 재시작 직후**
   - 모델 로딩 중 약 10초 동안 WebSocket 연결은 `MODEL_UNAVAILABLE`(close 1013)로 거절됩니다.
   - 재접속 간격을 1–2초 이상 두면 됩니다.
4. **`/phrase/transcribe`**
   - 지금은 항상 "물 좀 주세요"(0.82)를 돌려주는 가짜 응답입니다.
   - `POST /asr`로 바꾸면 실제 인식 결과가 나옵니다.

## 엔드포인트 전체 목록

| 메서드 | 경로 | 용도 |
|---|---|---|
| WS | `/asr-stream` | 실시간 전사 (`utterance` / `conversation`) |
| POST | `/asr` | 녹음 파일 전사 (**신규**) |
| GET | `/healthz` | 상태 (모델 준비 전 503) |
| GET | `/livez` | 프로세스 생존 확인 |
| GET | `/docs`, `/redoc`, `/openapi.json` | API 문서 |

문의는 ML 담당에게 주세요.
