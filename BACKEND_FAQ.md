# 백엔드 질문 답변: ML 스트림 스펙

백엔드에서 보낸 질문 4개에 대한 답이다. 기준은 GPU 서버에서 돌고 있는 현재 코드(`feat/asr-stream`)다.

- 작성: 2026-10-31
- 전체 연동 규칙과 에러 코드: [HANDOFF.md](HANDOFF.md) §3–§10
- 주소: `ws://10.1.2.11:8001/asr-stream` (백엔드 서버 `10.1.1.11`에서만 접속 가능)

---

## 1. `start` 메시지 전체 필드

```json
{"type": "start", "trace_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
 "sample_rate": 16000, "encoding": "pcm16",
 "patient_id": "305-KIM-7X3K", "language": "ko", "mode": "utterance"}
```

| 필드 | 필수 | 값 | 틀리면 |
|---|---|---|---|
| `type` | ✓ | `"start"` | — |
| `trace_id` | ✓ | 영문·숫자·`-`·`_` 1–64자 (UUIDv4 그대로 OK) | `INVALID_FORMAT` |
| `sample_rate` | ✓ | 정수 8000–48000, **16000 권장**. 실제 오디오와 같아야 한다 | `INVALID_FORMAT` |
| `encoding` | 선택 | `"pcm16"`만 (기본값) | `INVALID_FORMAT` |
| `patient_id` | 선택 | 아무 문자열 (ML은 로그에만 남김) | — |
| `language` | 선택 | `"ko"`만 (기본값) | `INVALID_FORMAT` |
| `mode` | 선택 | `"utterance"`(기본) 또는 `"conversation"` | `INVALID_FORMAT` |

### 오디오와 종료
| 항목 | 값 |
|---|---|
| 형식 | PCM16 little-endian, 16kHz, 모노, 헤더 없는 raw |
| 프레임 | 100ms = 3,200 bytes (1,600샘플), WebSocket **바이너리** 프레임 |
| 보내는 속도 | 실시간 (100ms마다 1프레임) |
| 최대 길이 | `utterance` 10초 / `conversation` 구간 15초, 세션 600초 |
| 종료 메시지 | `{"type": "end", "trace_id": "<start와 같은 값>"}` (`trace_id` 필수) |

`INVALID_FORMAT`이 나도 연결은 끊기지 않는다. 고쳐서 다시 `start`를 보내면 된다.

---

## 2. 녹음 중 중간 전사

**온다. 이름은 `partial`이 아니라 `transcript_interim`이다.**

```json
{"type": "transcript_interim", "trace_id": "a1b2…", "text": "믈 조세",
 "stable_text": "믈", "unstable_text": "조세",
 "confidence": 0.62, "is_final": false, "segment_id": 0, "audio_end_ms": 1312}
```

| 필드 | 설명 |
|---|---|
| `text` | 지금까지 전체 전사 (= `stable_text` + 공백 + `unstable_text`) |
| `stable_text` | 확정된 앞부분. 이후 interim에서 바뀌지 않음 |
| `unstable_text` | 아직 바뀔 수 있는 뒷부분 |
| `confidence` | 0–1, 참고용 |
| `is_final` | 항상 `false` |
| `segment_id` | `utterance` 모드는 항상 0. `conversation`은 구간마다 0, 1, 2… |
| `audio_end_ms` | 이 전사가 반영한 음성 위치 (세션 시작 기준 ms). 지연 측정용 |

- 말소리가 0.3초 쌓인 뒤부터 약 0.5초마다 온다. 간격은 최소 300ms, 말이 없을 때는 안 온다
- **interim에는 `is_last`가 없다.** `is_last`는 final에만 있다

### 최종 전사
```json
{"type": "transcript_final", "trace_id": "a1b2…", "text": "물 주세요",
 "confidence": 0.98, "duration_ms": 1420,
 "model_version": "yoona-J/ASR_Whisper_Disease_General", "no_speech_prob": 0.0,
 "segment_id": 0, "is_last": true}
```

### `is_last` 정리
| 모드 | final 개수 | `is_last` |
|---|---|---|
| **`utterance`** (기본, 버튼 누르고 한 문장) | **딱 1번** | 항상 `true`. 중간 final 없음 |
| `conversation` (버튼 없이 계속 말하기) | 1.75초 쉴 때마다 1번 + `end` 뒤 1번 | 중간은 `false`, 마지막만 `true` |

- ⭐ 정확도 모드는 **`utterance`**, ⚡ 실시간 모드는 **`conversation`** 을 쓴다. 실시간 모드는 `is_last: false` 구간 final이 올 때마다 바로 TTS로 보낸다(bypass)
- **"`is_last: true` final을 받으면 세션 끝"** 으로 처리하면 두 모드 모두 맞다
- final `text`는 빈 문자열일 수 있다 (무음). 이때는 후보 계산 대신 재질의(`ask_again`)로 보낸다
- 10초를 넘기면 `{"type":"error","code":"AUDIO_TOO_LONG",…}` 다음에 final(`is_last: true`)이 자동으로 온다

---

## 3. `text`는 delta인가 cumulative인가

**(b) cumulative다.** 매 interim의 `text`는 지금까지 들은 전체 전사다. diff 없이 **가장 최근 interim의 `text`로 통째로 교체**하면 된다.

| 순서 | 메시지 | `text` |
|---|---|---|
| 1 | interim | `"믈"` |
| 2 | interim | `"물 조세"` ← 앞 단어도 바뀔 수 있음 |
| 3 | final | `"물 주세요"` |

- Whisper가 다시 들으면서 앞 단어를 고칠 수 있다. **이어 붙이기(append)로 처리하면 안 된다**
- `stable_text`는 줄어들거나 바뀌지 않는다. 앱에서 진하게(확정) / 흐리게(미확정)로 나눠 보여 주고 싶으면 이 두 필드를 쓴다
- final도 그 발화 전체 문장이다. 화면을 final `text`로 바꾼다

---

## 4. 연결 지연과 keep-alive

**예상 지연:** 같은 VPC 안이라 WebSocket 새 연결(TCP + 업그레이드)은 **수 ms 수준으로 예상**한다. 서버도 연결과 `start` 처리에 무거운 작업이 없다 (모델은 미리 올라가 있음). **실측은 아직 안 했다.** 백엔드 서버에서 한 번 재 보면 정확하다.

두 방법 다 지원한다. 편한 쪽으로 하면 된다.

| 방법 | 방식 | 비고 |
|---|---|---|
| **1. 발화마다 새로 열기 (권장)** | 앱이 백엔드 WebSocket을 여는 순간(녹음 버튼 누름) ML 연결도 같이 연다 | 첫 오디오 프레임 전에 연결이 끝나 체감 지연이 거의 없다. 가장 단순 |
| 2. 미리 열어 두고 재사용 | 한 연결에서 세션을 **차례대로** 쓴다 (앞 세션의 `is_last: true` final을 받은 다음 `start`) | 연결을 오래 열어 둬도 서버가 끊지 않는다. WebSocket ping은 라이브러리가 자동 처리 |

- 앱 레벨 keep-alive가 필요하면: `{"type":"ping","trace_id":"keepalive-…"}` → `{"type":"pong",…}`
- **피할 것:** 한 연결에서 두 세션을 **겹쳐서** 쓰는 것. 오디오 프레임에는 세션 정보가 없어서, 겹치면 가장 최근 `start` 세션으로 오디오가 섞여 들어간다

### ML 서버 재시작 때
- 약 20초 동안 연결이 거부되거나, 연결 후 `MODEL_UNAVAILABLE` 메시지와 함께 종료 코드 1013으로 닫힌다
- 1, 2, 4, 8, 16초 간격으로 다시 연결한다
- 준비 여부는 `GET http://10.1.2.11:8001/healthz` → 200이면 준비 완료, 503이면 로딩 중

---

## 한 번에 보는 세션 흐름 (`utterance`)

```
백엔드                                         ML
  │── 연결 (ws://10.1.2.11:8001/asr-stream) ──▶│
  │── {"type":"start", trace_id, sample_rate:16000, …} ──▶│
  │── <PCM16 3,200 bytes> ─────────────────────▶│  100ms마다
  │── <PCM16 3,200 bytes> ─────────────────────▶│
  │◀── {"type":"transcript_interim", text:"믈", …} ──│  약 0.5초마다 (cumulative)
  │── <PCM16 …> ───────────────────────────────▶│
  │◀── {"type":"transcript_interim", text:"물 조세", …} ──│
  │── {"type":"end", trace_id} ────────────────▶│
  │◀── {"type":"transcript_final", text:"물 주세요", is_last:true, …} ──│  보통 0.1–0.4초 안
  │── 닫기 (또는 다음 start로 재사용) ──────────▶│
```
