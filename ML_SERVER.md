# 마리음 ML 서버 구현 정리

구음장애 환자의 한국어 음성을 실시간으로 텍스트로 바꾸는 GPU 서버다. 백엔드가 WebSocket으로 오디오를 흘려 보내면, 말하는 도중의 중간 전사(interim)와 최종 전사(final)를 돌려준다.

- 기준일: 2026-10-31
- 상태: GPU 서버에 배포되어 동작 중. 백엔드 서버에서 접속 확인 완료. 백엔드 코드 연동은 아직
- 관련 문서: 결정 근거와 실측 [CALIBRATION.md](CALIBRATION.md) §0, 백엔드 연동 [HANDOFF.md](HANDOFF.md), 프로토콜 원본 `design/ml-server-protocol.md` v1.0

---

## 1. 한눈에 보기

| 항목 | 내용 |
|---|---|
| 하는 일 | 음성 → 텍스트 (ASR) **하나만**. 의도 분류, 추천, RAG, LLM, TTS는 하지 않는다 |
| 모델 | `yoona-J/ASR_Whisper_Disease_General` (구음장애 음성으로 파인튜닝한 whisper-small). 가중치는 그대로 쓴다 |
| 실행 | PyTorch + transformers, **fp16**, NVIDIA L4 |
| 말소리 감지 | Silero VAD (세션마다 따로) |
| 통신 | WebSocket `/asr-stream` + HTTP `/healthz`, `/livez` (FastAPI, uvicorn 워커 1개) |
| 모드 | `utterance`(한 문장, v4 기본) / `conversation`(실시간 대화, 쉼마다 문장 단위 final) |
| 주소 | `ws://10.1.2.11:8001/asr-stream` (GPU 서버 `inno-gpu-006`) |
| 접근 | 백엔드 서버(`10.1.1.11`)와 GPU 서버 자신만. 인증 없음 |
| 운영 | systemd 서비스 2개 (서버, 방화벽). 죽으면 자동 재시작, 부팅 시 자동 시작 |

---

## 2. 전체 구조

```
 앱 ─(녹음)─▶ 백엔드 서버 (10.1.1.11)
                  │  WebSocket, 세션마다 연결 1개
                  │  start(JSON) → 오디오 PCM16 100ms 프레임(바이너리) … → end(JSON)
                  ▼
 ┌────────────── GPU 서버 (10.1.2.11:8001) ───────────────────────────┐
 │  iptables: 8001은 localhost, 10.1.1.11만 허용                        │
 │                                                                      │
 │  [이벤트 루프] app.py                                                 │
 │    Conn (연결) ── Session (세션) ── Segmenter (구간 상태)              │
 │       │                 │              └ Agreement (interim 확정분)   │
 │       │                 │                                            │
 │       │   오디오 프레임 ─▶ [VAD 스레드 2개] PCM 디코딩 + Silero VAD     │
 │       │                 │                                            │
 │       │                 ├─ interim 요청 ─┐                           │
 │       │                 └─ final 요청 ───┤                           │
 │       │                                  ▼                           │
 │       │                 [GPU 스레드 1개] scheduler.py → engine.py      │
 │       │                   최대 4건 묶음, final 먼저, fp16 Whisper     │
 │       ◀── transcript_interim / transcript_final / error (JSON) ──    │
 └──────────────────────────────────────────────────────────────────────┘
```

**스레드 3종류**

| 스레드 | 하는 일 | 이유 |
|---|---|---|
| 이벤트 루프 (1) | WebSocket 송수신, 세션·구간 상태 변경, 작업 예약 | 상태를 한 곳에서만 바꿔야 interim 스냅샷과 겹치지 않는다 |
| VAD 풀 (2) | PCM16 → float 변환, 리샘플링, Silero VAD | 32ms 창 하나에 약 1.6ms. 루프를 막지 않게 분리 |
| GPU (1) | Whisper 추론 | GPU 작업은 한 줄로 세워야 서로 느려지지 않는다 |

---

## 3. 파일 구성

| 파일 | 역할 |
|---|---|
| `server/app.py` | FastAPI 앱. 연결(`Conn`), 세션(`Session`), 메시지 처리, 모델·VAD 로딩, 워밍업, 로그, `/healthz`, `/livez` |
| `server/stream.py` | 구간 나누기(`Segmenter`)와 interim 확정 규칙(`Agreement`). **torch 없이** numpy만 쓴다 (단위 테스트 가능) |
| `server/scheduler.py` | GPU 작업자(`GpuScheduler`). 묶음 처리, final 우선, 세션당 interim 대기 1개 |
| `server/engine.py` | Whisper 추론(`AsrEngine`). 묶음 전사, confidence, `no_speech_prob` |
| `server/config.py` | 환경변수 설정과 기본값 (§8) |
| `deploy/marieum-ml.service` | 서버 systemd 유닛 |
| `deploy/marieum-ml-firewall.{sh,service}` | 8001 포트 iptables 규칙과 부팅 시 적용 유닛 |
| `scripts/stream_client.py` | 테스트·부하 클라이언트 (WAV를 실제 속도로 흘려 지연 측정) |
| `scripts/mock_asr_server.py` | GPU 없는 Mock 서버 (포트 9001, 같은 메시지 형식) |
| `scripts/verify_engine.py` | 엔진 결과가 연구 노트북 호출과 같은지, fp16, `no_speech_prob` 확인 |
| `scripts/smoke_test.py`, `slice_sample.py`, `run_chunks.py` | 초기 캘리브레이션 (모델 단독 시험, 샘플 자르기, 조각 전사) |
| `tests/test_stream.py` | 구간 나누기·확정 규칙 단위 테스트 11개 |
| `tests/test_app.py` | 가짜 엔진으로 실제 서버를 띄우는 WebSocket 통합 테스트 12개 |
| `requirements.txt` | GPU 서버에서 검증한 패키지 버전 |

---

## 4. 세션 하나가 처리되는 흐름

1. **연결.** 백엔드가 `/asr-stream`에 연결한다. 모델이 아직 준비되지 않았으면 `MODEL_UNAVAILABLE` 후 `close(1013)`.
2. **`start` 수신.** encoding(`pcm16`만), sample_rate(8–48kHz), mode, language(`ko`만), trace_id 형식을 검사한다. 틀리면 `INVALID_FORMAT`을 보내지만 **연결은 유지**한다. 통과하면 VAD 모델을 하나 빌려 세션을 만들고, 이 연결의 "현재 세션"으로 정한다.
3. **오디오 프레임 수신 (100ms마다).**
   - VAD 스레드: PCM16 → float, 16kHz가 아니면 리샘플링, 512샘플(32ms) 창으로 잘라 창마다 말소리 확률을 구한다. 남는 꼬리는 다음 프레임에 붙인다.
   - 이벤트 루프: 창과 확률을 `Segmenter`에 넣는다. 구간이 닫히면 final을 예약하고, interim 조건이 맞으면 interim을 예약한다.
4. **interim.** 구간 안 말소리가 300ms 이상이고, 마지막 interim 이후 새 말소리가 있고, 오디오가 500ms 더 쌓였으면 **그 구간 전체를 다시 전사**한다 (Whisper는 스트리밍 모델이 아니다). 결과를 `Agreement`에 넣어 `stable_text` / `unstable_text`로 나눠 보낸다. 세션당 동시에 1건만 돌고, 보내는 간격은 최소 300ms다.
5. **final.** 구간이 닫히면(대화 모드의 쉼·최대 길이, 발화 모드의 10초 초과, `end` 수신) 대기 중인 interim을 취소하고 final을 예약한다. 말소리가 없는 구간은 Whisper를 건너뛰고 바로 빈 final을 보낸다. final은 세션 안에서 구간 순서대로 나간다.
6. **정리.** `is_last: true` final을 보낸 뒤 세션을 지우고 VAD 모델을 초기화해서 반납한다. 요약 로그(`session_closed`)를 남긴다. 음성은 메모리에서 버린다.
7. **연결 끊김.** 그 연결의 진행 중 세션을 모두 버린다 (final을 보내지 않는다).

---

## 5. 구성 요소 상세

### 5.1 구간 나누기 (`Segmenter`, `server/stream.py`)

| 규칙 | utterance | conversation |
|---|---|---|
| 구간 시작 | 세션 시작부터 | 첫 말소리 창. 그 앞 200ms(패딩)를 붙인다 |
| 구간 끝 | `end` 또는 10초 초과 | 0.8초 쉼, 구간 6초, `end`, 세션 600초 초과 |
| 짧은 소리 | 말소리 150ms 미만이면 "말소리 없음" | 150ms 미만 구간은 버린다 (숨소리, 클릭음) |
| 최대 길이에서 자르기 | — | 마지막 5초 안의 **가장 늦은 쉼**에서 자르고 나머지는 다음 구간으로. 쉼이 없으면 그 자리에서 자른다 |
| final에 넣는 음성 | 첫 말소리 −200ms ~ 마지막 말소리 +200ms | 같음 |
| interim에 넣는 음성 | 첫 말소리 −200ms ~ 지금 | 구간 시작 ~ 지금 |

VAD 설정은 기존 캘리브레이션 값 그대로다: threshold 0.5, 앞뒤 여유 200ms, 최소 말소리 150ms.

### 5.2 interim 확정 규칙 (`Agreement`, LocalAgreement-2)

- 연속한 두 interim이 **같게 낸 앞 단어까지만** 확정(`stable_text`)한다. 단어 비교 때 문장부호(`. , ? ! ~ · …`)는 무시한다.
- 확정분은 늘기만 하고 줄지 않는다. 화면에서 진한 글자는 바뀌지 않는다.
- 현재 가설의 **마지막 단어는 확정하지 않는다** ("화장" → "화장실에"처럼 이어질 수 있다).
- 나머지는 `unstable_text`로 보낸다 (화면에서 흐리게).
- final은 구간 전체를 다시 전사한 값이라 드물게 확정분과 다를 수 있다. 화면은 final로 바꾼다.

예: `"미지근한"` → `"미지근한 물"` → `"미지근한 물 주"` → final `"미지근한 물 주세요"`
→ stable: `""` → `"미지근한"` → `"미지근한 물"` → (final로 교체)

### 5.3 GPU 작업자 (`GpuScheduler`, `server/scheduler.py`)

- 작업자는 **1개**, 한 번에 **최대 4건을 묶어** 돌린다 (Whisper 입력은 어차피 30초로 채워지므로 묶기 쉽다).
- **final이 먼저** (도착 순서대로), 남는 자리에 interim을 넣는다.
- interim은 **세션마다 대기 1개**만 둔다. 새 interim이 오면 대기 중인 것을 버리고 바꿔 끼운다. GPU가 밀리면 interim 간격이 저절로 늘고, 대기열이 쌓이지 않는다.
- 묶음의 토큰 상한은 묶인 작업 중 큰 값 (interim 96, final 128).
- 추론 오류가 나면 그 묶음의 작업만 실패시키고 작업자는 계속 돈다 (final 실패 → `INTERNAL`).

### 5.4 Whisper 엔진 (`AsrEngine`, `server/engine.py`)

1. 묶음 음성의 로그-멜 특징을 만든다.
2. **인코더를 한 번만** 돌린다.
3. `<|startoftranscript|>` 한 토큰으로 디코더를 한 번 돌려 `<|nocaptions|>` 확률을 구한다 → `no_speech_prob` (openai/whisper와 같은 정의).
4. 같은 인코더 출력으로 `generate()`: 연구 노트북과 같은 `forced_decoder_ids`(ko, transcribe), greedy.
5. 문장 끝 토큰 전까지의 텍스트 토큰 확률 평균 → `confidence`. 로그 확률 평균 → `avg_logprob` (내부용).

이 방식이 노트북 호출과 같은 결과를 내는지 조각 60개로 확인했다 (fp32·fp16 모두 60/60, 묶음 4건도 60/60).

**무음 판정:** `no_speech_prob > 0.6`이면 빈 전사로 바꾼다. openai/whisper 기본 규칙(`avg_logprob < -1` 조건 추가)은 이 모델에 맞지 않는다. 이 모델은 무음에서도 확신 있게 문장을 지어내기 때문이다 (§10).

**워밍업:** 시작할 때 묶음 1–4건 크기로 한 번씩 돌린다 (약 1.5초). 첫 추론이 856ms 걸리기 때문이다. 끝나야 `/healthz`가 200이 된다.

---

## 6. 두 가지 모드

| | `utterance` (기본) | `conversation` |
|---|---|---|
| 용도 | ⭐ 정확도 모드: 버튼 누르고 한 문장 → 후보 추천 | ⚡ 실시간 모드: 말하는 동안 구간 final마다 바로 TTS (bypass) |
| final | `end` 후 1회 (`segment_id: 0`, `is_last: true`) | 0.8초 쉴 때마다 구간 final (`is_last: false`), `end` 후 마지막 final (`is_last: true`) |
| 길이 상한 | 10초 → `AUDIO_TOO_LONG` + 자동 final | 구간 6초 (넘으면 쉼에서 끊음), 세션 600초 |
| interim 범위 | 세션 전체 | 현재 구간만 |

---

## 7. 메시지 요약

전체 필드와 에러 처리는 [HANDOFF.md](HANDOFF.md)에 있다.

| 방향 | 메시지 | 핵심 필드 |
|---|---|---|
| 백엔드 → ML | `start` | `trace_id`, `patient_id`(로그에만), `sample_rate`, `encoding: "pcm16"`, `mode` |
| 백엔드 → ML | (바이너리) | PCM16 LE 모노, 100ms = 3,200바이트 |
| 백엔드 → ML | `end`, `ping` | `trace_id` |
| ML → 백엔드 | `transcript_interim` | `text`, `confidence`, `is_final: false`, `segment_id`, `stable_text`, `unstable_text`, `audio_end_ms` |
| ML → 백엔드 | `transcript_final` | `text`, `confidence`, `duration_ms`, `model_version`, `no_speech_prob`, `segment_id`, `is_last` |
| ML → 백엔드 | `error` | `code`: `MODEL_UNAVAILABLE` / `INVALID_FORMAT` / `AUDIO_TOO_LONG` / `NO_SESSION` / `INTERNAL` |
| ML → 백엔드 | `pong` | `trace_id` |

프로토콜 v1.0의 필드는 모두 그대로 보낸다. 추가 필드를 무시하면 v1.0 클라이언트도 동작한다.

---

## 8. 설정 (환경변수)

| 변수 | 기본값 | 의미 |
|---|---|---|
| `ML_MODEL_DIR` | `models/ASR_Whisper_Disease_General` | 모델 경로 |
| `ML_MODEL_VERSION` | `yoona-J/ASR_Whisper_Disease_General` | 응답·로그의 `model_version` |
| `ML_DTYPE` | `fp16` | `fp16` / `fp32` |
| `ML_BATCH_MAX` | 4 | GPU 한 번에 묶는 최대 건수 |
| `ML_INTERIM_MAX_NEW_TOKENS` / `ML_FINAL_MAX_NEW_TOKENS` | 96 / 128 | 생성 토큰 상한 |
| `ML_INTERIM_STEP_MS` | 500 | interim 다시 전사 간격 (오디오 기준) |
| `ML_MIN_INTERIM_SPEECH_MS` | 300 | 첫 interim까지 필요한 말소리 |
| `ML_VAD_THRESHOLD` | 0.5 | 말소리 판정 |
| `ML_SPEECH_PAD_MS` | 200 | 말소리 앞뒤 여유 |
| `ML_MIN_SPEECH_MS` | 150 | 이보다 짧은 소리는 잡음 |
| `ML_END_SILENCE_MS` | 800 | 대화 모드에서 구간을 닫는 쉼 |
| `ML_UTTERANCE_MAX_S` | 10 | 발화 모드 최대 길이 |
| `ML_MAX_SEGMENT_S` | 6 | 대화 모드 구간 최대 길이 |
| `ML_SESSION_MAX_S` | 600 | 대화 모드 세션 최대 길이 |
| `ML_NO_SPEECH_THRESHOLD` / `ML_LOGPROB_THRESHOLD` | 0.6 / 0.0 | 무음 판정 (0.0 = logprob 조건 끔) |
| `ML_DEBUG_SAVE` | 0 | 1이면 세션 음성을 `/tmp/asr-debug/{trace_id}.wav`에 저장, 30분 뒤 삭제. **심사장에서는 반드시 0** |

코드 안 고정값: interim 최소 간격 300ms, 응답 텍스트 최대 1,000자, trace_id 1–64자(영문·숫자·`-`·`_`).

---

## 9. 성능 (2026-10-31, L4, GPU 서버 안에서 측정)

지연 = 클라이언트가 그 음성의 마지막 100ms 프레임을 보낸 시각 → 메시지 수신. **서버 간 네트워크와 앱 녹음은 빠져 있다.**

| 시험 (동시 4세션) | interim p50 / p95 | 마지막 final (end 후) p95 |
|---|---|---|
| utterance, 조각 20개 × 5회 | 123–157 / 283–380ms | 210–269ms |
| conversation, 실제 발화 90초 | 117 / 132ms | — |
| conversation, 쉼 없이 이어 붙인 말 60초 (최악) | 312–335 / 543–606ms | 380–407ms |
| 프로토콜 목표 | ≤ 300 / ≤ 500ms | ≤ 800ms |

| 엔진 (조각 60개) | fp32 | fp16 |
|---|---|---|
| GPU 시간 1건 | 233ms | 133ms |
| GPU 시간 4건 묶음 | 702ms | 248ms |
| GPU 메모리 최대 | 2,055MiB | 1,988MiB |

---

## 10. 알려진 한계

1. **무음·잡음에서 없는 말을 지어낸다.**
   - 무음 1초 → "아침 식사 시간은 몇 시까지예요" (confidence 0.73)
   - 잡음, 말소리 앞 0.5초 → "땅콩", "떡"
   - 무음은 VAD와 `no_speech_prob`로 막는다. 잡음과 아주 짧은 소리는 `no_speech_prob`가 0에 가까워 못 막는다. 확정 규칙 때문에 `stable_text`까지 가기는 어렵지만 `unstable_text`에는 잠깐 보일 수 있다.
2. **confidence가 높게 몰린다.** 정상 발화 중앙값 0.9996, 지어낸 말도 0.4–0.7. 백엔드의 재질의 기준(0.3/0.4)이 거의 걸리지 않는다.
3. **연결 하나에 세션 여러 개를 겹치면 오디오가 섞인다.** 프로토콜 v1.0 규칙(가장 최근 `start`에 귀속) 때문이다. 백엔드는 세션마다 연결을 따로 연다.
4. **쉼 없이 말하는 4명 동시**에서는 interim이 목표를 약간 넘는다. 구음장애 발화는 문장 사이 쉼이 길어서(샘플 기준 약 10초) 실제로는 드물다.
5. **Opus는 받지 않는다.** PCM16만.
6. **아직 재지 않은 것:** 백엔드 서버에서 잰 지연(네트워크 포함), 앱 마이크 음성, 정답 라벨 기준 오류율(CER).

---

## 11. 보류한 것

| 항목 | 상태 | 메모 |
|---|---|---|
| 화자 분리 / 화자 판별 | **보류 (2026-10-31, 필요해지면 진행)** | 대화 모드의 구간 final마다 목소리 특징값을 뽑아 붙이는 방식이 맞다. 후보 모델: 분리 pyannote community-1, 판별 ERes2NetV2 / CAM++ / ECAPA-TDNN. v4 기획은 "환자 1인 앱"으로 제외. 판별은 환자 목소리 특징값 저장이 필요해 음성 비보존 규칙과 충돌. 조사 내용: CALIBRATION.md §12, §13 (F-0, F-1) |
| 모델 교체 (large-v3-turbo) | 하지 않음 | v4 문서에 적혀 있으나 구음장애 특화 모델 유지. 팀 확인 필요 |
| Opus 입력 | 하지 않음 | 필요하면 디코더 추가 |
| 인증 / 외부 공개 (터널) | 하지 않음 | 심사장에서 백엔드가 VPC 밖에 있으면 토큰 인증부터 넣고 터널 |
| rpcbind(111) 끄기 | 하지 않음 | GPU 서버에서 쓰지 않는데 VPN에서 열려 있음. 팀 판단으로 유지 |

---

## 12. 운영

### 서버 위치
- GPU 서버 `inno-gpu-006` (NCP, `10.1.2.11`, SSL VPN 경유 SSH), 사용자 `ncloud`, 홈 `/home1/ncloud`
- 코드 `~/marieum-ml/{server,scripts,tests,deploy}`, 모델 `~/marieum-ml/models/ASR_Whisper_Disease_General`, venv `~/marieum-ml/.venv` (Python 3.11, uv)

### 자주 쓰는 명령
```bash
systemctl status marieum-ml                    # 상태
sudo journalctl -u marieum-ml -f               # 로그 (JSON 한 줄씩)
sudo systemctl restart marieum-ml              # 재시작 (약 20초 안에 /healthz 200)
curl -s localhost:8001/healthz                 # 준비 여부
sudo iptables -S INPUT | grep marieum-ml       # 방화벽 규칙
```

### 코드 갱신
1. `server/`(필요하면 `deploy/`)를 `~/marieum-ml/`에 덮어쓴다
2. 테스트: `cd ~/marieum-ml && source .venv/bin/activate && python -m pytest tests -q`
3. `sudo systemctl restart marieum-ml` → `/healthz` 200 확인

### 백엔드 서버 IP가 바뀌면
`/usr/local/sbin/marieum-ml-firewall.sh`의 `BACKEND_IP`를 고치고 (레포 `deploy/`도 같이) `sudo systemctl restart marieum-ml-firewall`

### 로그
- 이벤트: `start`, `transcript_interim`, `transcript_final`, `session_closed`, `error`, `model_loaded`, `warmup_done`
- 공통 필드: `ts`, `trace_id`, `endpoint`, `event`, `model_version` + `latency_ms`, `queue_ms`, `gpu_ms`, `batch`, `segment_id`, `interim_index`, `audio_bytes_total`, `interim_count`, `vad_segments`
- **전사 텍스트는 로그에 남기지 않는다.** `patient_id`는 남긴다

---

## 13. 테스트와 측정

```bash
python -m pytest tests -q                       # 23개, GPU 없이 돈다

# 엔진이 노트북과 같은 결과를 내는지 (GPU 서버)
python scripts/verify_engine.py --model models/ASR_Whisper_Disease_General --chunks data/chunks --dtype fp16

# 스트리밍 (한 세션 메시지 보기 / 동시 4세션 부하)
python scripts/stream_client.py --wav a.wav --show
python scripts/stream_client.py --wav data/chunks/chunk_*.wav --sessions 4 --rounds 5
python scripts/stream_client.py --mode conversation --wav long.wav --sessions 4 --offset-step 7 --max-seconds 60

# Mock (GPU 없이 같은 메시지)
python scripts/mock_asr_server.py               # ws://0.0.0.0:9001/asr-stream
```

테스트 음성(AI Hub 구음장애 데이터)은 재배포 금지라 레포에 없다. GPU 서버 `~/marieum-ml/data/`에 있다.
