# ML 서버 캘리브레이션 기록

구현 전에 사용자와 Claude가 맞춘 합의 내용이다. 구현은 이 문서를 기준으로 하고, 바뀌면 이 문서를 먼저 고친다.

- 작성일: 2026-10-30, 개정 2026-10-31 (§0)
- 상태: **v4 스트리밍 서버 구현 (§0)**. §1–§9의 HTTP `POST /asr` 내용 중 §0과 겹치는 것은 §0이 우선한다
- 반영한 리뷰: Codex 리뷰 + Claude 리뷰 통합본 (항목 번호 #1–#35는 §16 참고)

---

## 0. v4 전환: WebSocket 스트리밍 + 실시간 대화 모드 (2026-10-31)

기획 v4의 `design/ml-server-protocol.md` v1.0(HTTP → WebSocket)과 팀 요청("문장이 다 안 끝나도 인식한 단어가 화면에 쭉 나오는 실시간 대화")을 반영했다. **모델 가중치는 바꾸지 않았다.** 바뀐 것은 실행 방식(fp16, 배치)과 서빙 코드다.

### 0.1 바뀐 것
| 항목 | 이전 (§1–§9) | 지금 | 근거 |
|---|---|---|---|
| 통신 | HTTP `POST /asr`, WAV 업로드 → JSON 1회 | WebSocket `/asr-stream` + `GET /healthz`, `GET /livez` | 프로토콜 v1.0 |
| 세션 모드 | — | `utterance`(기본, v4 그대로: final 1회) / `conversation`(실시간 대화: 쉼마다 구간 final, 세션 유지) | 실시간 대화 요청 |
| 모델 | 구음장애 whisper-small | **그대로**. v4 문서의 faster-whisper large-v3-turbo는 채택하지 않음 (팀 확인 필요) | 구음장애 특화 유지 |
| 정밀도 | fp32 | **fp16** (`ML_DTYPE=fp32`로 되돌릴 수 있음) | 조각 60개 전사가 fp32 노트북과 60/60 같음. 4건 묶음 GPU 702ms → 248ms |
| GPU 처리 | 한 번에 1건 | 작업자 1개, **최대 4건 묶음(batch)**. final이 interim보다 먼저. interim은 세션당 대기 1개만 (새 것이 오면 바꿔 끼움) | GPU가 밀리면 interim 간격이 저절로 늘고 대기열이 쌓이지 않는다 |
| 워밍업 | 안 함 (#34) | 시작 시 batch 1–4로 한 번씩 (약 1.5초), 끝나야 `/healthz` 200 | 첫 추론 856ms. interim 목표 300ms |
| `no_speech_prob` | 뺌 (#16) | final에 넣음 (프로토콜 필수). `<|startoftranscript|>` 다음 자리의 `<|nocaptions|>` 확률 (openai/whisper 정의). VAD가 말소리를 못 찾아 Whisper를 건너뛰면 1.0 | 프로토콜 v1.0 |
| 무음 판정 | VAD만 | VAD + `no_speech_prob > 0.6`이면 빈 전사 | §0.5 환각 실측 |
| 길이 | 업로드 20초 | `utterance` 10초 (넘으면 `AUDIO_TOO_LONG` + 자동 final). `conversation` 구간 최대 6초, 세션 600초 | 프로토콜 v1.0, §0.5 |
| 입력 | WAV 파일 | raw PCM16 little-endian 모노, 8–48kHz (16kHz 권장), 100ms 프레임. **Opus는 받지 않음** (`INVALID_FORMAT`) | 프로토콜 열린 질문 1 → PCM16 제안 |
| Mock 서버 | 백엔드 담당 | ML이 제공: `scripts/mock_asr_server.py` (포트 9001, 아래 추가 필드 포함) | 프로토콜 §8 |
| 에러 | HTTP 상태 + `error_code` | `{"type":"error","code",...}`. 세션 하나의 형식 오류는 연결을 닫지 않음 (§0.6) | 프로토콜 §9 |

### 0.2 메시지 (프로토콜 v1.0 + 추가 필드)
v1.0 필드는 모두 그대로 보낸다. 추가 필드만 무시하면 v1.0 클라이언트도 그대로 동작한다.

| 메시지 | 추가 필드 | 의미 |
|---|---|---|
| `start` | `mode` (선택) | `"utterance"`(기본) 또는 `"conversation"` |
| `transcript_interim` | `segment_id` | 구간 번호 (utterance는 항상 0) |
| | `stable_text` | 확정된 앞부분. 화면에서 바뀌지 않는다 (진하게 표시) |
| | `unstable_text` | 아직 바뀔 수 있는 뒷부분 (흐리게 표시) |
| | `audio_end_ms` | 세션 시작 기준, 이 전사가 반영한 음성의 끝 위치. 지연 측정용 |
| `transcript_final` | `segment_id` | 어느 구간의 final인지. 같은 `segment_id`의 interim을 이 text로 바꾼다 |
| | `is_last` | 세션의 마지막 final인지. utterance는 항상 `true` |

- `text`는 `stable_text + " " + unstable_text`. v1.0의 "뒤 interim이 앞 interim을 교체" 규칙 그대로 쓸 수 있다
- `conversation`: 쉼(800ms)이나 구간 최대 길이에서 구간 final(`is_last: false`)이 온다. `end`를 받으면 남은 구간을 닫고 `is_last: true` final을 보낸다 (남은 말소리가 없으면 `text: ""`)
- 응답 순서 보장: 한 세션 안에서 final은 `segment_id` 순서대로 온다. 구간 final 뒤에 그 구간의 interim은 오지 않는다

### 0.3 interim 규칙
- 구간 안 말소리가 300ms 이상 쌓인 뒤부터, 새 말소리가 있을 때만 500ms마다 그 구간 전체를 다시 전사한다 (Whisper는 스트리밍 모델이 아니다)
- 보내는 간격은 최소 300ms (프로토콜 §7.1). GPU가 밀렸다 풀려도 몰려 나가지 않는다 (부하 시험 최소 간격 408ms)
- **확정 규칙 (LocalAgreement-2):** 연속한 두 interim이 같게 낸 앞 단어만 확정한다. 확정분은 늘기만 한다. 현재 가설의 마지막 단어는 확정하지 않는다 ("화장" → "화장실에")
- final은 구간 전체를 다시 전사한 결과라 드물게 확정분과 다를 수 있다. 화면은 final로 바꾼다
- 구간 나누기(`conversation`): Silero VAD, 세션마다 따로. 말소리가 150ms 미만이면 잡음으로 버린다. 6초에 닿으면 마지막 5초 안의 가장 늦은 쉼에서 자르고 나머지는 다음 구간으로 넘긴다

### 0.4 설정 (환경변수, `server/config.py`)
| 변수 | 기본값 | 의미 |
|---|---|---|
| `ML_MODEL_DIR` | `models/ASR_Whisper_Disease_General` | 모델 경로 |
| `ML_DTYPE` | `fp16` | `fp16` / `fp32` |
| `ML_BATCH_MAX` | 4 | GPU 한 번에 묶는 최대 건수 |
| `ML_INTERIM_STEP_MS` / `ML_MIN_INTERIM_SPEECH_MS` | 500 / 300 | interim 간격 / 첫 interim까지 필요한 말소리 |
| `ML_VAD_THRESHOLD`, `ML_SPEECH_PAD_MS`, `ML_MIN_SPEECH_MS` | 0.5, 200, 150 | §6 값 그대로 |
| `ML_END_SILENCE_MS` | 800 | 대화 모드에서 구간을 닫는 쉼 |
| `ML_UTTERANCE_MAX_S` / `ML_MAX_SEGMENT_S` / `ML_SESSION_MAX_S` | 10 / 6 / 600 | 길이 상한 |
| `ML_INTERIM_MAX_NEW_TOKENS` / `ML_FINAL_MAX_NEW_TOKENS` | 96 / 128 | 생성 토큰 상한 |
| `ML_NO_SPEECH_THRESHOLD` / `ML_LOGPROB_THRESHOLD` | 0.6 / 0.0 | 무음 판정 (0.0 = logprob 조건 끔) |
| `ML_DEBUG_SAVE` | 0 | 1이면 `/tmp/asr-debug/{trace_id}.wav`, 30분 뒤 삭제. 심사장에서는 반드시 0 |

실행: `uvicorn server.app:app --host 0.0.0.0 --port 8001 --workers 1` (워커 1개 고정, §8). 배포 전 보안그룹 조건(§3)은 그대로다.

### 0.5 실측 (2026-10-31, L4, GPU 서버 localhost)
지연은 "클라이언트가 그 음성의 마지막 100ms 프레임을 보낸 시각 → 메시지 수신"이다. 앱 녹음과 네트워크는 빠져 있다. 오디오는 실제 속도로 보냈다.

**엔진 (`scripts/verify_engine.py`, 조각 60개)**
| 항목 | fp32 | fp16 |
|---|---|---|
| 노트북 호출과 전사 일치 | 60/60 | 60/60 (fp32 노트북 기준) |
| batch 4 vs batch 1 일치 | 60/60 | 60/60 (confidence 차이 최대 0.0009) |
| GPU 시간 batch 1 (1건) | 233ms | 133ms |
| GPU 시간 batch 4 (4건 묶음) | 702ms | 248ms |
| GPU 메모리 최대 | 2,055MiB | 1,988MiB |

인코더를 한 번만 돌려 `no_speech_prob`과 `generate(encoder_outputs=...)`가 같이 쓰는 방식이 노트북과 같은 결과를 내는 것도 위 60/60으로 확인했다.

**환각 (이 모델의 특성, 중요)**
| 입력 | 결과 | `no_speech_prob` | confidence |
|---|---|---|---|
| 무음 1초, 3초 | "아침 식사 시간은 몇 시까지예요" | 0.69 | 0.73 |
| 백색잡음 1초 (-40 / -26dBFS) | "땅콩" / "떡" | 0.0002 / 0.0001 | 0.61 / 0.43 |
| 말소리 앞 0.5초 | "땅콩" | 0.002 | 0.61 |
| 말소리 조각 60개 | 정상 | 최대 0.0000 | — |

- 무음에서 확신 있게 문장을 지어낸다. openai/whisper 규칙(`no_speech > 0.6` **그리고** `avg_logprob < -1`)으로는 못 거른다 → `no_speech > 0.6`만으로 거른다
- 잡음과 아주 짧은 말소리는 `no_speech_prob`로 못 거른다. VAD(말소리 없으면 Whisper 생략), 첫 interim 전 말소리 300ms, 확정 규칙(두 번 연속 같아야 확정)으로 줄인다. **unstable 부분에는 지어낸 말이 잠깐 보일 수 있다**
- confidence는 지어낸 말에도 0.4–0.7이 나온다. 백엔드의 0.3(interim)·0.4(final) 기준으로는 거의 안 걸린다

**스트리밍 (`scripts/stream_client.py`)**
| 시험 | interim 지연 p50 / p95 / 최대 | 마지막 final 지연 (end 후) p50 / p95 | 에러 |
|---|---|---|---|
| utterance, 조각 20개, 동시 4 × 5회 (2회) | 123–157 / 283–380 / 311–408ms (55건씩) | 119–148 / 210–269ms | 0 |
| conversation, `sample.wav` 실제 발화 90초 × 동시 4 | 117 / 132 / 187ms (80건) | — | 0 |
| conversation, 쉼 없이 이어 붙인 말 60초 × 동시 4, 구간 최대 10초 | 561 / 851 / 986ms (384건) | 645 / 789ms | 0 |
| 같은 조건, 구간 최대 6초 (3회) | 312–335 / 543–606 / 706–1010ms | 272–336 / 380–407ms | 0 |

- 프로토콜 SLO(interim p50 ≤ 300, p95 ≤ 500, final p95 ≤ 800ms): **utterance와 실제 발화 대화는 통과**. 쉼 없이 말하는 4명 동시는 interim이 약간 넘는다. 구음장애 발화는 문장 사이 쉼이 길어서(이 샘플은 약 10초) 실제로는 드물다
- 쉼 없이 말할 때 서버 쪽 내역 (구간 10초): GPU 대기 p50 152 / p95 387ms, GPU 1회 p50 239 / p95 404ms, 평균 2.1건 묶음
- 대화 모드 `sample.wav` 45초: 문장마다 구간 final 1개 ("문 열어", "문 열어", "나무가 많아"), 마지막은 빈 final

### 0.6 프로토콜과 다르게 한 것, 백엔드에 알릴 점
1. **세션마다 WebSocket 연결을 따로 열기를 권장한다.** 프로토콜 §5.1.2는 바이너리 프레임을 "가장 최근 `start`의 세션"에 넣는다. 연결 하나에 두 세션이 겹치면 앞 세션의 오디오가 뒤 세션으로 들어간다. 서버는 둘 다 지원하고 규칙도 프로토콜 그대로지만, 겹치지 않게 하는 것은 백엔드 몫이다
2. **세션 형식 오류(`INVALID_FORMAT`)는 `close(4001)` 대신 error 메시지만 보낸다.** 연결을 같이 쓰는 다른 환자 세션이 끊기지 않게 하려는 것이다. 모델이 준비되지 않았을 때 연결하면 프로토콜대로 error 후 `close(1013)`
3. `NO_SESSION`: `start` 없이 온 오디오는 `start` 사이에 한 번만 알린다. `AUDIO_TOO_LONG`으로 자동 종료된 뒤 이어 오는 프레임과 늦은 `end`는 조용히 버린다
4. `trace_id`: 영문·숫자·`-`·`_` 1–64자 (UUIDv4 OK). 아니면 `INVALID_FORMAT`
5. Opus는 받지 않는다. PCM16 16kHz로 보내 달라
6. `/livez`(프로세스 생존)를 추가했다. 준비 여부는 `/healthz` 200으로 판단
7. `/healthz` 200 본문에 `sessions`, `gpu_pending`을 더 넣었다
8. 로그: JSON 한 줄씩 stdout. 이벤트 `start`, `transcript_interim`, `transcript_final`, `session_closed`, `error`. 공통 필드(§13.1) + `queue_ms`, `gpu_ms`, `batch`, `segment_id`, `interim_index`, `audio_bytes_total`, `interim_count`, `vad_segments`

### 0.7 남은 일
- ~~`0.0.0.0` 바인딩 + 접근 제한 + systemd 등록~~ **완료 (2026-10-31).** `deploy/`
  - `marieum-ml.service`: `0.0.0.0:8001`, 죽으면 재시작, 부팅 시 시작
  - `marieum-ml-firewall.service`: GPU 서버 iptables로 8001은 localhost와 백엔드 서버(`inno-server-006`, `10.1.1.11`)만 허용. ACG는 8001을 막지 않는다 (SSL VPN에서 온 요청이 GPU 서버까지 와서 iptables에 막힘)
  - 확인: 백엔드 서버 `curl http://10.1.2.11:8001/healthz` → 200, SSL VPN(Mac) → 연결 안 됨
- 백엔드 서버에서 지연 다시 재기
- 앱 마이크 음성(실제 경로)으로 시험. 지금은 AI Hub 샘플만
- 정답 라벨로 CER (라벨 JSON 필요)
- 앱: `stable_text`/`unstable_text` 표시, `segment_id`별로 final 교체
- 팀 확인: 모델(구음장애 small 유지 vs large-v3-turbo), 세션당 연결, Opus 미지원, 대화 모드 길이 상한

---

## 1. 목표

Hugging Face에 공개된 구음장애 특화 Whisper 모델을 L4 GPU 서버에 배포하고, 한국어 음성을 텍스트로 바꾸는 FastAPI `POST /asr`를 제공한다.

### 완료 기준
1. GPU 서버 안에서 파일을 첨부해 `curl -F "audio=@<조각>.wav" http://localhost:8001/asr` → 200 JSON
2. CPU(백엔드) 서버에서 같은 호출 → 200 JSON
3. 속도 측정 결과가 통과 기준(§9)을 만족
4. 백엔드 담당자에게 접속 정보와 호출 방법 전달 (`HANDOFF.md`)

---

## 2. 범위

### 포함
- GPU 서버 환경 확인 (§3)
- HF 가중치와 설정 파일을 GPU 서버에 직접 다운로드
- 모델 로딩과 한국어 WAV 추론 단독 검증
- FastAPI `POST /asr`, `GET /healthz`, `GET /livez`
- VAD 전처리: 앞뒤 무음 자르기, 무음이면 Whisper 생략 (§6)
- 입력 검증, 공통 에러 형식, 로그, 음성 비보존 (§7, §11)
- GPU 실행자, 마감 시각, 대기열 상한 (§8)
- 속도 측정 스크립트와 결과표 (§9)
- `HANDOFF.md`

### 제외
| 항목 | 상태 |
|---|---|
| `/intent-score`, `/refine-text` | 범위 밖 |
| Mock 서버 | 백엔드 담당 |
| WAV 외 포맷 (MP3/M4A/WebM) | Flutter 포맷 확정 후 추가 |
| WebSocket / SSE | 쓰지 않음 (§5) |
| 재학습 (전처리, LoRA) | 필요 없음 (§4) |
| WhisperX 전환 | 하지 않음, 연구 노트북 방식 유지 (§4) |
| `no_speech_prob` | 응답에 넣지 않음 (§7) |
| `flags`, `speech_ms` 필드 | 넣지 않음 |
| 시작 시 워밍업 추론 | 하지 않음 |
| 멈춤 감지 타이머 | 하지 않음, systemd 재시작만 (§8) |
| Flutter 녹음 형식·20초 자동 정지 요구 | Flutter에 강제하지 않음 |
| 화자 분리, 환자 식별 | 후속 이슈 F-0, F-1 (§13) |
| 심사장 준비 (ngrok) | 나중으로 미룸 (§14) |
| 형식 검증 (TLA+ 등) | 나중으로 미룸 (§14) |

---

## 3. 장비와 네트워크

| 항목 | 값 |
|---|---|
| 인스턴스 | `gp1l4-g3`, NVIDIA L4 24GB (Ada Lovelace) |
| 역할 분담 | Claude는 코드와 스크립트를 만든다. GPU 서버 실행은 사용자가 하고 출력을 공유한다 |
| 바인딩 | `0.0.0.0:8001` |
| **접근 제한 (배포 필수)** | 보안그룹에서 인바운드 TCP 8001을 **CPU 서버 IP만** 허용한다. 인증 없는 API라서 이 조건 없이 배포하지 않는다. 보안그룹 설정은 팀 내에서 한다 |
| 서버 주소 | TBD (저녁 미팅). CPU 서버와 같은 VPC면 사설 IP(`hostname -I`), 아니면 공인 IP(`curl ifconfig.me`). 구현에는 영향 없음, `HANDOFF.md`에만 기입 |

### 환경 확인 체크리스트 (사용자 실행)
- [x] GPU 서버 SSH 접속, `nvidia-smi`에서 L4 확인
- [x] `df -h .`로 디스크 가용 공간 기록 (모델 약 0.97GB + PyTorch 수 GB)
- [x] Python 3.10 또는 3.11, venv 설정
- [x] `nvidia-smi`의 CUDA 버전에 맞는 PyTorch 설치, `torch.cuda.is_available() == True` 확인

### 환경 확인 결과 (2026-10-30)
| 항목 | 값 |
|---|---|
| 호스트 | `inno-gpu-006` (NCP, 사설 IP `10.1.2.11`, SSL VPN 경유 접속) |
| OS | Ubuntu 24.04.1 LTS |
| GPU / 드라이버 | NVIDIA L4 23,034MiB / 580.159.04 (CUDA 13.0까지 지원) |
| CPU / 메모리 / 디스크 | 4코어 / 47GiB (스왑 없음) / 78GB 여유 |
| Python | 시스템은 3.12.3뿐 → uv 0.13.6으로 CPython 3.11.18 설치, venv `~/marieum-ml/.venv` |
| PyTorch | torch 2.11.0+cu128, torchaudio 2.11.0+cu128, `cuda.is_available() == True` |

---

## 4. 모델과 추론

| 항목 | 값 |
|---|---|
| 저장소 | `yoona-J/ASR_Whisper_Disease_General` |
| 기반 | `openai/whisper-small`, 6,780 step 학습, 최종 CER 24.2% / WER 29.9% (모델 카드) |
| 파일 | `model.safetensors`(967MB), `preprocessor_config.json`, 토크나이저, `generation_config.json` |
| 형태 | whisper-small 전체 fp32 크기, LoRA 어댑터 없음 → 학습 결과가 합쳐진 완성 모델. **재학습 필요 없음** |
| 로딩 | 서버 시작 시 1회, **백그라운드에서** 로딩 (§8). GPU 서버 로컬 경로에서 (재시작 시 재다운로드 없음) |
| 정밀도 | fp32 (연구 노트북과 동일) |
| 엔진 | transformers, 연구 노트북 `Predict/[Distribution]GeneralModel_Predict.ipynb` 방식 |

### 추론 코드 (노트북 + 옵션)
```python
forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
out = model.generate(
    input_features,                         # GPU로 옮긴 것만 노트북과 다름
    forced_decoder_ids=forced_decoder_ids,  # 노트북 그대로
    return_dict_in_generate=True,           # confidence 계산용
    output_scores=True,                     # confidence 계산용
    max_new_tokens=MAX_NEW_TOKENS,          # #3, 반복 생성 방지
    max_time=남은_시간_초,                   # #2, 마감 시 생성 중단
)
text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0]
```

### `confidence` (#14)
- 특수 토큰(언어, 작업, 타임스탬프 off, 문장 끝)을 뺀 **텍스트 토큰 확률의 평균** (0–1)
- 고정한 transformers 버전에서 `scores`와 생성 토큰이 순서대로 맞는지(강제 토큰이 `scores`에 들어가는지) 확인한 뒤 계산한다
- 빈 전사면 0.0
- 모델의 자기 확신도이지 정답 확률이 아니다. 백엔드 기준값은 실제 분포를 보고 정한다

### `max_new_tokens` (#3)
- 임시값 128
- 정답 라벨 문장을 토크나이저로 세어 최대 토큰 수를 구하고, 그 1.5배 정도로 다시 정한다 (한국어는 글자 하나가 토큰 1–3개)
- 정상 발화의 결과는 바뀌지 않고, 반복 생성만 끊는다
- **재검토 필요 (2026-10-30 실측):** 라벨 문장 하나는 끝 토큰 포함 4–8토큰이라, 위 규칙대로면 상한이 약 12가 된다. 하지만 업로드는 최대 20초이고 18.7초 꽉 찬 발화에서 44토큰이 나왔다. 문장 단위 상한으로 정하면 긴 정상 녹음이 잘린다 → **라벨 기준이 아니라 "20초 꽉 찬 발화" 기준으로 정하는 것을 제안**. 128은 44의 약 2.9배라 현재 임시값을 유지해도 된다. 확정은 사용자 합의 후

### 버전 고정과 디코딩 설정 (#15, #31)
- transformers, torch, torchaudio 버전을 고정하고 기록한다. 노트북의 `forced_decoder_ids`가 동작하는 버전으로 고르고 스모크 테스트로 확인한다
- 노트북은 디코딩 설정을 따로 주지 않았다 → 모델의 `generation_config.json` 기본값을 따른다. 그 값(`num_beams`, `do_sample` 등)을 확인해 기록하고, 같은 값으로 명시한다
- 배포 기록에 모델 revision(커밋 해시), 가중치 checksum, 라이브러리 버전을 남긴다

### 스모크 테스트 결과 (2026-10-30, `scripts/smoke_test.py`)
| 항목 | 값 |
|---|---|
| 모델 revision | `304fe2952840b5b96ca3d4399a43d024912d4096`, `~/marieum-ml/models/ASR_Whisper_Disease_General` (`runs/`, `training_args.bin` 제외) |
| 가중치 checksum | `model.safetensors` sha256 `8cff1f598ab8d0711cfcd2ad2c602ccf95ed9c1232a1898e5e6cbc7ca131d236` (HF LFS 값과 일치) |
| 고정 버전 | Python 3.11.18, torch 2.11.0+cu128, torchaudio 2.11.0+cu128, transformers 4.53.3, tokenizers 0.21.4, huggingface_hub 0.36.2, numpy 2.4.6 |
| 버전 선택 근거 | 모델 `config.json`의 `transformers_version`이 4.53.0.dev0 → 4.53.x 고정. `forced_decoder_ids`는 deprecated 경고만 나고 동작함 |
| 디코딩 기본값 (#15) | `num_beams=1`, `do_sample=false`, `temperature=1.0`, `max_length=448`, `repetition_penalty=1.0`, `no_repeat_ngram_size=0`, `return_timestamps=false` → greedy. 서버에서 같은 값을 명시한다 |
| 노트북 호출 vs 서버 호출 | 두 조각 모두 전사가 같음 (`"문 열어"`) |
| `scores` 정렬 (#14) | `scores`에는 **생성한 토큰만** 들어가고 강제 토큰 4개(`<\|startoftranscript\|> <\|ko\|> <\|transcribe\|> <\|notimestamps\|>`)는 빠진다. `sequences[-len(scores):]`와 `argmax(scores)`가 일치. 끝의 `<\|endoftext\|>`는 특수 토큰이라 confidence에서 뺀다 |
| `confidence` | 0.9962, 0.9948 (텍스트 토큰 3개). 이 조각에서는 후처리 전 logits와 값이 같았다 |
| GPU 메모리 | 로딩 후 923MiB, 최대 1,159MiB |
| 시간 (참고) | 2.7초 조각, 생성 토큰 4개: `feat` 5–11ms, `gen` 약 200ms. 10초 발화는 토큰이 많아 `gen`이 더 길어진다 → §9 625ms 예산 실측 필요 |

주의:
- 모델 `config.json`의 `forced_decoder_ids`는 `<|en|>`(50259)이다. **반드시 `language="ko"`로 만든 `forced_decoder_ids`를 넘긴다** (노트북과 같음)
- torchaudio 2.9+의 `torchaudio.load`는 torchcodec이 필요하다. 서버는 WAV를 표준 라이브러리 `wave`로 읽고 `torchaudio.functional.resample`만 쓴다
- `output_scores`/`output_logits`에 "not valid" 경고가 뜨지만 값은 정상으로 돌아온다
- 스모크 테스트 조각은 `sample.wav` 9.55–12.25초, 21.15–23.75초를 16kHz로 변환한 것. 원본 44.1kHz → `torchaudio.functional.resample` 경로는 §10 조각 전사에서 확인함

### 학습 데이터 (#10 조사)
AI Hub 구음장애 음성인식 데이터(608) 구성과 논문 사용 범위:

| 대분류 | 분량 | 논문 사용 |
|---|---|---|
| 뇌신경장애 | 약 1,043시간 (20.8%) | 일부: 말초신경병증 `TL01/26`·`TS01/26`, 뇌성마비 `VL01/13`, 뇌졸중 `VL01/11` |
| 언어청각장애 | 약 1,520시간 (30.4%) | 사용 안 함 |
| 후두장애 | 약 2,445시간 (48.8%) | 사용 안 함 |

- General 모델은 위 세 질환을 합친 데이터로 학습 (약 3.7만 발화, 학습 3.29만 / 검증 2천 / 테스트 2천)
- **`sample.wav`(`TL01_뇌신경장애/25.언어+뇌신경장애`)는 학습 폴더 목록에 없다** → 처음 보는 화자일 가능성이 높다
- 제외 이유는 논문(Speech Communication 2026, DOI `10.1016/j.specom.2026.103393`) 본문에만 있을 것으로 보이며, 본문이 유료라 확인하지 못했다. PDF를 구하면 확인한다
- **한계:** 언어청각장애와 후두장애 환자 발화는 학습하지 않았다. 요양시설 환자가 이쪽이면 정확도가 떨어질 수 있다

### 학습 전처리 (`Preprocess/` 노트북)
| 처리 | 학습 때 | 우리 추론 |
|---|---|---|
| 긴 녹음 자르기 | 자체 에너지 VAD, 3–5초 이상 무음에서 자름 | 서버는 자르지 않음. 테스트 스크립트만 자름 (§10) |
| 앞뒤 무음 | 따로 안 자름 (VAD 경계 500ms 단위만큼 남음) | Silero VAD, 앞뒤 여유 200ms |
| 문장 중간 쉼 | 그대로 남김 | 그대로 남김 (같음) |
| 최대 길이 | 제한 없음 | 업로드 20초 |
| 음량 정규화, 잡음 제거 | 안 함 | 안 함 (같음) |
| 리샘플링, 모노 | 특징 추출 단계에서 16kHz | 서버가 16kHz 모노 변환 (같음) |
| 정답 문장 | 문장부호 `+ * ( ) ? ! , . ~ - '` 제거 | 테스트 비교 때 같은 규칙 (§10) |

### 검토 후 하지 않기로 한 것
- **WhisperX:** 가중치 교체는 가능(CTranslate2 변환, `tokenizer.json` 생성 필요). 하지만 fp16/beam 5로 연구 결과와 달라질 수 있고, 짧은 발화에서는 강점이 거의 없다
- **재학습:** 배포에는 필요 없다

### Whisper 특성 메모
- 항상 30초 창으로 본다. 짧은 음성은 무음으로 채운다
- 인코더 비용은 길이와 무관하고, 처리 시간 차이는 생성 글자 수(디코더)에서 난다
- 무음이 많으면 없는 말을 지어낸다 → VAD로 줄인다

---

## 5. 통신 방식

- HTTP `POST /asr` (multipart 업로드 → JSON 응답 1회). 스펙 §3.1 그대로
- WebSocket / SSE는 쓰지 않는다. 오디오는 **보통 10초 이하, 최대 20초**이고(#30), 녹음이 끝난 파일을 한 번에 전사하며, ML 서버를 부르는 것은 백엔드다
- 다시 볼 조건: 말하는 도중의 중간 전사가 ML 서버에 필요해질 때 (새 작업)
- 속도는 반드시 측정한다 (§9)

---

## 6. VAD 전처리

- 모델: **Silero VAD** (MIT, 약 2MB, CPU에서 빠름. faster-whisper와 WhisperX도 사용)
- ML 서버 안에서 **앞뒤 무음만** 자른다. 문장 중간의 쉼은 자르지 않는다 (구음장애 발화는 중간 쉼이 길다)
- 서버 동작: 첫 말소리 시작 − 200ms부터 마지막 말소리 끝 + 200ms까지 남긴다
- **말소리를 못 찾으면 Whisper를 돌리지 않고** 200 `text: ""`, `confidence: 0.0` (#1 거부, 원래 방식 유지). GPU를 아끼고 무음에서 없는 말을 지어내는 것을 막는다
- **동시 사용 (#17):** Silero VAD는 앞의 소리를 내부에 기억한다. 요청끼리 섞이지 않게 한 번에 한 요청만 쓰게 하고(잠금), 쓸 때마다 상태를 초기화한다. VAD는 수십 ms라 속도 영향은 거의 없다

### VAD 설정
| 설정 | 보수적 (덜 자름) | 공격적 (많이 자름) | **확정 기본값** | 의미 |
|---|---|---|---|---|
| `threshold` | 0.3 | 0.6 | **0.5** | 이 값 이상이면 말소리 (Silero 기본값) |
| `speech_pad_ms` | 400 (faster-whisper 기본) | 30 (Silero 기본) | **200** | 말 앞뒤에 남기는 여유. 약한 첫소리와 끝 음절 보호 |
| `min_speech_duration_ms` | 100 | 250 | **150** | 이보다 짧은 소리는 잡음으로 무시 (숨소리, 클릭음) |
| `min_silence_duration_ms` | 2000 (faster-whisper 기본) | 100 (Silero 기본) | **800** | 서버에서는 앞뒤 경계 판단에만 쓰임 |

- 설정값은 환경변수로 바꿀 수 있게 한다. `sample.wav`로 측정한 뒤 조정한다

### 역할 경계
| 일 | 담당 |
|---|---|
| 발화 단위로 녹음 끊기 (녹음 시작과 정지) | Flutter / 백엔드 |
| 받은 파일의 앞뒤 무음 자르기, 무음이면 Whisper 생략 | ML 서버 (VAD) |
| 화자별 구간 나누기 | 후속 F-0 |
| `text`, `confidence`로 재질의 여부 판단 | 백엔드 |

---

## 7. API 계약

### 공통
- 오디오 업로드: `multipart/form-data`, 그 외 `application/json; charset=utf-8`
- **`X-Trace-Id` (#20):** 응답 헤더와 본문에 echo. 허용 문자는 영문·숫자·`-`·`_`, 길이 1–64자. 없거나 규칙에 안 맞으면 서버가 UUIDv4를 새로 발급한다
- `X-Patient-Id`: ML 서버는 사용 안 함, 로그에만 기록 (스펙 §2.2, #25 거부)
- **응답 크기 (#21):** UTF-8 JSON 본문 ≤ 32,768바이트. `text`는 최대 1,000자로 자른다. 에러 `message`는 최대 200자
- **에러 형식 (#11):** FastAPI 기본 422(`detail` 형식), `HTTPException`, multipart 파싱 오류, 처리되지 않은 예외를 **모두** 아래 형식으로 바꾼다. 에러 응답에도 `X-Trace-Id` 헤더를 붙인다
  ```json
  {"error_code": "AUDIO_TOO_LONG | INVALID_FORMAT | MODEL_UNAVAILABLE | TIMEOUT | INTERNAL",
   "message": "사람이 읽을 수 있는 설명",
   "trace_id": "echo back"}
  ```
- **백엔드는 상태 코드가 아니라 `error_code`로 분기한다** (#12, 계약서에 명시)

### `POST /asr`
- 폼 필드 `audio` (필수), 1차는 **WAV만**
- 서버가 16kHz 모노로 리샘플링 / 다운믹스
- 응답 (200):
  ```json
  {
    "text": "물 좀 주세요",
    "confidence": 0.87,
    "duration_ms": 1240,
    "language": "ko",
    "model_version": "yoona-J/ASR_Whisper_Disease_General",
    "trace_id": "..."
  }
  ```
- `duration_ms`: **업로드 원본 길이** (VAD로 자르기 전)
- `no_speech_prob`: **넣지 않는다** (#16, 스펙상 선택 필드). 무음은 `text: ""`, `confidence: 0.0`으로 알 수 있다
- 처리시간: 응답 헤더 `Server-Timing` (§9)과 로그에 남김

### 입력 검사 순서 (#7, #8, #19)
1. **바이트 상한 (받는 도중 누적 검사):** 요청 본문 4.5MB, 파일 4MB(4,194,304바이트) 초과 → 413 `AUDIO_TOO_LONG`. 20초 × 48kHz × 2채널 × 16bit ≈ 3.84MB 기준
2. **메모리에만 보관:** Starlette는 1MB가 넘는 업로드를 디스크 임시 파일로 넘긴다. 이 기준(spool 크기)을 바이트 상한보다 크게 올리거나 직접 파싱해서, 업로드가 디스크에 쓰이지 않게 한다. 고정한 버전에서 실제로 디스크에 안 쓰이는지 확인한다
3. **WAV 판별:** RIFF/WAVE 헤더, PCM 16-bit, 8–48kHz, 1–2채널만 허용. 아니면 400 `INVALID_FORMAT`. 확장자나 MIME만으로 판단하지 않는다
4. **헤더 기준 길이:** 20초 초과 → 400 `AUDIO_TOO_LONG`
5. **디코딩 후 실제 길이:** 다시 20초 초과 검사. 0프레임이면 400 `INVALID_FORMAT`
6. VAD로 앞뒤 자르기. **자른 후 길이로는 거절하지 않는다** (스펙 "10초 초과 시 `AUDIO_TOO_LONG`"을 완화. 백엔드 코드는 깨지지 않지만 Slack 공지 필요)
- 정상적인 무음 파일은 200 (`text: ""`, `confidence: 0.0`)
- 예: `[무음 3초][말소리 4초][쉼 2초][말소리 3초][무음 4초]` = 16초 업로드 → 앞뒤만 잘라 약 9.4초 → 전사

### 상태 코드 정리 (#12)
| 상황 | 상태 | `error_code` |
|---|---|---|
| 정상, 빈 전사, 무음 | 200 | — |
| 업로드 길이 20초 초과 | 400 | `AUDIO_TOO_LONG` |
| 업로드 바이트 초과 | 413 | `AUDIO_TOO_LONG` |
| WAV 아님, 허용 범위 밖, 0프레임, 필드 누락, 파싱 오류 | 400 | `INVALID_FORMAT` |
| 모델 로딩 중 / 로딩 실패 | 503 | `MODEL_UNAVAILABLE` |
| 대기열 포화 (#4) | 503 + `Retry-After: 1` | `MODEL_UNAVAILABLE` |
| 마감(2.5초) 초과, 마감 지난 대기 작업 | 504 | `TIMEOUT` |
| 그 외 | 500 | `INTERNAL` |

### `GET /healthz` (readiness, #6)
- **모델이 준비됐을 때만 200**, 로딩 중·실패·종료 중이면 503
- 본문은 상태와 관계없이 스펙 §4 형식: `status`(`ok`/`loading`/`error`), `models`(준비됐을 때만 `["asr"]`), `gpu`(name, memory_free_mb), `model_version`, `uptime_sec`
- GPU 메모리 값은 주기적으로(예: 10초) 갱신해 캐시한 값을 준다. 요청마다 `nvidia-smi`를 실행하지 않는다 (#32)

### `GET /livez` (liveness, #6)
- 프로세스가 살아 있으면 항상 200. 모델 상태와 무관

### 백엔드에 알릴 점 (`HANDOFF.md`)
- `error_code`로 분기할 것. 상태 코드 표 (위)
- `no_speech_prob`는 Mock과 달리 오지 않는다. 필드가 없어도 fallback이 동작해야 한다
- 1차 입력은 WAV만. 업로드는 20초까지, 자른 후 10초 제한은 두지 않음
- 준비 여부는 `/healthz` 200으로 판단
- 대기열 포화 시 503 `MODEL_UNAVAILABLE` + `Retry-After`

---

## 8. 서버 동작

| 항목 | 값 |
|---|---|
| 프로세스 | 단일 FastAPI 프로세스. **uvicorn 워커 1개 강제** (#33). 워커가 여러 개면 모델이 여러 번 올라가고 "GPU 1건씩" 규칙이 깨진다 |
| 모델 로딩 | 백그라운드에서 로딩. 그동안 `/healthz` 503, `/asr` 503 `MODEL_UNAVAILABLE` (#6) |
| 워밍업 | 하지 않음 (#34) |
| GPU 실행자 (#2) | 전용 GPU 작업자 1개. HTTP 요청과 GPU 작업의 수명을 분리한다. 504를 보낸 뒤에도 실행 중인 `generate()`가 끝날 때까지 작업자가 슬롯을 갖고, 늦게 나온 결과는 버린다. `max_time`으로 마감 시 생성을 멈춘다 |
| 마감 시각 (#4) | 요청을 받은 순간(업로드 포함) 단조 시계 기준 +2.5초. GPU에 올리기 직전에 마감이 지났으면 버리고 504 |
| 대기열 상한 (#4) | 4건. 넘으면 바로 503 `MODEL_UNAVAILABLE` + `Retry-After: 1` (3초를 기다리게 하지 않고 백엔드가 빨리 fallback) |
| GPU 동시성 | 한 번에 1건, 도착 순서대로 |
| 백엔드 타임아웃 | 3초 (보낸 순간부터 응답을 다 받을 때까지, 네트워크 포함). 초과 시 재질의 또는 일반 호출로 fallback |
| 자동 재시작 | systemd (죽으면 자동 재시작, 재부팅 후 자동 시작). 멈춤 감지 타이머는 두지 않음 (#24) |
| 재시작 다운타임 | ≤ 30초 |

발화 1개 = 파일 1개 = 요청 1개. 서버는 오디오를 나누지 않는다.

---

## 9. 성능 측정

### 타임라인과 `Server-Timing` (#22)
시작점은 요청을 받은 순간(업로드 포함)이다. 2.5초 마감도 여기서부터 잰다.

| `Server-Timing` 이름 | 구간 | 단위 |
|---|---|---|
| `recv` | 요청 수신과 업로드 읽기 | ms |
| `decode` | WAV 검사, 디코딩, 16kHz 모노 변환 | ms |
| `vad` | VAD, 앞뒤 자르기 | ms |
| `feat` | Whisper 특징 추출 (로그-멜) | ms |
| `queue` | GPU 차례 대기 | ms |
| `gen` | `generate()` | ms |
| `total` | 서버 전체 (직렬화 포함) | ms |

- 같은 값을 로그에도 남긴다
- CPU 서버에서 잰 왕복 시간 − `total` = 네트워크 시간

### 측정 조건
- GPU 서버(localhost)와 CPU 서버에서 각각 실행
- 동시 1건, 4건
- 첫 요청은 따로 기록하고 통계에서 뺀다
- 말이 꽉 찬 10초 샘플 포함. 20초(말이 꽉 찬 경우)도 참고값으로 측정

### 1건당 GPU 예산 (#5)
- GPU는 1건씩 처리하므로, 동시 4건이 2.5초 안에 끝나려면 **1건당 GPU 점유 시간이 625ms 미만**이어야 한다
- L4 fp32에서 1건당 GPU 점유 시간(`feat` + `gen`)과 생성 토큰 수를 실측한다
- 못 맞추면 결과를 공유하고 fp16 전환, `max_new_tokens`, 통과 기준을 다시 합의한다

#### 1차 실측 (2026-10-30, L4 fp32, 단독 실행, HTTP 없음)
| 입력 | 생성 토큰 | `feat` | `gen` |
|---|---|---|---|
| 짧은 조각 60개 (3–5초 파일) | 4–8 | 중앙값 4.3ms | 중앙값 202ms, 최대 222ms |
| 말소리 이어 붙임 9.4초 | 23 | 4.7ms | 358ms (3회 357–359ms) |
| 말소리 이어 붙임 18.7초 | 44 | 4.9ms | 552ms (3회 551–554ms) |
| 프로세스 첫 요청 | 4 | 9.5ms | 774ms (통계 제외) |

- 대략 `gen` ≈ 약 150ms(인코더, 길이 무관) + 토큰당 약 9.5ms
- 10초 꽉 찬 발화 기준 1건 약 363ms → **625ms 예산 안**. 동시 4건이면 GPU 약 1.45초
- 주의: "꽉 찬 발화"는 이 화자의 느린 발화를 0.3초 간격으로 이어 붙인 것이라 초당 약 2.4토큰이다. 말이 빠르면 토큰이 늘어난다. 최악의 경우(128토큰)는 약 1.37초로 예산을 넘는다
- 워밍업을 하지 않으므로(#34) 재시작 후 첫 요청은 약 0.8초 걸린다. 2.5초 마감 안이다
- HTTP, VAD, 대기열을 포함한 왕복 시간은 서버 구현 후 §9 조건으로 다시 잰다

### 통과 기준 (#23)
> 10초 음성 4건 동시 × 20회 = **80건**이 모두 200이고 응답이 완전하며, 5xx 0건, 각 왕복 시간(CPU 서버 측정) 3초 미만

503이나 504를 빨리 돌려준 것은 통과로 치지 않는다.

---

## 10. 테스트 음성과 정답

### 파일
| 항목 | 값 |
|---|---|
| 음성 | `D:\codyssey-hack\marieum-whisper\sample.wav` |
| 정답 | `D:\codyssey-hack\라벨링데이터\TL01_뇌신경장애\25.언어+뇌신경장애\ID-02-25-N-KSM-02-02-M-45-JL.json` |
| 출처 | AI Hub 다운로드 가능 데이터 (안심존 영역 데이터는 받지 못했음. 사용 가능) |
| 길이 | 744.3초 (12분 24초), 라벨 `playTime`과 일치 |
| 규격 | WAV, 44.1kHz, 모노 16-bit. 라벨 메타에는 48kHz로 적혀 있으나 실제 파일은 44.1kHz |
| 화자 | 남성, 45세, 뇌신경장애 (질환 코드 02-25) |
| 녹음 | 조용한 방, 녹음기, 대본 읽기 |
| 학습 포함 여부 | 학습 폴더 목록(26, 13, 11번)에 없음 → 처음 보는 화자일 가능성 높음 (§4) |

### 정답 구조
- `Transcript` 하나에 파일 전체 정답. **문장별 시간 정보는 없다**
- 30개 문장 × 평서문·의문문 = **60개 발화** (예: "문 열어." / "문 열어?")
- 2–5단어의 짧은 문장. 발화 사이에 긴 무음이 있다 (평균 약 12.4초 간격)

### 용도 (#10)
- `sample.wav`는 **동작 확인, 회귀 확인, 속도 측정용**이다
- 여기서 나온 CER은 참고값이다. 정확도는 다른 화자 데이터(VL 등)로 따로 잰다

### 테스트 방식
- **마이크 흉내:** `sample.wav`를 발화 단위로 잘라, 실제 녹음 속도에 맞춰 44.1kHz 원본 그대로 하나씩 보낸다. 동시 4건은 같은 방식을 4개 동시에 돌린다
- **자르는 기준:** 학습 전처리처럼 **3초 이상 무음**에서 자른다. 앞뒤에 무음 1초씩 붙여 보낸다 (서버 VAD가 앞뒤를 자르는지까지 확인). 목표 60조각
- **짝 확인 (#28):** 조각 5개 정도를 직접 들어 정답 문장과 짝이 맞는지 확인한다. 누락과 과분할이 겹치면 개수는 60개여도 짝이 한 칸씩 밀릴 수 있기 때문이다. trim 유무 CER 비교는 하지 않는다
- **전사 비교:**
  - 60조각이 맞으면 조각마다 정답 문장 1개와 CER 비교
  - 안 맞으면 조각별 전사를 순서대로 이어 붙여 정답 전체와 비교
  - 비교 전 학습과 같은 규칙으로 문장부호(`+ * ( ) ? ! , . ~ - '`)와 공백 제거
- **점수 분포:** 60개 발화의 `confidence` 분포를 기록해 백엔드 기준값 정하는 데 쓴다
- **한계:** 조용한 방 녹음이라 실제 병실보다 깨끗하다. 화자 1명이다

### 조각 전사 결과 (2026-10-30, `scripts/slice_sample.py`, `scripts/run_chunks.py`)
서버 VAD 없이, 앞뒤 무음 1초가 붙은 44.1kHz 조각을 서버와 같은 옵션으로 전사했다.

| 항목 | 값 |
|---|---|
| 자르기 | 에너지 기준(잡음 바닥 −82dBFS + 15dB), 3초 이상 무음에서 자름 → **정확히 60조각** |
| 말소리 길이 | 0.72–3.14초 (중앙값 1.27초), 발화 간격 10.2–11.4초 |
| 전사 | 60개 모두 `<\|endoftext\|>`로 정상 종료, 빈 전사 0 |
| 평서문·의문문 짝 | 조각 (1,2), (3,4), … 30쌍 모두 전사 일치 → 과분할·누락으로 짝이 밀리지 않았다는 간접 근거 |
| `confidence` | 최소 0.910, p10 0.996, 중앙값 0.9996. 낮은 건 "전화 왔었어요" 한 쌍(0.948, 0.910)뿐 |
| 생성 토큰 | 조각당 4–8개 (끝 토큰 포함) |

- **정답 라벨이 없어 CER은 아직 못 쟀다.** 라벨 JSON을 받으면 조각별 CER을 계산한다
- `confidence`가 거의 1에 몰려 있다. 쉬운 대본 문장, 조용한 녹음이라 그렇다. **이 분포로는 백엔드 기준값을 정할 수 없다** → 다른 화자 데이터로 다시 본다
- 짝 확인(#28): 사람이 직접 듣는 확인은 남아 있다

### 정답 문장 (모델 전사, 참고용. 라벨로 교체 예정)
문 열어 / 나무가 많아 / 내일 만나 / 이제 내려 / 머리 말려 / 여기 놔요 / 나비가 날아요 / 엄마가 노래해 / 머리 말려요 / 여기 놨어요 / 여기 놔 / 여기 놔요 / 머리 말려 / 머리 말려요 / 여기 놨어요 / 머리 말렸어요 / 여기 놓을 거예요 / 밥 먹었죠 / 집에 아무도 없겠지 / 발이 얼었어요 / 길이 많이 막혀요 / 선생님이 하셨어요 / 거기 갈래 / 그렇게 될 리가 없지 / 전화 왔었어요 / 여기까지 걸어 왔어요 / 방이 추운 거 같애 / 날씨가 추워요 / 음악 켜 / 날씨가 더워요

---

## 11. 로그와 음성 보존

- 로그: JSON 한 줄, `trace_id`, `timestamp`(ISO 8601 UTC), `endpoint`, `latency_ms`, `status_code`, `confidence`, `model_version`, `patient_id` (스펙대로 남김, #25 거부), `Server-Timing` 구간값
- 오디오는 메모리에만 두고 응답 후 폐기. 파일 저장 금지 (업로드 단계 포함, §7 입력 검사 2)
- `ML_DEBUG_SAVE=1`일 때만 로컬 임시 저장, 30분 뒤 삭제. 기본값 0. 서버 시작 시 정리 등은 추가하지 않음 (#26 거부)

---

## 12. 조사 기록: 겹치는 목소리 / 여러 화자

Whisper는 화자를 구분하지 않고, 동시에 나온 두 단어를 출력하지 못한다. WhisperX README도 겹치는 음성은 잘 처리하지 못한다고 적고 있다. 관리자가 제안한 대안은 겹친 구간에만 음성 분리를 적용하는 것이다.

| 방법 | 현업 대표 모델 | 성능 (일반인 영어 기준) | 비고 |
|---|---|---|---|
| ① 구간 나누기 (화자 분리) | **pyannote community-1** | DER AMI 17.0%, VoxConverse 11.2% (제조사 발표) | CC-BY-4.0, WhisperX 기본값. Sortformer v2가 일부 언어에서 경쟁 |
| ② 음성 분리 | **MossFormer2** (ClearerVoice-Studio) | SI-SNRi WSJ0-2mix 약 24dB, Libri2Mix 약 21.7dB | 바로 받아 쓸 수 있는 모델 중 가장 많이 쓰임 |
| ③ 환자 목소리만 추출 | WeSep, TargetVoice, SpeakerBeam | 13–16dB 수준, 처음 보는 화자에서 크게 떨어진다는 보고 | 같은 조건 벤치마크 없음, 연구 단계 |
| ④ 환자 목소리 판별 | CAM++, ERes2NetV2 (3D-Speaker), ECAPA-TDNN | EER 0.45–0.86% (VoxCeleb1-O) | ERes2NetV2는 짧은 발화에 강함 |

- 구음장애 음성으로 분리 성능을 잰 연구는 찾지 못했다
- 화자 판별은 구음장애 화자에서 성능이 떨어지고 보정이 필요하다는 연구가 있다 (Interspeech 2022)
- ③, ④는 환자 목소리 샘플이나 특징값 저장이 필요해서 §11 음성 비보존 규칙과 충돌한다

### pyannote community-1을 L4(gp1l4-g3)에서
- 돌릴 수 있다. `pyannote.audio` 4.x, PyTorch + torchaudio. `pipeline.to(torch.device("cuda"))`로 GPU 사용
- Whisper-small과 함께 올려도 24GB에 여유가 크다 (실측 필요)
- HF 약관 동의 + 토큰 필요 (gated). 받아두면 로컬 경로에서 오프라인 로드 가능
- 모델 카드에 속도·메모리 수치 없음 → 실측 필요. torch/transformers 버전 충돌 확인 필요

### 연구 노트북 방식 + 화자 분리
- 가능하다. 흐름: 오디오 → VAD → pyannote 화자별 구간 → 구간마다 잘라 Whisper 전사
- 반대 순서(전체 전사 후 시간으로 화자 붙이기)는 Whisper 타임스탬프가 필요한데, 노트북은 타임스탬프 없이 추론하므로 부정확할 수 있다
- 번갈아 말하기는 구분 가능. 동시에 겹친 구간은 여전히 망가진다
- pyannote는 `SPEAKER_00` 같은 이름표만 준다. 누가 환자인지는 모른다

---

## 13. 후속 이슈

| ID | 내용 | 메모 |
|---|---|---|
| F-0 | 화자 분리 (pyannote community-1) | 연구 노트북 방식 옆에 붙일 수 있음 (§12). 속도, 짧은 발화 안정성, 응답 `segments` 필드 추가, HF 토큰, torch 버전 충돌 확인 필요 |
| F-1 | 화자 분리 결과에서 누가 환자인지 알아보기 (F-0 이후) | 화자 판별(CAM++, ERes2NetV2 등) 또는 "가장 길게 말한 사람" 같은 규칙. 화자 판별은 환자 목소리 특징값 저장이 필요해 §11 규칙과 충돌 → 팀 규칙 재합의 필요. 구음장애 화자에서 성능 저하 연구 있음 |

---

## 14. 미뤄둔 것

- 심사장 준비 (ngrok 백업, URL 전환). ngrok을 쓰면 별도 인증이 필요하다 (#9)
- 겹치는 목소리 대응 (겹침 테스트 파일로 측정 → 필요하면 §12의 ④부터 검토)
- WAV 외 포맷 지원
- 형식 검증 스킬 (후보: GPU 실행자와 마감 시각의 슬롯 관리, §8)
- 논문 본문 확인 (학습 제외 이유, §4)
- 다른 화자 데이터(VL 등)로 정확도 측정 (§10)

---

## 15. 열린 질문

| # | 질문 | 메모 |
|---|---|---|
| 1 | 서버 주소 (`ML_ASR_URL`에 들어갈 GPU 서버 IP) | TBD (저녁 미팅). 구현에는 영향 없음 |

---

## 16. 리뷰 반영 기록

| # | 내용 | 결정 |
|---|---|---|
| 1 | VAD 미검출 시 Whisper 건너뛰기 조건 강화 | 거부 (원래대로 건너뜀) |
| 2 | GPU 실행자 분리, 504 뒤에도 슬롯 유지, 늦은 결과 버림, `max_time` | 수용 |
| 3 | `max_new_tokens` 제한 | 수용 (임시 128, 라벨로 재산정) |
| 4 | 마감 시각, 대기열 상한 4, 포화 시 503 | 수용 (`error_code`는 `MODEL_UNAVAILABLE`) |
| 5 | 1건당 GPU 625ms 미만 실측 | 수용 |
| 6 | 백그라운드 로딩, `/healthz` readiness, `/livez` | 수용 |
| 7 | 업로드 바이트 상한 누적 검사 | 수용 |
| 8 | 업로드 디스크 임시 파일 방지 | 수용 |
| 9 | CPU 서버 IP만 허용을 배포 필수로 | 수용 |
| 10 | 테스트 음성의 학습 포함 여부 | 수용. 조사 결과 학습 목록에 없음. `sample.wav`는 동작·회귀·속도용, 정확도는 다른 화자로 |
| 11 | 모든 에러를 공통 형식으로 | 수용 |
| 12 | 상태 코드 정리, `error_code`로 분기 | 수용 |
| 13 | Flutter 녹음 형식 요청 | 강제하지 않음 |
| 14 | `confidence` 토큰 정의 | 수용 |
| 15 | 버전 기록, 디코딩 설정 명시 | 수용 |
| 16 | `no_speech_prob` 정의 변경 | 거부. `no_speech_prob` 자체를 응답에서 뺌 |
| 17 | VAD 동시 사용 시 상태 섞임 방지 | 수용 |
| 18 | `flags`, `speech_ms` | 거부 |
| 19 | WAV 판별 기준 | 수용 |
| 20 | `X-Trace-Id` 검증 | 수용 |
| 21 | 응답 32KB 규칙화 | 수용 |
| 22 | 타임라인과 `Server-Timing` | 수용 |
| 23 | 통과 기준 80건 모두 200 | 수용 |
| 24 | 멈춤 감지 타이머 | 거부 |
| 25 | 로그에서 `patient_id` 빼기 | 거부 (스펙대로 남김) |
| 26 | 디버그 파일 시작 시 정리 | 거부 |
| 27 | Flutter 20초 자동 정지 | 강제하지 않음 |
| 28 | 조각 짝 확인, trim 유무 CER 비교 | 짝 확인만 수용 |
| 29 | VAD 표 열 이름 "확정 기본값" | 수용 |
| 30 | §5 "보통 10초, 최대 20초" | 수용 |
| 31 | 모델 revision, checksum, 라이브러리 버전 기록 | 수용 |
| 32 | `/healthz` GPU 메모리 캐시 | 수용 |
| 33 | uvicorn 워커 1개 강제 | 수용 |
| 34 | 시작 시 워밍업 | 거부 |
| 35 | 고위험 단어 환각 공지 | 거부 |

---

## 17. 근거 자료

- 모델: https://huggingface.co/yoona-J/ASR_Whisper_Disease_General
- 연구 코드: https://github.com/yoona-J/Diagnosis-Aware_Multitask_Fine-Tuning_of_Whisper_for_DSR
- 추론 예제: `Predict/[Distribution]GeneralModel_Predict.ipynb`
- 학습 전처리: https://github.com/yoona-J/Diagnosis-Aware_Multitask_Fine-Tuning_of_Whisper_for_DSR/tree/main/Preprocess
- 논문: https://www.sciencedirect.com/science/article/abs/pii/S0167639326000415 (DOI 10.1016/j.specom.2026.103393)
- 학습 데이터셋: https://huggingface.co/datasets/yoona-J/ASR_Preprocess_Disease_General_Dataset
- AI Hub 구음장애 음성인식 데이터: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=608
- Whisper 문서: https://huggingface.co/docs/transformers/model_doc/whisper
- HF 다운로드: https://huggingface.co/docs/huggingface_hub/guides/download
- PyTorch 설치: https://pytorch.org/get-started/locally/
- Silero VAD: https://github.com/snakers4/silero-vad
- pyannote community-1: https://huggingface.co/pyannote/speaker-diarization-community-1
- WhisperX 겹침 이슈: https://github.com/m-bain/whisperX/issues/9
- 화자 분리 비교 (SDBench): https://www.isca-archive.org/interspeech_2025/durmus25_interspeech.pdf
- MossFormer2 / ClearerVoice-Studio: https://www.isca-archive.org/interspeech_2025/zhao25f_interspeech.pdf
- 3D-Speaker: https://github.com/modelscope/3D-Speaker
- 구음장애 화자 판별: https://www.isca-archive.org/interspeech_2022/salim22_interspeech.pdf
