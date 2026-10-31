# ML 서버 시험 결과와 지표

마리음 ML 서버(구음장애 특화 Whisper, WebSocket `/asr-stream`)를 시험해서 얻은 숫자를 모아 둔 문서다. 무엇을, 어떤 조건에서, 어떻게 쟀는지와 목표 통과 여부를 함께 적는다.

- 기준일: 2026-10-31
- 원본 기록: [CALIBRATION.md](CALIBRATION.md) §0.5, §4, §9, §10
- 구조 설명: [ML_SERVER.md](ML_SERVER.md) / 팀원용 안내: [TEAM_GUIDE.md](TEAM_GUIDE.md)

---

## 0. 한눈에 보기

| 항목 | 결과 | 목표 | 판정 |
|---|---|---|---|
| 중간 전사(interim) 지연 p50, 동시 4세션 | 123–157ms | ≤ 300ms | ✅ |
| 중간 전사 지연 p95, 동시 4세션 | 283–380ms | ≤ 500ms | ✅ |
| 최종 전사(final) 지연 p95 (`end` 후) | 210–269ms | ≤ 800ms | ✅ |
| 실제 발화 대화 모드 interim p50 / p95 | 117 / 132ms | ≤ 300 / 500ms | ✅ |
| 쉼 없이 말하는 4명 동시 (최악) interim p95 | 543–606ms | ≤ 500ms | ⚠️ 약간 넘음 |
| fp16 전사가 원본(fp32 노트북)과 같음 | 60/60 | 모두 같음 | ✅ |
| 4건 묶음 GPU 시간 | 248ms (fp32 702ms) | — | 2.8배 빨라짐 |
| 시험 중 에러 | 0건 | 0건 | ✅ |
| 자동 테스트 | 23/23 통과 | 모두 통과 | ✅ |
| 무음 환각 차단 | 무음 입력 → 빈 전사 | 지어낸 말 없음 | ✅ |
| 잡음 환각 차단 | "땅콩", "떡"이 나올 수 있음 | 지어낸 말 없음 | ❌ 남은 문제 |
| 정확도(CER) | 측정 안 함 | — | ⏳ 정답 라벨 필요 |
| 네트워크 포함 지연 | 측정 안 함 | — | ⏳ 백엔드 서버에서 측정 필요 |

---

## 1. 시험 환경

### 1.1 장비
| 항목 | 값 |
|---|---|
| GPU 서버 | NCP `inno-gpu-006` (`10.1.2.11`), NVIDIA L4 |
| 측정 위치 | GPU 서버 안 (localhost). **서버 간 네트워크는 빠져 있다** |
| 서버 실행 | uvicorn 워커 1개, GPU 작업자 1개, VAD 스레드 2개 |

### 1.2 소프트웨어
| 항목 | 버전 |
|---|---|
| Python | 3.11.18 |
| torch / torchaudio | 2.11.0+cu128 |
| transformers / tokenizers | 4.53.3 / 0.21.4 |
| huggingface_hub | 0.36.2 |
| numpy | 2.4.6 |
| silero-vad | 6.2.3 |
| fastapi / starlette / uvicorn | 0.120.4 / 0.49.3 / 0.54.0 |

### 1.3 모델
| 항목 | 값 |
|---|---|
| 모델 | `yoona-J/ASR_Whisper_Disease_General` (whisper-small을 구음장애 음성으로 파인튜닝) |
| revision | `304fe2952840b5b96ca3d4399a43d024912d4096` |
| 가중치 sha256 | `8cff1f598ab8d0711cfcd2ad2c602ccf95ed9c1232a1898e5e6cbc7ca131d236` (HF 값과 일치) |
| 모델 카드 성능 | CER 24.2% / WER 29.9% (학습 측 보고값, 우리가 잰 값 아님) |
| 디코딩 | greedy (`num_beams=1`, `do_sample=false`), `language="ko"`, `task="transcribe"` |

### 1.4 시험 음성
| 항목 | 값 |
|---|---|
| 파일 | `sample.wav` (AI Hub 구음장애 음성인식 데이터, 재배포 금지라 레포에 없음) |
| 길이 / 형식 | 744.3초 (12분 24초), WAV 44.1kHz 모노 16-bit |
| 화자 | 남성 45세, 뇌신경장애 (질환 코드 02-25). 학습 폴더 목록에 없어 **처음 보는 화자일 가능성이 높다** |
| 녹음 | 조용한 방, 대본 읽기 |
| 내용 | 30문장 × 평서문·의문문 = 60발화, 2–5단어 짧은 문장, 발화 사이 약 10–11초 무음 |
| 조각 | 3초 이상 무음에서 잘라 **정확히 60조각**, 앞뒤 무음 1초씩 붙임. 말소리 0.72–3.14초 (중앙값 1.27초) |

**한계:** 화자 1명, 조용한 방, 쉬운 대본 문장이다. 실제 병실보다 깨끗한 조건이라 실제 정확도와 confidence는 더 나쁠 수 있다.

---

## 2. 측정 방법

| 지표 | 정의 |
|---|---|
| interim 지연 | 클라이언트가 그 interim에 반영된 음성의 마지막 100ms 프레임을 보낸 시각 → interim 수신 시각. 서버가 보내는 `audio_end_ms`로 어느 프레임인지 맞춘다 |
| final 지연 | 클라이언트가 `end`를 보낸 시각 → 마지막 final(`is_last: true`) 수신 시각 |
| GPU 시간 | 엔진 한 번 호출(특징 추출 제외, 인코더 + 무음 확률 + 생성)에 걸린 시간 |
| 동시 N세션 | N개 WebSocket 연결이 각각 실제 속도(100ms 프레임을 100ms마다)로 동시에 보낸다 |
| p50 / p95 | 중앙값 / 상위 5% 경계값 |

- 도구: `scripts/stream_client.py` (스트리밍), `scripts/verify_engine.py` (엔진), `scripts/run_chunks.py` (조각 전사), `scripts/smoke_test.py` (모델 단독)
- 첫 요청은 따로 기록하고 통계에서 뺀다 (워밍업 효과)
- 앱 녹음 시간과 서버 간 네트워크 시간은 들어 있지 않다

---

## 3. 엔진 정확성 (원본과 같은 결과를 내는가)

`scripts/verify_engine.py`, 조각 60개. 기준은 연구 노트북(`Predict/[Distribution]GeneralModel_Predict.ipynb`)의 fp32 호출 결과다.

| 확인 항목 | fp32 | fp16 |
|---|---|---|
| 노트북 호출과 전사가 같은 조각 | **60/60** | **60/60** |
| 4건 묶음 vs 1건씩 전사가 같은 조각 | 60/60 | 60/60 |
| 묶음 vs 1건 confidence 차이 최대 | — | 0.0009 |

- 서버는 인코더를 한 번만 돌리고 그 결과를 무음 확률 계산과 생성에 같이 쓴다. 이렇게 바꿔도 노트북과 결과가 같다는 것을 위 60/60으로 확인했다
- `scores`에는 생성한 토큰만 들어가고 강제 토큰 4개는 빠진다. `sequences[-len(scores):]`와 `argmax(scores)`가 일치해 confidence 계산 위치가 맞다는 것을 확인했다

---

## 4. 엔진 속도와 메모리

### 4.1 fp32 vs fp16 (조각 60개)
| 항목 | fp32 | fp16 | 변화 |
|---|---|---|---|
| GPU 시간, 1건 | 233ms | 133ms | −43% |
| GPU 시간, 4건 묶음 | 702ms | 248ms | −65% (2.8배) |
| GPU 메모리 최대 | 2,055MiB | 1,988MiB | 거의 같음 |

### 4.2 길이별 시간 (1차 실측, 2026-10-30, fp32, 1건씩)
| 입력 | 생성 토큰 | 특징 추출 | 생성 |
|---|---|---|---|
| 짧은 조각 60개 (3–5초 파일) | 4–8 | 중앙값 4.3ms | 중앙값 202ms, 최대 222ms |
| 말소리 이어 붙임 9.4초 | 23 | 4.7ms | 358ms (3회 357–359ms) |
| 말소리 이어 붙임 18.7초 | 44 | 4.9ms | 552ms (3회 551–554ms) |
| 프로세스 첫 요청 | 4 | 9.5ms | 774ms (통계 제외) |

- 대략 **생성 시간 ≈ 150ms(인코더, 길이와 무관) + 토큰당 약 9.5ms** (fp32)
- Whisper는 항상 30초 창으로 보기 때문에 인코더 시간은 길이와 상관없고, 차이는 생성 토큰 수에서 난다
- 최악의 경우(토큰 상한 128개까지 생성)는 fp32 약 1.37초로 추정한다. 실측하지 않은 값이다

### 4.3 시작과 워밍업
| 항목 | 값 |
|---|---|
| 첫 추론 (워밍업 없이) | 856ms |
| 워밍업 (묶음 1–4건 한 번씩) | 약 1.5초. 끝나야 `/healthz`가 200 |
| 모델 로딩 후 GPU 메모리 | 923MiB (최대 1,159MiB, fp32 스모크 테스트) |
| systemd 재시작 → `/healthz` 200 | 18초에 확인 (20초 이내) |
| VAD 32ms 창 하나 | 약 1.6ms (CPU) |

---

## 5. 전사 결과 (조각 60개)

`scripts/slice_sample.py` → `scripts/run_chunks.py`. 서버 VAD 없이 서버와 같은 옵션으로 전사했다.

| 항목 | 결과 |
|---|---|
| 정상 종료 (`<\|endoftext\|>`) | 60/60 |
| 빈 전사 | 0 |
| 평서문·의문문 짝 | 30쌍 모두 전사가 같다 → 조각이 밀리지 않았다는 간접 근거 |
| 생성 토큰 | 조각당 4–8개 (끝 토큰 포함) |
| `no_speech_prob` | 최대 0.0000 |

### 5.1 confidence 분포
| 통계 | 값 |
|---|---|
| 최소 | 0.910 |
| p10 | 0.996 |
| 중앙값 | 0.9996 |
| 낮은 것 | "전화 왔었어요" 한 쌍만 (0.948, 0.910) |

- **confidence가 거의 1에 몰려 있다.** 그래서 백엔드 기준값(interim 0.3, final 0.4)을 이 분포로는 정할 수 없다
- 지어낸 말도 0.4–0.7이 나온다 (§6). 기준값 0.4로는 지어낸 말을 거의 못 거른다

### 5.2 전사 문장 (모델 출력, 정답 라벨 아님)
문 열어 / 나무가 많아 / 내일 만나 / 이제 내려 / 머리 말려 / 여기 놔요 / 나비가 날아요 / 엄마가 노래해 / 머리 말려요 / 여기 놨어요 / 여기 놔 / 여기 놔요 / 머리 말려 / 머리 말려요 / 여기 놨어요 / 머리 말렸어요 / 여기 놓을 거예요 / 밥 먹었죠 / 집에 아무도 없겠지 / 발이 얼었어요 / 길이 많이 막혀요 / 선생님이 하셨어요 / 거기 갈래 / 그렇게 될 리가 없지 / 전화 왔었어요 / 여기까지 걸어 왔어요 / 방이 추운 거 같애 / 날씨가 추워요 / 음악 켜 / 날씨가 더워요

(30문장, 평서문·의문문 각 한 번씩 → 60발화)

- **정답 라벨과 비교한 오류율(CER)은 아직 없다.** 라벨 JSON을 받으면 조각별 CER을 계산한다
- 사람이 직접 듣고 짝을 맞춰 보는 확인도 아직 남아 있다

---

## 6. 무음·잡음 환각 시험

모델이 소리가 없거나 잡음뿐일 때 없는 말을 만들어 내는지 시험했다.

| 입력 | 모델 출력 | `no_speech_prob` | confidence | 서버 최종 결과 |
|---|---|---|---|---|
| 무음 1초 | "아침 식사 시간은 몇 시까지예요" | 0.69 | 0.73 | **빈 전사** (VAD에서 차단, 통과해도 no_speech > 0.6으로 차단) |
| 무음 3초 | "아침 식사 시간은 몇 시까지예요" | 0.69 | 0.73 | **빈 전사** |
| 백색잡음 1초, −40dBFS | "땅콩" | 0.0002 | 0.61 | VAD가 말소리로 보면 그대로 나갈 수 있음 |
| 백색잡음 1초, −26dBFS | "떡" | 0.0001 | 0.43 | 같음 |
| 말소리 앞 0.5초만 | "땅콩" | 0.002 | 0.61 | 같음 |
| 정상 말소리 60조각 | 정상 전사 | 최대 0.0000 | — | 정상 |

**알게 된 것**
1. 무음에서 문장 하나를 **확신 있게** 지어낸다. openai/whisper 기본 규칙(`no_speech > 0.6` **그리고** `avg_logprob < −1`)으로는 못 거른다. 이 모델은 지어낼 때도 logprob이 높기 때문이다 (−0.54) → 서버는 `no_speech > 0.6` 하나로만 거른다
2. 잡음과 아주 짧은 소리는 `no_speech_prob`가 0에 가까워 못 거른다. 서버는 세 가지로 줄인다:
   - VAD가 말소리가 없다고 보면 Whisper를 돌리지 않는다
   - 말소리가 300ms 이상 쌓여야 첫 interim을 보낸다
   - 확정 규칙: 두 번 연속 같아야 `stable_text`가 된다
3. 그래도 `unstable_text`(흐린 글자)에는 지어낸 말이 잠깐 보일 수 있다

---

## 7. 스트리밍 지연 시험

`scripts/stream_client.py`, 실제 속도로 보냄, GPU 서버 localhost, fp16.

### 7.1 결과
| 시험 | interim p50 / p95 / 최대 | 마지막 final p50 / p95 | 건수 | 에러 |
|---|---|---|---|---|
| ① utterance, 조각 20개, 동시 4 × 5회 (2회 반복) | 123–157 / 283–380 / 311–408ms | 119–148 / 210–269ms | interim 55건씩 | 0 |
| ② conversation, `sample.wav` 실제 발화 90초, 동시 4 | 117 / 132 / 187ms | — | interim 80건 | 0 |
| ③ conversation, 쉼 없이 이어 붙인 말 60초, 동시 4, **구간 최대 10초** | 561 / 851 / 986ms | 645 / 789ms | interim 384건 | 0 |
| ④ ③과 같은 조건, **구간 최대 6초** (3회 반복) | 312–335 / 543–606 / 706–1010ms | 272–336 / 380–407ms | — | 0 |

- ① ②: 앱의 실제 사용(버튼 누르고 한 문장, 쉼이 있는 대화)에 해당한다. 모두 목표 통과
- ③ ④: 4명이 동시에 한 번도 쉬지 않고 계속 말하는 **최악 조건**이다. 구음장애 발화는 문장 사이 쉼이 길어서(이 샘플은 약 10초) 실제로는 드물다

### 7.2 목표(프로토콜 v1.0 SLO)와 비교
| 시험 | interim p50 ≤ 300 | interim p95 ≤ 500 | final p95 ≤ 800 |
|---|---|---|---|
| ① utterance 4세션 | ✅ 123–157 | ✅ 283–380 | ✅ 210–269 |
| ② conversation 실제 발화 | ✅ 117 | ✅ 132 | — |
| ③ 쉼 없음, 구간 10초 | ❌ 561 | ❌ 851 | ✅ 789 |
| ④ 쉼 없음, 구간 6초 (현재 기본값) | ❌ 312–335 (약간 넘음) | ❌ 543–606 (약간 넘음) | ✅ 380–407 |

### 7.3 구간 최대 길이 10초 → 6초로 바꾼 근거
③에서 interim이 크게 늦었다. 원인은 interim이 "현재 구간 전체"를 다시 전사하기 때문에, 구간이 길수록 생성 토큰이 늘어 GPU 1회가 길어지는 것이다.

| 서버 내부 (③, 구간 10초) | p50 | p95 |
|---|---|---|
| GPU 대기 | 152ms | 387ms |
| GPU 1회 | 239ms | 404ms |
| 평균 묶음 크기 | 2.1건 | — |

구간 상한을 6초로 줄이자(④) interim p95가 851ms → 543–606ms, final p95가 789ms → 380–407ms로 줄었다. 그래서 기본값을 6초로 정했다 (`ML_MAX_SEGMENT_S=6`).

### 7.4 그 밖의 동작 확인
| 항목 | 결과 |
|---|---|
| interim 최소 간격 300ms 지키는지 | 부하 시험에서 가장 짧은 간격 408ms ✅ |
| 대화 모드 문장 나누기 (`sample.wav` 45초) | 문장마다 구간 final 1개: "문 열어", "문 열어", "나무가 많아", 마지막은 빈 final ✅ |
| `stable_text`가 줄어들지 않는지 | 자동 테스트로 확인 (§9) ✅ |
| 구간 final 뒤에 그 구간 interim이 오지 않는지 | 자동 테스트로 확인 ✅ |
| 워밍업 전 첫 interim | 856ms → 워밍업 추가 후 해결 |
| 140ms 간격으로 interim 두 개가 몰려 나감 | 300ms 최소 간격 타이머 추가 후 해결 |

---

## 8. 배포와 보안 확인

| 항목 | 방법 | 결과 |
|---|---|---|
| 서비스 자동 시작·재시작 | `systemctl` (`marieum-ml.service`) | 동작 중, 재시작 후 18초에 `/healthz` 200 |
| 방화벽 규칙 | `marieum-ml-firewall.service` (iptables) | 8001은 localhost와 백엔드 서버(`10.1.1.11`)만 허용 |
| 백엔드 서버에서 접속 | `curl http://10.1.2.11:8001/healthz` | **200** `status: ok` ✅ |
| 다른 곳(SSL VPN, Mac)에서 접속 | 같은 요청 | **연결 안 됨** ✅ (iptables에서 막힘) |
| 열린 포트 점검 | VPN에서 연결 시험 | 22(SSH), 111(rpcbind), 8001(백엔드만). rpcbind는 쓰지 않지만 팀 판단으로 유지 |
| 음성 저장 | `ML_DEBUG_SAVE` | 기본 0 (저장 안 함), 배포 유닛도 0 |
| 로그에 전사 텍스트 | 로그 필드 확인 | 남기지 않음 (`trace_id`, `patient_id`, 시간만) |

---

## 9. 자동 테스트 (23개, GPU 없이 동작)

`python -m pytest tests -q`. GPU 서버에서 **23/23 통과**했다.

### 9.1 구간 나누기·확정 규칙 단위 테스트 (`tests/test_stream.py`, 11개)
| 테스트 | 확인하는 것 |
|---|---|
| `test_utterance_one_final_trimmed` | 발화 모드에서 final 1개, 앞뒤 무음이 잘린다 |
| `test_utterance_silence_only_has_no_speech` | 무음만 있으면 "말소리 없음" |
| `test_utterance_too_long_closes_session` | 10초를 넘으면 자동 종료 |
| `test_conversation_splits_on_pause` | 대화 모드에서 0.8초 쉼마다 구간이 나뉜다 |
| `test_conversation_pause_inside_sentence_is_kept` | 문장 안의 짧은 쉼은 자르지 않는다 |
| `test_conversation_drops_short_noise` | 150ms 미만 소리는 버린다 |
| `test_conversation_max_segment_cut_keeps_rest` | 최대 길이에서 쉼으로 자르고 나머지를 다음 구간으로 넘긴다 |
| `test_interim_cadence_and_snapshot` | interim 간격과 다시 전사할 음성 범위 |
| `test_conversation_default_max_segment_hard_cut` | 쉼이 없으면 6초에서 바로 자른다 |
| `test_agreement_commits_only_repeated_prefix` | 두 번 연속 같은 앞부분만 확정, 마지막 단어는 확정 안 함 |
| `test_agreement_ignores_punctuation` | 확정 비교 때 문장부호를 무시한다 |

### 9.2 WebSocket 통합 테스트 (`tests/test_app.py`, 12개)
실제 uvicorn 서버를 띄우고 가짜 엔진과 에너지 기준 VAD로 시험한다.

| 테스트 | 확인하는 것 |
|---|---|
| `test_healthz_and_livez` | `/healthz`, `/livez` 응답 형식 |
| `test_utterance_interims_then_one_final` | interim 여러 개 → final 1개, 필드, `stable_text`가 줄지 않음 |
| `test_utterance_silence_skips_whisper` | 무음이면 Whisper 생략, 빈 final, `no_speech_prob: 1.0` |
| `test_conversation_final_per_pause` | 쉼마다 구간 final, `segment_id` 순서, 구간 final 뒤 interim 없음 |
| `test_utterance_too_long` | `AUDIO_TOO_LONG` + 자동 final, 이후 오디오·`end`는 조용히 버림, 연결 유지 |
| `test_invalid_start_keeps_connection` ×5 | Opus, 96kHz, 잘못된 mode, `en`, 잘못된 trace_id → `INVALID_FORMAT`, 연결 유지 |
| `test_no_session` | `start` 없는 오디오·`end` → `NO_SESSION` 한 번씩, 연결 유지 |
| `test_sessions_on_separate_connections_do_not_mix` | 두 연결에서 동시에 보내도 섞이지 않음 |

> 참고: 2026-10-31에 VAD 미리 만들기 버그(4개를 만든다고 했지만 실제 1개)를 고쳤다. 아래처럼 GPU 서버에서 검증했다. **운영 서비스(8001)에는 아직 배포하지 않았다.**

### 9.1 VAD 미리 만들기 수정 검증 (GPU 서버, 2026-10-31)

운영 서비스는 그대로 두고, 별도 폴더 `~/vadtest`에 수정본을 만들어 시험했다. 수정본 `server/app.py`의 sha256 앞자리는 `5da9acbe…`로 로컬 파일과 같다. 배포 중인 코드는 HEAD와 같은 `9fd92f41…`다.

| 시험 | 결과 |
|---|---|
| 자동 테스트 | 23 passed (1.66s) |
| 실제 모델 로드 뒤 미리 만든 VAD 수 | 수정 전 **1개** → 수정 후 **4개** |
| 수정본 서버를 8002에 띄워 utterance 동시 4세션 (`chunk_01–04`) | final 4개, 에러 0, end 후 final p50 153 / max 158ms |
| 같은 서버, conversation 1세션 (`sample.wav` 앞 30초) | 구간 final 3개, interim 지연 p50 134ms, 에러 0 |
| 서버 로그 | start 5 / session_closed 5, traceback·error 0 |
| 시험 후 | 8002 프로세스 종료, 운영 8001 `/healthz` ok 확인 |

---

## 10. HTTP `POST /asr` 공존 검증 (정적 검토, 2026-10-31)

CALIBRATION.md §5–§11의 원래 HTTP 계약을 지금 WebSocket 서버와 한 프로세스에서 같이 쓸 수 있는지, Codex가 코드를 읽기 전용으로 따로 검증했다. **실제 부하 측정은 하지 않았다.**

| 항목 | 결과 |
|---|---|
| 결론 | **조건부 가능.** 모델 1개, GPU 작업자 1개를 같이 쓰면 된다. 포트·방화벽 변경 없음 |
| 지금 코드 그대로 | 불가능. `/asr` 라우트, 업로드 검사, 대기열 상한, 마감 처리가 없다 |
| 충돌 지점 | 11개 (GPU 대기열 상한·마감, 20초 HTTP와 WebSocket 묶음 간섭, 공용 VAD 스레드 막힘, multipart 디스크 임시 저장, 에러 형식, 정확히 20초에서 잘림 등) |
| 코드로 확인한 사실 | 최종 전사 대기열 상한 없음, `python-multipart` 없음, 공용 VAD 스레드 2개, 정확히 20.0초 입력이 "너무 김"으로 처리됨, VAD 미리 만들기가 1개만 만듦(수정함, §9.1) |
| 미검증 | 실제 L4에서 HTTP 20초 작업이 WebSocket 지연을 얼마나 늘리는지, 토큰 상한 128이 20초에 충분한지, Starlette 0.49.3에서 업로드가 디스크에 안 쓰이는지 |
| 예상 규모 | 간소안 약 400–600줄 (추정) / 원래 §7 전부 복구 1,100–1,900줄 (Codex 추정) |

---

## 11. 아직 측정하지 않은 것

| 항목 | 왜 필요한가 | 필요한 것 |
|---|---|---|
| 정확도 (CER) | 모델이 얼마나 정확한지 숫자로 말하려면 | 정답 라벨 JSON |
| 다른 화자 정확도 | 지금은 화자 1명, 조용한 방 | AI Hub 검증 데이터(VL) 등 다른 화자 |
| 네트워크 포함 지연 | 실제 백엔드 → ML 경로 속도 | 백엔드 서버에서 `stream_client.py` 실행 |
| 앱 마이크 음성 | 실제 녹음 품질에서의 동작 | 앱·백엔드 연결 |
| 실제 병실 잡음 | 잡음 환각이 얼마나 자주 나오는지 | 잡음 섞인 녹음 |
| 5세션 이상 동시 | 최대 수용 인원 | 부하 시험 확대 |
| HTTP + WebSocket 공존 부하 | HTTP 추가 여부 결정 | HTTP 구현 후 |
| 사람이 직접 듣고 짝 확인 | 조각과 정답이 맞는지 | 사람 확인 5조각 정도 |

---

## 12. 다시 재는 방법

```bash
# GPU 서버에서
cd ~/marieum-ml && source .venv/bin/activate

# 자동 테스트 (GPU 없이도 됨)
python -m pytest tests -q

# 엔진이 노트북과 같은 결과를 내는지 + fp16 + GPU 시간
python scripts/verify_engine.py --model models/ASR_Whisper_Disease_General --chunks data/chunks --dtype fp16

# ① utterance 동시 4세션 × 5회
python scripts/stream_client.py --wav data/chunks/chunk_*.wav --sessions 4 --rounds 5

# ② ③ ④ conversation 동시 4세션 (시작 시점을 7초씩 어긋나게, 60초)
python scripts/stream_client.py --mode conversation --wav long.wav --sessions 4 --offset-step 7 --max-seconds 60

# 백엔드 서버에서 (네트워크 포함 지연)
python scripts/stream_client.py --url ws://10.1.2.11:8001/asr-stream --wav a.wav --sessions 4 --rounds 5
```

결과를 다시 재면 이 문서의 표와 [CALIBRATION.md](CALIBRATION.md) §0.5를 같이 고친다.
