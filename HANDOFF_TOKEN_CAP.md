# 핸드오프: 출력 토큰 상한 (feat/max-new-tokens)

2026-11-01. 브랜치 `feat/max-new-tokens` (main `d8c0c52` 기준). 코드는 끝났고, GPU 서버 검증과 배포는 사용자가 직접 한다.

## 1. 무엇을 바꿨나

- **목표:** 짧은 조각에서 Whisper가 같은 말을 반복하며 96~128토큰까지 폭주하는 걸 일찍 끊는다. 정상 문장은 바꾸지 않는다.
- **전:** GPU 묶음 전체가 상한 하나를 썼다 (묶음 안 최댓값, interim 96 / final 128). 0.5초 interim도 96토큰까지 갈 수 있었다.
- **후:** 작업마다 `cap = min(고정 상한, ceil(음성 길이(초) × 14) + 8)`. 같은 묶음 안에서도 줄마다 자기 상한에서 멈춘다.
  - 음성 길이 = Whisper에 넣는 음성 (final: 앞뒤 무음 자른 구간, interim: 스냅샷). R=14를 측정한 기준과 같다
  - 0.5초 → 15, 1초 → 22, 3초 → 50, 10초 이상 → 96 / 128

## 2. 근거 (실측)

환자 22명 실험 로그 두 개에 공식을 대입했다 (`token_stats.py --simulate`).

| | 0.8초/6초 | 2.5초/18초 |
|---|---|---|
| 정상 final 초당 토큰 p99 | 7.25 | 6.87 |
| 정상 final 최대 토큰 | 70 | 66 (17초) |
| R=14에서 정상 final 잘림 | 0% | 0% |
| 128에 닿은 final | 5개 | 8개 (전부 반복 루프) |

자세한 분석: `MAX_NEW_TOKENS.md`.

## 3. 코드

| 파일 | 변경 |
|---|---|
| `server/scheduler.py` | `token_cap()`, 작업마다 `Job.cap`. 묶음 최댓값 경로 삭제 |
| `server/engine.py` | `transcribe(audios, caps)`. 줄별 정지 `_RowCaps`, 줄 해석 `row_result` |
| `server/stream.py` | `Segment`/`Snapshot.vad_speech_ms` (로그용) |
| `server/app.py` | 음성 길이를 스케줄러에 전달. 로그 필드 추가 |
| `server/config.py` | `ML_TOKEN_CAP_RATE`(14), `ML_TOKEN_CAP_FLOOR`(8). rate 0이면 예전처럼 고정 상한만 |

**고친 버그:** Whisper는 pad 토큰이 끝 토큰(EOT)과 같은 id라, 상한으로 멈춘 줄 뒤에 EOT가 채워진다. 줄 전체를 보면 잘린 줄도 "정상 종료"로 잡힌다. 이제 앞 `cap`개만 본다.

**API 규격:** 클라이언트 메시지는 바뀌지 않았다. 바뀐 건 서버 로그뿐이다.

**새 로그 필드:**
- interim: `token_cap`, `ended_eot`, `vad_speech_ms`
- final: `token_cap`, `vad_speech_ms`
- POST /asr 구간별 `transcript_file_segment` 이벤트

## 4. 확인한 것 / 안 한 것

**확인함 (개발 PC)**
- 테스트 42개 통과 (`python -m pytest tests -q`)
- 작은 랜덤 Whisper로 실제 `AsrEngine.transcribe` 실행 (transformers 4.45.1): 줄마다 3, 7, 40에서 멈춤. 잘린 결과는 상한 없는 결과의 앞부분과 같음. 상한 경계에서 끝난 줄은 `ended_eot=True`
- 가짜 엔진 서버를 `ML_TOKEN_CAP_RATE=1 ML_TOKEN_CAP_FLOOR=0`으로 띄움: 모든 interim/final이 상한에 닿고, 로그의 `token_cap`이 공식과 일치하고, 위반 0

**안 함 (GPU 서버에서 할 일)**
- 서버 버전 transformers 4.53.3에서 줄별 정지 동작
- 실제 모델로 폭주 구간이 상한에서 멈추는지, 정상 final 텍스트가 기준선과 같은지
- 지연 변화

## 5. GPU 서버 검증 순서

절차 전체: `.claude/skills/verify-ml-server/SKILL.md`, 판정 기준: `features/token-cap.md`.

1. 사본 만들기: 후보 `~/mtk-cand`(이 브랜치), 기준선 `~/mtk-base`(운영 코드 복사). **운영 `~/marieum-ml`과 8001은 건드리지 않는다**
2. `cd ~/mtk-cand && $PY -m pytest tests -q`. 4.53.3에서 엔진 테스트까지 통과해야 한다
3. `launch.sh 8002`(후보), `REPO=~/mtk-base launch.sh 8003`(기준선) → `doctor.sh`
4. 폭주 재현 창 5개를 양쪽에 보낸다 → `token_stats.py`(후보 `위반 0개`), `compare_finals.py`(`정상 final 바뀜 0`)
5. 빠른 확인: `ML_TOKEN_CAP_RATE=1`로 띄운 인스턴스에서 실제 모델도 상한에서 멈추는지
6. `cleanup.sh`로 끄고, 운영 8001 healthz 확인

**배포:** 문제없으면 `server/`를 `~/marieum-ml/`에 덮어쓰고 재시작 (ML_SERVER.md §12). **되돌리기:** 재배포 없이 `ML_TOKEN_CAP_RATE=0`으로 재시작하면 예전 고정 상한 동작이 된다.

## 6. 남은 일 / 백엔드 전달 사항

- **지금 설정(1.75초/15초) 실험 결과로 다시 측정.** `gpu_1750.tgz`가 들어오면 PC에서 `token_stats.py --simulate 14`
- **15초 구간에서 128이 충분한지.** 정답 100자 넘는 긴 구간을 제대로 받아 적으면 약 90~150토큰이 필요하다. 일반 화자(말 빠름) 녹음으로 확인 필요
- **긴 구간의 반복 루프는 길이 상한으로 못 막는다.** 반복 감지(같은 n-gram 반복 시 중단)는 별도 과제
- **`ended_eot`를 클라이언트 메시지에 추가 (별도 작업, 필드 추가만이라 하위 호환).** 반복 루프 출력도 신뢰도가 0.92~0.97로 높게 나와 신뢰도로는 못 거른다. 백엔드 정의: "끝 토큰으로 스스로 마쳤으면 true, 길이 상한에 걸려 잘렸으면 false. false면 신뢰도와 상관없이 retry"
- **백엔드 정확도 모드 임계값 제안 (2.5초 실험 기준, 1.75초 결과로 재확인):**

  | 분기 | 백엔드 현재 | 제안 | 근거 (그 구간 틀린 비율) |
  |---|---|---|---|
  | retry | < 0.40 | **< 0.60** | 0.40~0.60도 99% 틀림. 0.40 기준은 0.3%만 걸림 |
  | 低 | 0.40~0.90 | 0.60~0.80 (+0.80~0.90 처리는 제품 판단) | 0.6~0.8은 90%, 0.8~0.9는 46% 틀림 |
  | 中 | 0.90~0.98 | 유지 | 17% |
  | 高 | ≥ 0.98 | 유지 | 1% |

  임계값은 백엔드에서 환경변수로 받는다. 구간 설정이 바뀌면 신뢰도 분포가 바뀐다 (0.98 이상 비율: 0.8초 25% → 2.5초 38%).
