# marieum-ai

구음장애 특화 Whisper 모델(`yoona-J/ASR_Whisper_Disease_General`)을 NCP의 L4 GPU 서버에서 돌려, 환자의 한국어 음성을 실시간으로 텍스트로 바꾸는 ML 서버다.

## 현재 상태 (프리즈, 2026-11-01)

- main `d98fcc7` (PR #6 출력 토큰 상한까지)이 GPU 서버 운영 인스턴스에 배포되어 동작 중이다
- 서버: NCP GPU Server `inno-gpu-006` (10.1.2.11), NVIDIA L4, systemd 서비스 `marieum-ml`, 포트 8001
- 백엔드 서버(10.1.1.11)에서만 접속된다 (방화벽 `marieum-ml-firewall`). 인증은 없다

## 구조

```
앱(Flutter) ──WSS──▶ 백엔드 FastAPI (10.1.1.11) ──ws / http──▶ ML 서버 (10.1.2.11:8001)
                     임베딩·Claude·DB(Docker PostgreSQL)        Whisper + Silero VAD
```

- ML 서버는 음성 인식만 한다. 실행 중에 외부로 나가는 호출이 없다
- 문장 임베딩(ko-sroberta), Claude 호출, DB는 백엔드 담당이다
- Whisper 가중치는 Hugging Face에서 받아 `~/marieum-ml/models/`에 두고 쓴다

## API

| 엔드포인트 | 용도 |
|---|---|
| `WS /asr-stream` | 실시간 전사. `utterance`(발화 1개, final 1회, 10초 한도) / `conversation`(쉼마다 구간 final, 세션 600초 한도) |
| `POST /asr` | 녹음 파일 한 번에 전사 (WAV, 60초까지) |
| `GET /healthz` | 준비 여부 (200 준비됨, 503 로딩 중) |
| `GET /livez` | 프로세스 생존 |

메시지 형식과 에러 코드는 [HANDOFF.md](HANDOFF.md), [BACKEND_NOTICE_20261101.md](BACKEND_NOTICE_20261101.md)에 있다.

## 주요 설정

환경변수로 바꾼다. 전체 목록은 [ML_SERVER.md](ML_SERVER.md) §8.

| 설정 | 값 | 뜻 |
|---|---|---|
| `ML_END_SILENCE_MS` | 1750 | 대화 모드에서 이만큼 쉬면 구간을 닫고 final을 보낸다 |
| `ML_MAX_SEGMENT_S` | 15 | 쉼 없이 말해도 이 길이에서 구간을 자른다 |
| `ML_INTERIM_MAX_NEW_TOKENS` / `ML_FINAL_MAX_NEW_TOKENS` | 96 / 128 | 출력 토큰 고정 상한 |
| `ML_TOKEN_CAP_RATE` / `ML_TOKEN_CAP_FLOOR` | 14 / 8 | 조각마다 상한 `min(고정 상한, ceil(음성 초 × 14) + 8)`. 같은 말 반복 폭주를 일찍 끊는다. **0이면 끈다** |
| `ML_DTYPE` | fp16 | Whisper 추론 정밀도 |

## 실행과 테스트

```bash
python -m pytest tests -q        # GPU 없이 돈다 (가짜 엔진, 42개)
ML_MODEL_DIR=models/ASR_Whisper_Disease_General \
  uvicorn server.app:app --host 0.0.0.0 --port 8001 --workers 1
python scripts/stream_client.py --mode conversation --wav long.wav --show
```

## 배포와 되돌리기

- 배포: `server/`를 GPU 서버 `~/marieum-ml/`에 덮어쓰고 `sudo systemctl restart marieum-ml` (약 25초 뒤 `/healthz` 200). 절차는 [ML_SERVER.md](ML_SERVER.md) §12
- 배포 전 검증: 운영 8001을 건드리지 않고 별도 포트에 띄워 확인하는 절차가 [.claude/skills/verify-ml-server/SKILL.md](.claude/skills/verify-ml-server/SKILL.md)에 있다
- 토큰 상한만 끄기: 환경변수 `ML_TOKEN_CAP_RATE=0`으로 재시작
- 코드 전체 되돌리기: 배포 때 만든 백업(`~/marieum-ml/server.bak-1101`)으로 바꾸고 재시작

## 문서 지도

| 읽는 사람 | 문서 |
|---|---|
| 팀원 (처음) | [TEAM_GUIDE.md](TEAM_GUIDE.md) 쉬운 안내 |
| 백엔드 | [HANDOFF.md](HANDOFF.md) 연동 안내, [BACKEND_NOTICE_20261101.md](BACKEND_NOTICE_20261101.md) 11-01 변경, [BACKEND_FAQ.md](BACKEND_FAQ.md) 질문 답변, [BACKEND_RELAY_PROMPT.md](BACKEND_RELAY_PROMPT.md) 중계 코드 프롬프트 |
| ML 운영 | [ML_SERVER.md](ML_SERVER.md) 구현·운영, [verify-ml-server](.claude/skills/verify-ml-server/SKILL.md) 검증 절차 |
| 기록 (결정 근거) | [CALIBRATION.md](CALIBRATION.md) 합의와 실측, [TEST_RESULTS.md](TEST_RESULTS.md) 시험 결과, [MAX_NEW_TOKENS.md](MAX_NEW_TOKENS.md) 토큰 상한 설계, [HANDOFF_TOKEN_CAP.md](HANDOFF_TOKEN_CAP.md) 토큰 상한 인수인계, [BACKEND_TOKEN_CAP_REQUEST.md](BACKEND_TOKEN_CAP_REQUEST.md) 백엔드에 요청한 통계 |

## 알려진 한계

- **말 끝에 말하지 않은 글자가 붙을 수 있다.** Whisper가 흐린 끝소리를 그럴듯한 단어로 채운다. 토큰 상한 안에 들어가서 상한으로는 못 막는다. 이런 출력은 신뢰도가 대체로 낮아(실험 중간값 0.72) 백엔드의 신뢰도 기준으로 거른다
- **같은 말 반복 출력은 신뢰도가 높다** (0.92~0.97). 짧은 구간은 토큰 상한이 끊지만, 15초 가까운 긴 구간은 상한(128)이 커서 거의 그대로 나온다
- 토큰 상한 R=14는 환자 실험 로그(검사 대본 읽기)로 정했다. 실제 사용 로그로 다시 확인해야 한다
- 신뢰도는 구간 길이 설정에 따라 분포가 바뀐다. 설정을 바꾸면 백엔드 기준값도 다시 봐야 한다

## 데이터 주의

테스트에 쓰는 AI Hub 구음장애 음성인식 데이터(오디오, 라벨, 전사)는 재배포가 금지되어 있어 이 레포에 올리지 않는다 (`.gitignore`의 `*.wav`, `data/`).
