# marieum-ai

구음장애 특화 Whisper 모델(`yoona-J/ASR_Whisper_Disease_General`)을 L4 GPU 서버에 올려, 한국어 음성을 실시간으로 텍스트로 바꾸는 ML 서버 (WebSocket `/asr-stream`).

- 현재 상태: v4 스트리밍 서버 구현, GPU 서버에 배포되어 동작 중, 백엔드 서버에서 접속 확인 완료
- 팀원용 쉬운 안내: [TEAM_GUIDE.md](TEAM_GUIDE.md)
- 구현 정리: [ML_SERVER.md](ML_SERVER.md)
- 시험 결과와 지표: [TEST_RESULTS.md](TEST_RESULTS.md)
- 합의 내용과 실측: [CALIBRATION.md](CALIBRATION.md) §0
- 백엔드 연동: [HANDOFF.md](HANDOFF.md), 질문 답변: [BACKEND_FAQ.md](BACKEND_FAQ.md), 중계 코드 프롬프트: [BACKEND_RELAY_PROMPT.md](BACKEND_RELAY_PROMPT.md)

## 구성

| 경로 | 내용 |
|---|---|
| `server/app.py` | FastAPI: `/asr-stream`, `/healthz`, `/livez` |
| `server/engine.py` | Whisper 추론 (fp16, 묶음, `no_speech_prob`) |
| `server/scheduler.py` | GPU 작업자 (final 우선, 세션당 interim 1개) |
| `server/stream.py` | VAD 구간 나누기, interim 확정 규칙 (torch 없음) |
| `scripts/stream_client.py` | 테스트·부하 클라이언트 |
| `scripts/mock_asr_server.py` | GPU 없는 Mock (포트 9001) |
| `scripts/verify_engine.py` | 엔진이 노트북과 같은 전사를 내는지 확인 |
| `tests/` | `python -m pytest tests -q` (GPU 없이 돈다: 구간 나누기 단위 테스트 + 가짜 엔진으로 WebSocket 통합 테스트) |

## 실행

```bash
ML_MODEL_DIR=models/ASR_Whisper_Disease_General \
  uvicorn server.app:app --host 0.0.0.0 --port 8001 --workers 1
python scripts/stream_client.py --wav a.wav --show                         # 발화 모드
python scripts/stream_client.py --mode conversation --wav long.wav --show  # 실시간 대화 모드
```

인증이 없는 서버다. 8001 포트는 보안그룹에서 CPU 서버 IP만 허용한 뒤 연다.

## 데이터 주의

테스트에 쓰는 AI Hub 구음장애 음성인식 데이터(오디오, 라벨)는 재배포가 금지되어 있어 이 레포에 올리지 않는다 (`.gitignore`).
