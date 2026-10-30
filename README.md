# marieum-ai

구음장애 특화 Whisper 모델(`yoona-J/ASR_Whisper_Disease_General`)을 L4 GPU 서버에 배포하고, 한국어 음성을 텍스트로 바꾸는 FastAPI ML 서버 (`POST /asr`).

- 현재 상태: 캘리브레이션 완료, 구현 시작 전
- 합의 내용: [CALIBRATION.md](CALIBRATION.md)

## 데이터 주의

테스트에 쓰는 AI Hub 구음장애 음성인식 데이터(오디오, 라벨)는 재배포가 금지되어 있어 이 레포에 올리지 않는다 (`.gitignore`).
