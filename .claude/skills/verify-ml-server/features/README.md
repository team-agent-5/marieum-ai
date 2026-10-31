# 마리음 ML 서버 검증 지도

ML 서버의 사용자(백엔드·앱) 관점 동작을 검증하는 기준 문서다. 서버를 몰기 전에 이 목록을 읽고, 맞는 기능 파일을 레시피로 쓴다. 실행·점검·정리 방법은 [../SKILL.md](../SKILL.md)에 있다.

## 기본 전제

- GPU 서버 `inno-gpu-006`에서 `../SKILL.md` 0절대로 후보 사본 `~/mtk-cand`, 기준선 사본 `~/mtk-base`를 만들었다.
- 셸 변수: `V=~/mtk-cand/.claude/skills/verify-ml-server/scripts`, `PY=~/marieum-ml/.venv/bin/python`, `ML_MODEL_DIR=$HOME/marieum-ml/models/ASR_Whisper_Disease_General`, `AUD=~/exp-1031/audio16k`.
- 후보는 `PY=$PY $V/launch.sh 8002` → `RUN_C`, 기준선은 `PY=$PY REPO=~/mtk-base $V/launch.sh 8003` → `RUN_B`.
- 몰기 전에 `$V/doctor.sh <RUN>`이 FAIL 없이 끝난다.
- 이 검증이 띄우지 않은 인스턴스(운영 8001 포함)는 절대 몰지 않는다.

## 몰기 규칙

- 음성은 항상 `scripts/stream_client.py`(WebSocket) 또는 `curl -X POST /asr`로 보낸다. 엔진을 직접 부르지 않는다.
- 각 레시피는 기본 전제 상태에서 시작한다. 설정을 바꿔 띄운 인스턴스는 레시피가 끝나면 cleanup한다.
- 명령은 그대로 쓴다. 파일 이름·offset·길이를 바꾸면 결과가 레시피의 기대값과 맞지 않는다.
- 지연을 비교할 때는 한 번에 한 인스턴스만 몬다 (GPU 1장 공유).

## 증명과 건너뜀 보고

- 보낸 음성(파일, offset, 길이, 모드, 세션 수) + 받은 메시지(`client_*.json`) + 서버 로그(`server.log`)를 한 쌍으로 남긴다.
- 숫자 판정은 `token_stats.py`·`compare_finals.py` 출력(`stats_*.txt`, `compare_*.txt`)으로 한다.
- 증거마다 기능 ID(예: `cap-runaway`)와 진입점(conversation / utterance / POST /asr)을 적는다.
- 전제가 없어서 못 간 경로는 시도한 명령과 빠진 전제를 적어 "증명 안 됨"으로 보고한다. 다른 진입점으로 대신 증명했다고 쓰지 않는다.
- 가짜 엔진(`FAKE=1`) 결과는 하네스 점검일 뿐, 어떤 기능도 증명하지 않는다.
- AI Hub 음성·전사는 GPU 서버 밖으로 가져가지 않는다.

## 기능 파일 형식

H1 제목과 사용자에게 보이는 동작 한 문단, 그 다음 H2 네 개를 이 순서로 둔다.

1. `Sub-features`: 짧은 ID와 한 줄 설명
2. `How to get to it (user POV)`: 백엔드·앱이 이 기능에 닿는 모든 진입점
3. `Driving it with verify-ml-server`: `Preconditions:`로 시작하고, 행동마다 정확한 명령과 관찰할 결과를 짝지은 굵은 제목 목록
4. `Gotchas`: 검증을 망치거나 무효로 만드는 함정

## 기능

- [출력 토큰 상한](./token-cap.md): 소리 길이에 비례하는 줄별 상한. 폭주를 일찍 끊고 정상 문장은 자르지 않는다. **개발 중**
- [대화 모드 스트리밍](./stream-conversation.md): 1.75초 쉼마다 구간 final, 15초 구간 상한, interim 확정분
- [발화 모드 스트리밍](./stream-utterance.md): 발화 하나에 final 한 번, 10초 한도
- [파일 전사 POST /asr](./file-asr.md): WAV 한 번에 올려 구간별 결과 받기, 60초 한도
- [상태 확인](./health.md): `/healthz` 준비 여부, `/livez`
