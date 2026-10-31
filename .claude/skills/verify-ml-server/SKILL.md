---
name: verify-ml-server
description: 마리음 ML 서버(Whisper ASR, WebSocket /asr-stream + POST /asr + /healthz)를 GPU 서버에 검증용 인스턴스로 따로 띄우고, 실제 음성을 보내 interim/final 메시지와 서버 로그(gen_tokens, ended_eot, token_cap, 지연)로 동작을 증명한다. 출력 토큰 상한(max_new_tokens), 구간 나누기, 지연처럼 모델·서버 동작을 바꾼 뒤 배포 전에 쓴다. GPU 없는 PC에서는 가짜 엔진으로 하네스만 점검한다.
---

# verify-ml-server

마리음 ML 서버를 실제 사용 경로(WebSocket 스트리밍, 파일 업로드)로 몰아서 동작을 증명하는 절차다. 기능별 레시피는 [features/README.md](features/README.md)에 있다. 지금 개발 중인 변경은 [출력 토큰 상한](features/token-cap.md)이다 (배경: 저장소 루트 `MAX_NEW_TOKENS.md`).

## 어디서 돌리나

| 대상 | 무엇을 증명하나 | 비고 |
|---|---|---|
| **GPU 서버** `inno-gpu-006` (`10.1.2.11`, SSL VPN 경유 SSH, 사용자 `ncloud`) | 모델 출력, 토큰 수, 상한, 지연. **진짜 검증은 여기서만** | 운영 인스턴스가 `0.0.0.0:8001`(systemd `marieum-ml`)에서 돈다. **절대 끄거나 재시작하거나 코드를 덮어쓰지 않는다** |
| 개발 PC (Windows, GPU 없음, 모델 없음) | 하네스 자체(launch/doctor/drive/cleanup, 메시지 형식) | `FAKE=1`. 모델 동작은 아무것도 증명하지 못한다 |

GPU 서버에서 쓰는 경로:

```bash
V=~/mtk-cand/.claude/skills/verify-ml-server/scripts   # 이 스킬의 스크립트 (후보 사본 안)
PY=~/marieum-ml/.venv/bin/python                         # 운영 venv를 읽기 전용으로 빌려 쓴다 (Python 3.11)
export ML_MODEL_DIR=$HOME/marieum-ml/models/ASR_Whisper_Disease_General
AUD=~/exp-1031/audio16k                                  # AI Hub 환자 음성 16kHz (42개). 없으면 ~/marieum-ml/data/
```

## 0. 코드 사본 준비 (GPU 서버)

검증 인스턴스는 운영 폴더 `~/marieum-ml`이 아니라 **별도 사본**에서 띄운다.

```bash
# PC (저장소 루트, Git Bash): 후보 브랜치를 묶는다. 커밋 안 한 변경도 들어간다
tar czf mtk.tgz server scripts tests .claude/skills/verify-ml-server
scp mtk.tgz ncloud@10.1.2.11:~/

# GPU 서버
mkdir -p ~/mtk-cand && tar xzf ~/mtk.tgz -C ~/mtk-cand              # 후보 (브랜치)
mkdir -p ~/mtk-base && cp -r ~/marieum-ml/server ~/marieum-ml/scripts ~/mtk-base/   # 기준선 (지금 운영 코드)
cd ~/mtk-cand && $PY -m pytest tests -q                              # GPU 없이 도는 테스트. 실패하면 멈춘다
```

scp가 안 되면 exp 브랜치의 `1031_test_model_confidence/scripts/transfer_server.py`로 PC에서 내려받게 한다 (그 파일 docstring 참고).

## 1. Launch

```bash
PY=$PY $V/launch.sh 8002                    # 후보 (REPO 기본값 = 스킬이 든 사본 = ~/mtk-cand)
PY=$PY REPO=~/mtk-base $V/launch.sh 8003    # 기준선
```

- 마지막 줄 `RUN=<폴더>`를 변수로 잡아 둔다: `RUN_C=...`, `RUN_B=...`.
- 준비 = `/healthz`가 200이고 `"status": "ok"`. 모델 로딩 + 묶음 1~4 워밍업이 끝나야 200이 된다 (약 20초, 최대 240초 대기).
- 포트가 사용 중이면 거부한다(종료 코드 3). **끄지 말고 다른 포트를 고른다.** 8001은 항상 거부한다.
- `build.txt`에 코드 해시(`server/*.py` sha256), git 정보(있으면), `ML_*` 환경변수가 남는다. 설정을 바꿔 띄우려면 `ML_INTERIM_MAX_NEW_TOKENS=64 PY=$PY $V/launch.sh 8004`처럼 앞에 붙인다.
- PC에서 하네스만 점검 (저장소 루트, Git Bash):
  ```bash
  export PYTHONUTF8=1   # Windows: 없으면 stream_client.py가 JSON을 cp949로 저장해 compare_finals.py가 못 읽는다
  V=.claude/skills/verify-ml-server/scripts PY=python AUD=1031_test_model_confidence/data/audio16k   # AUD는 exp 실험 데이터, PC에만 있음
  FAKE=1 PY=python bash $V/launch.sh 18002
  ```
  모델 폴더 검사를 건너뛰고 `fake_server.py`를 띄운다. 이후 명령의 `~/mtk-cand/scripts/`는 `scripts/`로, 포트는 18002로 바꿔 쓴다. Windows에는 `ss`가 없어 doctor의 포트 주인 확인은 SKIP된다.

## 2. Doctor (읽기 전용, 이상하면 먼저)

```bash
$V/doctor.sh $RUN_C
```

확인: pid 생존, 포트 주인이 그 pid(`ss -ltnp`), `/healthz` 200·`model_version`·`gpu`, **띄운 뒤 `server/*.py`가 바뀌지 않았는지**(바뀌었으면 옛 코드가 돌고 있다 → cleanup 후 다시 launch), `build.txt`. 하나라도 FAIL이면 종료 코드 1 → 몰지 않는다.

## 3. Drive

모든 요청은 `127.0.0.1:<PORT>`로만 보낸다.

**WebSocket 스트리밍 (주 경로, 앱이 쓰는 경로)** — 저장소의 `scripts/stream_client.py`가 WAV를 100ms PCM16 프레임으로 실제 속도로 보낸다.

```bash
$PY ~/mtk-cand/scripts/stream_client.py --url ws://127.0.0.1:8002/asr-stream \
  --mode conversation --wav $AUD/LSY_p01.wav --offset 232 --max-seconds 20 \
  --show --out $RUN_C/client_LSY.json | tee $RUN_C/client_LSY.txt
```

- `--mode utterance`(기본, 10초 한도, final 1회) / `conversation`(1.75초 쉼마다 구간 final, 구간 최대 15초)
- `--sessions 4 --offset-step 7`: 동시 4세션 (GPU 묶음 처리 확인용)
- 출력 JSON: 세션마다 `trace_id`, `finals[]`(text, confidence, segment_id, is_last, stable_delta …), `interim_ms[]`, `final_ms[]`, `errors[]`
- 서버 로그와는 `trace_id`로 잇는다

**POST /asr (녹음 파일 한 번에)**

```bash
$PY $V/make_clip.py $AUD/LSY_p01.wav 232 20 $RUN_C/clip_LSY.wav
curl -s -X POST --data-binary @$RUN_C/clip_LSY.wav -H 'Content-Type: audio/wav' \
  "http://127.0.0.1:8002/asr?trace_id=verify-asr-LSY" -o $RUN_C/asr_LSY.json -w '%{http_code}\n'
```

**상태**: `curl -s 127.0.0.1:8002/healthz`, `curl -s 127.0.0.1:8002/livez`

## 4. Evidence

증거 폴더 = `${VERIFY_ROOT:-~/verify-runs}/<RUN_ID>/` (저장소 밖). cleanup 뒤에도 남는다.

| 파일 | 내용 |
|---|---|
| `build.txt`, `server.sha256` | 무슨 코드·설정으로 띄웠나 |
| `healthz.json` | 준비 시점 healthz |
| `server.log` | 서버 JSON 로그 한 줄씩 (`transcript_interim`/`transcript_final`/`transcript_file`). **전사 텍스트는 없다** |
| `client_<이름>.json`, `.txt` | 클라이언트가 받은 메시지 (텍스트 포함) |
| `asr_<이름>.json`, `clip_<이름>.wav` | POST /asr 응답과 보낸 음성 |
| `stats_<이름>.txt` | `token_stats.py` 출력 |
| `compare_<이름>.txt` | `compare_finals.py` 출력 |

```bash
$PY $V/token_stats.py $RUN_C/server.log | tee $RUN_C/stats_all.txt
$PY $V/compare_finals.py $RUN_B $RUN_C client_LSY.json | tee $RUN_C/compare_LSY.txt
```

증명 기준:

- **실제 사용 경로로 몬다.** WebSocket/POST로 음성을 보낸다. 엔진 함수를 직접 부르거나 설정값을 코드에서 바꿔 끼우는 것은 증명이 아니다.
- **행동과 결과를 함께 남긴다.** 보낸 음성(파일·offset·길이) + 받은 메시지 + 서버 로그 한 쌍. 최종 화면 한 장만으로는 부족하다.
- **부수 효과도 본다.** 상한 기능이면 로그의 `gen_tokens`·`token_cap`·`ended_eot`가 1차 증거다. 메시지 텍스트만 보면 상한이 걸렸는지 알 수 없다.
- **기준선과 비교한다.** "안 바뀌어야 하는 것"(정상 final 텍스트)은 같은 음성을 기준선에도 보내 `compare_finals.py`로 확인한다.
- **가짜 엔진(`FAKE=1`) 결과는 모델 동작의 증거가 아니다.** 하네스와 메시지 형식만 증명한다.
- **AI Hub 음성·전사는 재배포 금지.** `client_*.json/.txt`, `clip_*.wav`, `--show-text` 출력은 GPU 서버 밖(PC, 저장소, 채팅, PR)으로 가져가지 않는다. 가져와도 되는 것: `stats_*.txt`, `compare_*.txt`(기본 출력은 길이·유사도만), `build.txt`, `healthz.json`, `server.log`(텍스트 없음).

## 5. Cleanup

```bash
$V/cleanup.sh $RUN_C
$V/cleanup.sh $RUN_B
```

`server.pid`의 프로세스만 끈다(15초 안에 안 꺼지면 그 pid만 `kill -9`). **`pkill`, `killall`, 포트로 찾아 끄기 금지** — 운영 8001이나 팀원 프로세스를 끌 수 있다. 끈 뒤 포트가 아직 응답하면 경고만 하고 손대지 않는다. 증거 파일은 지우지 않는다. 실패한 시도 뒤에도 매번 cleanup한다.

정리 확인: `curl -s localhost:8001/healthz` (운영이 그대로 200인지).

## 6. 격리 규칙

- 인스턴스 1개 = 포트 1개 = RUN 폴더 1개. 남이 띄운 인스턴스(포트 8002가 이미 사용 중 등)는 몰지 않는다.
- 기준선과 후보를 **동시에 띄워 둘 수는 있지만 동시에 몰지는 않는다.** GPU 1장을 나눠 써서 지연 비교가 무의미해진다. 지연을 볼 때는 하나씩 보낸다.
- 운영 8001과도 GPU를 나눠 쓴다. 지연 숫자에는 `curl -s localhost:8001/healthz`의 `sessions`를 같이 적는다.

## Helpers

| 스크립트 | 사용법 |
|---|---|
| `scripts/launch.sh` | `PY=$PY [REPO=<사본>] [FAKE=1] [ML_*=…] $V/launch.sh <PORT>` → `RUN=<폴더>` |
| `scripts/doctor.sh` | `$V/doctor.sh <RUN>` → OK/FAIL 목록, FAIL이면 종료 코드 1 |
| `scripts/cleanup.sh` | `$V/cleanup.sh <RUN>` → 그 pid만 종료, 증거 남김 |
| `scripts/token_stats.py` | `$PY $V/token_stats.py <server.log> [--trace ID] [--simulate R]…` → final/interim 토큰·상한·지연 집계. `token_cap` 위반이 있으면 종료 코드 1 |
| `scripts/compare_finals.py` | `$PY $V/compare_finals.py <BASE_RUN> <CAND_RUN> <client_x.json> [--show-text]` → 정상 final이 바뀌었으면 종료 코드 1 |
| `scripts/make_clip.py` | `$PY $V/make_clip.py <src.wav> <offset_s> <seconds> <out.wav>` 또는 `--synthetic "tone:2,sil:2.2" <out.wav>` |
| `scripts/fake_server.py` | `FAKE=1 launch.sh`가 부른다. 직접 부를 일 없음 |

`token_stats.py`는 journald 형식도 읽는다: `sudo journalctl -u marieum-ml --since today -o cat > ~/verify-runs/ops_today.log` (운영 로그 읽기는 읽기 전용이라 허용).

## 이 스킬의 검증 상태 (2026-11-01)

개발 PC(Windows, Git Bash, Python 3.10, websockets 15)에서 `FAKE=1`로 한 번 끝까지 돌렸다: 8001 거부(종료 코드 2) → launch(준비 확인) → 같은 포트 재launch 거부(3) → doctor OK → 코드 해시가 다르면 doctor FAIL(1) → [대화 모드](features/stream-conversation.md) 한 세션(`KJJ_p01.wav` 94초: final 13개, `is_last`는 마지막만, reason silence 12 + end 1, 확정분 줄어듦 0, errors 0) → POST /asr 200·빈 본문 400 → `token_stats.py`·`compare_finals.py` → cleanup(포트 닫힘, 증거 남음, 두 번째 cleanup은 "이미 정리됨").
`token_stats.py --simulate`는 2026-10-31 GPU 실험의 실제 `server.log`로 확인했다.

GPU 서버에서 아직 한 번도 안 돌린 것 (첫 실행 때 확인하고 이 절을 고친다):

- `~/mtk-cand`, `~/mtk-base` 사본 만들기와 tar/scp 전송
- 실제 모델 로딩: `ML_MODEL_DIR` 검사, 503 단계, 약 20초 준비 시간
- `PY` 자동 선택(`~/marieum-ml/.venv/bin/python`), Python 3.11·websockets 17.4에서 `stream_client.py`
- doctor의 `ss -ltnp` 포트 주인 확인 (PC에는 `ss`가 없어 SKIP)
- Linux에서 `kill`로 uvicorn 종료
- `compare_finals.py`가 실제로 다른 두 결과(기준선 vs 후보)를 가르는지 (PC에서는 같은 결과끼리만 비교)
- 실제 모델의 `token_cap` 검사 (PC 가짜 엔진에서는 ML_TOKEN_CAP_RATE=1·FLOOR=0으로 모든 interim/final이 상한에 닿고 위반 0을 확인)
