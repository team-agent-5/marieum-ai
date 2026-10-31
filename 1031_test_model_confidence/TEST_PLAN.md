# 구음장애 환자 음성 신뢰도 테스트 계획 (conversation 모드)

작성 2026-10-31. PR #2 서버(`yoona-J/ASR_Whisper_Disease_General`, `/asr-stream`) 기준. 이슈 #4, 브랜치 `exp/patient-confidence`.

## 1. 목적

- 학습 안 된 실제 환자 음성에서 서버의 **신뢰도(confidence)와 정확도**를 확인한다
- **사람이 느낀 난이도**와 신뢰도가 같이 움직이는지 본다
- conversation 모드의 **구간 final과 interim 신뢰도 기준값**을 정할 근거를 만든다 (v1.0의 interim 0.3 / final 0.4를 다시 정함)
- 0.8초 쉼 기준으로 구간이 **문장 중간에서 끊기는 비율**을 본다 (특히 `음절끊김` 화자)

## 2. 시나리오

- 앱의 실시간 대화 화면처럼, 환자가 계속 말하는 녹음을 **conversation 모드 세션**으로 실제 속도로 흘려 보낸다
- 서버가 0.8초 쉼(또는 쉼 없이 6초)마다 구간 final을 보내고, 말하는 동안 interim을 보낸다
- 세션은 최대 600초라서, 녹음을 **600초 이하 부분(part)**으로 나눠 부분 1개 = 세션 1개로 보낸다
- 라벨에 타임스탬프가 없다(파일 전체 대본만 있음). 그래서 구간별 정답은 출력과 대본을 글자 단위로 정렬해서 추정한다
- utterance 모드는 이번에 하지 않는다. 일부러 넣는 무음·잡음 환각 검사도 하지 않는다

## 3. 운영 서버 보호

GPU 서버(`inno-gpu-006`, NCP)의 운영 서비스는 건드리지 않는다.

| 항목 | 운영 (그대로 둠) | 실험 |
|---|---|---|
| 폴더 | `~/marieum-ml` (systemd, `Restart=always`) | `~/exp-1031` (scp로 따로 올림) |
| 주소 | `0.0.0.0:8001`, 방화벽으로 백엔드만 허용 | **`127.0.0.1:8002`** (비어 있을 때. 방화벽이 8001만 막으므로 `0.0.0.0` 금지) |
| 서버 코드 | `main`과 같음 | 운영 폴더의 `server/`를 복사해서 그대로 씀 (고치지 않음) |
| venv, 모델 | `~/marieum-ml/.venv`, `~/marieum-ml/models/...` | 같은 것을 읽기만 함 |
| 로그 | journald | `~/exp-1031/server.log` |

- GPU 메모리는 약 2GB 더 쓴다 (L4 23GB). GPU 칩은 같이 쓰므로, 실험 중 30초마다 8001 `/healthz`의 `sessions`를 기록해 운영 트래픽과 겹쳤는지 표시한다
- **8002는 팀이 같이 쓰는 시험 포트다.** PR #3 작성자도 수정한 서버를 8002에 띄워 검증했다
  - 시작 전에 8002가 비어 있는지 확인한다. 누가 쓰고 있으면 끄지 말고 그 사람에게 묻거나, 비어 있는 다른 포트(예: 8012)를 쓴다 (6절의 `PORT` 변수)
  - 실험 시작과 끝을 팀 채널에 알린다
  - 끌 때는 `pkill`을 쓰지 않는다. 띄울 때 저장한 PID(`server.pid`)로 **우리 프로세스만** 끈다
- 끝나면 우리 프로세스를 끄고 8001 `/healthz`가 ok인지 확인한다
- **서버 코드 버전:** PR #3이 운영에 반영되기 전 코드는 VAD 모델을 미리 1개, 반영 후 코드는 4개 만든다. 동시 4세션에서 세션 시작 지연이 조금 다를 수 있고, 인식 정확도와 신뢰도에는 영향이 거의 없다. 복사한 코드의 sha256과 VAD 미리 만들기 개수를 `server_sha.txt`에 남기고 보고서에 적는다
- **데이터:** AI Hub 오디오·전사는 재배포 금지다. 저장소에는 `raw/`(wav, `라벨링데이터/`)와 `data/`가 올라가지 않는다(`.gitignore`). GPU 서버에는 변환한 오디오만 올리고 정답 전사는 올리지 않는다

## 4. 데이터 (22명, 약 300분)

위치: `raw\원본데이터\<질환>\*.wav`, `raw\라벨링데이터\<질환>\*.json` (전사는 `Transcript`)

| 묶음 | 화자 (점수, 질환, 태그) |
|---|---|
| **쉬움 (0~2점), 8명** | LOS(0, 25), KJJ(1, 12, 음절끊김), CJK(1, 21), JJH(2, 21, 음절끊김), LGS(2, 25, 단조억양), PAS(2, 25, 음절끊김), AJH(2, 26 학습됨, 음절끊김), LUA(2, 28, 음절끊김) |
| **어려움 (3~5점), 14명** | HHS(5, 21), HJY(5, 21), KSW(5, 25), KMH(4, 21), LJG(4, 21), LSA(3, 25), LSY(3, 25), OTK(3, 25), BDS(3, 26 학습됨), BHL(3, 26 학습됨), BJS(3, 26 학습됨), BHN(3, 26 학습됨, ㅅ→th 발음), HMS(3, 28), KJM(3, 28) |
| **제외** | LMB(소리 작음), BDI(말 잘함), CBC. 파일은 `raw`에 두고 보내지 않는다 |

질환 코드: 12 뇌부상, 21 조음, 25 언어+뇌신경, 26 청각+뇌신경, 28 감음신경성

- **점수:** 평가자 1명(비전문가)이 들어보고 매긴 주관 점수(0~5)다. 0점은 장애가 거의 느껴지지 않음을 뜻한다. 분석은 두 묶음으로 한다
- **학습 여부:** 26의 파일은 논문 학습 목록(`preprocessing.csv`)에 그대로 들어 있다. 나머지 화자는 화자도 파일도 학습에 없다
- **같은 대본(02-01) 비교:** 학습 화자 AJH·BDS·BHL·BJS / 미학습 화자 OTK·PAS·KJM·LUA·KJJ
- **길이:** BHN(126분)과 HHS(135분)는 앞 600초(부분 1개)만 쓴다. 나머지는 전체. 합계 약 300분, 동시 4세션 실제 속도로 약 75~80분

### 화자별 메모

| 화자 | 파일 | 길이 | 메모 |
|---|---|---|---|
| LOS | ID-02-25-N-LOS-03-F-76-GS | 13.1분 | 말을 잘함. 03 준자유(자기소개)라 대본이 화자마다 다름 |
| KJJ | ID-01-12-N-KJJ-02-01-F-66-KK | 1.6분 | 음절 단위로 애매하게 끊어 말함. 스테레오. 학습된 뇌졸중 KJJ(남 63)와는 다른 사람 |
| CJK | ID-02-21-N-CJK-04-M-61-GS | 15.2분 | 들을만함. 스테레오 |
| JJH | ID-02-21-N-JJH-02-03-M-29-CC2 | 6.6분 | 음절끊김 |
| LGS | ID-02-25-N-LGS-02-03-F-75-GS | 39.7분 | 높낮이 없이 한 음으로 늘려 말함 |
| PAS | ID-02-25-N-PAS-02-01-F-84-JL | 13.7분 | 음절끊김 |
| AJH | ID-02-26-N-AJH-02-01-F-45-SU2 | 13.8분 | 음절끊김. 학습 파일 |
| LUA | ID-02-28-N-LUA-02-01-F-47-KK3 | 13.8분 | 음절끊김 |
| HHS | ID-02-21-N-HHS-06-01-M-30-SU | 135.0분 | 듣기 힘듦. 앞 600초만 |
| HJY | ID-02-21-N-HJY-04-F-13-GS | 13.1분 | 듣기 힘듦 |
| KSW | ID-02-25-N-KSW-07-F-31-JL | 5.9분 | 듣기 힘듦 |
| KMH | ID-02-21-N-KMH-02-03-M-26-GS2 | 6.2분 | 듣기 힘듦, 5점보다는 나음 |
| LJG | ID-02-21-N-LJG-52-04-M-44-KK | 19.0분 | 듣기 힘듦. 검사 코드 52-04(코드표에 없음), 32-bit float(`wave` 모듈로 안 열림, `soundfile` 사용), 라벨에 마침표 거의 없음 |
| LSA | ID-02-25-N-LSA-03-F-60-JL | 13.2분 | 라벨이 5문장뿐(녹음 13분 중 일부만 전사) |
| LSY | ID-02-25-N-LSY-06-02-F-60-JL | 26.2분 | |
| OTK | ID-02-25-N-OTK-02-01-M-30-SU | 11.8분 | |
| BDS, BHL, BJS | ID-02-26-N-…-02-01 | 각 13.6~13.7분 | 학습 파일 |
| BHN | ID-02-26-N-BHN-07-F-25-SU | 125.9분 | 목소리는 우렁찬데 혀가 짧음, 'ㅅ'을 'th'로 발음. 학습 파일. 앞 600초만 |
| HMS | ID-02-28-N-HMS-03-M-25-JL | 13.1분 | 00:20~00:23에 안내원 목소리. 스테레오 |
| KJM | ID-02-28-N-KJM-02-01-M-99-SU1 | 11.7분 | |

## 5. 절차

파일 위치 (`data/`는 git에 올라가지 않음):

```
1031_test_model_confidence/
  TEST_PLAN.md
  scripts/prepare.py           # 4.1 로컬
  scripts/run_conversation.py  # 4.2 GPU 서버
  scripts/analyze.py           # 4.3 로컬
  data/audio16k/<화자>_p01.wav …  # GPU 서버로 올림
  data/parts.json              # 부분 목록 (정답 없음, GPU 서버로 올림)
  data/refs.json               # 화자별 대본 (로컬에만)
  data/gpu/                    # GPU 서버에서 받아 온 결과
  data/segments.json, data/report_segments.md  # 분석 결과 (정답 포함)
  REPORT.md                    # 집계 보고서 (정답 전문 없음, 커밋 가능)
```

### 5.1 변환과 나누기 (로컬, `prepare.py`)

- 모두 16kHz 모노 16-bit로 바꾼다 (스테레오는 평균, LJG는 float에서 변환)
- 600초 이하 부분으로 나눈다. 590초 근처에서 끊지 않고, 540~590초 사이에서 가장 긴 쉼(에너지 기준)의 가운데에서 자른다
- BHN, HHS는 첫 부분만 만든다
- `parts.json`: 부분마다 `speaker`, `part`, `src`, `src_start_s`, `src_end_s`, `wav`
- `refs.json`: 화자마다 `Transcript`, 정렬 방식(`align`, 5.3), 메타(점수, 묶음, 태그, 질환, 학습 여부)

### 5.2 서버 전송 (GPU 서버, `run_conversation.py`)

- `ws://127.0.0.1:8002/asr-stream`, conversation 모드, 부분 1개 = 연결 1개 = 세션 1개
- 100ms 프레임을 실제 속도로, 동시 4세션 (대기열에서 4개씩 꺼내 보냄)
- `trace_id`는 `exp1031-<화자>-p<부분>`, `patient_id`는 `exp1031`
- 받은 메시지를 **전부** 도착 시각과 함께 세션마다 jsonl로 저장한다 (interim, final, error)
- 녹음 중 긴 쉼이 있을 수 있으므로 `stream_client.py`처럼 30초 무응답으로 끊지 않는다. `is_last: true` final을 받을 때까지 기다린다 (`end` 후 30초 넘으면 타임아웃 기록)
- interim 지연은 `audio_end_ms`로 클라이언트에서 계산한다 (`stream_client.py`와 같은 방식)
- 30초마다 `127.0.0.1:8001/healthz`의 `sessions`, `gpu_pending`을 `ops_8001.jsonl`에 기록한다

### 5.3 정답 붙이기와 분석 (로컬, `analyze.py`)

- **서버 로그 합치기:** final 메시지에는 구간이 끊긴 이유가 없으므로 `server.log`의 `transcript_final`을 `trace_id` + `segment_id`로 붙인다 (`reason`: silence / max_length / end, `latency_ms`, `speech_ms`, `vad_skip`, `gen_tokens`, `ended_eot`)
- **정규화:** 학습 전처리와 같이 `+ * ( ) ? ! , . ~ - '`와 공백을 뺀다
- **정렬:** 화자의 모든 부분의 구간 final을 순서대로 이어 붙여 대본과 글자 단위 편집거리로 정렬하고, 구간마다 대응하는 대본 부분(`ref_span`)을 잘라 낸다

  | `align` | 대상 | 방식 |
  |---|---|---|
  | `global` | 대부분 | 출력 전체 ↔ 대본 전체 |
  | `ref_prefix` | BHN, HHS (앞 600초만) | 대본 뒤쪽은 비용 없이 남겨도 됨 |
  | `ref_inside` | LSA, LJG (라벨이 녹음 일부만) | 대본 전체가 출력 안 어딘가에 들어감, 출력 앞뒤는 비용 없음. 보고서에 "추정" 표시 |

- **구간 CER** = 편집거리(구간 출력, `ref_span`) / `ref_span` 글자 수. `ref_span`이 비면 "정답 없음"으로 따로 센다
- **문장 중간 끊김:** 대본의 `. ? !` 위치를 문장 끝으로 본다. 구간 경계가 문장 끝에서 2글자보다 멀면 문장 중간에서 끊긴 것으로 센다 (LJG는 마침표가 거의 없어 제외)
- **interim 요약:** 구간마다 interim 수, 첫·마지막·최소 interim 신뢰도, 마지막 interim의 `stable_text`가 final과 다른지

### 5.4 결과 json (구간 1개당, `data/segments.json`)

```json
{"speaker": "OTK", "disease": "25", "test": "02-01", "trained": false,
 "score": 3, "group": "hard", "tags": [],
 "part": 1, "src_start_s": 0.0, "segment_id": 7, "reason": "silence",
 "duration_ms": 3120, "speech_ms": 2800,
 "hyp": "…", "confidence": 0.97, "no_speech_prob": 0.0, "vad_skip": false,
 "ref_span": "…", "align": "global", "cer": 0.0, "mid_sentence_cut": false,
 "interim_count": 5, "interim_conf_first": 0.81, "interim_conf_last": 0.95, "interim_conf_min": 0.62,
 "stable_changed": false, "final_latency_ms": 180, "ops_8001_sessions": 0, "error": null}
```

## 6. GPU 서버 실행 순서 (사용자가 직접)

SSH가 끊겨도 계속 돌도록 `tmux` 안에서 실행한다.

```bash
# 로컬 → GPU 서버
scp -r 1031_test_model_confidence/scripts 1031_test_model_confidence/data/audio16k 1031_test_model_confidence/data/parts.json ncloud@10.1.2.11:~/exp-1031/

# GPU 서버: 운영 코드를 복사하고 버전 기록 (PR #3 반영 여부 = VAD 미리 만들기 1개/4개)
cd ~/exp-1031
cp -r ~/marieum-ml/server .
sha256sum server/*.py > server_sha.txt
if grep -q "vads.free.extend" server/app.py; then echo "vad_preload=4 (PR #3 반영)"; else echo "vad_preload=1 (PR #3 이전)"; fi | tee -a server_sha.txt
curl -s localhost:8001/healthz; nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# GPU 서버: 포트가 비어 있는지 확인. 무언가 나오면 끄지 말고 주인에게 묻거나 PORT를 바꾼다
PORT=8002
ss -ltnp | grep ":$PORT " || echo "$PORT free"

# GPU 서버: 실험 서버 (127.0.0.1만). PID를 저장해 두고 끌 때 이것만 끈다
ML_MODEL_DIR=$HOME/marieum-ml/models/ASR_Whisper_Disease_General \
  nohup ~/marieum-ml/.venv/bin/uvicorn server.app:app --host 127.0.0.1 --port $PORT --workers 1 > server.log 2>&1 &
echo $! > server.pid
until curl -sf localhost:$PORT/healthz; do sleep 2; done

# GPU 서버: 전송 (약 75~80분)
~/marieum-ml/.venv/bin/python scripts/run_conversation.py --url ws://127.0.0.1:$PORT/asr-stream \
  --parts parts.json --out results --sessions 4

# GPU 서버: 정리 (pkill 금지: 같은 포트를 쓰는 팀원 프로세스까지 끌 수 있음)
kill "$(cat server.pid)"
ss -ltnp | grep ":$PORT " || echo "$PORT closed"
curl -s localhost:8001/healthz

# GPU 서버 → 로컬
scp -r "ncloud@10.1.2.11:~/exp-1031/{results,server.log,server_sha.txt}" 1031_test_model_confidence/data/gpu/
```

## 7. 보고서 항목

`REPORT.md`에는 집계만 넣고 대본 문장 전문은 넣지 않는다. 구간별 표(정답 포함)는 `data/report_segments.md`에 둔다.

1. **구간별 표** (`data/report_segments.md`): 정답 / 출력 / 신뢰도 / 오류율 / 끊긴 이유
2. **신뢰도 구간별 평균 오류율:** 0.9 이상, 0.8~0.9, 0.6~0.8, 0.6 미만. 구간 final과 interim(마지막·최소) 각각
3. **기준값 후보 표:** 기준 t = 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9마다 t 미만 구간 비율, t 미만·이상 평균 오류율
4. **확신하며 틀린 구간:** 신뢰도 0.8 이상인데 오류율 30% 이상
5. **비교:** 쉬움과 어려움, 태그별(음절끊김·단조억양·조음), 02-01의 학습 화자와 미학습 화자. 화자 단위 CER(출력 전체 ↔ 대본 전체)도 같이
6. **구간 나누기:** 화자별 구간 수, 구간 길이 분포, 끊긴 이유(silence / max_length) 비율, 문장 중간 끊김 비율 (음절끊김 화자 vs 나머지)
7. **서버 동작**
   - 말했는데 결과가 빈 구간 (`vad_skip`이나 `no_speech_prob > 0.6`에 막힌 경우)
   - 정답 없는 구간(기침, 숨, HMS 안내원 목소리, 라벨 밖 구간)의 출력과 신뢰도
   - interim의 `stable_text`가 final에서 바뀐 비율
8. **지연 시간:** interim p50·p95(클라이언트), 구간 final `latency_ms` p50·p95(서버 로그). HANDOFF §7의 conversation 실제 발화 4세션(interim 117 / 132ms)과 비교. 8001에 운영 세션이 있던 시간은 따로 표시. 서버 코드 버전(`server_sha.txt`의 `vad_preload`)을 적고, 세션 첫 interim은 VAD 준비 지연이 섞일 수 있어 따로 본다
9. **한계**
   - 26은 학습 파일이라 상한선 참고용이다
   - 구간별 정답은 정렬로 추정한 것이다. 정렬이 틀리면 경계 근처 오류율이 실제보다 나쁘게 나온다
   - LSA와 LJG는 라벨이 녹음 일부만 덮어 `ref_inside`로 추정했다
   - 뇌부상은 1명(1.6분)뿐이라 경향을 말할 수 없다
   - 난이도는 평가자 1명의 주관 점수다
   - 서버 응답에 `min_token_prob`이 없다 (`no_speech_prob`은 final에 있음)
   - 녹음은 대본을 읽는 검사 상황이라 실제 대화와 쉼 패턴이 다를 수 있다
   - 대본(02-01, 02-03, 04)은 모든 화자가 같아서, 모델이 대본 문장을 알고 있을 수 있다
   - 8002는 운영 8001과 GPU 칩을 같이 쓴다
   - 서버 코드는 실험 당시 운영 코드를 복사한 것이다. PR #3 반영 전후로 세션 시작 지연이 조금 다를 수 있다

## 8. 다음 단계

1. 5.1 `prepare.py` 작성과 실행 (로컬)
2. 5.2 `run_conversation.py` 작성. 로컬에서 Mock 서버(`scripts/mock_asr_server.py`, 9001)로 먼저 확인
3. GPU 서버 실행과 결과 회수는 사용자가 직접 (6절)
4. 5.3 `analyze.py` 작성, 보고서 md 작성
