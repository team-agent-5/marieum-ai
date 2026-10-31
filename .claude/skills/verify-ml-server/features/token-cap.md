# 출력 토큰 상한

Whisper가 한 번 전사할 때 만들 수 있는 토큰 수에 조각마다 상한을 둔다. 상한은 말소리 길이에 비례하고(`cap = min(고정 상한, ceil(초 × R) + floor)`), 고정 상한(interim 96 / final 128)을 넘지 않는다. 사용자에게 보이는 효과는 두 가지다. 같은 말을 반복하는 폭주 출력이 짧게 끊겨 화면에 긴 쓰레기 글자가 덜 뜨고 다른 세션의 interim이 덜 밀린다. 정상 문장의 텍스트는 전혀 바뀌지 않는다. 설계 배경은 저장소 루트 `MAX_NEW_TOKENS.md`에 있다.

## Sub-features

- `cap-log`: interim·final 로그에 그 작업에 적용된 `token_cap`이 남고, interim에도 `ended_eot`가 남는다.
- `cap-short`: 짧은 조각의 상한이 고정값이 아니라 길이에 비례해 작다 (예: 말소리 0.5초 interim의 `token_cap`이 96보다 훨씬 작다).
- `cap-runaway`: 폭주한 조각이 자기 `token_cap`에서 멈춘다 (`gen_tokens == token_cap`, `ended_eot=false`, `token_cap < 128`).
- `cap-normal`: 끝 토큰으로 끝나던 정상 final은 텍스트가 기준선과 똑같다.
- `cap-long`: 15초 가까이 쉼 없이 말한 구간의 final이 상한에 잘리지 않는다 (`ended_eot=true`).
- `cap-batch`: 동시 세션의 조각이 한 GPU 묶음에 들어가도 각자 자기 상한을 지킨다 (어떤 로그에서도 `gen_tokens > token_cap`이 없다).
- `cap-file`: POST /asr의 구간 전사도 같은 상한을 쓴다.

## How to get to it (user POV)

- 앱이 실시간 대화 화면을 열면 백엔드가 `/asr-stream`에 `mode: "conversation"`으로 접속해 interim·final을 받는다.
- 앱이 한 번 말하기(버튼 누르고 말하기)를 쓰면 `/asr-stream`에 `mode: "utterance"`로 접속한다.
- 백엔드가 녹음 파일을 통째로 `POST /asr`에 올린다.

세 경로 모두 같은 GPU 작업자와 엔진을 거치므로, 세 경로 각각에서 증거를 남겨야 기능이 증명된다.

## Driving it with verify-ml-server

Preconditions:

- 후보(`RUN_C`, 포트 8002)와 기준선(`RUN_B`, 포트 8003)이 떠 있고 둘 다 doctor OK.
- `$AUD`에 `LJG_p01.wav LJG_p02.wav LSY_p01.wav BHN_p01.wav HJY_p02.wav`가 있다 (`ls $AUD | wc -l` → 42).
- 후보 로그에 `token_cap` 필드가 나온다 (아래 첫 단계에서 확인). 없으면 `cap-log` 실패로 보고하고 멈춘다.

폭주 재현 창 (2026-10-31 GPU 실험, 당시 설정 끝 무음 0.8초·구간 6초에서 final이 128토큰까지 반복된 자리):

| 이름 | 파일 | 당시 구간 | 보낼 창 |
|---|---|---|---|
| `LJG15` | `LJG_p01.wav` | seg 15, 말소리 3.7초, 94–97초 | `--offset 85 --max-seconds 20` |
| `LJG22` | `LJG_p01.wav` | seg 22, 5.5초(구간 상한 컷), 136–141초 | `--offset 128 --max-seconds 20` |
| `LSY20` | `LSY_p01.wav` | seg 20, 1.5초, 243–244초 | `--offset 232 --max-seconds 20` |
| `BHN50` | `BHN_p01.wav` | seg 50, 2.8초, 366–367초 | `--offset 355 --max-seconds 20` |
| `HJY26` | `HJY_p02.wav` | seg 26, 4.5초(구간 상한 컷), 81–86초 | `--offset 70 --max-seconds 25` |

`LJG_p02.wav`는 interim이 96에 닿은 자리가 7개(seg 12, 13, 15, 19, 29, 31, 84)라 폭주를 찾기 가장 좋은 파일이다.

- **cap-log.** 짧은 창 하나를 후보에 보낸다. Run `$PY ~/mtk-cand/scripts/stream_client.py --url ws://127.0.0.1:8002/asr-stream --mode conversation --wav $AUD/LSY_p01.wav --offset 232 --max-seconds 20 --out $RUN_C/client_LSY20.json > $RUN_C/client_LSY20.txt` 후 `grep -c '"token_cap"' $RUN_C/server.log`. 0보다 크고, `grep '"transcript_interim"' $RUN_C/server.log | grep -c '"ended_eot"'`도 0보다 크다.
- **기준선에서 폭주 재현.** 다섯 창을 기준선에 보낸다. Run `for c in "LJG15 LJG_p01 85 20" "LJG22 LJG_p01 128 20" "LSY20 LSY_p01 232 20" "BHN50 BHN_p01 355 20" "HJY26 HJY_p02 70 25"; do set -- $c; $PY ~/mtk-cand/scripts/stream_client.py --url ws://127.0.0.1:8003/asr-stream --mode conversation --wav $AUD/$2.wav --offset $3 --max-seconds $4 --out $RUN_B/client_$1.json > $RUN_B/client_$1.txt; done` 후 `$PY $V/token_stats.py $RUN_B/server.log | tee $RUN_B/stats_runaway.txt`. `상한에 닿음(ended_eot=false)`이 1개 이상이거나 interim `상한에 닿음`이 1개 이상이다. 둘 다 0이면 이 창들로는 재현이 안 된 것이다 → `LJG_p02.wav` 전체(`--max-seconds 0`)를 같은 방식으로 보내 본다.
- **cap-runaway.** 같은 다섯 창을 후보에 보낸다 (위 루프에서 포트를 8002, 폴더를 `$RUN_C`로). Run `$PY $V/token_stats.py $RUN_C/server.log | tee $RUN_C/stats_runaway.txt`. 기준선에서 상한에 닿았던 구간이 후보에서는 `gen=<token_cap>`이고 `cap`이 128(final)·96(interim)보다 작다. `[token_cap 검사] 위반 0개`.
- **cap-short.** Run `grep '"transcript_interim"' $RUN_C/server.log | $PY -c "import sys,json; r=[json.loads(l[l.find('{'):]) for l in sys.stdin]; s=[x for x in r if x['audio_ms']<1000]; print(len(s), max(x['token_cap'] for x in s) if s else None)"`. 1초 미만 interim이 1개 이상 있고 그중 최대 `token_cap`이 96보다 작다 (설계값 R=14, floor 8이면 22 이하).
- **cap-normal.** 다섯 창 각각을 비교한다. Run `for n in LJG15 LJG22 LSY20 BHN50 HJY26; do $PY $V/compare_finals.py $RUN_B $RUN_C client_$n.json | tee $RUN_C/compare_$n.txt; done`. 모든 줄이 `정상 final 바뀜 0`이고 `한쪽에만 있는 구간 0`이다. 바뀐 final은 `폭주 구간(바뀌는 게 목적)`뿐이다.
- **cap-long.** 말이 빠른 일반 화자 녹음으로 15초 구간을 만든다. Run `ls ~/marieum-ml/data/`로 `sample.wav`(일반 대화 녹음, 44.1kHz)를 찾고 `$PY ~/mtk-cand/scripts/stream_client.py --url ws://127.0.0.1:8002/asr-stream --mode conversation --wav ~/marieum-ml/data/sample.wav --max-seconds 120 --out $RUN_C/client_long.json > $RUN_C/client_long.txt` 후 `grep '"reason": "max_length"' $RUN_C/server.log`. `max_length`로 닫힌 final이 1개 이상 있고 모두 `"ended_eot": true`다. 기준선에도 같은 명령을 보내고 `compare_finals.py $RUN_B $RUN_C client_long.json`이 `정상 final 바뀜 0`이다.
- **cap-batch.** 동시 4세션. Run `$PY ~/mtk-cand/scripts/stream_client.py --url ws://127.0.0.1:8002/asr-stream --mode conversation --wav $AUD/LJG_p02.wav --sessions 4 --offset-step 30 --max-seconds 60 --out $RUN_C/client_batch.json > $RUN_C/client_batch.txt` 후 `$PY $V/token_stats.py $RUN_C/server.log --trace <client_batch.json의 trace_id 4개> | tee $RUN_C/stats_batch.txt`. `[token_cap 검사] 위반 0개`이고, `grep -c '"batch": [234]' $RUN_C/server.log`가 0보다 크다 (실제로 묶여 돌았다).
- **cap-file.** Run `$PY $V/make_clip.py $AUD/LSY_p01.wav 232 20 $RUN_C/clip_LSY20.wav` 후 `curl -s -X POST --data-binary @$RUN_C/clip_LSY20.wav -H 'Content-Type: audio/wav' "http://127.0.0.1:8002/asr?trace_id=verify-cap-file" -o $RUN_C/asr_LSY20.json -w '%{http_code}\n'`. 응답 200이고, `grep verify-cap-file $RUN_C/server.log`에 구간별 `gen_tokens`·`token_cap`이 남는다.
- **Proof.** Run `$PY $V/token_stats.py $RUN_C/server.log | tee $RUN_C/stats_all.txt` 그리고 `$PY $V/token_stats.py $RUN_B/server.log --simulate 14 | tee $RUN_B/stats_all.txt`. 후보의 `위반 0개`, 각 `compare_*.txt`의 `정상 final 바뀜 0`, 기준선 시뮬레이션의 `정상 잘림` 비율을 함께 보고한다.

## Gotchas

- **재현 창은 옛 설정(0.8초·6초)에서 찾은 자리다.** 지금 기본값(1.75초·15초)은 구간 경계가 달라 같은 폭주가 안 날 수 있다. 서버는 구간마다 따로 전사하므로(앞 구간 문맥 없음) 경계가 다르면 입력이 다르다. 기준선에서 재현이 안 되면 `cap-runaway`는 "증명 안 됨"으로 보고하고, 후보만 보고 통과라고 쓰지 않는다.
- **R을 환경변수로 받게 구현했다면** 아주 작은 값(예: 1)으로 띄운 인스턴스에서 정상 음성도 상한에 닿는지 보면 폭주 재현 없이 `cap-runaway`의 동작(`gen_tokens == token_cap`, `ended_eot=false`)을 결정적으로 확인할 수 있다. 이 인스턴스의 결과는 `cap-normal` 증거로 쓰지 않는다.
- **텍스트 비교는 단일 세션 결과로 한다.** fp16 묶음 처리는 같이 묶인 조각에 따라 수치가 아주 조금 달라질 수 있다. `cap-batch` 결과를 `compare_finals.py`에 넣으면 상한과 무관한 차이가 섞인다.
- **상한은 Whisper에 넣는 음성 길이로 계산한다.** final은 `speech_ms`(앞뒤 무음 자른 길이), interim은 `audio_ms`(스냅샷 길이)가 상한 기준이다. `vad_speech_ms`는 로그용이다. 공식 검산은 이 두 필드로 한다.
- **기준선 폭주 final은 신뢰도가 높다** (2026-10-31 실험에서 0.94~0.97). 신뢰도로는 폭주를 구분할 수 없으니 `ended_eot`·`gen_tokens`로 판단한다.
- **POST /asr 로그(`transcript_file`)에는 원래 구간별 `gen_tokens`가 없다.** 구현에서 구간별 로그를 넣지 않았다면 `cap-file`은 로그로 증명할 수 없다 → "증명 안 됨"으로 보고한다.
- **`--show`/`.txt` 출력과 `--show-text`에는 전사 텍스트가 있다.** GPU 서버 밖으로 가져가지 않는다.
- **interim은 덮어쓰기라 하나 놓쳐도 화면은 괜찮다.** 그래도 interim 폭주는 GPU 시간을 잡아먹어 다른 세션 지연을 늘린다. 지연 비교는 한 인스턴스씩, `latency_ms`의 p95·max로 한다.
