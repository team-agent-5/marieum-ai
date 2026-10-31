# 발화 모드 스트리밍

한 번 말하기 버튼에서 쓰는 경로다. 세션 하나가 발화 하나이고, 말하는 동안 interim이 오다가 `end`를 보내면 final이 한 번(`segment_id: 0`, `is_last: true`) 온다. 10초를 넘으면 `AUDIO_TOO_LONG` 에러와 함께 그때까지의 final이 오고 세션이 닫힌다.

## Sub-features

- `utt-final`: `end` 뒤 final 한 번, `segment_id: 0`, `is_last: true`.
- `utt-interim`: 말하는 동안 interim이 온다.
- `utt-too-long`: 10초 초과 시 `error` `AUDIO_TOO_LONG` + `is_last: true` final.

## How to get to it (user POV)

- 앱에서 말하기 버튼을 누르면 백엔드가 `/asr-stream`에 `{"type":"start","mode":"utterance",...}`(또는 `mode` 생략)를 보내고, 버튼을 떼면 `{"type":"end"}`를 보낸다.

## Driving it with verify-ml-server

Preconditions:

- 후보 인스턴스(`RUN_C`, 8002)가 doctor OK.
- `$AUD/KJJ_p01.wav`가 있다.

- **한 발화.** Run `$PY ~/mtk-cand/scripts/stream_client.py --url ws://127.0.0.1:8002/asr-stream --wav $AUD/KJJ_p01.wav --offset 10 --max-seconds 8 --show --out $RUN_C/client_utt.json | tee $RUN_C/client_utt.txt`. `.txt`에 `interim` 줄이 1개 이상, `final seg=0` 한 줄, 요약 `[errors] 0`.
- **final 형식.** Run `$PY -c "import json; f=json.load(open('$RUN_C/client_utt.json'))[0]['finals']; print(len(f), f[0]['segment_id'], f[0]['is_last'], sorted(f[0]))"`. `1 0 True`이고 키에 `confidence duration_ms model_version no_speech_prob stable_delta stable_revised text trace_id`가 있다.
- **10초 초과.** Run `$PY ~/mtk-cand/scripts/stream_client.py --url ws://127.0.0.1:8002/asr-stream --wav $AUD/KJJ_p01.wav --offset 10 --max-seconds 12 --out $RUN_C/client_utt_long.json > $RUN_C/client_utt_long.txt`. 요약 `[errors] 1 ['AUDIO_TOO_LONG']`, final 1개(`is_last: true`).
- **Proof.** 두 `client_utt*.json`과 `grep '"reason": "too_long"' $RUN_C/server.log` 결과를 남긴다.

## Gotchas

- 발화 모드는 세션 시작부터 음성을 쌓으므로, 앞쪽 무음은 interim 입력에서 잘라 낸다. offset을 말소리 중간으로 잡으면 첫 interim이 빨리 온다.
- 10초 초과 뒤 이어 오는 프레임은 에러 없이 버린다 (클라이언트가 바로 멈추지 않아도 된다).
- 같은 연결에서 이전 세션이 끝나기 전에 새 `start`를 보내면 오디오는 가장 최근 세션으로 간다. `stream_client.py`는 세션마다 연결을 따로 열어 이 문제가 없다.
