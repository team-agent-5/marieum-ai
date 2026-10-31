# 대화 모드 스트리밍

실시간 대화 화면에서 쓰는 경로다. 백엔드가 WebSocket으로 음성을 계속 보내면, 말하는 동안 interim(덮어쓰는 중간 결과)이 오고, 1.75초 이상 쉬면 그 구간의 final이 온다. 쉼 없이 15초를 넘게 말하면 마지막 5초 안의 가장 늦은 쉼에서 구간을 자른다. `end`를 보내면 남은 구간을 닫고 `is_last: true` final을 보낸다.

## Sub-features

- `conv-interim`: 새 말소리가 있을 때 약 0.5초마다 interim이 온다 (최소 간격 300ms, 말소리 300ms 이상부터).
- `conv-stable`: interim의 `stable_text`는 한 구간 안에서 줄어들지 않고, `stable_delta`는 새로 확정된 단어만 담는다.
- `conv-final-silence`: 1.75초 쉼마다 구간 final(`is_last: false`, 로그 `reason: silence`)이 온다.
- `conv-final-cap`: 15초를 넘는 연속 발화는 구간 상한에서 잘린다 (로그 `reason: max_length`).
- `conv-end`: `end` 뒤 마지막 final은 `is_last: true`다. 남은 말소리가 없으면 `text: ""`.

## How to get to it (user POV)

- 앱에서 실시간 대화 화면을 열면 백엔드가 `ws://<ML>/asr-stream`에 접속해 `{"type":"start","mode":"conversation",...}`을 보내고 PCM16 프레임을 보낸다.
- 화면을 닫으면 백엔드가 `{"type":"end"}`를 보낸다.

## Driving it with verify-ml-server

Preconditions:

- 후보 인스턴스(`RUN_C`, 8002)가 doctor OK.
- `$AUD/KJJ_p01.wav`(환자 음성, 약 1.6분)가 있다.

- **대화 한 세션.** Run `$PY ~/mtk-cand/scripts/stream_client.py --url ws://127.0.0.1:8002/asr-stream --mode conversation --wav $AUD/KJJ_p01.wav --show --out $RUN_C/client_conv.json | tee $RUN_C/client_conv.txt`. 마지막 요약에 `[errors] 0`, `finals=` 2개 이상.
- **interim과 확정분.** `client_conv.txt`에서 `interim seg=N` 줄의 `[확정|미확정]` 중 확정 부분이 같은 구간 안에서 짧아지지 않는다. 구간이 바뀌면(seg 번호 증가) 새로 시작한다.
- **구간 final.** Run `$PY -c "import json; s=json.load(open('$RUN_C/client_conv.json'))[0]; f=s['finals']; print([(m['segment_id'], m['is_last']) for m in f])"`. `segment_id`가 0부터 순서대로 늘고, 마지막만 `is_last: True`다.
- **닫힌 이유.** Run `grep '"transcript_final"' $RUN_C/server.log | grep -o '"reason": "[a-z_]*"' | sort | uniq -c`. `silence`가 대부분이고, 마지막 하나가 `end`다.
- **구간 상한.** 쉼 없는 긴 발화를 보낸다. Run `$PY ~/mtk-cand/scripts/stream_client.py --url ws://127.0.0.1:8002/asr-stream --mode conversation --wav ~/marieum-ml/data/sample.wav --max-seconds 120 --out $RUN_C/client_long.json > $RUN_C/client_long.txt` 후 `grep '"reason": "max_length"' $RUN_C/server.log`. `max_length` final이 있으면 `duration_ms`가 15000 이하다.
- **Proof.** `client_conv.json`, `client_conv.txt`, `server.log`, `$PY $V/token_stats.py $RUN_C/server.log | tee $RUN_C/stats_conv.txt`(지연 p50/p95)를 남긴다.

## Gotchas

- final은 말이 끝나고 약 1.75초 + GPU 0.1~0.4초 뒤에 온다. 그 전에 연결을 끊으면 마지막 구간을 잃는다. `stream_client.py`는 `is_last` final을 받을 때까지 기다린다.
- 환자 음성은 쉼이 많아 `max_length`가 잘 안 나온다. 구간 상한은 일반 화자 녹음(`sample.wav`)으로 확인한다. 파일이 없으면 "증명 안 됨"으로 보고한다.
- `stream_client.py`의 JSON에는 interim 메시지 본문이 없고 지연만 있다. interim 내용은 `--show` 출력(`.txt`)으로 본다.
- interim은 GPU가 밀리면 간격이 저절로 늘어난다 (세션마다 대기 1개만 둠). interim 개수로 성능을 판단하지 않는다.
- 음성이 44.1kHz면 클라이언트가 16kHz로 바꿔 보낸다 (torchaudio 필요, 운영 venv에 있음).
