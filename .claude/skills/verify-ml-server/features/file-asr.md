# 파일 전사 POST /asr

녹음 파일을 한 번에 올려 전사 결과를 받는 경로다. 본문은 WAV(PCM 8/16/24/32비트, 채널·샘플레이트 무관, 8–48kHz) 또는 PCM16 LE 모노 + `?sample_rate=`. 서버는 대화 모드와 같은 기준(1.75초 쉼, 구간 15초)으로 나눠 구간마다 전사하고, 전체 텍스트·신뢰도·구간별 시작/끝을 돌려준다. 60초까지 받는다.

## Sub-features

- `file-ok`: WAV를 올리면 200과 `text`, `confidence`, `duration_ms`, `segments[]`(`start_ms`, `end_ms`)를 받는다.
- `file-empty`: 본문이 비면 400 `INVALID_FORMAT`.
- `file-too-long`: 60초를 넘으면 413 `AUDIO_TOO_LONG`.
- `file-raw-no-rate`: WAV가 아닌 바이트를 `sample_rate` 없이 보내면 415 `INVALID_FORMAT`.

## How to get to it (user POV)

- 백엔드가 `POST http://<ML>/asr?trace_id=<id>`에 `Content-Type: audio/wav`로 파일 본문을 보낸다. `trace_id`는 `X-Trace-Id` 헤더로도 줄 수 있다.

## Driving it with verify-ml-server

Preconditions:

- 후보 인스턴스(`RUN_C`, 8002)가 doctor OK.
- `$AUD/LSY_p01.wav`가 있다.

- **정상 업로드.** Run `$PY $V/make_clip.py $AUD/LSY_p01.wav 232 20 $RUN_C/clip_file.wav` 후 `curl -s -X POST --data-binary @$RUN_C/clip_file.wav -H 'Content-Type: audio/wav' "http://127.0.0.1:8002/asr?trace_id=verify-file-ok" -o $RUN_C/asr_file.json -w '%{http_code}\n'`. `200`이 찍힌다.
- **응답 형식.** Run `$PY -c "import json; b=json.load(open('$RUN_C/asr_file.json')); print(b['trace_id'], b['duration_ms'], len(b['segments']), [(s['start_ms'], s['end_ms']) for s in b['segments']])"`. `verify-file-ok`, `duration_ms`가 약 20000, 구간 1개 이상, `start_ms < end_ms`이고 순서대로 늘어난다.
- **서버 로그.** Run `grep verify-file-ok $RUN_C/server.log`. `"event": "transcript_file"` 한 줄에 `segments`, `text_segments`, `latency_ms`가 있다.
- **빈 본문.** Run `curl -s -X POST --data-binary '' "http://127.0.0.1:8002/asr?trace_id=verify-file-empty" -w '\n%{http_code}\n'`. `400`과 `"code": "INVALID_FORMAT"`.
- **60초 초과.** Run `$PY $V/make_clip.py $AUD/LSY_p01.wav 0 65 $RUN_C/clip_65s.wav` 후 `curl -s -X POST --data-binary @$RUN_C/clip_65s.wav -H 'Content-Type: audio/wav' "http://127.0.0.1:8002/asr?trace_id=verify-file-long" -w '\n%{http_code}\n'`. `413`과 `"code": "AUDIO_TOO_LONG"`.
- **WAV 아님.** Run `head -c 32000 /dev/urandom > $RUN_C/raw.bin; curl -s -X POST --data-binary @$RUN_C/raw.bin "http://127.0.0.1:8002/asr?trace_id=verify-file-raw" -w '\n%{http_code}\n'`. `415`와 `"code": "INVALID_FORMAT"`.
- **Proof.** `asr_file.json`, 각 curl의 상태 코드 출력, `grep '"/asr"' $RUN_C/server.log` 결과를 남긴다.

## Gotchas

- 긴 파일 하나가 구간 수만큼 GPU 작업을 한꺼번에 넣는다. 스트리밍 세션과 같이 돌리면 그 세션 지연이 튄다. 지연 측정 중에는 업로드하지 않는다.
- 응답 `segments`에는 텍스트가 있는 구간만 들어간다. 로그의 `segments`(전체 구간 수)와 다를 수 있다.
- 지금 `transcript_file` 로그에는 구간별 `gen_tokens`가 없다. 토큰 상한 증명에는 쓸 수 없다 ([token-cap.md](./token-cap.md)의 `cap-file`).
