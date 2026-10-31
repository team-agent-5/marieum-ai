# 상태 확인

백엔드가 ML 서버를 쓸 수 있는지 판단하는 경로다. `/healthz`는 모델 로딩과 워밍업이 끝나야 200을 주고, 그 전에는 503과 이유를 준다. `/livez`는 프로세스가 살아 있으면 늘 200이다.

## Sub-features

- `health-loading`: 띄운 직후 `/healthz`는 503, `{"status": "loading", "message": "모델 로딩 중 (예상 20s)"}`.
- `health-ok`: 준비되면 200, `status: ok`, `models: ["asr"]`, `model_version.asr`, `gpu`(이름·메모리), `uptime_sec`, `sessions`, `gpu_pending`.
- `health-sessions`: 스트리밍 세션이 열려 있으면 `sessions`가 그 수만큼 늘었다가 끝나면 돌아온다.
- `live`: `/livez`는 `{"status": "alive"}`.

## How to get to it (user POV)

- 백엔드가 시작할 때와 주기적으로 `GET http://<ML>/healthz`를 부른다. 200이 아니면 ML 기능을 끈다.
- 프로세스 감시는 `GET /livez`.

## Driving it with verify-ml-server

Preconditions:

- 이 검증에서 새로 띄운 인스턴스가 있다 (`launch.sh`가 503 단계를 지나며 로그에 남긴다).

- **로딩 중 503.** Run `grep -c 'GET /healthz HTTP/1.1" 503' $RUN_C/server.log`. 실제 모델이면 1 이상이다 (`launch.sh`가 2초마다 물어봄).
- **준비 완료.** Run `cat $RUN_C/healthz.json`. `"status": "ok"`, `"models": ["asr"]`, `model_version.asr`가 `yoona-J/ASR_Whisper_Disease_General`, `gpu.name`이 GPU 이름이다.
- **세션 수.** 긴 세션을 뒤에서 돌리며 본다. Run `$PY ~/mtk-cand/scripts/stream_client.py --url ws://127.0.0.1:8002/asr-stream --mode conversation --wav $AUD/KJJ_p01.wav --sessions 2 --max-seconds 20 --out $RUN_C/client_health.json > $RUN_C/client_health.txt & sleep 5; curl -s 127.0.0.1:8002/healthz; wait; curl -s 127.0.0.1:8002/healthz`. 첫 응답 `"sessions": 2`, 둘째 응답 `"sessions": 0`.
- **livez.** Run `curl -s 127.0.0.1:8002/livez`. `{"status":"alive"}`.
- **Proof.** `healthz.json`, 위 curl 출력, 503 줄 수를 남긴다.

## Gotchas

- 모델 로딩이 실패하면 `/healthz`가 503 `status: error`로 머문다. `launch.sh`는 240초 뒤 포기한다. 로그의 `model_load_failed`를 본다.
- `gpu.memory_free_mb`는 10초마다 갱신된다. 운영 8001과 GPU를 같이 쓰므로 두 인스턴스를 띄우면 그만큼 줄어든다.
- `FAKE=1` 인스턴스는 바로 준비돼 503 단계가 거의 없고 `gpu.name`이 `fake`다.
