#!/usr/bin/env bash
# 검증용 ML 서버 인스턴스 1개를 127.0.0.1:<PORT>에 띄우고 /healthz 200(status ok)까지 기다린다.
#
#   launch.sh [PORT]                  # 기본 8002. 8001(운영)은 거부한다
#   REPO=~/mtk-base launch.sh 8003    # 다른 코드 사본으로 띄우기 (기준선 비교)
#   FAKE=1 launch.sh 18002            # GPU 없는 PC: 가짜 엔진으로 하네스만 점검 (모델 동작 증명 아님)
#
# 환경변수: REPO(띄울 코드, 기본 이 스킬이 든 저장소), VERIFY_ROOT(기본 ~/verify-runs), RUN_ID,
#           PY(파이썬, 기본 ~/marieum-ml/.venv/bin/python → python3 → python), ML_MODEL_DIR, ML_* 설정
# 마지막 줄에 RUN=<증거 폴더>를 출력한다. 이후 doctor.sh / cleanup.sh에 그 폴더를 넘긴다.
set -euo pipefail

SKILL=$(cd "$(dirname "$0")/.." && pwd)
REPO=${REPO:-$(cd "$SKILL/../../.." && pwd)}
REPO=$(cd "$REPO" && pwd)
PORT=${1:-8002}

if [ "$PORT" = 8001 ]; then
  echo "8001은 운영 인스턴스(marieum-ml.service) 포트다. 다른 포트를 써라" >&2; exit 2
fi
if [ ! -f "$REPO/server/app.py" ]; then
  echo "REPO=$REPO 에 server/app.py가 없다" >&2; exit 2
fi

if [ -z "${PY:-}" ]; then
  for c in "$HOME/marieum-ml/.venv/bin/python" python3 python; do
    if command -v "$c" >/dev/null 2>&1; then PY=$c; break; fi
  done
fi

# 포트가 비어 있는지: 접속되면 누가 쓰는 중이다. 끄지 말고 다른 포트를 고른다
if "$PY" - "$PORT" <<'EOF'
import socket, sys
p = int(sys.argv[1])
s = socket.socket()
s.settimeout(1)
busy = s.connect_ex(("127.0.0.1", p)) == 0
s.close()
if not busy:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", p))
    except OSError:
        busy = True
    finally:
        s.close()
sys.exit(0 if busy else 1)
EOF
then
  echo "포트 $PORT 이 사용 중이다. 끄지 말고 다른 포트를 골라라" >&2
  command -v ss >/dev/null 2>&1 && ss -ltnp 2>/dev/null | grep ":$PORT " >&2 || true
  exit 3
fi

if [ -z "${FAKE:-}" ]; then
  MD=${ML_MODEL_DIR:-models/ASR_Whisper_Disease_General}
  case "$MD" in /*) ABS_MD=$MD ;; *) ABS_MD=$REPO/$MD ;; esac
  if [ ! -d "$ABS_MD" ]; then
    echo "모델 폴더가 없다: $ABS_MD  (GPU 서버: ML_MODEL_DIR=\$HOME/marieum-ml/models/ASR_Whisper_Disease_General)" >&2
    exit 4
  fi
  export ML_MODEL_DIR=$ABS_MD
fi

RUN_ID=${RUN_ID:-$(date +%m%d-%H%M%S)-p$PORT}
RUN=${VERIFY_ROOT:-$HOME/verify-runs}/$RUN_ID
mkdir -p "$RUN"
if [ -f "$RUN/server.pid" ] && kill -0 "$(cat "$RUN/server.pid")" 2>/dev/null; then
  echo "$RUN 의 서버가 이미 돌고 있다 (pid $(cat "$RUN/server.pid"))" >&2; exit 5
fi

{
  echo "run_id: $RUN_ID"
  echo "repo: $REPO"
  echo "port: $PORT"
  echo "fake: ${FAKE:-0}"
  echo "python: $PY ($("$PY" --version 2>&1))"
  echo "launched_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if git -C "$REPO" rev-parse HEAD >/dev/null 2>&1; then
    echo "git: $(git -C "$REPO" rev-parse --abbrev-ref HEAD) $(git -C "$REPO" rev-parse --short HEAD)"
    echo "git_dirty_server: $(git -C "$REPO" status --short -- server | tr '\n' ' ')"
  else
    echo "git: (없음. 파일 해시로 판단)"
  fi
  echo "env:"; env | grep '^ML_' | sort | sed 's/^/  /' || true
  echo "server_sha256:"
  (cd "$REPO" && sha256sum server/*.py | sed 's/^/  /')
} > "$RUN/build.txt"
(cd "$REPO" && sha256sum server/*.py) > "$RUN/server.sha256"
echo "$PORT" > "$RUN/port"

cd "$REPO"
if [ -n "${FAKE:-}" ]; then
  REPO=$REPO PYTHONUNBUFFERED=1 nohup "$PY" "$SKILL/scripts/fake_server.py" --port "$PORT" > "$RUN/server.log" 2>&1 &
else
  PYTHONUNBUFFERED=1 nohup "$PY" -m uvicorn server.app:app --host 127.0.0.1 --port "$PORT" --workers 1 \
    > "$RUN/server.log" 2>&1 &
fi
PID=$!
echo "$PID" > "$RUN/server.pid"

# 준비 = /healthz 200 + status ok (모델 로딩과 워밍업이 끝나야 200이 된다). 실제 모델은 약 20초
for _ in $(seq 1 120); do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "서버가 시작 중에 죽었다. 로그 끝:" >&2; tail -20 "$RUN/server.log" >&2; exit 6
  fi
  code=$(curl -s -o "$RUN/healthz.json" -w '%{http_code}' "http://127.0.0.1:$PORT/healthz" || true)
  if [ "$code" = 200 ] && grep -q '"status": *"ok"' "$RUN/healthz.json"; then
    echo "ready: $(cat "$RUN/healthz.json")"
    echo "RUN=$RUN"
    exit 0
  fi
  sleep 2
done
echo "240초 안에 준비되지 않았다. cleanup.sh $RUN 으로 끄고 로그를 봐라" >&2
tail -20 "$RUN/server.log" >&2
exit 7
