#!/usr/bin/env bash
# launch.sh가 띄운 프로세스만 끈다 (pkill/killall 금지: 같은 포트대의 팀원·운영 프로세스를 끌 수 있다).
# 증거(server.log, client_*.json, stats_*.txt 등)는 RUN 폴더에 그대로 남긴다.
#
#   cleanup.sh <RUN>
set -uo pipefail

RUN=${1:?사용법: cleanup.sh <RUN 폴더>}
if [ ! -f "$RUN/server.pid" ]; then
  echo "$RUN/server.pid 없음: 이미 정리됨"; exit 0
fi
PID=$(cat "$RUN/server.pid")
PORT=$(cat "$RUN/port")

if kill -0 "$PID" 2>/dev/null; then
  kill "$PID" 2>/dev/null || true
  for _ in $(seq 1 15); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
  if kill -0 "$PID" 2>/dev/null; then
    echo "15초 안에 안 꺼져서 강제 종료: pid $PID"; kill -9 "$PID" 2>/dev/null || true; sleep 1
  fi
fi

if curl -s -m 2 -o /dev/null "http://127.0.0.1:$PORT/livez"; then
  echo "경고: 포트 $PORT 이 아직 응답한다. 우리 pid($PID)는 꺼졌으니 다른 프로세스다. 끄지 말고 확인하라"
else
  echo "포트 $PORT 닫힘"
fi
mv "$RUN/server.pid" "$RUN/server.pid.stopped"
echo "stopped_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN/build.txt"
echo "증거 남김: $RUN"
ls -1 "$RUN"
