#!/usr/bin/env bash
# 읽기 전용 점검: "이 인스턴스를 몰아도 되는가?" 하나라도 FAIL이면 종료 코드 1.
#
#   doctor.sh <RUN>        # launch.sh가 출력한 RUN 폴더
set -uo pipefail

RUN=${1:?사용법: doctor.sh <RUN 폴더>}
fail=0
ok() { echo "OK    $*"; }
bad() { echo "FAIL  $*"; fail=1; }

[ -f "$RUN/server.pid" ] || { echo "FAIL  $RUN/server.pid 없음 (이미 정리했거나 launch 실패)"; exit 1; }
PID=$(cat "$RUN/server.pid")
PORT=$(cat "$RUN/port")
REPO=$(sed -n 's/^repo: //p' "$RUN/build.txt")

if kill -0 "$PID" 2>/dev/null; then ok "프로세스 pid $PID 살아 있음"; else bad "프로세스 pid $PID 없음"; fi

if command -v ss >/dev/null 2>&1; then
  line=$(ss -ltnp 2>/dev/null | grep ":$PORT " || true)
  if echo "$line" | grep -q "pid=$PID,"; then ok "포트 $PORT 주인이 pid $PID"
  elif [ -z "$line" ]; then bad "포트 $PORT 에 아무도 없음"
  else bad "포트 $PORT 주인이 우리 프로세스가 아님: $line"; fi
else
  echo "SKIP  포트 주인 확인 (ss 없음. Windows PC 등)"
fi

body=$(curl -s -m 5 -w '\n%{http_code}' "http://127.0.0.1:$PORT/healthz" || true)
code=$(echo "$body" | tail -1)
json=$(echo "$body" | sed '$d')
if [ "$code" = 200 ] && echo "$json" | grep -q '"status": *"ok"'; then ok "healthz 200: $json"
else bad "healthz $code: $json"; fi

if [ -n "$REPO" ] && [ -f "$RUN/server.sha256" ]; then
  if (cd "$REPO" && sha256sum -c --quiet "$RUN/server.sha256" >/dev/null 2>&1); then
    ok "server/*.py가 띄울 때와 같음 ($REPO)"
  else
    bad "띄운 뒤 server/*.py가 바뀌었다. 지금 프로세스는 옛 코드다 → cleanup 후 다시 launch"
  fi
fi

echo "----- build.txt"; cat "$RUN/build.txt"
if [ "$PORT" != 8001 ]; then
  ops=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8001/healthz || true)
  echo "참고: 운영 8001 healthz=$ops (건드리지 않는다)"
fi
exit $fail
