#!/bin/sh
# ML 서버 8001 포트: localhost와 CPU(백엔드) 서버만 허용, 나머지는 차단 (CALIBRATION.md §3, §0.7)
# 여러 번 실행해도 같은 결과: 이 스크립트가 넣은 규칙(comment marieum-ml)을 지우고 다시 넣는다
set -e
BACKEND_IP="${BACKEND_IP:-10.1.1.11}"
PORT="${PORT:-8001}"

iptables -S INPUT | grep -- '--comment marieum-ml' | sed 's/^-A /-D /' | while read -r rule; do
  iptables $rule
done
iptables -I INPUT 1 -p tcp --dport "$PORT" -i lo -m comment --comment marieum-ml -j ACCEPT
iptables -I INPUT 2 -p tcp --dport "$PORT" -s "$BACKEND_IP" -m comment --comment marieum-ml -j ACCEPT
iptables -I INPUT 3 -p tcp --dport "$PORT" -m comment --comment marieum-ml -j DROP
