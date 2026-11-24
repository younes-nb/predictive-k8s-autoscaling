#!/bin/bash
# P4.2 — validate the exact ts-user-service liveness probe command against
# three connector states:
#   healthy   (answers 200/403)  -> probe exit 0
#   blackhole (accept, no reply) -> probe exit 1   <- the Acceptor-wedge case
#   closed    (connection refused) -> probe exit 1
# No cluster object is touched.
set -u
cd /tmp/opencode
setsid python3 blackhole.py 12399 > /tmp/opencode/blackhole.log 2>&1 < /dev/null &
BH=$!
setsid python3 -m http.server 12397 --bind 127.0.0.1 > /dev/null 2>&1 < /dev/null &
HS=$!
sleep 2

probe() {  # port
  curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$1/" | grep -qE '^[1-5][0-9][0-9]$'
}

for state in "12397 healthy" "12399 blackhole" "12398 closed"; do
  set -- $state
  port=$1; name=$2
  probe "$port"; rc=$?
  echo "$name (port $port): probe exit=$rc  -> $([ $rc -eq 0 ] && echo 'PASS (connector answered)' || echo 'FAIL (no answer -> kubelet would restart)')"
done

echo "--- raw curl to the blackhole (should print 000 and no status):"
curl -s -o /dev/null -m 3 -w "http=%{http_code}\n" http://127.0.0.1:12399/ || echo "curl exit=$? (no HTTP status)"

kill $BH $HS 2>/dev/null
exit 0