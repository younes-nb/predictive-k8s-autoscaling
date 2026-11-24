# Readiness test harness (2026-11-16/17)

Bounded, synthetic tests used to close the remediation. None of them sends
traffic to TrainTicket and none uses the 25k/32k benchmark shape against the
cluster.

| Script | Purpose |
|---|---|
| `mock_server.py` | async TrainTicket stand-in (real HS256-shaped JWT). `PORT`, `LATENCY_MS`, `LOGIN_LATENCY_MS`, `FAIL_PCT`, `TOKEN_TTL` (test-only), `LOG`. |
| `blackhole.py` | TCP listener that accepts and never replies — simulates a wedged application connector for the liveness probe test. |
| `test_orch_sshtest.sh` | full `--local` orchestrator run with a poisoned `ssh`/`scp` in PATH (must be 0 invocations). |
| `test_orch_fast.sh` | same, short shape. |
| `test_orch_fail.sh` | k6 non-zero exit preservation + dead-collector → `incomplete`. |
| `test_leak.sh` | stale-collector cleanup + no leftover collectors/kubectl streams. |
| `test_vu_sizing.sh` | 2 999 pre-allocated VUs against the mock; records peak RSS/fds. |
| `test_population.sh` | renewal behaviour at N sessions with an accelerated token TTL. |
| `test_probe.sh` | ts-user-service liveness probe command vs healthy / blackhole / closed. |
| `audit_ssh.py` | static audit: no ssh on the `--local` path. |
| `ssh`, `scp` | poisoned stubs: log any invocation to /tmp/opencode/{ssh,scp}_calls.log and exit 255. |

Example:

```bash
export KUBECONFIG=/proj/k8sautoscaledl-PG0/src/cluster-access/admin.conf
bash benchmark/tests/test_orch_sshtest.sh
PORT=18100 TOKEN_TTL=3600 python3 benchmark/tests/mock_server.py &
bash benchmark/tests/test_vu_sizing.sh 3 prod
bash benchmark/tests/test_probe.sh
python3 benchmark/tests/audit_ssh.py
```
