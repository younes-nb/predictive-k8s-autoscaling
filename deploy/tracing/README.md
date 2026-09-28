# Distributed tracing (Tempo + Istio, no app changes)

Mesh sidecars emit OTLP spans; Tempo stores them (48h, emptyDir).

Apply in order:
  kubectl apply -f deploy/tracing/tempo.yaml                 # Tempo (monitoring ns)
  kubectl -n istio-system patch configmap istio --type merge \
    --patch-file deploy/istio/proxy-stats-matcher-patch.json # (queues; also keep)
  # extensionProviders.otel-tempo must exist in the istio ConfigMap mesh:
  #   extensionProviders:
  #   - name: otel-tempo
  #     opentelemetry: {port: 4317, service: tempo.monitoring.svc.cluster.local}
  kubectl apply -f deploy/tracing/destinationrule-tempo.yaml # plaintext (Tempo has no sidecar)
  kubectl apply -f deploy/tracing/telemetry-otel.yaml        # sampling (10%)
  kubectl -n online-boutique rollout restart deploy/<each>    # sequential

Hard-won gotchas (all hit during setup):
- Service ports MUST be Istio-protocol-named (`grpc-otlp`, not `otlp-grpc`),
  or the OTLP cluster negotiates HTTP/1.1 and gRPC export RSTs with
  "protocol error". Verify: explicit_http_config.http2 in the cluster dump.
- Tempo OTLP receivers bind localhost by default; set explicit
  `endpoint: 0.0.0.0:4317/4318` in tempo.yaml.
- Pre-1.22-style `defaultConfig.tracing.provider` does nothing on Istio 1.30;
  use the Telemetry API. Telemetry changes are XDS-dynamic (no restart).
- Search API needs TraceQL `q=` + explicit start/end; tag-only search misses.
- kubectl port-forward hits localhost-in-netns (bypasses Envoy); curling a
  Service ClusterIP from a sideless pod is the honest connectivity test.
