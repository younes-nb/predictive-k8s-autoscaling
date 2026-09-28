# adservice JMX runtime metrics (Tier-1)

Agentless jmx_exporter (no image rebuild): apply in order
  kubectl apply -f deploy/jmx/configmap.yaml
  kubectl -n online-boutique patch deploy adservice --type strategic --patch-file deploy/jmx/adservice-patch.yaml
  kubectl apply -f deploy/jmx/podmonitor.yaml   # PodMonitor CRD in monitoring ns, 10s interval

Verify: `count(jvm_memory_bytes_used{namespace="online-boutique"})` > 0.
Go/Python/Node services expose no runtime metrics without code changes
(rebuilds) — out of reach agentlessly, intentionally not done.
