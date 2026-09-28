# Istio mesh change (Tier-1 queue stats)

Widens Envoy stats exposure so per-cluster queue gauges are scraped.
Applied with:
  kubectl -n istio-system patch configmap istio --type merge \
    --patch-file deploy/istio/proxy-stats-matcher-patch.json

The ConfigMap is Helm-managed (`release-name: istiod`); a future
`helm upgrade` reverts this — re-apply afterwards. Verify with:
  count(envoy_cluster_upstream_rq_active{cluster_name!="xds-grpc"})  # was 0, now ~57
  count(envoy_cluster_upstream_rq_pending_active{cluster_name!="xds-grpc"})

Note: the matcher takes effect on proxies only after they restart
(`kubectl -n online-boutique rollout restart deploy/<name>`, sequential).
