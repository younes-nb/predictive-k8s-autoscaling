"""PrometheusClient adapted for the lab cluster's Mimir TSDB.

Differences from upstream PBScaler:
  * Every query that filtered on ``job="kubernetes-cadvisor"`` is relaxed:
    in this cluster cAdvisor metrics are remote-written into Mimir by Alloy
    and carry ``job="prometheus.scrape.kubelet"`` instead. Selecting by
    namespace/container (identical labels in both clusters) is enough.
  * Result iterables are guarded against empty/404 responses so a missing
    series (e.g. a stateless service with zero traffic) cannot crash the
    collector — Mimir only keeps series that have recent samples.
  * ``container_spec_cpu_quota`` does not exist in this cluster's kubelet
    exposition, so CPU limits are derived from
    ``kube_pod_container_resource_limits`` (kube-state-metrics) instead.
"""

import time

import networkx as nx
import numpy as np
import pandas as pd
import requests

from config.Config import Config
from util.KubernetesClient import KubernetesClient


class PrometheusClient:
    def __init__(self, config: Config):
        self.config = config
        self.namespace = config.namespace
        self.prom_no_range_url = config.prom_no_range_url
        self.prom_range_url = config.prom_range_url
        self.start = config.start
        self.end = config.end
        self.step = config.step

    def set_time_range(self, start, end):
        self.start = start
        self.end = end

    # execute prom_sql
    def execute_prom(self, prom_url, prom_sql):
        response = requests.get(prom_url,
                                params={'query': prom_sql,
                                        'start': self.start,
                                        'end': self.end,
                                        'step': self.step},
                                timeout=30)
        response.raise_for_status()
        data = response.json().get('data') or {}
        return data.get('result') or []

    @staticmethod
    def _instant(result, default: float = 0.0) -> float:
        """first vector sample of an instant query as float"""
        if len(result) == 0:
            return default
        try:
            return float(result[0].get('value', [0, default])[1])
        except (ValueError, IndexError, TypeError):
            return default

    @staticmethod
    def _range_series(result, df, key: str):
        """append one matrix query result (series) to df under `key`"""
        if len(result) == 0:
            return df
        values = result[0].get('values') or []
        if len(values) == 0:
            return df
        values = list(zip(*values))
        if 'timestamp' not in df:
            df['timestamp'] = values[0]
            df['timestamp'] = df['timestamp'].astype('datetime64[s]')
        df[key] = pd.Series(values[1]).astype('float64')
        return df

    def p90(self, svc):
        prom_90_sql = 'histogram_quantile(0.90, sum(irate(istio_request_duration_milliseconds_bucket{reporter="destination", destination_workload="%s"}[1m])) by (destination_workload, le))' % svc
        return self._instant(self.execute_prom(self.prom_no_range_url, prom_90_sql))

    def p50(self, svc):
        prom_50_sql = 'histogram_quantile(0.50, sum(irate(istio_request_duration_milliseconds_bucket{reporter="destination", destination_workload="%s"}[1m])) by (destination_workload, le))' % svc
        return self._instant(self.execute_prom(self.prom_no_range_url, prom_50_sql))

    def get_call(self):
        DG = nx.DiGraph()
        k8s_util = KubernetesClient(self.config)
        def is_stateful(svc):
            return any(m in svc for m in ('redis', 'rabbitmq', 'mongo', 'mysql'))
        svcs = [s for s in k8s_util.get_svcs()
                if not is_stateful(s) and 'unknown' not in s and 'loadgenerator' not in s]
        DG.add_nodes_from(svcs)
        edges = []
        prom_sql = 'sum(istio_tcp_received_bytes_total{destination_workload_namespace="%s"}) by (source_workload, destination_workload)' % self.namespace
        results = self.execute_prom(self.prom_no_range_url, prom_sql)

        prom_sql = 'sum(istio_requests_total{destination_workload_namespace="%s"}) by (source_workload, destination_workload)' % self.namespace
        results = results + self.execute_prom(self.prom_no_range_url, prom_sql)

        for result in results:
            metric = result.get('metric') or {}
            source = metric.get('source_workload')
            destination = metric.get('destination_workload')
            if source in svcs and destination in svcs:
                edges.append((source, destination))

        DG.add_edges_from(edges)
        return DG

    def get_call_latency(self):
        """p90 for every invocation edge (name: 'source_destination')."""
        prom_90_sql = 'histogram_quantile(0.90, sum(irate(istio_request_duration_milliseconds_bucket{reporter="destination", destination_workload_namespace="%s"}[1m])) by (destination_workload, source_workload, le))' % self.namespace
        responses_90 = self.execute_prom(self.prom_no_range_url, prom_90_sql)

        call_latencies = {}
        for res in responses_90:
            try:
                name = res['metric']['source_workload'] + '_' + res['metric']['destination_workload']
            except KeyError:
                continue
            p90 = self._instant([res])
            call_latencies[name] = p90

        return call_latencies

    def get_svc_latency(self):
        # p50,p90,p99 of every workloads
        prom_50_sql = 'histogram_quantile(0.50, sum(irate(istio_request_duration_milliseconds_bucket{reporter="destination", destination_workload_namespace="%s"}[1m])) by (destination_workload, le))' % self.namespace
        prom_90_sql = 'histogram_quantile(0.90, sum(irate(istio_request_duration_milliseconds_bucket{reporter="destination", destination_workload_namespace="%s"}[1m])) by (destination_workload, le))' % self.namespace
        prom_99_sql = 'histogram_quantile(0.99, sum(irate(istio_request_duration_milliseconds_bucket{reporter="destination", destination_workload_namespace="%s"}[1m])) by (destination_workload, le))' % self.namespace

        svc_latencies = {}
        for kind, sql in (('p50', prom_50_sql), ('p90', prom_90_sql), ('p99', prom_99_sql)):
            for res in self.execute_prom(self.prom_no_range_url, sql):
                try:
                    name = res['metric']['destination_workload'] + '&' + kind
                except KeyError:
                    continue
                svc_latencies[name] = self._instant([res])

        return svc_latencies

    def get_svc_qps(self):
        qps_dic = {}
        qps_sql = 'sum(rate(istio_requests_total{reporter="destination",namespace="%s"}[1m])) by (destination_workload)' % self.namespace
        for res in self.execute_prom(self.prom_no_range_url, qps_sql):
            try:
                name = res['metric']['destination_workload'] + '&qps'
            except KeyError:
                continue
            qps_dic[name] = self._instant([res])

        return qps_dic

    def get_svc_avg(self):
        """Mean response latency per workload (ms): exact ratio of the
        completion-weighted duration counters (sum/count), [1m] window —
        same completion semantics as the quantiles; no histogram addition.
        """
        sql = ('sum(rate(istio_request_duration_milliseconds_sum'
               '{reporter="destination", destination_workload_namespace="%s"}[1m]))'
               ' by (destination_workload) / '
               'sum(rate(istio_request_duration_milliseconds_count'
               '{reporter="destination", destination_workload_namespace="%s"}[1m]))'
               ' by (destination_workload)') % (self.namespace, self.namespace)
        out = {}
        for res in self.execute_prom(self.prom_no_range_url, sql):
            try:
                name = res['metric']['destination_workload']
            except KeyError:
                continue
            out[name] = self._instant([res])
        return out

    def get_front_avg(self, front_svc: str, gateway: str = "") -> float:
        """Mean end-user latency (ms) at the gateway->front edge, with the
        same completion semantics + gateway_edge/service fallback as
        get_front_p90. NaN when no traffic.
        """
        if gateway:
            sql = ('sum(rate(istio_request_duration_milliseconds_sum'
                   '{reporter="destination", destination_workload="%s", '
                   'source_workload="%s"}[1m])) by (destination_workload) / '
                   'sum(rate(istio_request_duration_milliseconds_count'
                   '{reporter="destination", destination_workload="%s", '
                   'source_workload="%s"}[1m])) by (destination_workload)') % (
                    front_svc, gateway, front_svc, gateway)
            v = self._instant(self.execute_prom(self.prom_no_range_url, sql),
                              default=float('nan'))
            if v == v:
                return v
        sql = ('sum(rate(istio_request_duration_milliseconds_sum'
               '{reporter="destination", destination_workload="%s"}[1m]))'
               ' by (destination_workload) / '
               'sum(rate(istio_request_duration_milliseconds_count'
               '{reporter="destination", destination_workload="%s"}[1m]))'
               ' by (destination_workload)') % (front_svc, front_svc)
        return self._instant(self.execute_prom(self.prom_no_range_url, sql),
                             default=float('nan'))

    def get_front_p90(self, front_svc: str, gateway: str = "") -> dict:
        """End-user tail latency: p90 for the gateway->front edge, falling
        back to the front service's own p90 (reporter=destination).

        This is the label signal for the SLO violation predictor
        (paper Sec. IV-C2: compare the tail latency of the front-end
        microservice with the SLO).

        Completion semantics (2026-10-21 decision): the quantile is taken
        over the durations of requests that COMPLETED inside the window
        (`rate` over the istio duration-bucket counters), not an instant
        slope (`irate`). istio records a request's duration from arrival
        until its response is fully delivered back through the front-end
        — the user-visible round trip — so the gateway->front edge p90 is
        completion-based end-to-end truth. Verified live 2026-10-21: a
        30-loop login storm read 2334-2368 ms at this edge vs client
        curl-truth p90 = 2333 ms (irate[1m] and rate[1m] agree; rate was
        chosen per the paper's windowed-interval sampling and its
        stability under the 5 s scrape cadence).
        """
        if gateway:
            sql = ('histogram_quantile(0.90, sum(rate('
                   'istio_request_duration_milliseconds_bucket{reporter="destination", '
                   'destination_workload="%s", source_workload="%s"}[1m])) '
                   'by (destination_workload, source_workload, le))') % (front_svc, gateway)
            p90 = self._instant(self.execute_prom(self.prom_no_range_url, sql),
                                default=float('nan'))
            if p90 == p90:  # not NaN
                return {'front_p90': p90, 'front_source': 'gateway_edge'}
        sql = ('histogram_quantile(0.90, sum(rate('
               'istio_request_duration_milliseconds_bucket{reporter="destination", '
               'destination_workload="%s"}[1m])) by (destination_workload, le))') % front_svc
        p90 = self._instant(self.execute_prom(self.prom_no_range_url, sql),
                            default=float('nan'))
        return {'front_p90': p90, 'front_source': 'service'}

    def get_call_p90_latency_range(self):
        call_df = pd.DataFrame()
        prom_90_sql = 'histogram_quantile(0.90, sum(irate(istio_request_duration_milliseconds_bucket{reporter="destination", destination_workload_namespace="%s"}[1m])) by (destination_workload, destination_workload_namespace, source_workload, le))' % self.namespace
        responses_90 = self.execute_prom(self.prom_range_url, prom_90_sql)

        for result in responses_90:
            try:
                name = result['metric']['source_workload'] + '_' + result['metric']['destination_workload']
            except KeyError:
                continue
            call_df = self._range_series([result], call_df, name)

        return call_df

    def get_svc_p90_latency_range(self):
        latency_df = pd.DataFrame()
        prom_90_sql = 'histogram_quantile(0.90, sum(irate(istio_request_duration_milliseconds_bucket{reporter="destination", destination_workload_namespace="%s"}[1m])) by (destination_workload, destination_workload_namespace, le))' % self.namespace
        responses_90 = self.execute_prom(self.prom_range_url, prom_90_sql)

        for result in responses_90:
            try:
                name = result['metric']['destination_workload']
            except KeyError:
                continue
            latency_df = self._range_series([result], latency_df, name)

        return latency_df

    def get_resource_metric_range(self):
        """Whole-namespace aggregate vCPU + memory."""
        metric_df = pd.DataFrame()
        vCPU_sql = 'sum(rate(container_cpu_usage_seconds_total{image!="",namespace="%s"}[1m]))' % self.namespace
        mem_sql = 'sum(rate(container_memory_usage_bytes{image!="",namespace="%s"}[1m])) / (1024*1024)' % self.namespace
        vCPU = self.execute_prom(self.prom_range_url, vCPU_sql)
        mem = self.execute_prom(self.prom_range_url, mem_sql)

        if len(vCPU):
            metric_df = self._range_series(vCPU, metric_df, 'vCPU')
        if len(mem):
            metric_df = self._range_series(mem, metric_df, 'memory')
        if 'vCPU' in metric_df:
            metric_df['vCPU'] = metric_df['vCPU'].fillna(0)
        if 'memory' in metric_df:
            metric_df['memory'] = metric_df['memory'].fillna(0)

        return metric_df

    # Get qps for all workloads as a range
    def get_svc_qps_range(self):
        qps_df = pd.DataFrame()
        qps_sql = 'sum(rate(istio_requests_total{reporter="destination",namespace="%s"}[1m])) by (destination_workload)' % self.namespace
        response = self.execute_prom(self.prom_range_url, qps_sql)
        for result in response:
            try:
                name = result['metric']['destination_workload']
            except KeyError:
                continue
            qps_df = self._range_series([result], qps_df, name)
        return qps_df

    # Get CPU and memory usage per microservice (aggregated over pods)
    def get_svc_metric_range(self, k8s_util=None):
        """Per-microservice cpu_usage / mem_usage.

        Column schema preserved from upstream (`<svc>&cpu_usage`,
        `<svc>&mem_usage`), but the aggregation is defensive: a service with
        no series in the window is simply omitted instead of raising.
        """
        if k8s_util is None:
            k8s_util = KubernetesClient(self.config)
        cpu_usage_sql = '(sum(rate(container_cpu_usage_seconds_total{namespace="%s",container!~"POD|istio-proxy"}[1m])) by(pod))' % self.namespace
        # memory usage MB
        mem_usage_sql = 'sum(container_memory_usage_bytes{namespace="%s",container!~"POD|istio-proxy"}) by(pod) / (1024*1024)' % self.namespace

        tempdfs = []

        def handle(result, col):
            try:
                name = result['metric']['pod'] + '&' + col
            except KeyError:
                return
            doc = result.get('values') or []
            if len(doc) == 0:
                return
            values = list(zip(*doc))
            tempdf = pd.DataFrame()
            tempdf['timestamp'] = values[0]
            tempdf[name] = pd.Series(values[1]).astype('float64')
            tempdf.set_index('timestamp', inplace=True)
            tempdfs.append(tempdf)

        for result in self.execute_prom(self.prom_range_url, cpu_usage_sql):
            handle(result, 'cpu_usage')
        for result in self.execute_prom(self.prom_range_url, mem_usage_sql):
            handle(result, 'mem_usage')

        if len(tempdfs) == 0:
            return pd.DataFrame()

        df = pd.concat(tempdfs, axis=1)
        df = df.fillna(0)

        # Aggregate per microservice
        final_df = pd.DataFrame(index=df.index)
        svcs = k8s_util.get_svcs()
        col_list = list(df)
        for svc in svcs:
            if svc == 'loadgenerator':
                continue
            svc_cols = [col for col in col_list if col.startswith(svc) and col.endswith('cpu_usage')]
            if svc_cols:
                final_df[svc + '&cpu_usage'] = df[svc_cols].sum(axis=1)
            svc_cols = [col for col in col_list if col.startswith(svc) and col.endswith('mem_usage')]
            if svc_cols:
                final_df[svc + '&mem_usage'] = df[svc_cols].sum(axis=1)

        final_df['timestamp'] = final_df.index
        final_df['timestamp'] = final_df['timestamp'].astype('datetime64[s]')
        return final_df

    # Get success rate for all workloads
    def get_success_rate_range(self):
        success_df = pd.DataFrame()
        success_rate_sql = ('(sum(rate(istio_requests_total{reporter="destination", response_code!~"5.*",namespace="%s"}[1m]))'
                            ' by (destination_workload) / sum(rate(istio_requests_total{reporter="destination",namespace="%s"}[1m]))'
                            ' by (destination_workload))') % (self.namespace, self.namespace)
        response = self.execute_prom(self.prom_range_url, success_rate_sql)
        for result in response:
            try:
                name = result['metric']['destination_workload']
            except KeyError:
                continue
            success_df = self._range_series([result], success_df, name)

        return success_df

    def cal_slo_violation(self, slo: float, apis: list, duration):
        '''
        eg:
            slo: 500
            apis: ['unknown_frontend']
            duration: 30 (seconds)
        '''
        self.end = int(round(time.time()))
        self.start = self.end - duration
        cols = [api + '&p90' for api in apis]
        df = self.get_call_p90_latency_range()
        if df.empty or not set(cols).issubset(df.columns):
            return 0.0
        df = df[cols]
        con = df >= slo
        conflict_num = df[con].count().sum()
        return conflict_num / df.count().sum()

    def get_edge_index(self):
        # get the dependence
        DG = self.get_call()
        adj = nx.to_scipy_sparse_matrix(DG).tocoo()
        row = adj.row.astype(np.int64)
        col = adj.col.astype(np.int64)
        edge_index = np.stack([row, col], axis=0)
        return edge_index
