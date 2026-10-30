"""PBScaler core (upstream code, adapted to the lab cluster).

Loop unchanged from the paper (IEEE TSC 2024, Sec. III):
  * every AB_CHECK_INTERVAL(15s): SLO violation detection on all invocation
    edges (P90 via Istio in Prometheus) -> TopoRank bottleneck localization
    -> GA offline optimization guided by the SLO violation predictor ->
    patch replicas of the top-K bottlenecks,
  * every WASTE_CHECK_INTERVAL(120s): redundancy check via a one-sided
    t-test between recent/older QPS -> scale in.

Cluster adaptations (all keyed off config/Config.py):
  * bottleneck filters use the cluster's Istio gateway workload name
    ('train-ticket-gateway-istio') instead of 'istio-ingressgateway'.
  * the predictor model path comes from the config, not a hardcoded
    /home/ubuntu/... path (upstream loaded it in TWO places — here and in
    GA.__init__ — both go through config.slo_predictor_path now).
  * HPA suspend/restore around the run (see KubernetesClient), since PBScaler
    patches Deployment .spec.replicas directly.
  * dry_run mode executes every phase and only *prints* the would-be action.
"""

import json
import math
import os
import time
import numpy as np
import schedule
from copy import deepcopy
import networkx as nx
import scipy.stats

from config.Config import Config
from util.GA import GA
import warnings
from util.KubernetesClient import KubernetesClient
from util.PrometheusClient import PrometheusClient

warnings.filterwarnings("ignore")

CONF = 0.05
# anomaly detection window -- 15 seconds
AB_CHECK_INTERVAL = 15
# waste detection window -- 120 seconds
WASTE_CHECK_INTERVAL = 120
ALPHA = 0.2
BETA = 0.9
K = 2


class PBScaler:
    def __init__(self, config: Config, simulation_model):
        # the prometheus client && k8s client
        self.config = config
        self.prom_util = PrometheusClient(config)
        self.k8s_util = KubernetesClient(config)
        # simulation environment (joblib path or loaded predictor)
        self.predictor = simulation_model
        # args: paper Sec. IV-A4 (alpha, beta, sigma, k)
        self.SLO = config.SLO
        self.max_num = config.max_pod
        self.min_num = config.min_pod
        self.ab_check_interval = getattr(config, 'ab_check_interval', AB_CHECK_INTERVAL)
        self.waste_check_interval = getattr(config, 'waste_check_interval', WASTE_CHECK_INTERVAL)
        self.ALPHA = getattr(config, 'alpha', ALPHA)
        self.BETA = getattr(config, 'beta', BETA)
        self.K = getattr(config, 'k_top', K)
        self.CONF = getattr(config, 'conf', CONF)
        # microservices
        self.mss = self.k8s_util.get_svcs_without_state()
        self.roots = None
        self.svc_counts = None
        # the exact service ORDER the SLO predictor was trained on — the GA
        # reproduces the training feature row against it (see _load_model_svcs)
        self.model_svcs = self._load_model_svcs(config)
        self.model_index = {s: i for i, s in enumerate(self.model_svcs or [])}

    @staticmethod
    def coast_time(func):
        def fun(*args, **kwargs):
            t = time.perf_counter()
            result = func(*args, **kwargs)
            print(f'func {func.__name__} coast time:{time.perf_counter() - t:.8f} s')
            return result
        return fun

    @staticmethod
    def _log_timing(module: str, seconds: float):
        """Parseable per-module timing line (Table III measurement, v1.2.5+).

        Format: '[timing] <UTC ISO> module=<name> seconds=<t>' — one line per
        executed module stage; consumed post-run by the metrics analysis.
        """
        ts = time.strftime('%FT%TZ', time.gmtime())
        print(f'[timing] {ts} module={module} seconds={seconds:.6f}', flush=True)

    def _load_model_svcs(self, config) -> list:
        """Per-service list (order-sensitive) the SLO predictor was trained on.

        RandomForestClassify.py writes it next to the model file
        ('<model>.svcs'); the GA's paper feature row ([i, qps, count] per
        service) must be built against this exact order for the RF's
        predictions to be meaningful.
        """
        path = getattr(config, 'slo_predictor_svcs_path', None)
        if path and os.path.exists(path):
            with open(path) as f:
                svcs = json.load(f)
            print(f'loaded predictor svc list: {path} ({len(svcs)} svcs)')
            return svcs
        if config.dry_run:
            print('WARNING: no predictor svc list found — GA falls back to the '
                  'legacy feature layout (dry-run stub only)')
            return None
        raise SystemExit(
            f"predictor svc list not found at '{path}'. Re-run "
            f"simulation/RandomForestClassify.py (it writes '<model>.svcs' "
            f"next to the model file).")

    def anomaly_detect(self):
        """ SLO check
        """
        # get service count
        # exact spec.replicas per Deployment (get_svcs_counts' pod-name
        # substring count runs ahead of the spec during churn, and echoing
        # that count back as the scale target inflates the Deployment every
        # cycle — the 2026-10-23 ts-station-service runaway, 9 -> 664)
        self.svc_counts = self.k8s_util.get_all_counts()
        t0 = time.perf_counter()
        ab_calls = self.get_abnormal_calls()
        self._log_timing('slo_detection', time.perf_counter() - t0)
        print(f"[anomaly] SLO-violating calls: {len(ab_calls)} (SLO={self.SLO} ms)")
        # abnormal
        if len(ab_calls) > 0:
            self.root_analysis(ab_calls=ab_calls)

    def waste_detection(self):
        """waste detection
        """
        t0 = time.perf_counter()
        ab_calls = self.get_abnormal_calls()
        self._log_timing('slo_detection', time.perf_counter() - t0)
        if len(ab_calls) > 0:
            return
        # check qps
        t0 = time.perf_counter()
        cur_time = int(round(time.time()))
        self.prom_util.set_time_range(cur_time - 60, cur_time)
        now_qps_df = self.prom_util.get_svc_qps_range()

        self.prom_util.set_time_range(cur_time - 300, cur_time - 60)
        old_qps_df = self.prom_util.get_svc_qps_range()

        # select the waste pods
        waste_mss = []

        '''Hypothesis µ testing
        a: now_qps µ
        b: old_qps µ0
        H0: µ >= µ0
        H1: µ < µ0
        '''
        for svc in self.mss:
            if svc in now_qps_df.columns and svc in old_qps_df.columns:
                try:
                    t, p = scipy.stats.ttest_ind(now_qps_df[svc], old_qps_df[svc] * self.BETA, equal_var=False)
                except ValueError:
                    continue
                if t < 0 and p <= self.CONF:
                    waste_mss.append(svc)
        self.roots = list(filter(lambda ms: self.svc_counts[ms] > self.min_num, waste_mss))
        self._log_timing('redundancy_check', time.perf_counter() - t0)
        if len(self.roots) != 0:
            self.choose_action(option='reduce')

    def get_abnormal_calls(self):
        # get all call latency last 1min
        begin = int(round((time.time() - 60)))
        end = int(round(time.time()))
        self.prom_util.set_time_range(begin, end)
        # slo Hypothesis testing
        ab_calls = []
        call_latency = self.prom_util.get_call_latency()
        for call, latency in call_latency.items():
            if latency > self.SLO * (1 + self.ALPHA / 2):
                ab_calls.append(call)
        return ab_calls

    def root_analysis(self, ab_calls):
        """ locate the root cause
        1. build the abnormal subgraph
        2. calculate pr score
        3. sort and return the root cause
        Args:
            :param ab_calls: abnormal call edge
        """
        t0 = time.perf_counter()
        ab_dg, personal_array = self.build_abnormal_subgraph(ab_calls)
        nodes = [node for node in ab_dg.nodes]
        if len(nodes) == 1:
            res = [(nodes[0], 1)]
        else:
            res = nx.pagerank(ab_dg, alpha=0.85, personalization=personal_array, max_iter=1000)
            res = sorted(res.items(), key=lambda x: x[1], reverse=True)
        res = [ms for ms, _ in res]
        self.roots = list(filter(lambda root: self.svc_counts[root] + 1 < self.max_num, res))[:self.K]
        self._log_timing('bottleneck_analysis', time.perf_counter() - t0)
        # trigger the choose_action
        if len(self.roots) != 0:
            self.choose_action('add')

    def choose_action(self, option='add'):
        """ choose action
        Args:
            :param option: [add, reduce], add or reduce the replica
        """
        mss = deepcopy(self.mss)
        roots = deepcopy(self.roots)
        r = deepcopy(self.svc_counts)

        # keep only roots the predictor's feature set knows about (the 3-feature
        # row describes the model's svc order; anomaly roots outside it should
        # be impossible — the gateway is filtered from anomaly sets upstream —
        # but stay defensive). Filter FIRST: every bound array below must be
        # built over the same final root list — misaligned min/max lengths
        # feed the GA out-of-band bounds (pre-2026-10-23 bug).
        if self.model_svcs is not None:
            not_in_model = [t for t in roots if t not in self.model_index]
            if not_in_model:
                print('roots outside the predictor feature set, skipped: '
                      + ', '.join(not_in_model))
            roots = [t for t in roots if t in self.model_index]
            if not roots:
                print('no actionable roots in the predictor feature set; no action')
                return

        if option == 'add':
            print('begin scale out')
            roots = [t for t in roots if r[t] < self.max_num]
            if not roots:
                print('all candidate roots already at the max replica cap; no action')
                return
            min_array = [r[t] + 1 for t in roots]
            max_array = [self.max_num for _ in roots]
        elif option == 'reduce':
            print('begin scale in')
            roots = [t for t in roots if r[t] > self.min_num]
            if not roots:
                print('all candidate roots already at the min replica floor; no action')
                return
            min_array = [max(self.min_num, r[t] - 1) for t in roots]
            max_array = [r[t] for t in roots]
        else:
            raise NotImplementedError()
        dim = len(roots)

        # paper Sec. III: the GA fitness evaluates the SLO-violation predictor
        # on a candidate resource vector with the paper's 3-per-service feature
        # layout [i, qps, count]. The row is built over the svc ORDER the
        # predictor was trained on (baked '<model>.svcs'; the namespace list is
        # the dry-run fallback).
        svcs_for_row = self.model_svcs if self.model_svcs is not None else mss
        qps = self.prom_util.get_svc_qps()
        workloads = [qps.get(svc + '&qps', 0.0) for svc in svcs_for_row]

        # Decision Maker: GA search (population-based replica search against
        # the SLO predictor) — Table III's dominant cost, timed separately
        # from the k8s write so post-run analysis can split search vs commit.
        t_ga0 = time.perf_counter()
        opter = GA(self.predictor, dim, min_array, max_array, 'max', size_pop=50, max_iter=5, prob_cross=0.9, prob_mut=0.01, precision=1, encoding='BG', selectStyle='tour', recStyle='xovdp', mutStyle='mutbin', seed=1)
        opter.set_env(workloads, svcs_for_row, roots, r)

        res = opter.evolve()
        self._log_timing('decision_maker_ga', time.perf_counter() - t_ga0)

        # commit: clamp every action into its per-root band before writing —
        # the patch is an absolute spec.replicas, so any decoded value outside
        # the band would write a wrong count straight into the Deployment
        # (defense in depth over the GA decode).
        actions = deepcopy(self.svc_counts)
        for i, t in enumerate(roots):
            cur = int(r[t])
            act = int(res[i])
            if option == 'add':
                lo, hi = cur + 1, self.max_num
            else:
                lo, hi = max(self.min_num, cur - 1), cur
            clamped = int(max(lo, min(act, hi)))
            if clamped != act:
                print(f'clamped out-of-band action for {t}: {act} -> {clamped} (band [{lo},{hi}])')
            actions[t] = clamped

        t_ct0 = time.perf_counter()
        self.execute_task(actions)
        self._log_timing('action_commit', time.perf_counter() - t_ct0)

    def build_abnormal_subgraph(self, ab_calls):
        """
            1. collect metrics for all abnormal services
            2. build the abnormal subgraph with abnormal calls
            3. weight the edges by Pearson correlation coefficient
        """
        ab_sets = set()
        for ab_call in ab_calls:
            ab_sets.update(ab_call.split('_'))
        if 'unknown' in ab_sets: ab_sets.remove('unknown')
        if self.config.gateway_name in ab_sets: ab_sets.remove(self.config.gateway_name)
        ab_mss = list(ab_sets)
        ab_mss.sort()
        begin = int(round((time.time() - 60)))
        end = int(round(time.time()))
        self.prom_util.set_time_range(begin, end)
        ab_metric_df = self.prom_util.get_svc_metric_range()
        ab_svc_latency_df = self.prom_util.get_svc_p90_latency_range()
        if not ab_svc_latency_df.empty:
            ab_svc_latency_df = ab_svc_latency_df[[col for col in ab_svc_latency_df.columns if col in ab_mss]]

        ab_dg = nx.DiGraph()
        ab_dg.add_nodes_from(ab_mss)
        edges = []
        for ab_call in ab_calls:
            edge = ab_call.split('_')
            if 'unknown' in edge or self.config.gateway_name in edge:
                continue
            if not ab_svc_latency_df.empty and edge[0] in ab_svc_latency_df.columns:
                metric_df = ab_metric_df[[col for col in ab_metric_df.columns if col.startswith(edge[1])]]
                weight = self.cal_weight(ab_svc_latency_df[edge[0]], metric_df) if not metric_df.empty else 0
            else:
                weight = 0
            edges.append((edge[0], edge[1], weight))
        ab_dg.add_weighted_edges_from(edges)

        # calculate topology potential
        anomaly_score_map = {}
        for node in ab_mss:
            if not ab_svc_latency_df.empty and node in ab_svc_latency_df.columns:
                e_latency_array = ab_svc_latency_df[node]
                ef = e_latency_array[e_latency_array > self.SLO].count()
            else:
                ef = 0
            anomaly_score_map[node] = ef
        personal_array = self.cal_topology_potential(ab_dg, anomaly_score_map)

        return ab_dg, personal_array

    def cal_weight(self, latency_array, metric_df):
        max_corr = 0
        for col in metric_df.columns:
            try:
                temp = abs(metric_df[col].corr(latency_array))
            except Exception:
                temp = 0
            if not np.isfinite(temp):
                temp = 0
            if temp > max_corr:
                max_corr = temp
        return max_corr

    def cal_topology_potential(self, ab_DG, anomaly_score_map: dict):
        personal_array = {}
        for node in ab_DG.nodes:
            # calculate topological potential
            sigma = 1
            potential = anomaly_score_map[node]
            pre_nodes = ab_DG.predecessors(node)
            for pre_node in pre_nodes:
                potential += (anomaly_score_map[pre_node] * math.exp(-1 * math.pow(1 / sigma, 2)))
                for pre2_node in ab_DG.predecessors(pre_node):
                    if pre2_node != node:
                        potential += (anomaly_score_map[pre2_node] * math.exp(-1 * math.pow(2 / sigma, 2)))
            personal_array[node] = potential
        return personal_array

    def execute_task(self, actions):
        """commit the optimized replica counts (paper Sec. III, 'Scaling
        Decision' -> kubernetes-client)."""
        for ms in self.mss:
            if self.config.dry_run:
                print('[dry-run] would scale {} to {}'.format(ms, int(actions[ms])))
                continue
            self.k8s_util.patch_scale(ms, int(actions[ms]))
            print('scale {} to {}'.format(ms, int(actions[ms])))

    # ------------------------------------------------------------------ run
    @staticmethod
    def _maybe_suspend_hpas(k8s_util, config: Config):
        if getattr(config, 'suspend_hpas_on_start', False) and not config.dry_run:
            try:
                k8s_util.suspend_hpas(config.hpa_state_file)
            except Exception as e:
                print(f'WARNING: HPA suspend failed ({e}); '
                      f'scaling decisions may be overwritten by the HPAs')

    @staticmethod
    def _maybe_restore_hpas(k8s_util, config: Config):
        if getattr(config, 'suspend_hpas_on_start', False) \
                and getattr(config, 'restore_hpas_on_exit', False) and not config.dry_run:
            try:
                k8s_util.restore_hpas(config.hpa_state_file)
            except Exception as e:
                print(f'WARNING: HPA restore failed ({e}); '
                      f'restore manually: pbscaler/scripts/hpa_control.py restore')

    def start(self):
        print("PBScaler is running...")
        self._maybe_suspend_hpas(self.k8s_util, self.config)
        schedule.clear()
        schedule.every(self.ab_check_interval).seconds.do(self.anomaly_detect)
        schedule.every(self.waste_check_interval).seconds.do(self.waste_detection)
        time_start = time.time()

        try:
            while True:
                time_c = time.time() - time_start
                if time_c > self.config.duration:
                    schedule.clear()
                    break
                schedule.run_pending()
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("interrupted — restoring cluster state")
        finally:
            self._maybe_restore_hpas(self.k8s_util, self.config)
