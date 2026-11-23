"""Reproducible model-localization cost benchmark; excludes telemetry/backend costs.

This is NOT a benchmark against PBScaler and does NOT establish an empirical
speedup over PBScaler until both run on the same cluster and graph sizes.
"""
import statistics,time
import numpy as np
import torch
from .model import SparseTemporalGraph
from .schema import NODE_FEATURES,EDGE_FEATURES

def run(node_counts=(16,64,128,256,512),window=24,degree=4,repeats=8):
    torch.set_num_threads(min(4,torch.get_num_threads()))
    model=SparseTemporalGraph(dim=48).eval()
    rng=np.random.default_rng(2026)
    rows=[]
    for n in node_counts:
        t=window;f=len(NODE_FEATURES);ef=len(EDGE_FEATURES)
        x=torch.tensor(rng.uniform(0,.9,(n,t,f)),dtype=torch.float32)
        m=torch.ones_like(x)
        src=np.repeat(np.arange(n),degree)
        dst=(src+rng.integers(1,n,size=src.size))%n
        ei=torch.tensor(np.stack([src,dst]),dtype=torch.long)
        ex=torch.tensor(rng.uniform(0,1,(len(src),t,ef)),dtype=torch.float32)
        em=torch.ones_like(ex)
        with torch.inference_mode():
            model(x,m,ei,ex,em,torch.tensor([.02,.01,0.,1.])) # warm up
            times=[]
            for _ in range(repeats):
                start=time.perf_counter();model(x,m,ei,ex,em,torch.tensor([.02,.01,0.,1.]))
                times.append((time.perf_counter()-start)*1000.)
        rows.append({'nodes':n,'edges':len(src),'timesteps':t,
                     'graph_ms_p50':round(statistics.median(times),3),
                     'graph_ms_p95':round(sorted(times)[int(.95*(len(times)-1))],3),
                     'graph_ms_max':round(max(times),3)})
    return {'benchmark':'model-only, local CPU (not Mimir or Kubernetes)','results':rows}