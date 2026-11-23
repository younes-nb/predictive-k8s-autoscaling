"""Few-shot calibration of the heads on top of a frozen encoder.

  1. encode every window with the pretrained encoder;
  2. fit one prototype per bottleneck class from the weak labels
     (balanced, so ``healthy`` does not swamp the rare classes);
  3. calibrate the OOD distance threshold from the nearest-prototype distances;
  4. fit the independent addressability head;
  5. publish an immutable, versioned model.

The encoder is never trained here. Adding a new service later means adding its
prototype at run time (``PrototypicalHead.set_prototype``), not re-running this.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.build import stack_windows                     # noqa: E402
from features.groups import FEATURE_ORDER                    # noqa: E402
from models.encoder import TS2VecEncoder                     # noqa: E402
from models.proto import (ADDRESSABLE_PRIOR, AddressabilityHead,  # noqa: E402
                          AuroraModel, BOTTLENECK_CLASSES, PrototypicalHead)
from models.registry import ModelRegistry                    # noqa: E402


def encode(enc, x, mask, batch=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.tensor(x[i:i + batch], dtype=torch.float32)
            mb = torch.tensor(mask[i:i + batch], dtype=torch.float32)
            out.append(enc.sequence(xb, mb).numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../data/corpus_labelled.parquet")
    ap.add_argument("--encoder", default="../data/encoder.pt")
    ap.add_argument("--registry", default="../data/registry")
    ap.add_argument("--t-len", type=int, default=24)
    ap.add_argument("--per-class", type=int, default=200,
                    help="max windows per class used to fit prototypes")
    ap.add_argument("--version", default=None)
    a = ap.parse_args()

    torch.manual_seed(20)
    import pandas as pd
    df = pd.read_parquet(a.corpus)

    ck = torch.load(a.encoder, map_location="cpu", weights_only=False)
    norm = ck["norm"]
    d_model = ck["d_model"]
    n_feat = ck["n_features"]

    enc = TS2VecEncoder(n_feat, d_model=d_model)
    enc.load_state_dict(ck["state_dict"])
    enc.eval()

    # normalise exactly as the controller will
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.sqrt(np.asarray(norm["var"], dtype=np.float32))

    # Window construction is shared with training (features.build), so the
    # temporal boundaries are identical. The previous manual groupby("service")
    # loop could join the end of one experiment to the start of another.
    from features.build import group_windows
    xs, ms, ys, ss = [], [], [], []
    for svc, x, m, label in group_windows(df, a.t_len, label_col="bottleneck"):
        xs.append((x - mean) / std)
        ms.append(m)
        ys.append(label if label else "unknown")
        ss.append(svc)

    x = np.stack(xs)
    m = np.stack(ms)
    print(f"[calibrate] windows={x.shape[0]}  classes={sorted(set(ys))}")
    z = encode(enc, x, m)

    # ---- prototypes, balanced per class --------------------------------
    proto = PrototypicalHead(d_model, classes=list(BOTTLENECK_CLASSES))
    counts = {}
    for ci, cls in enumerate(proto.classes):
        idx = [i for i, y in enumerate(ys) if y == cls]
        if not idx:
            continue
        if len(idx) > a.per_class:
            sel = np.random.choice(idx, a.per_class, replace=False)
        else:
            sel = np.array(idx)
        vec = z[sel].mean(axis=0)
        proto.set_prototype(cls, vec, ema=1.0)
        counts[cls] = int(len(idx))
    print(f"[calibrate] prototypes: {counts}")
    proto.set_log_prior(counts)

    # ---- OOD threshold: 99th pct of nearest-prototype distance ----------
    with torch.no_grad():
        # must use the head's own metric (cosine), otherwise the threshold is
        # expressed in a different space than the one used at inference time
        d = proto.distances(torch.tensor(z, dtype=torch.float32))
    nearest = d.min(dim=-1).values.numpy()
    thr = float(np.percentile(nearest, 99))
    proto._dist_threshold = thr
    print(f"[calibrate] ood threshold (p99 nearest dist) = {thr:.3f}")

    # ---- addressability head -------------------------------------------
    # Balance the addressability fit. The corpus is ~75% healthy, so an
    # unweighted fit degenerates to "never addressable" (measured: p always
    # < 0.4), which would silently disable scale-out. Sample equally from the
    # two classes instead.
    y_all = np.array([1.0 if c in ADDRESSABLE_PRIOR else 0.0 for c in ys])
    pos = np.where(y_all > 0.5)[0]
    neg = np.where(y_all <= 0.5)[0]
    rng = np.random.default_rng(20)
    k = min(len(pos), len(neg))
    if k > 0:
        sel = np.concatenate([rng.choice(pos, k, replace=False),
                              rng.choice(neg, k, replace=False)])
    else:
        sel = np.arange(len(ys))
    y_addr = torch.tensor(y_all[sel], dtype=torch.float32)
    z_fit = torch.tensor(z[sel], dtype=torch.float32)
    print(f"[calibrate] addressability fit: {k} per class "
          f"(pos={len(pos)}, neg={len(neg)})")
    addr = AddressabilityHead(d_model)
    opt = torch.optim.Adam(addr.parameters(), lr=5e-3, weight_decay=1e-4)
    zt = z_fit
    for epoch in range(300):
        opt.zero_grad()
        out = addr(zt)["addressable_prob"]
        loss = F.binary_cross_entropy(out.clamp(1e-6, 1 - 1e-6), y_addr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (addr(zt)["addressable_prob"] > 0.5).float()
        acc = float((pred == y_addr).float().mean())
        allp = addr(torch.tensor(z, dtype=torch.float32))["addressable_prob"]
        print(f"[calibrate] addressability on the full corpus: "
              f"mean={float(allp.mean()):.3f} max={float(allp.max()):.3f}")
    print(f"[calibrate] addressability head: train acc={acc:.3f} "
          f"(prior = class in {sorted(ADDRESSABLE_PRIOR)})")

    model = AuroraModel(enc, proto, addr)
    reg = ModelRegistry(a.registry)
    version = a.version or ("v" + __import__("time").strftime("%Y%m%dT%H%M%SZ",
                                                    __import__("time").gmtime()))
    validation = {
        "windows": int(x.shape[0]),
        "services": int(len(set(ss))),
        "class_counts": counts,
        "ood_threshold": thr,
        "addressability_train_acc": acc,
        "addressability_prior": sorted(ADDRESSABLE_PRIOR),
        "t_len": a.t_len,
        "n_features": n_feat,
    }
    reg.publish(model, version,
                feature_schema={"order": FEATURE_ORDER, "groups": "features/groups.py"},
                validation=validation,
                notes="few-shot calibration on telemetry-derived weak labels",
                norm=norm)
    print(f"[calibrate] published {version} -> {a.registry}")


if __name__ == "__main__":
    main()