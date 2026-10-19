"""Train the PBScaler SLO violation predictor on TrainTicket data.

Upstream: the repo ships two `svcs` lists (online-boutique active,
TrainTicket commented). This repo targets the TrainTicket deployment of the
lab cluster, so the TrainTicket list is active; the list is re-derived from
the training CSV columns when possible so it stays in sync with whatever the
collector recorded (svc set frozen per run in meta.json).

Input — written by pbscaler/scripts/collect_training_data.py during a k6 run
(NASA replay) with the namespace's native HPAs performing elastic scaling
(paper Sec. IV-C2: "historical metric data ... generated using ... the
Kubernetes autoscaler by default"):

    <data>/train_data.csv   columns: <svc>&qps, <svc>&count, slo_reward
                            (1 = no SLO violation, 0 = violation)

Rows with slo_reward < 0 (no front traffic in that interval) are skipped.

Output — model + stats to simulation/<bench>/(RandomForestClassify.model, rf.pkl)
(train_irs dirs created here, upstream required making them by hand).
"""

import argparse
import os
import random
import pickle

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# fallback list for TrainTicket (the 43 stateless services of the demo app)
TT_SVCS = ['ts-admin-basic-info-service', 'ts-admin-order-service', 'ts-admin-route-service',
           'ts-admin-travel-service', 'ts-admin-user-service', 'ts-assurance-service',
           'ts-auth-service', 'ts-avatar-service', 'ts-basic-service', 'ts-cancel-service',
           'ts-config-service', 'ts-consign-price-service', 'ts-consign-service',
           'ts-contacts-service', 'ts-delivery-service', 'ts-execute-service',
           'ts-food-map-service', 'ts-food-service', 'ts-inside-payment-service',
           'ts-news-service', 'ts-notification-service', 'ts-order-other-service',
           'ts-order-service', 'ts-payment-service', 'ts-preserve-other-service',
           'ts-preserve-service', 'ts-price-service', 'ts-rebook-service',
           'ts-route-plan-service', 'ts-route-service', 'ts-seat-service',
           'ts-security-service', 'ts-station-service', 'ts-ticket-office-service',
           'ts-ticketinfo-service', 'ts-train-service', 'ts-travel-plan-service',
           'ts-travel-service', 'ts-travel2-service', 'ts-ui-dashboard',
           'ts-user-service', 'ts-verification-code-service', 'ts-voucher-service']
BOUTIQUE_SVCS = ['adservice', 'cartservice', 'checkoutservice', 'currencyservice', 'emailservice',
                 'frontend', 'paymentservice', 'productcatalogservice', 'recommendationservice',
                 'shippingservice']


def svcs_from_csv(df: pd.DataFrame) -> list:
    qps_cols = sorted(c for c in df.columns if c.endswith('&qps'))
    return [c[:-4] for c in qps_cols]


def data_loader(path, svcs):
    df = pd.read_csv(path)
    cols = [col for col in df.columns
            if col.endswith('&qps') or col.endswith('&count') or col == 'slo_reward']
    df = df[cols].fillna(0)
    # build dataset: [index, qps, count] per stateless microservice + label
    datas = []
    for _, row in df.iterrows():
        x = []
        for i, svc in enumerate(svcs):
            try:
                x.extend([i, row[svc + '&qps'], row[svc + '&count']])
            except KeyError:
                x.extend([i, 0, row.get(svc + '&count', 0)])
        x.append(row['slo_reward'])
        datas.append(x)
    datas = [d for d in datas if d[-1] >= 0]  # drop '-1 = no traffic' rows
    random.shuffle(datas)
    datas = np.array(datas)
    datas_x, datas_y = datas[:, 0:-1], datas[:, -1]
    return datas_x, datas_y


def setup_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='../data/train_ticket/train_data.csv',
                    help='path to train_data.csv from the collector')
    ap.add_argument('--bench', default='train_ticket', choices=['train_ticket', 'boutique'])
    ap.add_argument('--seed', type=int, default=20)
    args = ap.parse_args()

    setup_seed(args.seed)
    df = pd.read_csv(args.data)
    svcs = svcs_from_csv(df) or (TT_SVCS if args.bench == 'train_ticket' else BOUTIQUE_SVCS)
    print(f'using {len(svcs)} services from {args.data}')
    datas_x, datas_y = data_loader(args.data, svcs)
    print(f'samples: {len(datas_y)} (violations={(1 - datas_y).sum() if datas_y.max() <= 1 else "?"})')

    model = RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(datas_x, datas_y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Test set score:{:.2f}".format(model.score(X_test, y_test)))
    print('acc', accuracy_score(y_test, y_pred))
    print('recall', recall_score(y_test, y_pred, pos_label=0))
    print('auc', roc_auc_score(y_test, y_pred))

    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=None,
                                     sample_weight=None, drop_intermediate=True)
    res = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.bench)
    os.makedirs(out_dir, exist_ok=True)
    seed_dump = out_dir + '/rf.pkl'
    model_path = out_dir + '/RandomForestClassify.model'
    pickle.dump(res, open(seed_dump, 'wb'))
    joblib.dump(model, model_path)
    print(f'model written: {model_path}\n(pickled ROC: {seed_dump})')
