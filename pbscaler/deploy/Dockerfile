# PBScaler controller image (in-cluster deployment of pbscaler/main.py).
# Build context: this pbscaler/ directory (see deploy/build_and_push.sh).
FROM python:3.9-slim

WORKDIR /app

# deps first for layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# code: needs the package root (config/, util/, monitor/, simulation/, scripts/)
COPY PBScaler.py main.py /app/
COPY config /app/config
COPY util /app/util
COPY monitor /app/monitor
COPY scripts /app/scripts
RUN mkdir -p /app/simulation/train_ticket /app/data

ENV PBS_DRY_RUN=false \
    PBS_SUSPEND_HPAS=true \
    PBS_RESTORE_HPAS=true \
    PBS_HPA_STATE_FILE=/app/data/hpa_state.json \
    PBS_SLO_PREDICTOR=/app/simulation/train_ticket/RandomForestClassify.model \
    PBS_NAMESPACE=train-ticket

# Mimir is reachable in-cluster; PBS_* overrides come from the Deployment env.
ENTRYPOINT ["python", "main.py"]
