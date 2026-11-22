# AURORA-HPA controller image.
#
# Runtime dependencies only: the controller does not need pandas/scikit-learn
# (those are training-time only), and dropping them keeps the RSS near 310 MiB
# instead of 372 MiB -- which matters because the container is deliberately
# capped at PBScaler's allocation (requests 100m/256Mi, limits 500m/512Mi).
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt /app/requirements-runtime.txt
# Two separate invocations: torch only exists on the PyTorch CPU index, and
# forcing every package through that index breaks source builds of the pure
# Python deps (flit_core is not mirrored there).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        --only-binary=:all: torch
RUN pip install --no-cache-dir --only-binary=:all: -r /app/requirements-runtime.txt

COPY . /app/

# The published model is staged into build_model/ by build_and_push.sh and baked
# in, so the image is self-contained (like the PBScaler RF). A build without a
# published model fails in the script rather than shipping a hold-only image.
COPY build_model/registry /opt/aurora/model/registry

# The model is published into the image at build time (like the PBScaler RF).
# build_and_push.sh fails if it is missing, so a stale image can never deploy.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    AURORA_MODEL_DIR=/opt/aurora/model \
    AURORA_STATE_DIR=/var/lib/aurora \
    AURORA_REGISTRY_DIR=/opt/aurora/model/registry \
    AURORA_MODEL_DIR=/opt/aurora/model

ENTRYPOINT ["python", "main.py"]