# AURORA-R controller image.
#
# The model bundle is baked in (like PBScaler's RF) so the image is
# self-contained and a stale image can never deploy.
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        --only-binary=:all: torch \
 && pip install --no-cache-dir --only-binary=:all: -r /app/requirements.txt

COPY aurora_r /app/aurora_r
COPY bundle/bundle.pt /opt/aurora-r/registry/bundle.pt

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

ENTRYPOINT ["python", "-m", "aurora_r"]
CMD ["run", "--config", "/etc/aurora-r/config.json"]