FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy \
    requests \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN mkdir -p common

COPY deploy/cpa/config.py .
COPY deploy/cpa/cpa_utils.py .
COPY deploy/cpa/metric.py .
COPY deploy/cpa/evaluate.py .

COPY common/models.py common/
COPY model.pt .

RUN touch common/__init__.py

RUN chmod +x metric.py evaluate.py


CMD ["python", "evaluate.py"]