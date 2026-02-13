FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy \
    requests \
    torch --index-url https://download.pytorch.org/whl/cpu

ADD https://github.com/jthomperoo/custom-pod-autoscaler/releases/download/v2.12.2/custom-pod-autoscaler-linux-amd64 /custom-pod-autoscaler
RUN chmod +x /custom-pod-autoscaler

RUN mkdir -p common

COPY deploy/cpa/config.py .
COPY deploy/cpa/utils.py .
COPY deploy/cpa/metric.py .
COPY deploy/cpa/evaluate.py .
COPY common/models.py common/
COPY model.pt .

RUN touch common/__init__.py
RUN chmod +x metric.py evaluate.py

ENTRYPOINT ["/custom-pod-autoscaler"]