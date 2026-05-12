FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y curl tar && \
    pip install --no-cache-dir numpy requests torch --index-url https://download.pytorch.org/whl/cpu && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -L https://github.com/jthomperoo/custom-pod-autoscaler/releases/download/v2.12.2/custom-pod-autoscaler-linux-amd64.tar.gz -o cpa.tar.gz && \
    tar -xzf cpa.tar.gz && \
    find . -name "custom-pod-autoscaler*" -type f -exec mv {} /custom-pod-autoscaler \; && \
    chmod +x /custom-pod-autoscaler && \
    rm cpa.tar.gz

COPY deploy/cpa/*.py ./
COPY core/models.py core/
COPY model.pt .
COPY deploy/cpa/config.yaml /config.yaml

RUN touch core/__init__.py && chmod +x metric.py evaluate.py

ENTRYPOINT ["/custom-pod-autoscaler"]
