FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y curl tar && \
    pip install --no-cache-dir numpy requests torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir pywavelets==1.8.0 prometheus_client && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -L https://github.com/jthomperoo/custom-pod-autoscaler/releases/download/v2.12.2/custom-pod-autoscaler-linux-amd64.tar.gz -o cpa.tar.gz && \
    tar -xzf cpa.tar.gz && \
    find . -name "custom-pod-autoscaler*" -type f -exec mv {} /custom-pod-autoscaler \; && \
    chmod +x /custom-pod-autoscaler && \
    rm cpa.tar.gz

COPY deploy/cpa/*.py ./
COPY core/models.py core/
COPY core/architectures/ core/architectures/
COPY preprocessing/__init__.py preprocessing/__init__.py
COPY preprocessing/swt/ preprocessing/swt/
COPY shared/__init__.py shared/__init__.py
COPY shared/config_preprocessing_defaults.py shared/config_preprocessing_defaults.py
COPY deploy/model.pt .
COPY deploy/cpa/config.yaml /config.yaml

RUN touch core/__init__.py && chmod +x metric.py evaluate.py

ENTRYPOINT ["/custom-pod-autoscaler"]
