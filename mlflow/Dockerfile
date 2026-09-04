FROM python:3.11-slim

WORKDIR /mlflow

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    mkdir -p /mlflow/artifacts && \
    chmod -R 777 /mlflow

EXPOSE 5000

CMD ["sh", "-c", "mkdir -p ${ARTIFACT_ROOT} && chmod -R 777 ${ARTIFACT_ROOT} 2>/dev/null || true; mlflow server --backend-store-uri ${BACKEND_STORE_URI} --default-artifact-root ${ARTIFACT_ROOT} --host 0.0.0.0 --port 5000 --allowed-hosts '*'"]