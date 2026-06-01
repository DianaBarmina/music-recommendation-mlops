FROM python:3.12-slim AS builder

WORKDIR /build

COPY services/mlflow/requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.12-slim AS runtime

RUN groupadd --gid 1001 mlflow && \
    useradd --uid 1001 --gid mlflow --shell /bin/bash --create-home mlflow

COPY --from=builder /install /usr/local

RUN mkdir -p /mlflow/artifacts && \
    chown -R mlflow:mlflow /mlflow

WORKDIR /mlflow

USER mlflow

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://mlops:mlops_password@postgres-service:5432/mlops_db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--serve-artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--allowed-hosts", "mlflow-service,mlflow-service:5000,mlflow-service.music-rec.svc.cluster.local,mlflow-service.music-rec.svc.cluster.local:5000,localhost,127.0.0.1"]
