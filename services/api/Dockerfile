FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
        --prefix=/install \
        -r requirements-api.txt

FROM python:3.12-slim AS api

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appgroup \
    && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

COPY --from=builder /install /usr/local

COPY src/ ./src/
COPY services/api/ ./services/api/
COPY configs/ ./configs/
COPY params.yaml .

RUN mkdir -p \
        data/raw \
        data/interim \
        data/processed/reference \
        data/predictions \
        models \
        reports \
    && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "services.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
