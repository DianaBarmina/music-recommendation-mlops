FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appgroup && useradd -m -r -g appgroup appuser


FROM python:3.12-slim AS builder-api
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ cmake \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-api.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir --prefix=/install -r requirements-api.txt


FROM python:3.12-slim AS builder-ui
WORKDIR /app
COPY requirements-ui.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir --prefix=/install -r requirements-ui.txt


FROM base AS api
COPY --from=builder-api /install /usr/local
COPY src/ ./src/
COPY services/api/ ./services/api/
COPY configs/ ./configs/
COPY params.yaml .

RUN mkdir -p data/predictions data/processed models reports && \
    chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn","services.api.main:app","--host","0.0.0.0", "--port","8000","--workers","1"]


FROM base AS ui
COPY --from=builder-ui /install /usr/local
COPY services/ui/ ./services/ui/
COPY configs/ ./configs/

RUN chown -R appuser:appgroup /app
USER appuser

ENV API_URL=http://api:8000

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit","run","services/ui/main.py","--server.port=8501", "--server.address=0.0.0.0","--server.headless=true"]
