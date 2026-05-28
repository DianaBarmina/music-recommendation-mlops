FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements-ui.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
        --prefix=/install \
        -r requirements-ui.txt

FROM python:3.12-slim AS ui

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appgroup \
    && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

COPY --from=builder /install /usr/local

COPY services/ui/ ./services/ui/
COPY configs/     ./configs/

RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8501

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=30s \
    --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "services/ui/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
