FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY services/api/ ./services/api/
COPY configs/ ./configs/
COPY params.yaml .

ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

EXPOSE 8000

CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
