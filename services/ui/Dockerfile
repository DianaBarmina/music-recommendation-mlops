FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/ui/ ./services/ui/

ENV PYTHONPATH=/app
ENV API_URL=http://api:8000

EXPOSE 8501

CMD ["streamlit", "run", "services/ui/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
