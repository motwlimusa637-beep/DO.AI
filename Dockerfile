FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY doai ./doai

# Run as a non-root user
RUN useradd --create-home appuser
USER appuser

# Railway sets $PORT. The cron service overrides this command in railway.worker.json.
CMD ["sh", "-c", "uvicorn doai.api:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
