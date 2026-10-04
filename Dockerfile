FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN grep -v '^xgboost==' requirements.txt > requirements-docker.txt \
    && pip install --no-cache-dir -r requirements-docker.txt \
    && pip install --no-cache-dir --no-deps xgboost==3.0.4

COPY api/ ./api/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"] 