# syntax=docker/dockerfile:1

# Pin the language runtime so local, CI and hosted environments use the same
# Python family as the environment that produced the model artifact.
FROM python:3.13-slim

# Python should write logs immediately and should not create .pyc files inside
# the immutable application image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    CHURN_MODEL_PATH=/app/models/churn_logistic_v1.joblib

WORKDIR /app

# Copy dependencies first. Docker can reuse this expensive installation layer
# whenever application code changes but requirements do not.
COPY requirements-api.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt

# The service needs only package code, the bootstrap command, model artifact
# and verified decision snapshot. Raw transactions and notebooks stay outside.
COPY customer_intelligence ./customer_intelligence
COPY scripts/load_decisions_to_database.py ./scripts/load_decisions_to_database.py
COPY models/churn_logistic_v1.joblib ./models/churn_logistic_v1.joblib
COPY data/processed/latest_customer_decisions.csv ./data/processed/latest_customer_decisions.csv

# Running as root would give an exploited API unnecessary operating-system
# privileges. The application only needs read access to its packaged files.
RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup appuser \
    && chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000

# Readiness checks both the model and database, not merely the Python process.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8000')+'/health/ready', timeout=4)"

CMD ["python", "-m", "customer_intelligence.api.server"]
