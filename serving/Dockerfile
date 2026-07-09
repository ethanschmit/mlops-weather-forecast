# serving/Dockerfile
# Packages the FastAPI app + its environment into a portable container.
# The container connects to MLflow on startup to load the Production model.

FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy and install dependencies FIRST — Docker caches this layer
# so rebuilds are fast if you only change code, not requirements
COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY serving/app.py .
COPY config.yaml .
COPY src/ ./src/

# Where MLflow tracking server lives — overridden at runtime if needed
# host.docker.internal resolves to your host machine from inside a container
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

EXPOSE 8000

# Healthcheck: Docker will ping /health every 30s
# Container is marked unhealthy if this fails 3 times
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]