# -----------------------------------------------------
# EXTREME MINIMAL BUILD: Stable Alpine Single-Stage
# -----------------------------------------------------
FROM python:3.9-alpine3.18
WORKDIR /app

# Install build dependencies (Final CRITICAL FIX)
# Required for compiling native dependencies like xgboost on Alpine Linux.
RUN apk add --no-cache \
    build-base \
    libstdc++ \
    cmake \
    && rm -rf /var/cache/apk/*

# Copy requirements and application code
COPY requirements.txt .
COPY . .

# CRITICAL FIX: Add the current working directory (/app) to the Python search path.
# This ensures that imports like 'from src.logger import logging' resolve correctly.
ENV PYTHONPATH=/app:$PYTHONPATH

# Install Python packages (This step will now succeed)
RUN pip install --no-cache-dir -r requirements.txt

# Expose port as defined in Dockerrun.aws.json
EXPOSE 5000

# Run Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]