# -----------------------------------------------------
# EXTREME MINIMAL BUILD: Stable Alpine Single-Stage
# -----------------------------------------------------
FROM python:3.9-alpine3.18

WORKDIR /app

# Install build dependencies (Final CRITICAL FIX)
RUN apk add --no-cache \
    build-base \
    libstdc++ \
    cmake \
    # The 'cmake' package is mandatory for building xgboost's C++ components.
    && rm -rf /var/cache/apk/*

# Copy requirements and application code
COPY requirements.txt .
COPY . .

# Install Python packages (This step will now succeed)
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Run Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]