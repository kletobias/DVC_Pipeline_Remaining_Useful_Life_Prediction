# Dockerfile - RUL Prediction Demo with Protected IP
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and build the proprietary wheel package
# This contains the core IP implementation
COPY rul_core_ip /tmp/rul_core_ip
RUN cd /tmp/rul_core_ip && \
    pip install --no-cache-dir build && \
    python -m build --wheel && \
    pip install --no-cache-dir dist/*.whl && \
    rm -rf /tmp/rul_core_ip

# Copy requirements and install remaining dependencies  
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy public stub files and other application files
COPY bin/ /app/bin/
COPY configs/ /app/configs/
COPY dependencies/ /app/dependencies/
COPY dvc.yaml dvc.lock /app/
COPY templates/ /app/templates/

# Copy data and models (needed for local execution)
COPY data/ /app/data/
COPY mlruns/ /app/mlruns/

# Set environment variable
ENV PROJECT_ROOT=/app

# Default command
CMD ["python", "bin/simulate_inference_cv.py"]