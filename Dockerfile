# ============================================================================
# Dockerfile — SphinxSkynet Node
# Builds container with Python, Circom, SnarkJS, FastAPI, and Prometheus
# client for multi-cloud zk-hypercube deployment.
# ============================================================================
FROM python:3.11-slim

# Install Node.js (for SnarkJS) and build essentials (for Circom)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential git \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g snarkjs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Circom from prebuilt binary
RUN curl -fsSL https://github.com/iden3/circom/releases/latest/download/circom-linux-amd64 \
        -o /usr/local/bin/circom \
    && chmod +x /usr/local/bin/circom

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and circuits
COPY node_main.py .
COPY rarity_api.py .
COPY circuits/ circuits/
COPY frontend/ frontend/

# Expose FastAPI port and Prometheus metrics port
EXPOSE 8000 8001

# Run the rarity API backend
CMD ["uvicorn", "rarity_api:app", "--host", "0.0.0.0", "--port", "8000"]
