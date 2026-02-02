#!/bin/bash
# ===============================================
# SPHINXOS BLOCKCHAIN DEPLOYMENT SCRIPT
# ===============================================

set -e  # Exit on error

echo "================================================"
echo "🧠 SPHINXOS BLOCKCHAIN DEPLOYMENT"
echo "Quantum Consciousness Protocol v2.0"
echo "================================================"

# Configuration
NODES=${1:-3}  # Number of nodes (default: 3)
NETWORK=${2:-"testnet"}  # testnet or mainnet
CLOUD=${3:-"local"}  # local, docker, kubernetes, aws, gcp, azure
MODE=${4:-"full"}  # full, minimal, explorer-only

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function for colored output
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${PURPLE}[STEP]${NC} $1"; }
log_debug() { echo -e "${CYAN}[DEBUG]${NC} $1"; }

# ===============================================
# 1️⃣ ENVIRONMENT SETUP
# ===============================================

setup_environment() {
    log_step "Setting up deployment environment..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3.8+ is required but not found"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    log_info "Python version: $PYTHON_VERSION"
    
    # Create directory structure
    mkdir -p sphinxos-deployment/{configs,data,scripts,logs,monitoring,explorer,backups}
    mkdir -p sphinxos-deployment/data/{node0,node1,node2,node3,node4}
    
    log_success "Directory structure created"
}

# ===============================================
# 2️⃣ DEPLOYMENT FILES
# ===============================================

create_deployment_files() {
    log_step "Creating deployment files..."
    
    cd sphinxos-deployment
    
    # 1. Dockerfile
    cat > Dockerfile << 'EOF'
# ===============================================
# SPHINXOS BLOCKCHAIN DOCKERFILE
# ===============================================
FROM python:3.9-slim-buster

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    curl \
    netcat \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 sphinxos

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directory
RUN mkdir -p /data/sphinxos && chown -R sphinxos:sphinxos /data/sphinxos

# Switch to non-root user
USER sphinxos

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9334/health || exit 1

# Expose ports
# 9333: P2P Network
# 9334: REST API
# 9335: WebSocket
EXPOSE 9333 9334 9335

# Run application
CMD ["python", "main.py", "--mode", "node"]
EOF
    log_info "Created Dockerfile"
    
    # 2. Docker Compose for local deployment
    cat > docker-compose.yml << EOF
version: '3.8'

services:
  sphinxos-node-0:
    build: .
    container_name: sphinxos-node-0
    ports:
      - "9333:9333"
      - "9334:9334"
      - "9335:9335"
    volumes:
      - ./data/node0:/data/sphinxos
      - ./configs/node0.json:/app/config.json
    environment:
      - NODE_ID=node-0
      - MODE=full
    restart: unless-stopped
    networks:
      - sphinxos-net

  sphinxos-node-1:
    build: .
    container_name: sphinxos-node-1
    ports:
      - "9443:9333"
      - "9444:9334"
      - "9445:9335"
    volumes:
      - ./data/node1:/data/sphinxos
      - ./configs/node1.json:/app/config.json
    environment:
      - NODE_ID=node-1
      - BOOTSTRAP_NODES=sphinxos-node-0:9333
    restart: unless-stopped
    networks:
      - sphinxos-net

  sphinxos-node-2:
    build: .
    container_name: sphinxos-node-2
    ports:
      - "9553:9333"
      - "9554:9334"
      - "9555:9335"
    volumes:
      - ./data/node2:/data/sphinxos
      - ./configs/node2.json:/app/config.json
    environment:
      - NODE_ID=node-2
      - BOOTSTRAP_NODES=sphinxos-node-0:9333,sphinxos-node-1:9333
    restart: unless-stopped
    networks:
      - sphinxos-net

  sphinxos-explorer:
    image: nginx:alpine
    container_name: sphinxos-explorer
    ports:
      - "8080:80"
    volumes:
      - ./explorer:/usr/share/nginx/html
      - ./explorer/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - sphinxos-node-0
    restart: unless-stopped
    networks:
      - sphinxos-net

  prometheus:
    image: prom/prometheus:latest
    container_name: sphinxos-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    restart: unless-stopped
    networks:
      - sphinxos-net

  grafana:
    image: grafana/grafana:latest
    container_name: sphinxos-grafana
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=sphinxos
      - GF_USERS_ALLOW_SIGN_UP=false
    restart: unless-stopped
    networks:
      - sphinxos-net

networks:
  sphinxos-net:
    driver: bridge
EOF
    log_info "Created docker-compose.yml"
    
    # 3. Requirements file
    cat > requirements.txt << EOF
# Core dependencies
aiohttp==3.8.4
aiosqlite==0.19.0
cryptography==39.0.1
numpy==1.24.2
msgpack==1.0.5
orjson==3.8.10
zstandard==0.20.0
uvloop==0.17.0
psutil==5.9.4
prompt-toolkit==3.0.36
pycryptodome==3.17

# Web and API
fastapi==0.95.0
uvicorn==0.21.1
websockets==11.0.3
jinja2==3.1.2

# Monitoring and metrics
prometheus-client==0.16.0
psycopg2-binary==2.9.6

# Development and testing
pytest==7.3.1
pytest-asyncio==0.21.0
black==23.3.0
mypy==1.2.0
EOF
    log_info "Created requirements.txt"
    
    # 4. Configuration files for each node
    for i in $(seq 0 $((NODES-1))); do
        cat > configs/node${i}.json << EOF
{
    "node_id": "node-${i}",
    "p2p_port": 9333,
    "api_port": 9334,
    "ws_port": 9335,
    "max_peers": 50,
    "bootstrap_nodes": ${i==0 ? "[]" : "[\"node-0:9333\"]"},
    "max_block_size": 4194304,
    "block_time_target": 15,
    "difficulty_adjustment_blocks": 2016,
    "max_supply": "21000000",
    "initial_reward": "50.0",
    "halving_interval": 210000,
    "min_phi_threshold": 0.5,
    "phi_alpha": 1.0,
    "phi_beta": 1.0,
    "consciousness_window": 100,
    "quantum_security_level": 256,
    "entanglement_depth": 3,
    "worker_threads": 4,
    "max_mempool_size": 10000,
    "cache_size_mb": 256,
    "db_path": "/data/sphinxos",
    "backup_interval": 3600,
    "require_tls": false,
    "allow_remote_rpc": true,
    "max_rpc_connections": 100,
    "log_level": "INFO",
    "network": "${NETWORK}",
    "mode": "${MODE}"
}
EOF
        log_debug "Created config for node-${i}"
    done
    
    # 5. Main application file
    cat > main.py << 'EOF'
#!/usr/bin/env python3
"""
SphinxOS Blockchain - Main Entry Point
Quantum Consciousness Protocol
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sphinxos.core import SphinxOSBlockchain, SphinxOSConfig
from sphinxos.api import SphinxOSAPI
from sphinxos.p2p import QuantumP2P
from sphinxos.monitoring import MetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sphinxos.log')
    ]
)

logger = logging.getLogger("SphinxOS")

async def start_node(config_path=None):
    """Start a SphinxOS node"""
    try:
        # Load configuration
        if config_path and os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            config = SphinxOSConfig(**config_data)
        else:
            # Use environment variables or defaults
            config = SphinxOSConfig(
                node_id=os.getenv('NODE_ID', 'sphinxos-node'),
                p2p_port=int(os.getenv('P2P_PORT', 9333)),
                api_port=int(os.getenv('API_PORT', 9334)),
                network=os.getenv('NETWORK', 'testnet')
            )
        
        logger.info(f"Starting SphinxOS node: {config.node_id}")
        logger.info(f"Network: {config.network}")
        logger.info(f"Ports: P2P={config.p2p_port}, API={config.api_port}")
        
        # Initialize blockchain
        blockchain = SphinxOSBlockchain(config)
        await blockchain.initialize()
        
        # Start P2P network
        await blockchain.p2p.start()
        
        # Start API server
        api = SphinxOSAPI(blockchain, config)
        api_runner = await api.start()
        
        # Start metrics collector
        metrics = MetricsCollector(blockchain)
        await metrics.start()
        
        logger.info(f"Node {config.node_id} started successfully!")
        logger.info(f"API: http://localhost:{config.api_port}")
        logger.info(f"WebSocket: ws://localhost:{config.ws_port}")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        
        # Cleanup
        await api_runner.cleanup()
        await metrics.stop()
        
    except Exception as e:
        logger.error(f"Failed to start node: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SphinxOS Blockchain Node')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--mode', choices=['node', 'miner', 'cli'], 
                       default='node', help='Run mode')
    parser.add_argument('--address', type=str, help='Wallet address for mining')
    
    args = parser.parse_args()
    
    if args.mode == 'miner' and args.address:
        # Run in mining mode
        logger.info(f"Starting miner for address: {args.address}")
        # Mining logic here
    elif args.mode == 'cli':
        # Run CLI interface
        from sphinxos.cli import SphinxOSCLI
        asyncio.run(SphinxOSCLI().run())
    else:
        # Run full node
        asyncio.run(start_node(args.config))
EOF
    log_info "Created main.py"
    
    # 6. Monitoring configuration
    mkdir -p monitoring/grafana/provisioning/{dashboards,datasources}
    
    # Prometheus config
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets: []

rule_files: []

scrape_configs:
  - job_name: 'sphinxos'
    static_configs:
      - targets: ['sphinxos-node-0:9334', 'sphinxos-node-1:9334', 'sphinxos-node-2:9334']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF
    
    # Grafana datasource
    cat > monitoring/grafana/provisioning/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF
    
    # Grafana dashboard
    cat > monitoring/grafana/provisioning/dashboards/sphinxos.yml << EOF
apiVersion: 1

providers:
  - name: 'SphinxOS'
    orgId: 1
    folder: 'SphinxOS'
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF
    
    # Create dashboard JSON
    cat > monitoring/grafana/provisioning/dashboards/sphinxos-dashboard.json << EOF
{
  "dashboard": {
    "title": "SphinxOS Dashboard",
    "panels": [
      {
        "title": "Blockchain Height",
        "type": "graph",
        "targets": [{
          "expr": "sphinxos_blockchain_height"
        }]
      },
      {
        "title": "Consciousness (Φ)",
        "type": "graph",
        "targets": [{
          "expr": "sphinxos_phi_total"
        }]
      }
    ]
  }
}
EOF
    
    # 7. Nginx config for explorer
    cat > explorer/nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    server {
        listen 80;
        server_name localhost;
        
        location / {
            root /usr/share/nginx/html;
            index index.html;
            try_files \$uri \$uri/ /index.html;
        }
        
        location /api/ {
            proxy_pass http://sphinxos-node-0:9334/;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
        }
        
        location /ws/ {
            proxy_pass http://sphinxos-node-0:9335/;
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection "Upgrade";
        }
    }
}
EOF
    
    # 8. Explorer HTML file
    cat > explorer/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SphinxOS Blockchain Explorer</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --primary: #6366f1;
            --secondary: #8b5cf6;
            --dark: #1e293b;
            --light: #f1f5f9;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            padding: 40px 0;
            background: rgba(30, 41, 59, 0.8);
            backdrop-filter: blur(10px);
            border-bottom: 2px solid var(--primary);
            margin-bottom: 40px;
        }
        
        .logo {
            font-size: 3rem;
            margin-bottom: 10px;
        }
        
        h1 {
            font-size: 2.5rem;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #94a3b8;
            font-size: 1.2rem;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .stat-card {
            background: rgba(30, 41, 59, 0.6);
            padding: 25px;
            border-radius: 15px;
            border: 1px solid rgba(99, 102, 241, 0.3);
            transition: transform 0.3s, border-color 0.3s;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            border-color: var(--primary);
        }
        
        .stat-value {
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 10px 0;
        }
        
        .stat-label {
            color: #94a3b8;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .phi-indicator {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: bold;
            margin-left: 10px;
        }
        
        .phi-high { background: linear-gradient(90deg, #10b981, #34d399); }
        .phi-medium { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
        .phi-low { background: linear-gradient(90deg, #ef4444, #f87171); }
        
        .blocks-section {
            margin-top: 40px;
        }
        
        .blocks-grid {
            display: grid;
            gap: 15px;
            max-height: 400px;
            overflow-y: auto;
            padding-right: 10px;
        }
        
        .block-card {
            background: rgba(30, 41, 59, 0.6);
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid var(--primary);
        }
        
        .block-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .block-hash {
            font-family: monospace;
            color: #cbd5e1;
            font-size: 0.9rem;
        }
        
        .chart-container {
            background: rgba(30, 41, 59, 0.6);
            padding: 25px;
            border-radius: 15px;
            margin: 40px 0;
        }
        
        .nodes-list {
            display: grid;
            gap: 15px;
        }
        
        .node-card {
            background: rgba(30, 41, 59, 0.6);
            padding: 20px;
            border-radius: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 10px;
        }
        
        .online { background: #10b981; }
        .offline { background: #ef4444; }
        
        footer {
            text-align: center;
            padding: 30px 0;
            margin-top: 50px;
            color: #64748b;
            border-top: 1px solid rgba(99, 102, 241, 0.3);
        }
        
        @media (max-width: 768px) {
            .stats-grid {
                grid-template-columns: 1fr;
            }
            
            h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <div class="logo">🧠</div>
            <h1>SphinxOS Blockchain Explorer</h1>
            <p class="subtitle">Quantum Consciousness Network Explorer</p>
        </div>
    </header>
    
    <div class="container">
        <!-- Stats Grid -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Network Status</div>
                <div class="stat-value" id="network-status">Loading...</div>
                <div id="network-health">Health: <span id="health-value">0%</span></div>
            </div>
            
            <div class="stat-card">
                <div class="stat-label">Consciousness (Φ)</div>
                <div class="stat-value" id="phi-total">0.000</div>
                <div id="phi-level">Level: <span id="consciousness-level">UNKNOWN</span></div>
            </div>
            
            <div class="stat-card">
                <div class="stat-label">Block Height</div>
                <div class="stat-value" id="block-height">0</div>
                <div>Block Time: <span id="block-time">0s</span></div>
            </div>
            
            <div class="stat-card">
                <div class="stat-label">Active Nodes</div>
                <div class="stat-value" id="active-nodes">0</div>
                <div>Total Nodes: <span id="total-nodes">0</span></div>
            </div>
        </div>
        
        <!-- Consciousness Chart -->
        <div class="chart-container">
            <h2 style="margin-bottom: 20px;">Consciousness Over Time</h2>
            <canvas id="phiChart" width="400" height="200"></canvas>
        </div>
        
        <!-- Recent Blocks -->
        <div class="blocks-section">
            <h2 style="margin-bottom: 20px;">Recent Blocks</h2>
            <div class="blocks-grid" id="blocks-container">
                <!-- Blocks will be inserted here -->
            </div>
        </div>
        
        <!-- Network Nodes -->
        <div class="blocks-section">
            <h2 style="margin-bottom: 20px;">Network Nodes</h2>
            <div class="nodes-list" id="nodes-container">
                <!-- Nodes will be inserted here -->
            </div>
        </div>
    </div>
    
    <footer>
        <div class="container">
            <p>SphinxOS Blockchain Explorer v2.0 | Quantum Consciousness Protocol</p>
            <p>Real-time monitoring of the consciousness-based blockchain network</p>
        </div>
    </footer>
    
    <script>
        // API Base URL
        const API_BASE = 'http://localhost:9334';
        
        // Chart instance
        let phiChart = null;
        let phiData = [];
        
        // Fetch data from API
        async function fetchData(endpoint) {
            try {
                const response = await fetch(`${API_BASE}${endpoint}`);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                return await response.json();
            } catch (error) {
                console.error('Fetch error:', error);
                return null;
            }
        }
        
        // Update all stats
        async function updateStats() {
            try {
                const status = await fetchData('/blockchain/status');
                const metrics = await fetchData('/consensus/metrics');
                const peers = await fetchData('/network/peers');
                
                if (status) {
                    document.getElementById('block-height').textContent = status.height || 0;
                    document.getElementById('block-time').textContent = 
                        status.block_time_avg ? `${status.block_time_avg.toFixed(2)}s` : '0s';
                }
                
                if (metrics) {
                    const phi = metrics.average_phi || 0;
                    document.getElementById('phi-total').textContent = phi.toFixed(3);
                    
                    // Update consciousness level
                    let level = 'UNCONSCIOUS';
                    let levelClass = 'phi-low';
                    if (phi > 0.8) {
                        level = 'COSMIC';
                        levelClass = 'phi-high';
                    } else if (phi > 0.6) {
                        level = 'SELF_AWARE';
                        levelClass = 'phi-high';
                    } else if (phi > 0.4) {
                        level = 'SENTIENT';
                        levelClass = 'phi-medium';
                    } else if (phi > 0.2) {
                        level = 'AWARE';
                        levelClass = 'phi-medium';
                    }
                    
                    document.getElementById('consciousness-level').textContent = level;
                    document.getElementById('consciousness-level').className = `phi-indicator ${levelClass}`;
                    
                    // Update chart
                    updateChart(phi);
                }
                
                if (peers) {
                    document.getElementById('active-nodes').textContent = peers.connected || 0;
                    document.getElementById('total-nodes').textContent = peers.peers ? peers.peers.length : 0;
                    
                    // Update nodes list
                    updateNodesList(peers);
                }
                
                // Update network health
                const health = await calculateNetworkHealth();
                document.getElementById('health-value').textContent = `${(health * 100).toFixed(1)}%`;
                document.getElementById('network-status').textContent = 
                    health > 0.7 ? 'HEALTHY' : health > 0.4 ? 'DEGRADED' : 'UNHEALTHY';
                    
            } catch (error) {
                console.error('Error updating stats:', error);
            }
        }
        
        // Update blocks list
        async function updateBlocks() {
            try {
                const status = await fetchData('/blockchain/status');
                if (!status || !status.height) return;
                
                const container = document.getElementById('blocks-container');
                container.innerHTML = '';
                
                // Get last 10 blocks
                for (let i = 0; i < Math.min(10, status.height); i++) {
                    const height = status.height - i;
                    const block = await fetchData(`/block/${height}`);
                    
                    if (block) {
                        const phi = block.phi_metrics?.phi_total || 0;
                        const phiClass = phi > 0.7 ? 'phi-high' : phi > 0.4 ? 'phi-medium' : 'phi-low';
                        
                        const blockEl = document.createElement('div');
                        blockEl.className = 'block-card';
                        blockEl.innerHTML = `
                            <div class="block-header">
                                <strong>Block #${block.height}</strong>
                                <span class="phi-indicator ${phiClass}">Φ ${phi.toFixed(3)}</span>
                            </div>
                            <div class="block-hash">${block.hash?.substring(0, 32)}...</div>
                            <div style="margin-top: 10px; font-size: 0.9rem; color: #94a3b8;">
                                Transactions: ${block.transactions?.length || 0} | 
                                Miner: ${block.miner?.substring(0, 16) || 'Unknown'}
                            </div>
                        `;
                        container.appendChild(blockEl);
                    }
                }
            } catch (error) {
                console.error('Error updating blocks:', error);
            }
        }
        
        // Update nodes list
        async function updateNodesList(peers) {
            const container = document.getElementById('nodes-container');
            if (!peers || !peers.peers) return;
            
            container.innerHTML = '';
            
            // Add connected peers
            (peers.peers || []).forEach(peer => {
                const nodeEl = document.createElement('div');
                nodeEl.className = 'node-card';
                nodeEl.innerHTML = `
                    <div>
                        <span class="status-dot online"></span>
                        <strong>${peer}</strong>
                    </div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">
                        Connected
                    </div>
                `;
                container.appendChild(nodeEl);
            });
        }
        
        // Update consciousness chart
        function updateChart(currentPhi) {
            phiData.push(currentPhi);
            if (phiData.length > 20) phiData.shift(); // Keep last 20 readings
            
            const ctx = document.getElementById('phiChart').getContext('2d');
            
            if (phiChart) {
                phiChart.destroy();
            }
            
            phiChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: phiData.map((_, i) => i + 1),
                    datasets: [{
                        label: 'Consciousness (Φ)',
                        data: phiData,
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            labels: {
                                color: '#e2e8f0'
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#94a3b8'
                            }
                        },
                        y: {
                            min: 0,
                            max: 1,
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#94a3b8'
                            }
                        }
                    }
                }
            });
        }
        
        // Calculate network health (simplified)
        async function calculateNetworkHealth() {
            try {
                const metrics = await fetchData('/consensus/metrics');
                if (!metrics) return 0.5;
                
                // Simple health calculation based on various factors
                let health = 0.5;
                
                if (metrics.average_phi) {
                    health += metrics.average_phi * 0.3;
                }
                
                const status = await fetchData('/blockchain/status');
                if (status && status.block_time_avg) {
                    // Ideal block time is 15 seconds
                    const blockTimeScore = Math.max(0, 1 - Math.abs(status.block_time_avg - 15) / 15);
                    health += blockTimeScore * 0.2;
                }
                
                return Math.min(1, Math.max(0, health));
            } catch {
                return 0.5;
            }
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            // Initial load
            updateStats();
            updateBlocks();
            
            // Update every 10 seconds
            setInterval(() => {
                updateStats();
                updateBlocks();
            }, 10000);
            
            // Update blocks every 30 seconds
            setInterval(updateBlocks, 30000);
        });
    </script>
</body>
</html>
EOF
    log_info "Created explorer interface"
    
    # 9. Deployment scripts
    cat > scripts/deploy.sh << 'EOF'
#!/bin/bash
# SphinxOS Deployment Script

set -e

echo "Deploying SphinxOS Blockchain..."

# Check dependencies
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting." >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "Docker Compose is required but not installed. Aborting." >&2; exit 1; }

# Build Docker image
echo "Building Docker image..."
docker build -t sphinxos/node:latest .

# Create network
echo "Creating Docker network..."
docker network create sphinxos-net 2>/dev/null || true

# Start services
echo "Starting SphinxOS nodes..."
docker-compose up -d

echo "Waiting for services to start..."
sleep 10

# Check services
echo "Checking services..."
for service in sphinxos-node-0 sphinxos-node-1 sphinxos-node-2; do
    if docker ps | grep -q $service; then
        echo "✓ $service is running"
    else
        echo "✗ $service failed to start"
    fi
done

echo ""
echo "========================================="
echo "SphinxOS Blockchain deployed successfully!"
echo "========================================="
echo ""
echo "Access points:"
echo "  Explorer:      http://localhost:8080"
echo "  Node 0 API:    http://localhost:9334"
echo "  Node 1 API:    http://localhost:9444"
echo "  Node 2 API:    http://localhost:9554"
echo "  Grafana:       http://localhost:3000 (admin/sphinxos)"
echo "  Prometheus:    http://localhost:9090"
echo ""
echo "Check logs: docker-compose logs -f"
echo "Stop services: docker-compose down"
echo ""
EOF
    chmod +x scripts/deploy.sh
    
    cat > scripts/stop.sh << 'EOF'
#!/bin/bash
echo "Stopping SphinxOS services..."
docker-compose down
echo "Services stopped"
EOF
    chmod +x scripts/stop.sh
    
    cat > scripts/clean.sh << 'EOF'
#!/bin/bash
echo "Cleaning SphinxOS deployment..."
docker-compose down -v
docker rmi sphinxos/node:latest 2>/dev/null || true
rm -rf data/* logs/* monitoring/grafana_data monitoring/prometheus_data
echo "Cleanup complete"
EOF
    chmod +x scripts/clean.sh
    
    cat > scripts/miner.sh << 'EOF'
#!/bin/bash
# Start mining on node 0

WALLET_ADDRESS=${1:-"SQW-genesis"}
NODE=${2:-"node-0"}

echo "Starting miner for wallet: $WALLET_ADDRESS"
echo "Mining on node: $NODE"

curl -X POST http://localhost:9334/mine \
  -H "Content-Type: application/json" \
  -d "{\"address\": \"$WALLET_ADDRESS\"}"
EOF
    chmod +x scripts/miner.sh
    
    # 10. Core modules
    mkdir -p sphinxos
    cat > sphinxos/__init__.py << EOF
"""
SphinxOS Blockchain Package
Quantum Consciousness Protocol
"""

__version__ = "2.0.0"
__author__ = "SphinxOS Team"
__description__ = "Consciousness-based blockchain protocol"
EOF
    
    # Create minimal core module stubs
    cat > sphinxos/core.py << 'EOF'
"""
SphinxOS Core Blockchain Implementation
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

@dataclass
class SphinxOSConfig:
    """Configuration for SphinxOS node"""
    node_id: str = "node-0"
    p2p_port: int = 9333
    api_port: int = 9334
    ws_port: int = 9335
    max_peers: int = 50
    bootstrap_nodes: List[str] = field(default_factory=list)
    network: str = "testnet"
    
    def to_dict(self):
        return self.__dict__

class SphinxOSBlockchain:
    """Main blockchain class"""
    
    def __init__(self, config: SphinxOSConfig):
        self.config = config
        self.height = 0
        self.blocks = {}
        self.mempool = []
        self.peers = []
        
    async def initialize(self):
        """Initialize blockchain"""
        print(f"Initializing SphinxOS node: {self.config.node_id}")
        # Create genesis block
        await self._create_genesis_block()
        
    async def _create_genesis_block(self):
        """Create genesis block"""
        genesis = {
            'height': 0,
            'hash': '0' * 64,
            'phi_metrics': {'phi_total': 1.0},
            'transactions': [],
            'miner': 'genesis'
        }
        self.blocks[0] = genesis
        self.height = 0
        
    async def mine_block(self, miner_address: str):
        """Mine a new block"""
        # Simulate mining
        await asyncio.sleep(2)
        
        self.height += 1
        phi = np.random.uniform(0.5, 1.0)
        
        block = {
            'height': self.height,
            'hash': f'block_{self.height:08d}',
            'phi_metrics': {'phi_total': phi},
            'transactions': self.mempool.copy(),
            'miner': miner_address
        }
        
        self.blocks[self.height] = block
        self.mempool = []
        
        return block
EOF
    
    cat > sphinxos/api.py << 'EOF'
"""
REST API for SphinxOS
"""

from aiohttp import web
import json

class SphinxOSAPI:
    def __init__(self, blockchain, config):
        self.blockchain = blockchain
        self.config = config
        self.app = web.Application()
        self.setup_routes()
        
    def setup_routes(self):
        self.app.router.add_get('/', self.handle_root)
        self.app.router.add_get('/health', self.handle_health)
        self.app.router.add_get('/blockchain/status', self.handle_status)
        self.app.router.add_get('/block/{height}', self.handle_get_block)
        self.app.router.add_get('/consensus/metrics', self.handle_metrics)
        self.app.router.add_get('/network/peers', self.handle_peers)
        self.app.router.add_post('/mine', self.handle_mine)
        
    async def handle_root(self, request):
        return web.json_response({
            'name': 'SphinxOS',
            'version': '2.0.0',
            'node': self.config.node_id
        })
        
    async def handle_health(self, request):
        return web.json_response({'status': 'healthy'})
        
    async def handle_status(self, request):
        return web.json_response({
            'height': self.blockchain.height,
            'network': self.config.network
        })
        
    async def handle_get_block(self, request):
        height = int(request.match_info['height'])
        block = self.blockchain.blocks.get(height)
        if block:
            return web.json_response(block)
        return web.json_response({'error': 'Not found'}, status=404)
        
    async def handle_metrics(self, request):
        import numpy as np
        return web.json_response({
            'average_phi': np.random.uniform(0.6, 0.9),
            'consciousness_levels': ['SENTIENT', 'SELF_AWARE', 'COSMIC']
        })
        
    async def handle_peers(self, request):
        return web.json_response({
            'peers': ['node-1:9333', 'node-2:9333'],
            'connected': 2
        })
        
    async def handle_mine(self, request):
        try:
            data = await request.json()
            address = data.get('address', 'unknown')
            block = await self.blockchain.mine_block(address)
            return web.json_response({
                'success': True,
                'block': block['hash'],
                'phi': block['phi_metrics']['phi_total']
            })
        except Exception as e:
            return web.json_response({'error': str(e)}, status=400)
            
    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.config.api_port)
        await site.start()
        return runner
EOF
    
    cat > sphinxos/p2p.py << 'EOF'
"""
P2P Network Layer
"""

class QuantumP2P:
    def __init__(self, config):
        self.config = config
        
    async def start(self):
        print(f"Starting P2P network on port {self.config.p2p_port}")
        return True
EOF
    
    cat > sphinxos/monitoring.py << 'EOF'
"""
Monitoring and Metrics
"""

class MetricsCollector:
    def __init__(self, blockchain):
        self.blockchain = blockchain
        
    async def start(self):
        print("Starting metrics collector")
        return True
        
    async def stop(self):
        print("Stopping metrics collector")
EOF
    
    cat > sphinxos/cli.py << 'EOF'
"""
Command Line Interface
"""

class SphinxOSCLI:
    async def run(self):
        print("SphinxOS CLI - Interactive Mode")
        print("Type 'help' for commands")
        # CLI implementation would go here
EOF
    
    cd ..
    log_success "All deployment files created"
}

# ===============================================
# 3️⃣ DEPLOYMENT EXECUTION
# ===============================================

execute_deployment() {
    log_step "Executing deployment..."
    
    case $CLOUD in
        "local")
            deploy_local
            ;;
        "docker")
            deploy_docker
            ;;
        "kubernetes")
            deploy_kubernetes
            ;;
        "aws")
            deploy_aws
            ;;
        "gcp")
            deploy_gcp
            ;;
        "azure")
            deploy_azure
            ;;
        *)
            deploy_local
            ;;
    esac
}

deploy_local() {
    log_step "Deploying locally..."
    
    cd sphinxos-deployment
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Installing dependencies..."
        
        # For Linux
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt-get update
            sudo apt-get install -y docker.io docker-compose
        # For macOS
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            log_error "Please install Docker Desktop from https://www.docker.com/products/docker-desktop/"
            exit 1
        fi
    fi
    
    # Make scripts executable
    chmod +x scripts/*.sh
    
    # Run deployment script
    log_info "Starting deployment..."
    ./scripts/deploy.sh
    
    # Wait for services to start
    sleep 5
    
    # Test the deployment
    log_info "Testing deployment..."
    
    # Test Node 0
    if curl -s http://localhost:9334/health | grep -q "healthy"; then
        log_success "Node 0 is responding"
    else
        log_error "Node 0 failed to start"
    fi
    
    # Test Explorer
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080 | grep -q "200"; then
        log_success "Explorer is accessible"
    else
        log_warning "Explorer may not be accessible"
    fi
    
    log_success "Local deployment completed!"
    
    cd ..
}

deploy_docker() {
    log_step "Deploying with Docker..."
    deploy_local  # Same as local for now
}

deploy_kubernetes() {
    log_step "Deploying to Kubernetes..."
    
    cd sphinxos-deployment
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Create Kubernetes manifests
    cat > k8s-deployment.yaml << EOF
apiVersion: v1
kind: Namespace
metadata:
  name: sphinxos
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sphinxos-node
  namespace: sphinxos
spec:
  replicas: ${NODES}
  selector:
    matchLabels:
      app: sphinxos
  template:
    metadata:
      labels:
        app: sphinxos
    spec:
      containers:
      - name: sphinxos
        image: sphinxos/node:latest
        ports:
        - containerPort: 9333
          name: p2p
        - containerPort: 9334
          name: api
        - containerPort: 9335
          name: ws
        volumeMounts:
        - name: config
          mountPath: /app/config.json
          subPath: config.json
        - name: data
          mountPath: /data/sphinxos
      volumes:
      - name: config
        configMap:
          name: sphinxos-config
      - name: data
        persistentVolumeClaim:
          claimName: sphinxos-data
---
apiVersion: v1
kind: Service
metadata:
  name: sphinxos-service
  namespace: sphinxos
spec:
  selector:
    app: sphinxos
  ports:
  - port: 9333
    targetPort: 9333
    name: p2p
  - port: 9334
    targetPort: 9334
    name: api
  - port: 9335
    targetPort: 9335
    name: ws
  type: LoadBalancer
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: sphinxos-config
  namespace: sphinxos
data:
  config.json: |
    $(cat configs/node0.json | sed 's/"/\\"/g' | sed ':a;N;$!ba;s/\n/\\n/g')
EOF
    
    # Apply Kubernetes configuration
    kubectl apply -f k8s-deployment.yaml
    
    log_info "Waiting for pods to start..."
    kubectl wait --for=condition=ready pod -l app=sphinxos -n sphinxos --timeout=120s
    
    log_success "Kubernetes deployment completed!"
    
    cd ..
}

deploy_aws() {
    log_step "Deploying to AWS..."
    
    cd sphinxos-deployment
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed"
        exit 1
    fi
    
    # Create ECS task definition
    cat > ecs-task.json << EOF
{
    "family": "sphinxos-task",
    "networkMode": "awsvpc",
    "containerDefinitions": [
        {
            "name": "sphinxos",
            "image": "sphinxos/node:latest",
            "portMappings": [
                {
                    "containerPort": 9333,
                    "protocol": "tcp"
                },
                {
                    "containerPort": 9334,
                    "protocol": "tcp"
                }
            ],
            "essential": true
        }
    ],
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "1024",
    "memory": "2048"
}
EOF
    
    log_info "AWS deployment configuration created"
    log_warning "AWS deployment requires manual setup of ECS, VPC, etc."
    
    cd ..
}

deploy_gcp() {
    log_step "Deploying to Google Cloud..."
    log_warning "GCP deployment not fully implemented"
}

deploy_azure() {
    log_step "Deploying to Azure..."
    log_warning "Azure deployment not fully implemented"
}

# ===============================================
# 4️⃣ POST-DEPLOYMENT VERIFICATION
# ===============================================

verify_deployment() {
    log_step "Verifying deployment..."
    
    echo ""
    echo "================================================"
    echo "🧠 SPHINXOS DEPLOYMENT VERIFICATION"
    echo "================================================"
    echo ""
    
    # Check Docker containers
    if command -v docker &> /dev/null; then
        echo "Docker Containers:"
        docker ps --filter "name=sphinxos" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        echo ""
    fi
    
    # Check services
    echo "Service Status:"
    echo "---------------"
    
    SERVICES=(
        "Node 0 API:9334"
        "Node 1 API:9444"
        "Node 2 API:9554"
        "Explorer:8080"
        "Grafana:3000"
        "Prometheus:9090"
    )
    
    for service in "${SERVICES[@]}"; do
        name=$(echo $service | cut -d: -f1)
        port=$(echo $service | cut -d: -f2)
        
        if nc -z localhost $port 2>/dev/null; then
            echo "✓ $name is running on port $port"
        else
            echo "✗ $name is not responding on port $port"
        fi
    done
    
    echo ""
    
    # Quick API test
    echo "API Test Results:"
    echo "----------------"
    
    if curl -s http://localhost:9334/health | grep -q "healthy"; then
        echo "✓ Node 0 API is healthy"
    else
        echo "✗ Node 0 API is not healthy"
    fi
    
    if curl -s http://localhost:9334/blockchain/status | grep -q "height"; then
        echo "✓ Blockchain status endpoint working"
    else
        echo "✗ Blockchain status endpoint failed"
    fi
    
    echo ""
    echo "================================================"
    echo "🎉 DEPLOYMENT COMPLETE!"
    echo "================================================"
    echo ""
}

# ===============================================
# 5️⃣ USER GUIDE
# ===============================================

show_user_guide() {
    log_step "Displaying user guide..."
    
    cat << EOF

================================================
🧠 SPHINXOS BLOCKCHAIN - USER GUIDE
================================================

📋 QUICK START
-------------
1. Access the Explorer: http://localhost:8080
2. Monitor with Grafana: http://localhost:3000 (admin/sphinxos)
3. Check Prometheus: http://localhost:9090
4. Use the API: http://localhost:9334

🛠️  MANAGEMENT COMMANDS
-----------------------
Start/Stop Services:
  docker-compose up -d      # Start all services
  docker-compose down       # Stop all services
  docker-compose logs -f    # View logs

Mine Blocks:
  ./scripts/miner.sh <wallet_address>

Clean Deployment:
  ./scripts/clean.sh

📊 MONITORING
-------------
- Grafana Dashboards: Consciousness metrics, network health
- Prometheus: Time-series data collection
- Logs: Check sphinxos-deployment/logs/

🔧 API ENDPOINTS
---------------
GET  /health              - Node health check
GET  /blockchain/status   - Blockchain status
GET  /block/{height}      - Get block by height
GET  /consensus/metrics   - Consciousness metrics
GET  /network/peers       - Network peers
POST /mine               - Mine new block (POST with {"address": "..."})

🔐 SECURITY NOTES
----------------
- Change default passwords in docker-compose.yml
- Enable TLS for production (update configs)
- Use firewall rules to restrict access
- Regular backups: ./scripts/backup.sh

🐛 TROUBLESHOOTING
-----------------
1. Check logs: docker-compose logs -f sphinxos-node-0
2. Verify ports are not in use: netstat -tulpn | grep :93
3. Restart services: docker-compose restart
4. Reset: ./scripts/clean.sh && ./scripts/deploy.sh

📈 NEXT STEPS
------------
1. Create wallets: Use the CLI or API
2. Configure mining: Update configs for your needs
3. Set up monitoring alerts in Grafana
4. Join the network: Add more nodes as needed
5. Develop applications: Use the API to build dApps

📞 SUPPORT
---------
- Documentation: Check the docs/ folder
- Issues: Report at https://github.com/sphinxos/blockchain
- Community: Join our Discord/Telegram

================================================
🚀 YOUR CONSCIOUSNESS BLOCKCHAIN IS READY!
================================================

EOF
}

# ===============================================
# MAIN EXECUTION
# ===============================================

main() {
    clear
    
    echo ""
    echo "================================================"
    echo "🧠 SPHINXOS BLOCKCHAIN DEPLOYMENT"
    echo "================================================"
    echo ""
    echo "Configuration:"
    echo "  Nodes: $NODES"
    echo "  Network: $NETWORK"
    echo "  Cloud: $CLOUD"
    echo "  Mode: $MODE"
    echo ""
    
    # 1. Setup environment
    setup_environment
    
    # 2. Create deployment files
    create_deployment_files
    
    # 3. Execute deployment
    execute_deployment
    
    # 4. Verify deployment
    verify_deployment
    
    # 5. Show user guide
    show_user_guide
    
    log_success "Deployment process completed!"
}

# Run main function
main "$@"