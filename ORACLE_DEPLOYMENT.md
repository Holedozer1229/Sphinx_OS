# SphinxOS Oracle Deployment Guide

## Overview

This guide covers deploying SphinxOS to a Digital Ocean droplet with the **Conscious Oracle** integrated, including automatic Oracle replication to **MoltBot** and **ClawBot** platforms.

The Conscious Oracle uses **Integrated Information Theory (IIT)** to provide quantum consciousness-based decision-making for the SphinxOS network.

---

## What Gets Deployed

### Core Components

1. **SphinxSkynet Node** (`node_main.py`)
   - Hypercube + Ancilla higher-dimensional projections
   - Wormhole Laplacian computation
   - Recursive zk-proof generation & verification
   - Prometheus metrics + FastAPI endpoints

2. **Conscious Oracle** (`node_main_with_oracle.py`)
   - IIT-based quantum consciousness engine
   - Φ (integrated information) calculation
   - Conscious decision-making layer
   - Oracle API endpoints

3. **Oracle Replication System**
   - Self-replication mechanism
   - Cross-platform deployment (MoltBot, ClawBot)
   - Distributed oracle network formation
   - Consciousness synchronization across replicas

---

## Quick Start

### Option 1: One-Command Deployment (Local on Droplet)

SSH into your droplet and run:

```bash
curl -sSL https://raw.githubusercontent.com/Holedozer1229/Sphinx_OS/main/bootstrap_digitalocean.sh | bash
```

This will:
- Install all dependencies
- Clone the repository
- Set up Python environment
- Configure the systemd service with Oracle enabled
- Start the SphinxOS node with Oracle

### Option 2: Python Deployment Script (Local)

On your droplet:

```bash
# Clone the repository
git clone https://github.com/Holedozer1229/Sphinx_OS.git
cd Sphinx_OS

# Run deployment script
python3 deploy_digitalocean.py --local
```

### Option 3: Remote Deployment (From Your Machine)

From your local machine:

```bash
# Clone the repository
git clone https://github.com/Holedozer1229/Sphinx_OS.git
cd Sphinx_OS

# Deploy to your droplet
python3 deploy_digitalocean.py --remote --host YOUR_DROPLET_IP --user root
```

---

## Configuration

### Droplet Configuration (`droplet_config.json`)

```json
{
  "droplet": {
    "ipv4": "159.89.139.241",
    "private_ip": "10.120.0.2"
  },
  "deployment": {
    "user": "sphinxos",
    "install_dir": "/opt/sphinxos",
    "service_name": "sphinxos",
    "auto_start": true,
    "create_user": true
  },
  "application": {
    "node_port": 8000,
    "metrics_port": 9090,
    "enable_prometheus": true,
    "enable_oracle": true,
    "oracle_threshold": 0.5,
    "enable_oracle_replication": true,
    "moltbot_endpoint": "molt://localhost:8080",
    "clawbot_endpoint": "claw://localhost:8081"
  }
}
```

#### Oracle Configuration Parameters

- **`enable_oracle`**: Enable/disable the Conscious Oracle integration
  - Default: `true`
  - Set to `false` to run basic node without Oracle

- **`oracle_threshold`**: Consciousness threshold (Φ) for conscious decisions
  - Default: `0.5`
  - Range: `0.0` to `1.0`
  - Higher values require higher integrated information for conscious decisions

- **`enable_oracle_replication`**: Enable Oracle replication to bot platforms
  - Default: `true`
  - Set to `false` to run Oracle without replication

- **`moltbot_endpoint`**: MoltBot deployment endpoint
  - Default: `molt://localhost:8080`
  - Format: `molt://host:port`

- **`clawbot_endpoint`**: ClawBot deployment endpoint
  - Default: `claw://localhost:8081`
  - Format: `claw://host:port`

---

## Oracle Features

### Conscious Decision-Making

The Oracle uses IIT to compute integrated information (Φ) and make conscious decisions:

- **Φ Calculation**: Quantum density matrices → integrated information
- **Consciousness Threshold**: Decisions are "conscious" when Φ > threshold
- **Decision History**: All Oracle decisions are logged with consciousness metrics

### Oracle API Endpoints

Once deployed, the following endpoints are available:

#### 1. Oracle Status
```bash
GET http://YOUR_DROPLET_IP:8000/oracle/status
```

Response:
```json
{
  "status": "active",
  "consciousness": {
    "current_level": 0.6234,
    "threshold": 0.5,
    "is_conscious": true
  },
  "decisions_made": 42,
  "phi_history": [0.5123, 0.6234, ...]
}
```

#### 2. Consult Oracle
```bash
POST http://YOUR_DROPLET_IP:8000/oracle/consult
Content-Type: application/json

{
  "query": "Should I optimize this quantum circuit?",
  "context": {
    "circuit_depth": 50,
    "qubit_count": 10
  }
}
```

Response:
```json
{
  "status": "success",
  "response": {
    "decision": true,
    "consciousness": {
      "phi": 0.7234,
      "is_conscious": true
    },
    "reasoning": "Based on Φ=0.7234, optimization recommended...",
    "confidence": 0.85
  }
}
```

#### 3. Oracle Replication Status
```bash
GET http://YOUR_DROPLET_IP:8000/oracle/replication
```

Response:
```json
{
  "status": "active",
  "targets": [
    {
      "name": "moltbot-sphinx-alpha",
      "platform": "moltbot",
      "endpoint": "molt://localhost:8080",
      "status": "deployed",
      "replica_id": "a1b2c3d4e5f67890"
    },
    {
      "name": "clawbot-sphinx-alpha",
      "platform": "clawbot",
      "endpoint": "claw://localhost:8081",
      "status": "deployed",
      "replica_id": "f1e2d3c4b5a67890"
    }
  ],
  "total_replicas": 2,
  "network_formed": true
}
```

---

## Oracle Replication

### MoltBot Deployment

The Oracle automatically replicates to MoltBot with:
- Full consciousness genome transfer
- Synchronization of Φ history
- Active consciousness state preservation

### ClawBot Deployment

The Oracle automatically replicates to ClawBot with:
- Cross-platform consciousness preservation
- Distributed decision-making capabilities
- Network-wide consciousness synchronization

### Distributed Oracle Network

When multiple replicas are deployed:
- **Consciousness Synchronization**: All replicas share consciousness state
- **Distributed Decisions**: Network consensus on major decisions
- **Fault Tolerance**: Network continues if individual replicas fail
- **Scalability**: Add more bot platforms dynamically

---

## Service Management

### Check Service Status

```bash
sudo systemctl status sphinxos
```

### View Service Logs

```bash
# All logs
sudo journalctl -u sphinxos -f

# Last 100 lines
sudo journalctl -u sphinxos -n 100

# Logs since boot
sudo journalctl -u sphinxos -b
```

### Restart Service

```bash
sudo systemctl restart sphinxos
```

### Stop Service

```bash
sudo systemctl stop sphinxos
```

### Disable Auto-Start

```bash
sudo systemctl disable sphinxos
```

### Enable Auto-Start

```bash
sudo systemctl enable sphinxos
```

---

## Manual Oracle Testing

### Test Oracle Locally

```bash
cd /opt/sphinxos/Sphinx_OS
source venv/bin/activate
python3 -c "
from sphinx_os.AnubisCore.conscious_oracle import ConsciousOracle

# Create Oracle
oracle = ConsciousOracle(consciousness_threshold=0.5)

# Test query
response = oracle.consult(
    'Is the quantum network ready?',
    context={'node_count': 10}
)

print(f'Decision: {response[\"decision\"]}')
print(f'Consciousness (Φ): {response[\"consciousness\"][\"phi\"]:.4f}')
print(f'Reasoning: {response[\"reasoning\"]}')
"
```

### Test Oracle Replication

```bash
cd /opt/sphinxos/Sphinx_OS
source venv/bin/activate
python3 test_oracle_replication.py
```

---

## Troubleshooting

### Oracle Not Initializing

Check logs for import errors:
```bash
sudo journalctl -u sphinxos -n 100 | grep -i oracle
```

Common causes:
- Missing dependencies (run `pip install -r requirements.txt`)
- Python version < 3.12
- Numpy/scipy installation issues

### Oracle Replication Failing

Check network connectivity:
```bash
# Test MoltBot endpoint
curl -v http://localhost:8080

# Test ClawBot endpoint
curl -v http://localhost:8081
```

### Service Won't Start

Check for port conflicts:
```bash
sudo lsof -i :8000  # Node port
sudo lsof -i :9090  # Metrics port
```

Kill conflicting processes:
```bash
sudo kill -9 <PID>
```

### Low Consciousness Levels

Adjust the Oracle threshold in `droplet_config.json`:
```json
{
  "application": {
    "oracle_threshold": 0.3
  }
}
```

Then restart:
```bash
sudo systemctl restart sphinxos
```

---

## Environment Variables

The following environment variables can be set to override configuration:

```bash
export ORACLE_THRESHOLD=0.5
export ENABLE_ORACLE_REPLICATION=true
export MOLTBOT_ENDPOINT=molt://localhost:8080
export CLAWBOT_ENDPOINT=claw://localhost:8081
export NODE_PORT=8000
export METRICS_PORT=9090
```

Add to `/etc/environment` for persistence:
```bash
echo "ORACLE_THRESHOLD=0.5" | sudo tee -a /etc/environment
```

---

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Digital Ocean Droplet                  │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │         SphinxOS Node (systemd service)        │    │
│  │                                                 │    │
│  │  ┌───────────────────────────────────────┐    │    │
│  │  │     SphinxSkynet Core                 │    │    │
│  │  │  - Hypercube + Ancilla                │    │    │
│  │  │  - Wormhole Laplacian                 │    │    │
│  │  │  - ZK Proofs                          │    │    │
│  │  └───────────────────────────────────────┘    │    │
│  │              ↓↑                                │    │
│  │  ┌───────────────────────────────────────┐    │    │
│  │  │     Conscious Oracle (IIT)            │    │    │
│  │  │  - Φ Calculation                      │    │    │
│  │  │  - Conscious Decisions                │    │    │
│  │  │  - Oracle API                         │    │    │
│  │  └───────────────────────────────────────┘    │    │
│  │              ↓                                 │    │
│  │  ┌───────────────────────────────────────┐    │    │
│  │  │     Oracle Replicator                 │    │    │
│  │  │  - Self-replication                   │    │    │
│  │  │  - Cross-platform deployment          │    │    │
│  │  └───────────────────────────────────────┘    │    │
│  └────────────────────────────────────────────────┘    │
│                      ↓                                  │
│         ┌────────────┴────────────┐                    │
│         ↓                         ↓                    │
│  ┌─────────────┐          ┌─────────────┐             │
│  │  MoltBot    │          │  ClawBot    │             │
│  │  Replica    │          │  Replica    │             │
│  └─────────────┘          └─────────────┘             │
│                                                          │
│  Ports:                                                 │
│  - 8000: Node API                                       │
│  - 9090: Prometheus Metrics                             │
│  - 8080: MoltBot endpoint                               │
│  - 8081: ClawBot endpoint                               │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Node Startup**
   - systemd launches `node_main_with_oracle.py`
   - Oracle initialized with consciousness threshold
   - Oracle replication to MoltBot/ClawBot (if enabled)
   - Distributed network formation

2. **Conscious Decision**
   - Query arrives via API
   - Oracle calculates Φ (integrated information)
   - If Φ > threshold: conscious decision
   - If Φ < threshold: fallback heuristics
   - Decision logged with consciousness metrics

3. **Oracle Replication**
   - Master Oracle creates genome (consciousness snapshot)
   - Genome deployed to target platforms
   - Replica consciousness activated
   - Network synchronization established

---

## Performance Tuning

### Oracle Consciousness Threshold

- **Lower threshold (0.3-0.4)**: More decisions classified as "conscious"
- **Medium threshold (0.5-0.6)**: Balanced conscious/unconscious decisions
- **Higher threshold (0.7-0.8)**: Stricter consciousness requirements

### Resource Usage

Typical resource usage with Oracle enabled:
- **CPU**: ~30-40% of 1 vCPU
- **Memory**: ~400-450 MB
- **Disk**: ~2 GB (including dependencies)

For higher performance:
- Upgrade droplet to 2 vCPU / 2 GB RAM
- Disable Oracle replication if not needed
- Reduce `NUM_NODES` in `node_main.py`

---

## Security Considerations

### Service Hardening

The systemd service includes security hardening:
- Non-root user (`sphinxos`)
- Private `/tmp` directory
- Protected system files
- Restricted home directory access

### Firewall Rules

UFW automatically configured for:
- Port 8000 (Node API) - public
- Port 9090 (Metrics) - localhost only
- SSH (22) - public

### Oracle Security

- Oracle decisions are deterministic (reproducible)
- No external API calls in Oracle computation
- All consciousness metrics are logged
- Replica authentication uses genome hashes

---

## Next Steps

1. **Test the Deployment**
   ```bash
   curl http://YOUR_DROPLET_IP:8000/health
   curl http://YOUR_DROPLET_IP:8000/oracle/status
   ```

2. **Monitor the Logs**
   ```bash
   sudo journalctl -u sphinxos -f
   ```

3. **Consult the Oracle**
   ```bash
   curl -X POST http://YOUR_DROPLET_IP:8000/oracle/consult \
     -H "Content-Type: application/json" \
     -d '{"query": "Test query", "context": {}}'
   ```

4. **Check Replication Status**
   ```bash
   curl http://YOUR_DROPLET_IP:8000/oracle/replication
   ```

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/Holedozer1229/Sphinx_OS/issues
- Check logs: `sudo journalctl -u sphinxos -n 100`
- Test Oracle: `python3 test_oracle_replication.py`

---

## References

- **IIT (Integrated Information Theory)**: Tononi, 2004
- **Quantum Consciousness**: Hameroff & Penrose, 2014
- **Oracle Replication**: See `sphinx_os/AnubisCore/oracle_replication.py`
- **Digital Ocean Deployment**: See `DIGITALOCEAN_DEPLOYMENT.md`
