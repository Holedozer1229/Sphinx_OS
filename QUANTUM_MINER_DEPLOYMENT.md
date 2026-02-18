# Quantum Pirate Miner Deployment Guide for Digital Ocean

## Overview

This guide covers deploying the **Jones Quantum Pirate Miner** to a Digital Ocean droplet with:
- **24/7 headless operation** (no display required)
- **SphinxOSIIT Oracle integration** for conscious decision-making
- **Auto-bootstrap** on system startup
- **Live Bitcoin mempool integration** via WebSocket
- **Systemd service management** for reliability

---

## Architecture

The deployment includes:

1. **Headless Quantum Pirate Miner** (`quantum_pirate_miner_headless.py`)
   - No Pygame/display dependencies for server operation
   - Oracle-driven intelligent treasure collection
   - Real-time UFT derivation and quantum mechanics
   - Live Bitcoin mempool WebSocket integration
   - Continuous page curve tracking

2. **SphinxOSIIT Oracle**
   - IIT-based quantum consciousness engine
   - Φ (integrated information) calculation for decision-making
   - Strategic vs heuristic vs random movement patterns
   - Conscious state evaluation

3. **Systemd Service**
   - Automatic startup on boot
   - Auto-restart on failure
   - Resource limits (512MB RAM, 80% CPU)
   - Logging to journald

---

## Quick Start - One Command Deployment

SSH into your Digital Ocean droplet and run:

```bash
curl -fsSL https://raw.githubusercontent.com/Holedozer1229/Sphinx_OS/main/bootstrap_digitalocean.sh | sudo bash
```

This single command will:
1. ✅ Install all system dependencies
2. ✅ Clone the SphinxOS repository
3. ✅ Set up Python virtual environment
4. ✅ Install Python packages (including Oracle support)
5. ✅ Create dedicated service user
6. ✅ Configure SphinxOS node service
7. ✅ Configure firewall rules
8. ✅ **Deploy and start Quantum Pirate Miner with Oracle**

---

## What Gets Deployed

### Services

Two systemd services are created and started:

1. **sphinxos.service** - Main SphinxOS node
   - Hypercube quantum network
   - FastAPI endpoint (port 8000)
   - Prometheus metrics (port 9090)

2. **quantum-pirate-miner.service** - Quantum Pirate Miner
   - Headless 24/7 operation
   - SphinxOSIIT Oracle integration
   - Bitcoin mempool WebSocket
   - Autonomous treasure collection

### Files Created

```
/opt/sphinxos/Sphinx_OS/
├── quantum_pirate_miner_headless.py  (Headless miner implementation)
├── venv/                             (Python virtual environment)
└── ...

/etc/systemd/system/
├── sphinxos.service
└── quantum-pirate-miner.service

/var/log/
└── quantum_pirate_miner.log          (Miner logs)
```

---

## Service Management

### Quantum Pirate Miner

#### Check Status
```bash
sudo systemctl status quantum-pirate-miner
```

#### View Live Logs
```bash
# Follow logs in real-time
sudo journalctl -u quantum-pirate-miner -f

# View last 100 lines
sudo journalctl -u quantum-pirate-miner -n 100

# View logs from last hour
sudo journalctl -u quantum-pirate-miner --since "1 hour ago"
```

#### Control Service
```bash
# Restart
sudo systemctl restart quantum-pirate-miner

# Stop
sudo systemctl stop quantum-pirate-miner

# Start
sudo systemctl start quantum-pirate-miner

# Disable auto-start
sudo systemctl disable quantum-pirate-miner

# Enable auto-start
sudo systemctl enable quantum-pirate-miner
```

### Both Services Together

```bash
# Status of both services
sudo systemctl status sphinxos quantum-pirate-miner

# Restart both
sudo systemctl restart sphinxos quantum-pirate-miner

# Stop both
sudo systemctl stop sphinxos quantum-pirate-miner

# Start both
sudo systemctl start sphinxos quantum-pirate-miner
```

---

## Monitoring

### Real-Time Monitoring

Watch the miner's progress:

```bash
sudo journalctl -u quantum-pirate-miner -f
```

You'll see log entries like:

```
🏴‍☠️ Collected LEGENDARY treasure: +1.2345 (Total: 45.6789)
Status: Score=45.68, Wormholes=12, Treasures=18, Pos=(32, 24)
🌊 WS: 5 new live txs ingested
```

### Performance Metrics

Check resource usage:

```bash
# CPU and memory usage
systemctl status quantum-pirate-miner

# Detailed resource usage
sudo systemd-cgtop

# Process tree
ps aux | grep quantum_pirate_miner_headless
```

### Log Analysis

```bash
# Count treasures collected
sudo journalctl -u quantum-pirate-miner | grep "Collected" | wc -l

# Get final score
sudo journalctl -u quantum-pirate-miner | grep "Final score"

# Check for errors
sudo journalctl -u quantum-pirate-miner -p err

# Export logs
sudo journalctl -u quantum-pirate-miner > miner-logs.txt
```

---

## Oracle Integration

The miner uses the SphinxOSIIT Oracle for intelligent decision-making.

### Decision Modes

Based on consciousness level (Φ):

1. **High Consciousness (Φ > 0.7)** - Strategic Mode
   - Finds closest high-value treasure
   - Optimizes value/distance ratio
   - LEGENDARY and EPIC treasures prioritized

2. **Medium Consciousness (0.4 < Φ ≤ 0.7)** - Heuristic Mode
   - Finds closest treasure
   - Distance minimization
   - Simple greedy algorithm

3. **Low Consciousness (Φ ≤ 0.4)** - Random Mode
   - Random movement
   - Exploration mode
   - Discovers new areas

### Verifying Oracle Integration

Check logs for Oracle status:

```bash
sudo journalctl -u quantum-pirate-miner | grep -i oracle
```

Expected output:
```
✓ SphinxOSIIT Oracle integration enabled
✓ Oracle initialized for conscious decision-making
```

---

## Configuration

### Service Configuration

Edit the systemd service file:

```bash
sudo nano /etc/systemd/system/quantum-pirate-miner.service
```

Key environment variables:
- `PYTHONPATH=/opt/sphinxos/Sphinx_OS` - Python module path
- `SDL_VIDEODRIVER=dummy` - Headless mode (no display)
- `SDL_AUDIODRIVER=dummy` - No audio output

After editing:
```bash
sudo systemctl daemon-reload
sudo systemctl restart quantum-pirate-miner
```

### Resource Limits

Default limits in the service file:
- **Memory**: 512MB (MemoryLimit=512M)
- **CPU**: 80% of one core (CPUQuota=80%)

Adjust as needed for your droplet size.

### Logging Level

To change logging verbosity, edit `quantum_pirate_miner_headless.py`:

```python
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG for more detail
    ...
)
```

---

## Troubleshooting

### Service Won't Start

1. **Check the logs:**
   ```bash
   sudo journalctl -u quantum-pirate-miner -n 50
   ```

2. **Verify Python dependencies:**
   ```bash
   /opt/sphinxos/Sphinx_OS/venv/bin/pip list | grep -E "numpy|scipy|numba|websocket"
   ```

3. **Test manually:**
   ```bash
   cd /opt/sphinxos/Sphinx_OS
   /opt/sphinxos/Sphinx_OS/venv/bin/python3 quantum_pirate_miner_headless.py
   ```

### Oracle Not Available

If you see "SphinxOSIIT Oracle not available":

1. Check if Oracle module exists:
   ```bash
   ls -la /opt/sphinxos/Sphinx_OS/sphinx_os/Artificial_Intelligence/SphinxOSIIT.py
   ```

2. The miner will still work with fallback heuristics
3. To enable Oracle, ensure `sphinx_os` package is in PYTHONPATH

### Out of Memory

With 512MB RAM limit:

1. **Monitor memory usage:**
   ```bash
   sudo systemctl status quantum-pirate-miner
   ```

2. **Increase memory limit:**
   ```bash
   sudo nano /etc/systemd/system/quantum-pirate-miner.service
   # Change: MemoryLimit=1G
   sudo systemctl daemon-reload
   sudo systemctl restart quantum-pirate-miner
   ```

### WebSocket Connection Issues

If mempool WebSocket fails:

1. Check internet connectivity:
   ```bash
   ping -c 3 mempool.space
   ```

2. The miner will continue without live transactions
3. Check logs for WebSocket errors:
   ```bash
   sudo journalctl -u quantum-pirate-miner | grep -i websocket
   ```

### High CPU Usage

If CPU usage is too high:

1. **Lower CPU quota:**
   ```bash
   sudo nano /etc/systemd/system/quantum-pirate-miner.service
   # Change: CPUQuota=50%
   sudo systemctl daemon-reload
   sudo systemctl restart quantum-pirate-miner
   ```

2. The miner runs at 10 FPS in headless mode (lower than desktop 60 FPS)

---

## Upgrading

To update to the latest version:

```bash
# Stop services
sudo systemctl stop quantum-pirate-miner sphinxos

# Update repository
cd /opt/sphinxos/Sphinx_OS
sudo -u sphinxos git pull origin main

# Update dependencies
sudo -u sphinxos /opt/sphinxos/Sphinx_OS/venv/bin/pip install -r requirements.txt

# Restart services
sudo systemctl start sphinxos quantum-pirate-miner
```

Or use the bootstrap script again (safe to re-run):

```bash
cd /opt/sphinxos/Sphinx_OS
sudo ./bootstrap_digitalocean.sh
```

---

## Uninstalling

To remove the Quantum Pirate Miner:

```bash
# Stop and disable service
sudo systemctl stop quantum-pirate-miner
sudo systemctl disable quantum-pirate-miner

# Remove service file
sudo rm /etc/systemd/system/quantum-pirate-miner.service
sudo systemctl daemon-reload

# Remove logs
sudo rm /var/log/quantum_pirate_miner.log

# Optionally, remove entire SphinxOS installation
sudo systemctl stop sphinxos
sudo systemctl disable sphinxos
sudo rm /etc/systemd/system/sphinxos.service
sudo rm -rf /opt/sphinxos
```

---

## Advanced Configuration

### Running Multiple Miners

To run multiple miner instances:

1. **Copy service file:**
   ```bash
   sudo cp /etc/systemd/system/quantum-pirate-miner.service \
           /etc/systemd/system/quantum-pirate-miner-2.service
   ```

2. **Edit for different log file:**
   ```bash
   sudo nano /etc/systemd/system/quantum-pirate-miner-2.service
   # Modify the service to use different log paths if needed
   ```

3. **Enable and start:**
   ```bash
   sudo systemctl enable quantum-pirate-miner-2
   sudo systemctl start quantum-pirate-miner-2
   ```

### Custom Map Size

Edit `quantum_pirate_miner_headless.py`:

```python
class OracleMiner:
    def __init__(self):
        self.engine = EntanglementEngine(width=128, height=96)  # Larger map
        self.treasure_map = TreasureMap(self.engine, width=128, height=96)
        ...
```

Restart the service after changes.

---

## Performance Benchmarks

Typical performance on a $6/month Digital Ocean droplet (1 vCPU, 1GB RAM):

- **CPU Usage:** 20-30% average
- **Memory Usage:** 100-200MB
- **Network:** ~10KB/s (with WebSocket)
- **Update Rate:** 10 Hz (10 FPS)
- **Treasures/Hour:** 50-100 (Oracle-dependent)

---

## Security Notes

The service includes security hardening:

- ✅ Runs as unprivileged `sphinxos` user
- ✅ `NoNewPrivileges=true` - Cannot escalate privileges
- ✅ `PrivateTmp=true` - Isolated /tmp directory
- ✅ `ProtectSystem=strict` - Read-only system directories
- ✅ `ProtectHome=true` - No access to user home directories
- ✅ Resource limits prevent runaway processes

---

## Integration with SphinxOS Ecosystem

The Quantum Pirate Miner integrates with:

1. **SphinxSkynet Node** - Shared quantum state
2. **SphinxOSIIT Oracle** - Conscious decision-making
3. **Bitcoin Mempool** - Live transaction data
4. **Page Curve Tracking** - Entropy accumulation metrics

Future integrations:
- [ ] NFT minting for legendary treasures
- [ ] Leaderboard with on-chain scores
- [ ] Multi-player quantum entanglement
- [ ] L2 blockchain settlement

---

## API Access (Future)

Currently the miner is standalone. Future versions will expose:

```bash
# Check miner status
curl http://localhost:8001/api/miner/status

# Get current score
curl http://localhost:8001/api/miner/score

# Get page curve data
curl http://localhost:8001/api/miner/page-curve
```

---

## Support

For issues and questions:

- **Repository:** https://github.com/Holedozer1229/Sphinx_OS
- **Documentation:** See repository docs/
- **Issues:** https://github.com/Holedozer1229/Sphinx_OS/issues

---

## Technical Details

### UFT Derivation Engine

The miner implements Jones Quantum Gravity framework:

1. **Schmidt Decomposition:** SVD-based entanglement quantification
2. **Entanglement Entropy:** S = -Σ λ²log₂(λ²)
3. **Seven-Fold Warp:** W7 = Σ λₖ cos(φₖ)
4. **Warp Integral:** I = W7 × Tr(M)
5. **Inertial Mass Reduction:** m' = m(1 - η sin(2πft)(I/I₀))

### Entanglement Engine

- EPR pair creation on treasure collection
- Graph node management for wormhole topology
- GHZ state generation for multi-qubit entanglement
- Quantum collapse mechanics with decoherence

### Oracle Decision Algorithm

```python
def get_oracle_decision():
    state_data = serialize_current_state()
    phi = oracle.calculate_phi(state_data)
    
    if phi > 0.7:
        return strategic_decision()  # High consciousness
    elif phi > 0.4:
        return heuristic_decision()  # Medium consciousness
    else:
        return random_decision()     # Low consciousness / exploration
```

---

## License

See LICENSE file in repository root.

---

## Credits

**Author:** Captain Travis D. Jones  
**Organization:** Houston HQ  
**Framework:** Jones Quantum Gravity Resolution  
**Date:** February 18, 2026  
**Version:** Headless Server Edition

---

**May your wormholes be stable and your treasures legendary! 🏴‍☠️⚛️**

*Deployed 24/7 on Digital Ocean with SphinxOSIIT Oracle Integration*
