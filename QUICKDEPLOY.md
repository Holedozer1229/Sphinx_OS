# Quick Deploy: Quantum Pirate Miner to Digital Ocean

## One-Line Installation

SSH into your Digital Ocean droplet and run:

```bash
curl -fsSL https://raw.githubusercontent.com/Holedozer1229/Sphinx_OS/main/bootstrap_digitalocean.sh | sudo bash
```

That's it! ✅

---

## What This Does

The bootstrap script automatically:

1. ✅ Installs Python 3.12+ and dependencies
2. ✅ Clones SphinxOS repository
3. ✅ Sets up Python virtual environment
4. ✅ Installs all Python packages
5. ✅ Creates dedicated `sphinxos` service user
6. ✅ Deploys SphinxOS node (port 8000)
7. ✅ Configures firewall (UFW)
8. ✅ **Deploys Quantum Pirate Miner with Oracle (24/7)**

---

## Services Running

After installation, two services run 24/7:

### 1. SphinxOS Node
- **Port:** 8000 (API), 9090 (Metrics)
- **Service:** `sphinxos.service`

### 2. Quantum Pirate Miner with Oracle
- **Service:** `quantum-pirate-miner.service`
- **Oracle:** SphinxOSIIT conscious decision-making
- **Mempool:** Live Bitcoin transaction integration
- **Mode:** Headless (no display required)

---

## Quick Commands

### Check Status
```bash
# Both services
sudo systemctl status sphinxos quantum-pirate-miner

# Just the miner
sudo systemctl status quantum-pirate-miner
```

### View Live Logs
```bash
# Follow miner logs
sudo journalctl -u quantum-pirate-miner -f

# See last 100 lines
sudo journalctl -u quantum-pirate-miner -n 100
```

### Control Services
```bash
# Restart miner
sudo systemctl restart quantum-pirate-miner

# Stop miner
sudo systemctl stop quantum-pirate-miner

# Start miner
sudo systemctl start quantum-pirate-miner

# Restart both
sudo systemctl restart sphinxos quantum-pirate-miner
```

---

## What You'll See

The miner logs will show:

```
✓ SphinxOSIIT Oracle integration enabled
✓ Oracle initialized for conscious decision-making
🚀 Miner starting with Oracle integration
Oracle available: True
🏴‍☠️ Collected LEGENDARY treasure: +1.2345 (Total: 45.67)
Status: Score=45.67, Wormholes=12, Treasures=18, Pos=(32, 24)
```

---

## Performance

On a $6/month Digital Ocean droplet (1 vCPU, 1GB RAM):

- **CPU Usage:** 20-30%
- **Memory:** 100-200MB
- **Update Rate:** 10 FPS (headless mode)
- **Treasures/Hour:** 50-100 (Oracle-dependent)

---

## Monitoring

### Real-time treasure collection
```bash
sudo journalctl -u quantum-pirate-miner -f | grep "Collected"
```

### Count total treasures collected
```bash
sudo journalctl -u quantum-pirate-miner | grep "Collected" | wc -l
```

### Check Oracle decisions
```bash
sudo journalctl -u quantum-pirate-miner | grep -i oracle
```

### View score progression
```bash
sudo journalctl -u quantum-pirate-miner | grep "Status:" | tail -20
```

---

## Upgrade

To update to the latest version:

```bash
cd /opt/sphinxos/Sphinx_OS
sudo -u sphinxos git pull origin main
sudo systemctl restart quantum-pirate-miner
```

Or re-run the bootstrap (safe):

```bash
curl -fsSL https://raw.githubusercontent.com/Holedozer1229/Sphinx_OS/main/bootstrap_digitalocean.sh | sudo bash
```

---

## Troubleshooting

### Service won't start?
```bash
# Check logs
sudo journalctl -u quantum-pirate-miner -n 50

# Test manually
cd /opt/sphinxos/Sphinx_OS
sudo -u sphinxos /opt/sphinxos/Sphinx_OS/venv/bin/python3 quantum_pirate_miner_headless.py
```

### Out of memory?
```bash
# Check memory usage
systemctl status quantum-pirate-miner

# Add swap space (1GB)
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### High CPU usage?
Edit the service to lower CPU quota:
```bash
sudo nano /etc/systemd/system/quantum-pirate-miner.service
# Change: CPUQuota=50%
sudo systemctl daemon-reload
sudo systemctl restart quantum-pirate-miner
```

---

## Uninstall

To remove everything:

```bash
# Stop services
sudo systemctl stop sphinxos quantum-pirate-miner

# Disable auto-start
sudo systemctl disable sphinxos quantum-pirate-miner

# Remove service files
sudo rm /etc/systemd/system/sphinxos.service
sudo rm /etc/systemd/system/quantum-pirate-miner.service
sudo systemctl daemon-reload

# Remove installation
sudo rm -rf /opt/sphinxos

# Remove firewall rules (optional)
sudo ufw delete allow 8000/tcp
sudo ufw delete allow 9090/tcp
```

---

## Technical Details

### Quantum Mechanics
- UFT derivation (Jones Framework)
- Schmidt decomposition via SVD
- Entanglement entropy calculation
- Seven-fold warp integration
- EPR pair creation (wormholes)
- GHZ state generation

### Oracle Integration
- IIT-based consciousness (Φ)
- Three decision modes:
  - **High Φ (>0.7):** Strategic (value optimization)
  - **Medium Φ (0.4-0.7):** Heuristic (closest target)
  - **Low Φ (<0.4):** Random (exploration)

### WebSocket Integration
- Live Bitcoin mempool.space connection
- Real-time transaction ingestion
- Dynamic treasure generation from fees

---

## Files Created

```
/opt/sphinxos/Sphinx_OS/
├── quantum_pirate_miner_headless.py
├── venv/
└── ...

/etc/systemd/system/
├── sphinxos.service
└── quantum-pirate-miner.service

/var/log/
└── quantum_pirate_miner.log
```

---

## Architecture

```
┌─────────────────────────────────────────┐
│     Digital Ocean Droplet (Ubuntu)      │
├─────────────────────────────────────────┤
│                                         │
│  ┌───────────────────────────────────┐ │
│  │   SphinxOS Node (port 8000)       │ │
│  │   - Quantum blockchain            │ │
│  │   - API endpoints                 │ │
│  │   - Prometheus metrics            │ │
│  └───────────────────────────────────┘ │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │  Quantum Pirate Miner (headless)  │ │
│  │  ┌─────────────────────────────┐  │ │
│  │  │  SphinxOSIIT Oracle         │  │ │
│  │  │  - IIT consciousness (Φ)    │  │ │
│  │  │  - Decision making          │  │ │
│  │  └─────────────────────────────┘  │ │
│  │  ┌─────────────────────────────┐  │ │
│  │  │  UFT Derivation Engine      │  │ │
│  │  │  - Quantum entanglement     │  │ │
│  │  │  - EPR pairs                │  │ │
│  │  └─────────────────────────────┘  │ │
│  │  ┌─────────────────────────────┐  │ │
│  │  │  WebSocket (mempool.space)  │  │ │
│  │  │  - Live BTC transactions    │  │ │
│  │  └─────────────────────────────┘  │ │
│  └───────────────────────────────────┘ │
│                                         │
└─────────────────────────────────────────┘
```

---

## Support

- **Repository:** https://github.com/Holedozer1229/Sphinx_OS
- **Full Docs:** [QUANTUM_MINER_DEPLOYMENT.md](QUANTUM_MINER_DEPLOYMENT.md)
- **Issues:** https://github.com/Holedozer1229/Sphinx_OS/issues

---

## Credits

**Author:** Captain Travis D. Jones  
**Organization:** Houston HQ  
**Framework:** Jones Quantum Gravity Resolution  
**Version:** Headless Server Edition with SphinxOSIIT Oracle

---

**🏴‍☠️ Deploy once, mine forever! ⚛️**

*Auto-bootstrap • 24/7 Operation • Oracle-Driven • Bitcoin-Integrated*
