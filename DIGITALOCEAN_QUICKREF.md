# SphinxOS Digital Ocean Quick Reference

## Quick Deploy Commands

### 1. One-Line Bootstrap (Recommended)
```bash
ssh root@159.89.139.241 "curl -fsSL https://raw.githubusercontent.com/Holedozer1229/Sphinx_OS/main/bootstrap_digitalocean.sh | bash"
```

### 2. Manual Bootstrap
```bash
# SSH into droplet
ssh root@159.89.139.241

# Download and run
wget https://raw.githubusercontent.com/Holedozer1229/Sphinx_OS/main/bootstrap_digitalocean.sh
chmod +x bootstrap_digitalocean.sh
./bootstrap_digitalocean.sh
```

### 3. Python Remote Deploy
```bash
# From your local machine
git clone https://github.com/Holedozer1229/Sphinx_OS.git
cd Sphinx_OS
python3 deploy_digitalocean.py --remote --host 159.89.139.241 --user root
```

---

## Service Commands

```bash
# Status
sudo systemctl status sphinxos

# Start
sudo systemctl start sphinxos

# Stop
sudo systemctl stop sphinxos

# Restart
sudo systemctl restart sphinxos

# View logs
sudo journalctl -u sphinxos -f

# Enable auto-start
sudo systemctl enable sphinxos

# Disable auto-start
sudo systemctl disable sphinxos
```

---

## Access Points

- **API:** http://159.89.139.241:8000
- **Metrics:** http://159.89.139.241:9090
- **Health:** http://159.89.139.241:8000/health

---

## Configuration Files

- **Service:** `/etc/systemd/system/sphinxos.service`
- **Installation:** `/opt/sphinxos/Sphinx_OS/`
- **Config:** `/opt/sphinxos/Sphinx_OS/droplet_config.json`
- **Logs:** `journalctl -u sphinxos`

---

## Common Tasks

### Update SphinxOS
```bash
sudo systemctl stop sphinxos
cd /opt/sphinxos/Sphinx_OS
sudo git pull
sudo /opt/sphinxos/Sphinx_OS/venv/bin/pip install -r requirements.txt
sudo systemctl start sphinxos
```

### Check Resource Usage
```bash
# CPU and Memory
htop

# Disk usage
df -h

# Network
netstat -tulpn | grep 8000
```

### Create Swap (if low memory)
```bash
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Backup Configuration
```bash
sudo tar -czf ~/sphinxos-backup-$(date +%Y%m%d).tar.gz /opt/sphinxos/Sphinx_OS
```

---

## Troubleshooting

### Service not starting
```bash
# Check detailed logs
sudo journalctl -u sphinxos -n 100 --no-pager

# Test manually
cd /opt/sphinxos/Sphinx_OS
/opt/sphinxos/Sphinx_OS/venv/bin/python3 node_main.py
```

### Port conflicts
```bash
# Check what's using port 8000
sudo lsof -i :8000

# Change port in service file
sudo nano /etc/systemd/system/sphinxos.service
# Edit NODE_PORT value
sudo systemctl daemon-reload
sudo systemctl restart sphinxos
```

### Memory issues
```bash
# Check memory
free -h

# Add swap (see above)

# Monitor in real-time
watch -n 1 free -h
```

---

## Firewall Management

```bash
# Check status
sudo ufw status

# Allow port
sudo ufw allow 8000/tcp

# Delete rule
sudo ufw delete allow 8000/tcp

# Enable firewall
sudo ufw enable

# Reset firewall
sudo ufw reset
```

---

## Monitoring

### System Health
```bash
# System load
uptime

# Process list
ps aux | grep sphinxos

# Resource usage
top

# Network connections
ss -tuln | grep 8000
```

### Application Metrics
```bash
# Health check
curl http://localhost:8000/health

# Prometheus metrics
curl http://localhost:9090/metrics
```

---

## Security Hardening

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Enable automatic security updates
sudo apt-get install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades

# Check open ports
sudo netstat -tulpn

# Review firewall
sudo ufw status numbered

# Check failed login attempts
sudo journalctl -u ssh | grep "Failed password"
```

---

## Complete Uninstall

```bash
# Stop service
sudo systemctl stop sphinxos
sudo systemctl disable sphinxos

# Remove service
sudo rm /etc/systemd/system/sphinxos.service
sudo systemctl daemon-reload

# Remove installation
sudo rm -rf /opt/sphinxos

# Remove firewall rules
sudo ufw delete allow 8000/tcp
sudo ufw delete allow 9090/tcp
```

---

📚 **Full Documentation:** [DIGITALOCEAN_DEPLOYMENT.md](DIGITALOCEAN_DEPLOYMENT.md)
