# SphinxOS Digital Ocean Deployment Guide

> **Important:** This guide uses `159.89.139.241` as an example IP address. 
> Replace it with your actual Digital Ocean droplet IP address throughout.

## Quick Start - One Command Deployment

### Option 1: Direct Droplet Setup (Recommended)

SSH into your Digital Ocean droplet and run:

```bash
curl -fsSL https://raw.githubusercontent.com/Holedozer1229/Sphinx_OS/main/bootstrap_digitalocean.sh | sudo bash
```

Or manually:

```bash
# SSH into your droplet
ssh root@159.89.139.241

# Download and run the bootstrap script
wget https://raw.githubusercontent.com/Holedozer1229/Sphinx_OS/main/bootstrap_digitalocean.sh
chmod +x bootstrap_digitalocean.sh
sudo ./bootstrap_digitalocean.sh
```

That's it! SphinxOS will be automatically installed and configured.

### Option 2: Remote Deployment from Your Machine

```bash
# Clone the repository
git clone https://github.com/Holedozer1229/Sphinx_OS.git
cd Sphinx_OS

# Deploy to your droplet
python3 deploy_digitalocean.py --remote --host 159.89.139.241 --user root
```

### Option 3: Python-Based Local Deployment

On the droplet itself:

```bash
# Clone repository
git clone https://github.com/Holedozer1229/Sphinx_OS.git
cd Sphinx_OS

# Run Python deployment script
python3 deploy_digitalocean.py --local
```

---

## Droplet Specifications

Your Digital Ocean droplet configuration:

- **Name:** ubuntu-s-1vcpu-512mb-10gb-sfo2-01
- **Plan:** LaunchNFT / 512 MB Memory / 10 GB Disk / SFO2
- **OS:** Ubuntu 24.04 (LTS) x64
- **IPv4:** 159.89.139.241
- **Private IP:** 10.120.0.2
- **Region:** SFO2 (San Francisco)

---

## What Gets Installed

The deployment script automatically:

1. ✅ Installs Python 3.12+ and required system packages
2. ✅ Clones the SphinxOS repository
3. ✅ Creates a Python virtual environment
4. ✅ Installs all Python dependencies from requirements.txt
5. ✅ Creates a systemd service for auto-start
6. ✅ Configures firewall rules (UFW)
7. ✅ Starts the SphinxOS node service

---

## Access Your Deployment

After deployment completes, access your SphinxOS node at:

- **API Endpoint:** http://159.89.139.241:8000
- **Prometheus Metrics:** http://159.89.139.241:9090
- **Health Check:** http://159.89.139.241:8000/health

---

## Service Management

### Check Service Status

```bash
sudo systemctl status sphinxos
```

### View Live Logs

```bash
sudo journalctl -u sphinxos -f
```

### Restart Service

```bash
sudo systemctl restart sphinxos
```

### Stop Service

```bash
sudo systemctl stop sphinxos
```

### Start Service

```bash
sudo systemctl start sphinxos
```

### Enable Auto-Start on Boot

```bash
sudo systemctl enable sphinxos
```

### Disable Auto-Start

```bash
sudo systemctl disable sphinxos
```

---

## Configuration

### Customize Deployment Settings

Edit `droplet_config.json` before running deployment:

```json
{
  "droplet": {
    "ipv4": "159.89.139.241",
    "private_ip": "10.120.0.2"
  },
  "deployment": {
    "install_dir": "/opt/sphinxos",
    "service_name": "sphinxos",
    "auto_start": true
  },
  "application": {
    "node_port": 8000,
    "metrics_port": 9090,
    "enable_prometheus": true
  }
}
```

### Environment Variables

You can set environment variables in the systemd service file:

```bash
sudo nano /etc/systemd/system/sphinxos.service
```

Add environment variables under the `[Service]` section:

```ini
Environment="NODE_PORT=8000"
Environment="METRICS_PORT=9090"
Environment="LOG_LEVEL=INFO"
```

Then reload and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart sphinxos
```

---

## Firewall Configuration

The deployment script automatically configures UFW firewall with these rules:

- **Port 22:** SSH (TCP)
- **Port 8000:** SphinxOS API (TCP)
- **Port 9090:** Prometheus Metrics (TCP)

### Manual Firewall Management

```bash
# Check firewall status
sudo ufw status

# Allow additional ports
sudo ufw allow 443/tcp

# Remove a rule
sudo ufw delete allow 9090/tcp

# Enable firewall
sudo ufw enable

# Disable firewall
sudo ufw disable
```

---

## Troubleshooting

### Service Won't Start

1. Check the logs:
   ```bash
   sudo journalctl -u sphinxos -n 50
   ```

2. Verify Python dependencies:
   ```bash
   /opt/sphinxos/Sphinx_OS/venv/bin/pip list
   ```

3. Test manually:
   ```bash
   cd /opt/sphinxos/Sphinx_OS
   /opt/sphinxos/Sphinx_OS/venv/bin/python3 node_main.py
   ```

### Port Already in Use

If port 8000 is already in use, modify the service configuration:

```bash
sudo nano /etc/systemd/system/sphinxos.service
```

Change `NODE_PORT` to a different port, then:

```bash
sudo systemctl daemon-reload
sudo systemctl restart sphinxos
```

### Out of Memory

With 512MB RAM, you may need to create swap space:

```bash
# Create 1GB swap file
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Network Issues

1. Test connectivity:
   ```bash
   curl http://localhost:8000/health
   ```

2. Check if service is listening:
   ```bash
   sudo netstat -tulpn | grep 8000
   ```

3. Verify firewall:
   ```bash
   sudo ufw status numbered
   ```

---

## Updating SphinxOS

To update to the latest version:

```bash
# Stop the service
sudo systemctl stop sphinxos

# Pull latest changes
cd /opt/sphinxos/Sphinx_OS
sudo git pull origin main

# Update dependencies
sudo /opt/sphinxos/Sphinx_OS/venv/bin/pip install -r requirements.txt

# Restart service
sudo systemctl start sphinxos
```

Or use the automated script:

```bash
cd /opt/sphinxos/Sphinx_OS
sudo ./bootstrap_digitalocean.sh
```

---

## Uninstall

To completely remove SphinxOS:

```bash
# Stop and disable service
sudo systemctl stop sphinxos
sudo systemctl disable sphinxos

# Remove service file
sudo rm /etc/systemd/system/sphinxos.service
sudo systemctl daemon-reload

# Remove installation directory
sudo rm -rf /opt/sphinxos

# Remove firewall rules (optional)
sudo ufw delete allow 8000/tcp
sudo ufw delete allow 9090/tcp
```

---

## Advanced Configuration

### Running Multiple Instances

To run multiple SphinxOS instances on different ports:

1. Copy the service file:
   ```bash
   sudo cp /etc/systemd/system/sphinxos.service /etc/systemd/system/sphinxos-2.service
   ```

2. Edit the new service file:
   ```bash
   sudo nano /etc/systemd/system/sphinxos-2.service
   ```

3. Change the port:
   ```ini
   Environment="NODE_PORT=8001"
   Environment="METRICS_PORT=9091"
   ```

4. Enable and start:
   ```bash
   sudo systemctl enable sphinxos-2
   sudo systemctl start sphinxos-2
   ```

### Using a Reverse Proxy (Nginx)

For production deployments, use Nginx as a reverse proxy:

```bash
# Install Nginx
sudo apt-get install -y nginx

# Create config
sudo nano /etc/nginx/sites-available/sphinxos
```

Add this configuration:

```nginx
server {
    listen 80;
    server_name 159.89.139.241;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

Enable and restart:

```bash
sudo ln -s /etc/nginx/sites-available/sphinxos /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### SSL/TLS with Let's Encrypt

```bash
# Install Certbot
sudo apt-get install -y certbot python3-certbot-nginx

# Get certificate (requires domain name)
sudo certbot --nginx -d yourdomain.com
```

---

## Monitoring and Metrics

### Prometheus Metrics

Access Prometheus metrics at: http://159.89.139.241:9090

Available metrics include:
- Node uptime
- Request counts
- Response times
- Quantum circuit execution stats
- Memory usage
- CPU usage

### Log Management

View logs in real-time:

```bash
sudo journalctl -u sphinxos -f
```

View logs from the last hour:

```bash
sudo journalctl -u sphinxos --since "1 hour ago"
```

Export logs to file:

```bash
sudo journalctl -u sphinxos > sphinxos-logs.txt
```

---

## Security Best Practices

1. **Change Default Ports:** Modify NODE_PORT in the service configuration
2. **Enable Firewall:** Ensure UFW is enabled with minimal rules
3. **Regular Updates:** Keep system and dependencies updated
4. **Use SSH Keys:** Disable password authentication for SSH
5. **Monitor Logs:** Regularly check logs for suspicious activity
6. **Backup Data:** Regular backups of configuration and data

---

## Support

For issues and questions:

- **Repository:** https://github.com/Holedozer1229/Sphinx_OS
- **Documentation:** See repository docs/ folder
- **Issues:** https://github.com/Holedozer1229/Sphinx_OS/issues

---

## Architecture

SphinxOS on Digital Ocean includes:

- **Quantum Core:** 64-qubit simulation engine
- **Blockchain Node:** Hypercube network with 12-face x 50-layer architecture
- **API Server:** FastAPI-based REST API
- **Metrics:** Prometheus metrics exposition
- **Auto-scaling:** Systemd-based auto-restart

Installation location: `/opt/sphinxos/Sphinx_OS`

---

Built with ❤️ by the SphinxOS team  
© 2026 Travis D. Jones
