#!/bin/bash
# =============================================================================
# SphinxOS Digital Ocean Quick Setup Script
# Run this script directly on your Ubuntu 24.04 droplet
# =============================================================================

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║           SphinxOS Digital Ocean Auto-Bootstrap              ║"
echo "║           Ubuntu 24.04 LTS - One Command Setup               ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/Holedozer1229/Sphinx_OS.git"
INSTALL_DIR="/opt/sphinxos"
BRANCH="main"

echo -e "${BLUE}[1/6] Checking system requirements...${NC}"
echo ""

# Check if we're on Ubuntu
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" != "ubuntu" ]]; then
        echo -e "${RED}✗ This script is designed for Ubuntu 24.04${NC}"
        echo "  Current OS: $PRETTY_NAME"
        exit 1
    fi
    echo -e "${GREEN}✓ Running on $PRETTY_NAME${NC}"
else
    echo -e "${YELLOW}⚠ Could not detect OS version${NC}"
fi

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}⚠ This script should be run as root or with sudo${NC}"
    echo "  Attempting to use sudo for privileged operations..."
    SUDO="sudo"
else
    echo -e "${GREEN}✓ Running as root${NC}"
    SUDO=""
fi
echo ""

echo -e "${BLUE}[2/6] Installing system dependencies...${NC}"
echo ""

# Update package lists
echo "  Updating package lists..."
$SUDO apt-get update -qq

# Install required packages
echo "  Installing Python 3, git, and build tools..."
$SUDO apt-get install -y -qq \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    curl \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    pkg-config

echo -e "${GREEN}✓ System dependencies installed${NC}"
echo ""

echo -e "${BLUE}[3/6] Cloning SphinxOS repository...${NC}"
echo ""

# Create installation directory
$SUDO mkdir -p $INSTALL_DIR

# Clone repository if not already present
if [ -d "$INSTALL_DIR/Sphinx_OS/.git" ]; then
    echo "  Repository already exists, pulling latest changes..."
    cd $INSTALL_DIR/Sphinx_OS
    $SUDO git pull origin $BRANCH
else
    echo "  Cloning from $REPO_URL..."
    $SUDO git clone $REPO_URL $INSTALL_DIR/Sphinx_OS
    cd $INSTALL_DIR/Sphinx_OS
    $SUDO git checkout $BRANCH
fi

echo -e "${GREEN}✓ Repository cloned/updated${NC}"
echo ""

echo -e "${BLUE}[4/7] Creating service user...${NC}"
echo ""

# Create dedicated service user for security
if id -u sphinxos &>/dev/null; then
    echo -e "${GREEN}✓ Service user 'sphinxos' already exists${NC}"
else
    echo "  Creating system user 'sphinxos'..."
    $SUDO useradd --system --no-create-home --shell /bin/false sphinxos
    echo -e "${GREEN}✓ Service user 'sphinxos' created${NC}"
fi
echo ""

echo -e "${BLUE}[5/7] Setting up Python environment...${NC}"
echo ""

# Create virtual environment
if [ ! -d "$INSTALL_DIR/Sphinx_OS/venv" ]; then
    echo "  Creating virtual environment..."
    $SUDO python3 -m venv $INSTALL_DIR/Sphinx_OS/venv
fi

# Set ownership to service user
echo "  Setting ownership to sphinxos user..."
$SUDO chown -R sphinxos:sphinxos $INSTALL_DIR

# Activate virtual environment and install dependencies
echo "  Installing Python dependencies..."
$SUDO -u sphinxos $INSTALL_DIR/Sphinx_OS/venv/bin/pip install --upgrade pip -q
$SUDO -u sphinxos $INSTALL_DIR/Sphinx_OS/venv/bin/pip install -r $INSTALL_DIR/Sphinx_OS/requirements.txt -q

echo -e "${GREEN}✓ Python environment configured${NC}"
echo ""

echo -e "${BLUE}[6/7] Creating systemd service...${NC}"
echo ""

# Create systemd service file
cat << EOF | $SUDO tee /etc/systemd/system/sphinxos.service > /dev/null
[Unit]
Description=SphinxOS Quantum Blockchain Node
After=network.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=sphinxos
Group=sphinxos
WorkingDirectory=$INSTALL_DIR/Sphinx_OS
Environment="PATH=$INSTALL_DIR/Sphinx_OS/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="NODE_PORT=8000"
Environment="METRICS_PORT=9090"
ExecStart=$INSTALL_DIR/Sphinx_OS/venv/bin/python3 node_main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=sphinxos

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=$INSTALL_DIR/Sphinx_OS
ProtectHome=true

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
$SUDO systemctl daemon-reload

# Enable service for auto-start
echo "  Enabling service for auto-start on boot..."
$SUDO systemctl enable sphinxos

echo -e "${GREEN}✓ Systemd service created and enabled${NC}"
echo ""

echo -e "${BLUE}[7/7] Configuring firewall...${NC}"
echo ""

# Configure UFW if available
if command -v ufw &> /dev/null; then
    echo "  Configuring UFW firewall..."
    
    # Ensure SSH is allowed
    $SUDO ufw allow 22/tcp > /dev/null 2>&1
    
    # Allow application ports
    $SUDO ufw allow 8000/tcp > /dev/null 2>&1  # Node API
    $SUDO ufw allow 9090/tcp > /dev/null 2>&1  # Metrics
    
    # Check if UFW is enabled
    if $SUDO ufw status | grep -q "Status: active"; then
        echo -e "${GREEN}✓ Firewall rules configured${NC}"
    else
        echo -e "${YELLOW}⚠ UFW is installed but not enabled${NC}"
        echo "  To enable: sudo ufw enable"
    fi
else
    echo -e "${YELLOW}⚠ UFW not installed, skipping firewall configuration${NC}"
fi
echo ""

# Start the service
echo -e "${BLUE}Starting SphinxOS service...${NC}"
echo ""
$SUDO systemctl start sphinxos

# Wait a moment for service to start
sleep 3

# Check service status
if $SUDO systemctl is-active --quiet sphinxos; then
    echo -e "${GREEN}✓ SphinxOS service is running!${NC}"
else
    echo -e "${YELLOW}⚠ Service started but may be experiencing issues${NC}"
    echo "  Check logs with: sudo journalctl -u sphinxos -f"
fi
echo ""

# Get droplet IP
DROPLET_IP=$(hostname -I | awk '{print $1}')

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║                   ✅  DEPLOYMENT COMPLETE!                     ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}SphinxOS is now running on your Digital Ocean droplet!${NC}"
echo ""
echo "📊 Access Points:"
echo "  • API Endpoint:    http://$DROPLET_IP:8000"
echo "  • Metrics:         http://$DROPLET_IP:9090"
echo ""
echo "🔧 Service Management:"
echo "  • Status:          sudo systemctl status sphinxos"
echo "  • View Logs:       sudo journalctl -u sphinxos -f"
echo "  • Restart:         sudo systemctl restart sphinxos"
echo "  • Stop:            sudo systemctl stop sphinxos"
echo ""
echo "📁 Installation Directory: $INSTALL_DIR/Sphinx_OS"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Check the service status: sudo systemctl status sphinxos"
echo "  2. View the logs: sudo journalctl -u sphinxos -f"
echo "  3. Test the API: curl http://$DROPLET_IP:8000/health"
echo ""
echo "🌌 SphinxOS: Quantum • Blockchain • AI"
echo ""
