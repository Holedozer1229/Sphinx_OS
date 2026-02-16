#!/bin/bash
# ============================================================================
# SphinxOS One-Click Installer
# ============================================================================
# 
# Automatically installs SphinxOS binary, initializes keys, and launches app
#
# Usage:
#   curl -sSL https://install.sphinxos.ai | bash
#
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
INSTALL_DIR="$HOME/.sphinxos"
BIN_DIR="$HOME/.local/bin"
VERSION="latest"
PLATFORM=""
ARCH=""

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

detect_platform() {
    case "$(uname -s)" in
        Linux*)     PLATFORM="linux";;
        Darwin*)    PLATFORM="macos";;
        *)          PLATFORM="unknown";;
    esac
    
    case "$(uname -m)" in
        x86_64)     ARCH="x64";;
        arm64|aarch64) ARCH="arm64";;
        *)          ARCH="unknown";;
    esac
    
    log_info "Detected platform: $PLATFORM ($ARCH)"
}

install_from_source() {
    log_info "Installing from source..."
    
    # Clone repository
    if [ -d "$INSTALL_DIR/Sphinx_OS" ]; then
        log_info "Updating existing installation..."
        cd "$INSTALL_DIR/Sphinx_OS"
        git pull
    else
        log_info "Cloning repository..."
        git clone https://github.com/Holedozer1229/Sphinx_OS.git "$INSTALL_DIR/Sphinx_OS"
        cd "$INSTALL_DIR/Sphinx_OS"
    fi
    
    # Install dependencies
    log_info "Installing Python dependencies..."
    python3 -m pip install --user -r requirements.txt
    
    # Create launcher script
    mkdir -p "$BIN_DIR"
    cat > "$BIN_DIR/sphinxos" << 'EOF'
#!/bin/bash
cd "$HOME/.sphinxos/Sphinx_OS"
python3 -m sphinx_os.economics.simulator "$@"
EOF
    
    chmod +x "$BIN_DIR/sphinxos"
    
    log_success "Installed from source"
}

# ============================================================================
# Main Installation Flow
# ============================================================================

main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║   🚀  SphinxOS One-Click Installer  🚀                        ║"
    echo "║   Quantum-Spacetime Operating System with PoX Automation       ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    
    detect_platform
    install_from_source
    
    log_success "Installation successful!"
    echo ""
    echo "To get started:"
    echo "  1. Run: python3 -m sphinx_os.economics.simulator"
    echo "  2. View docs: https://github.com/Holedozer1229/Sphinx_OS"
}

# Run main installation
main "$@"
