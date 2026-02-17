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

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

detect_platform() {
    case "$(uname -s)" in
        Linux*)     PLATFORM="linux";;
        Darwin*)    PLATFORM="macos";;
        MINGW*|MSYS*|CYGWIN*) PLATFORM="windows";;
        *)          PLATFORM="unknown";;
    esac
    
    case "$(uname -m)" in
        x86_64|amd64)     ARCH="x64";;
        arm64|aarch64)    ARCH="arm64";;
        i386|i686)        ARCH="x86";;
        *)                ARCH="unknown";;
    esac
    
    log_info "Detected platform: $PLATFORM ($ARCH)"
    
    if [ "$PLATFORM" = "unknown" ] || [ "$ARCH" = "unknown" ]; then
        log_warning "Unsupported platform. Installing from source..."
    fi
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed. Please install Python 3.11+ first."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    log_info "Found Python $PYTHON_VERSION"
    
    # Check git
    if ! command -v git &> /dev/null; then
        log_error "Git is required but not installed. Please install Git first."
        exit 1
    fi
    
    # Check pip
    if ! python3 -m pip --version &> /dev/null; then
        log_warning "pip not found. Attempting to install..."
        python3 -m ensurepip --user || {
            log_error "Failed to install pip. Please install pip manually."
            exit 1
        }
    fi
}

install_binary() {
    log_info "Attempting binary installation for $PLATFORM-$ARCH..."
    
    BINARY_URL="https://github.com/Holedozer1229/Sphinx_OS/releases/latest/download/sphinxos-${PLATFORM}-${ARCH}"
    
    if [ "$PLATFORM" = "windows" ]; then
        BINARY_URL="${BINARY_URL}.exe"
        BINARY_NAME="sphinxos.exe"
    elif [ "$PLATFORM" = "macos" ]; then
        BINARY_URL="${BINARY_URL}.app.tar.gz"
        BINARY_NAME="SphinxOS.app"
    else
        BINARY_NAME="sphinxos"
    fi
    
    log_info "Downloading from: $BINARY_URL"
    
    # Try to download binary
    if command -v curl &> /dev/null; then
        curl -sSLf "$BINARY_URL" -o "$INSTALL_DIR/$BINARY_NAME" 2>/dev/null
        if [ $? -eq 0 ]; then
            log_success "Binary downloaded successfully"
            
            if [ "$PLATFORM" = "macos" ]; then
                tar -xzf "$INSTALL_DIR/$BINARY_NAME" -C "$INSTALL_DIR/"
                rm "$INSTALL_DIR/$BINARY_NAME"
            fi
            
            chmod +x "$INSTALL_DIR/$BINARY_NAME"
            
            # Create symlink
            mkdir -p "$BIN_DIR"
            ln -sf "$INSTALL_DIR/$BINARY_NAME" "$BIN_DIR/sphinxos"
            
            log_success "Binary installation complete"
            return 0
        fi
    fi
    
    log_warning "Binary not available for this platform. Installing from source..."
    return 1
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
        mkdir -p "$INSTALL_DIR"
        git clone https://github.com/Holedozer1229/Sphinx_OS.git "$INSTALL_DIR/Sphinx_OS"
        cd "$INSTALL_DIR/Sphinx_OS"
    fi
    
    # Install dependencies
    log_info "Installing Python dependencies..."
    python3 -m pip install --user -r requirements.txt
    
    log_success "Dependencies installed"
    
    # Create launcher script
    mkdir -p "$BIN_DIR"
    cat > "$BIN_DIR/sphinxos" << 'EOF'
#!/bin/bash
# SphinxOS launcher script
SPHINXOS_DIR="$HOME/.sphinxos/Sphinx_OS"

if [ ! -d "$SPHINXOS_DIR" ]; then
    echo "Error: SphinxOS installation not found at $SPHINXOS_DIR"
    exit 1
fi

cd "$SPHINXOS_DIR"

# Default to economic simulator if no arguments
if [ $# -eq 0 ]; then
    python3 -m sphinx_os.economics.simulator
else
    # Pass through commands
    case "$1" in
        simulator)
            python3 -m sphinx_os.economics.simulator "${@:2}"
            ;;
        node)
            python3 node_main.py "${@:2}"
            ;;
        *)
            python3 -m sphinx_os.economics.simulator "$@"
            ;;
    esac
fi
EOF
    
    chmod +x "$BIN_DIR/sphinxos"
    
    log_success "Installed from source"
}

initialize_keys() {
    log_info "Initializing cryptographic keys..."
    
    KEYS_DIR="$INSTALL_DIR/keys"
    mkdir -p "$KEYS_DIR"
    
    # Generate keys if they don't exist
    if [ ! -f "$KEYS_DIR/wallet.json" ]; then
        log_info "Generating new wallet..."
        cd "$INSTALL_DIR/Sphinx_OS"
        
        # Try wallet creation with error capture
        WALLET_OUTPUT=$(python3 -c "
from sphinx_os.wallet.wallet_manager import WalletManager
import json

try:
    wm = WalletManager()
    wallet_data = wm.create_wallet()
    print('Wallet created successfully')
except Exception as e:
    print(f'Error: {e}')
" 2>&1)
        
        if echo "$WALLET_OUTPUT" | grep -q "successfully"; then
            log_success "Wallet created successfully"
        else
            log_warning "Wallet creation skipped (will initialize on first run)"
            if [ ! -z "$WALLET_OUTPUT" ]; then
                log_info "Details: $WALLET_OUTPUT"
            fi
        fi
    else
        log_info "Existing keys found"
    fi
}

verify_installation() {
    log_info "Verifying installation..."
    
    if [ -f "$BIN_DIR/sphinxos" ]; then
        log_success "SphinxOS command installed at $BIN_DIR/sphinxos"
        
        # Check if BIN_DIR is in PATH
        if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
            log_warning "$BIN_DIR is not in PATH"
            echo ""
            echo "Add this to your ~/.bashrc or ~/.zshrc:"
            echo "  export PATH=\"\$PATH:$BIN_DIR\""
        fi
    else
        log_error "Installation verification failed"
        return 1
    fi
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
    check_dependencies
    
    # Try binary installation first
    if ! install_binary; then
        install_from_source
    fi
    
    initialize_keys
    verify_installation
    
    log_success "Installation successful!"
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  ✅  SphinxOS is ready to use!                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "To get started:"
    echo ""
    echo "  1. Run economic simulator:"
    echo "     $ sphinxos simulator"
    echo ""
    echo "  2. Start local node:"
    echo "     $ sphinxos node"
    echo ""
    echo "  3. View documentation:"
    echo "     https://github.com/Holedozer1229/Sphinx_OS"
    echo ""
    echo "  4. Test PoX integration:"
    echo "     $ python3 -c 'from sphinx_os.economics.yield_calculator import YieldCalculator; print(YieldCalculator())'"
    echo ""
}

# Run main installation
main "$@"
