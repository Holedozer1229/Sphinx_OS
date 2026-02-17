#!/bin/bash
# ============================================================================
# SphinxOS Binary Build Script
# ============================================================================
# 
# Builds standalone binaries for all supported platforms using PyInstaller
#
# Targets:
#   - macOS (x64, arm64)
#   - Linux (x64, arm64)
#   - Windows (x64)
#
# Usage:
#   ./scripts/build_binaries.sh [platform]
#
# Examples:
#   ./scripts/build_binaries.sh          # Build for current platform
#   ./scripts/build_binaries.sh macos    # Build macOS bundle
#   ./scripts/build_binaries.sh linux    # Build Linux binary
#   ./scripts/build_binaries.sh windows  # Build Windows executable
#   ./scripts/build_binaries.sh all      # Build for all platforms (requires Docker)
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
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$PROJECT_ROOT/dist"
BUILD_DIR="$PROJECT_ROOT/build"
SPEC_FILE="$PROJECT_ROOT/sphinxos.spec"

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
        Linux*)     echo "linux";;
        Darwin*)    echo "macos";;
        MINGW*|MSYS*|CYGWIN*) echo "windows";;
        *)          echo "unknown";;
    esac
}

detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64)     echo "x64";;
        arm64|aarch64)    echo "arm64";;
        *)                echo "unknown";;
    esac
}

check_dependencies() {
    log_info "Checking build dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed."
        exit 1
    fi
    
    # Check PyInstaller
    if ! python3 -c "import PyInstaller" 2>/dev/null; then
        log_warning "PyInstaller not found. Installing..."
        pip install pyinstaller
    fi
    
    log_success "All dependencies satisfied"
}

clean_build() {
    log_info "Cleaning previous builds..."
    rm -rf "$BUILD_DIR"
    rm -rf "$DIST_DIR"
    mkdir -p "$DIST_DIR"
}

build_current_platform() {
    local platform=$(detect_platform)
    local arch=$(detect_arch)
    
    log_info "Building for $platform-$arch..."
    
    cd "$PROJECT_ROOT"
    
    # Run PyInstaller
    pyinstaller "$SPEC_FILE" --clean --noconfirm
    
    if [ $? -eq 0 ]; then
        log_success "Build completed successfully"
        
        # Rename output based on platform
        if [ "$platform" = "macos" ]; then
            if [ -d "$DIST_DIR/SphinxOS.app" ]; then
                tar -czf "$DIST_DIR/sphinxos-${platform}-${arch}.app.tar.gz" -C "$DIST_DIR" SphinxOS.app
                log_success "macOS bundle: $DIST_DIR/sphinxos-${platform}-${arch}.app.tar.gz"
            fi
        elif [ "$platform" = "windows" ]; then
            if [ -f "$DIST_DIR/sphinxos.exe" ]; then
                mv "$DIST_DIR/sphinxos.exe" "$DIST_DIR/sphinxos-${platform}-${arch}.exe"
                log_success "Windows executable: $DIST_DIR/sphinxos-${platform}-${arch}.exe"
            fi
        else
            if [ -f "$DIST_DIR/sphinxos" ]; then
                mv "$DIST_DIR/sphinxos" "$DIST_DIR/sphinxos-${platform}-${arch}"
                log_success "Linux binary: $DIST_DIR/sphinxos-${platform}-${arch}"
            fi
        fi
    else
        log_error "Build failed"
        exit 1
    fi
}

build_docker() {
    local platform=$1
    
    log_info "Building for $platform using Docker..."
    
    # TODO: Implement Docker-based cross-compilation
    log_warning "Docker-based builds not yet implemented"
    log_info "Please build on the target platform directly"
}

show_usage() {
    cat << EOF
SphinxOS Binary Build Script

Usage:
    $0 [platform]

Platforms:
    current     Build for current platform (default)
    macos       Build macOS .app bundle
    linux       Build Linux binary
    windows     Build Windows .exe
    all         Build for all platforms (requires Docker)

Examples:
    $0              # Build for current platform
    $0 macos        # Build macOS bundle
    $0 all          # Build all platforms

Output:
    Binaries are placed in: $DIST_DIR/

EOF
}

# ============================================================================
# Main Build Flow
# ============================================================================

main() {
    local target="${1:-current}"
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║   📦  SphinxOS Binary Builder  📦                             ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    
    check_dependencies
    clean_build
    
    case "$target" in
        current)
            build_current_platform
            ;;
        macos|linux|windows)
            if [ "$(detect_platform)" = "$target" ]; then
                build_current_platform
            else
                build_docker "$target"
            fi
            ;;
        all)
            log_info "Building for all platforms..."
            for platform in macos linux windows; do
                if [ "$(detect_platform)" = "$platform" ]; then
                    build_current_platform
                else
                    build_docker "$platform"
                fi
            done
            ;;
        help|-h|--help)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown target: $target"
            show_usage
            exit 1
            ;;
    esac
    
    echo ""
    log_success "Build process complete!"
    echo ""
    echo "Output directory: $DIST_DIR"
    echo ""
    
    # List built binaries
    if [ -d "$DIST_DIR" ]; then
        log_info "Built binaries:"
        ls -lh "$DIST_DIR" | grep -v "^total" | grep -v "^d" | awk '{print "  - " $9 " (" $5 ")"}'
    fi
}

# Run main
main "$@"
