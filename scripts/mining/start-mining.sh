#!/bin/bash
# Start SphinxSkynet Mining

set -e

echo "========================================="
echo "SphinxSkynet Mining Starter"
echo "========================================="
echo ""

# Get miner address
if [ -z "$1" ]; then
    echo "Usage: $0 <miner_address> [algorithm]"
    echo ""
    echo "Example: $0 SPHINX_ADDRESS_123 spectral"
    echo ""
    echo "Available algorithms:"
    echo "  - spectral (default, Φ-boosted)"
    echo "  - sha256 (Bitcoin-compatible)"
    echo "  - ethash (Ethereum-compatible)"
    echo "  - keccak256 (ETC-compatible)"
    exit 1
fi

MINER_ADDRESS=$1
ALGORITHM=${2:-spectral}

echo "Miner Address: $MINER_ADDRESS"
echo "Algorithm: $ALGORITHM"
echo ""

# Start mining
cd "$(dirname "$0")/../.."
python3 -m sphinx_os.mining.auto_miner "$MINER_ADDRESS" --algorithm "$ALGORITHM"
