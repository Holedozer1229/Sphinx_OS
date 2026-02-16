#!/bin/bash
# Start SphinxSkynet Merge Mining

set -e

echo "========================================="
echo "SphinxSkynet Merge Mining Starter"
echo "========================================="
echo ""

# Get miner address
if [ -z "$1" ]; then
    echo "Usage: $0 <miner_address> --chains btc,eth,etc"
    echo ""
    echo "Example: $0 SPHINX_ADDRESS_123 --chains btc,eth,etc"
    exit 1
fi

MINER_ADDRESS=$1

echo "Miner Address: $MINER_ADDRESS"
echo "Merge Mining: ENABLED"
echo ""

# Start merge mining
cd "$(dirname "$0")/../.."
python3 -m sphinx_os.mining.auto_miner "$MINER_ADDRESS" --merge "$@"
