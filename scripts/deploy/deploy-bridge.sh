#!/bin/bash
# Deploy SphinxSkynet Cross-Chain Bridge

set -e

echo "========================================="
echo "SphinxSkynet Bridge Deployment"
echo "========================================="
echo ""

# Check prerequisites
echo "✅ Checking prerequisites..."

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q fastapi uvicorn

# Start bridge API
echo "🌉 Starting bridge API..."
cd "$(dirname "$0")/../.."
mkdir -p logs

nohup python3 -m sphinx_os.api.bridge_api > logs/bridge-api.log 2>&1 &
BRIDGE_PID=$!
echo $BRIDGE_PID > data/bridge-api.pid

echo ""
echo "========================================="
echo "✅ Bridge deployed successfully!"
echo "========================================="
echo "Bridge PID: $BRIDGE_PID"
echo "API: http://localhost:8001"
echo "Logs: logs/bridge-api.log"
echo ""
echo "Supported chains:"
echo "  - Bitcoin (BTC)"
echo "  - Ethereum (ETH)"
echo "  - Ethereum Classic (ETC)"
echo "  - Polygon (MATIC)"
echo "  - Avalanche (AVAX)"
echo "  - BNB Chain (BNB)"
echo "  - Stacks (STX)"
echo ""
echo "Check status: curl http://localhost:8001/api/bridge/supported-chains"
