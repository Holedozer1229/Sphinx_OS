#!/bin/bash
# Deploy SphinxSkynet Blockchain

set -e

echo "========================================="
echo "SphinxSkynet Blockchain Deployment"
echo "========================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Python 3 found"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q fastapi uvicorn prometheus_client numpy scipy

# Initialize blockchain
echo "🔧 Initializing blockchain..."
cd "$(dirname "$0")/../.."
python3 -c "
from sphinx_os.blockchain.core import SphinxSkynetBlockchain
blockchain = SphinxSkynetBlockchain()
blockchain.save_to_file('data/blockchain.json')
print('✅ Blockchain initialized with genesis block')
print(f'Genesis hash: {blockchain.chain[0].hash}')
"

mkdir -p data logs

# Start blockchain node
echo "🚀 Starting blockchain node..."
nohup python3 -m sphinx_os.api.mining_api > logs/blockchain-node.log 2>&1 &
NODE_PID=$!
echo $NODE_PID > data/blockchain-node.pid

echo ""
echo "========================================="
echo "✅ Blockchain deployed successfully!"
echo "========================================="
echo "Node PID: $NODE_PID"
echo "API: http://localhost:8000"
echo "Logs: logs/blockchain-node.log"
echo ""
echo "Check status: curl http://localhost:8000/api/chain/stats"
