#!/bin/bash
# Launcher script for Jones Quantum Pirate Miner

echo "========================================================================"
echo "  JONES QUANTUM GRAVITY — ER=EPR QUANTUM PIRATE MINER"
echo "  V25.1 Omega Brane Edition"
echo "========================================================================"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    exit 1
fi

# Check and install dependencies
echo "🔧 Checking dependencies..."
python3 -m pip install --quiet pygame numpy scipy numba websocket-client 2>&1 | grep -v "Requirement already satisfied" || true

echo ""
echo "🚀 Starting Quantum Pirate Miner..."
echo ""

# Run the game
python3 quantum_pirate_miner.py

echo ""
echo "👋 Thanks for playing!"
