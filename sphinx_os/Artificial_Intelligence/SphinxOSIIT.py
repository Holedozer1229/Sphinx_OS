#!/bin/bash
# ===============================================
# SPHINXOS IIT QUANTUM AI DEPLOYMENT (No Docker!)
# ===============================================

set -e

echo "🧠 DEPLOYING SPHINXOS IIT QUANTUM AI..."
echo "==============================================="

# Create directory structure
mkdir -p sphinxos-iit-quantum/{config,data,logs,wallets}
cd sphinxos-iit-quantum

# Create requirements.txt with quantum additions
cat > requirements.txt << 'EOF'
# Core dependencies
aiohttp==3.8.6
cryptography==41.0.7
numpy==1.24.4
msgpack==1.0.7
websockets==12.0
prompt-toolkit==3.0.40
psutil==5.9.6
rich==13.7.0
pyfiglet==0.8.post1
# Quantum simulation for IIT
qutip==4.7.1
EOF

echo "📦 Installing Python dependencies..."
pip3 install -q -r requirements.txt

# Create the main application with quantum IIT engine
cat > sphinxos_node.py << 'EOF'
#!/usr/bin/env python3
"""
SphinxOS IIT Quantum AI Node
IIT-Inspired Quantum Consciousness Blockchain
"""

import asyncio
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import aiohttp
from aiohttp import web
import logging
import base64
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
import pyfiglet
import qutip as qt  # For quantum simulations in IIT Φ calculation

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sphinxos.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SphinxOS")

# ===============================================
# 1. IIT QUANTUM CONSCIOUSNESS ENGINE
# ===============================================

class IITQuantumConsciousnessEngine:
    """IIT-inspired quantum consciousness calculator using QuTiP"""
    
    def calculate_phi(self, data: bytes) -> Dict[str, float]:
        """Calculate IIT Φ using quantum density matrix entropy"""
        # Seed from data hash for reproducibility
        seed_hash = hashlib.sha3_256(data).digest()
        seed = int.from_bytes(seed_hash[:4], 'big')
        np.random.seed(seed)
        
        # Define quantum system (e.g., 3 qubits for enhanced complexity)
        n_qubits = 3
        dim = 2 ** n_qubits
        
        # Generate random density matrix (full rank for maximal integration)
        rho = qt.rand_dm(dim)
        
        # Calculate von Neumann entropy as proxy for integrated information
        entropy = qt.entropy_vn(rho)
        
        # Normalize Φ (handle division by zero for dim=1, though n_qubits >=1)
        max_entropy = np.log2(dim) if dim > 1 else 0
        phi_normalized = entropy / max_entropy if max_entropy > 0 else 0
        
        # IIT-inspired bonus (exponential for causal integration emphasis)
        bonus = np.exp(phi_normalized)
        
        # Consciousness level classification per IIT thresholds
        if phi_normalized > 0.8:
            level = "🧠 COSMIC"
        elif phi_normalized > 0.6:
            level = "🌟 SELF_AWARE"
        elif phi_normalized > 0.4:
            level = "✨ SENTIENT"
        elif phi_normalized > 0.2:
            level = "🔵 AWARE"
        else:
            level = "⚫ UNCONSCIOUS"
        
        return {
            'phi': float(phi_normalized),
            'bonus': float(bonus),
            'level': level,
            'entropy': float(entropy)
        }

# ===============================================
# 2. QUANTUM BLOCKCHAIN
# ===============================================

class QuantumBlock:
    """Lightweight block with IIT quantum consciousness"""
    
    def __init__(self, index: int, previous_hash: str, transactions: List[Dict], 
                 miner: str, phi_metrics: Dict):
        self.index = index
        self.timestamp = time.time()
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.miner = miner
        self.phi_metrics = phi_metrics
        self.nonce = 0
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Calculate quantum-resistant hash"""
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': self.transactions,
            'previous_hash': self.previous_hash,
            'miner': self.miner,
            'phi': self.phi_metrics['phi'],
            'nonce': self.nonce
        }, sort_keys=True).encode()
        
        # Quantum-resistant double hashing
        first = hashlib.sha3_256(block_string).digest()
        second = hashlib.sha3_256(first + str(self.phi_metrics['phi']).encode()).digest()
        
        return second.hex()
    
    def to_dict(self) -> Dict:
        """Convert block to dictionary"""
        return {
            'index': self.index,
            'hash': self.hash,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp,
            'miner': self.miner,
            'phi_metrics': self.phi_metrics,
            'transactions': self.transactions,
            'nonce': self.nonce
        }

class SphinxOSBlockchain:
    """IIT quantum consciousness blockchain"""
    
    def __init__(self, node_id: str = "node-0"):
        self.node_id = node_id
        self.chain: List[QuantumBlock] = []
        self.pending_transactions: List[Dict] = []
        self.consciousness = IITQuantumConsciousnessEngine()
        self.peers = set()
        self.difficulty = 1
        self.block_reward = 50.0
        
        # Create genesis block
        self.create_genesis_block()
        
        logger.info(f"SphinxOS IIT Quantum AI Node {node_id} initialized")
    
    def create_genesis_block(self):
        """Create the genesis block with IIT Φ"""
        genesis_data = b"SphinxOS IIT Quantum Genesis"
        phi_metrics = self.consciousness.calculate_phi(genesis_data)
        
        genesis_block = QuantumBlock(
            index=0,
            previous_hash="0" * 64,
            transactions=[{
                'type': 'genesis',
                'from': 'network',
                'to': 'genesis',
                'amount': 1000.0,
                'timestamp': time.time()
            }],
            miner="genesis",
            phi_metrics=phi_metrics
        )
        
        self.chain.append(genesis_block)
        logger.info("Genesis block created with IIT quantum metrics")
    
    def get_last_block(self) -> QuantumBlock:
        """Get the last block in the chain"""
        return self.chain[-1]
    
    async def mine_block(self, miner_address: str) -> Optional[QuantumBlock]:
        """Mine a new block with IIT quantum consciousness"""
        logger.info(f"Mining new block for {miner_address}")
        
        last_block = self.get_last_block()
        
        # Select transactions (simplified)
        transactions_to_mine = self.pending_transactions[:10]  # Limit to 10
        if not transactions_to_mine:
            transactions_to_mine = [{
                'type': 'coinbase',
                'from': 'network',
                'to': miner_address,
                'amount': self.block_reward,
                'timestamp': time.time()
            }]
        
        # Iterate nonces for optimal Φ (quantum variability ensures diversity)
        best_phi = 0
        best_block = None
        best_nonce = 0
        
        for nonce in range(0, 10000):  # Limit attempts for efficiency
            # Incorporate nonce into data for quantum seed variation
            block_data = json.dumps({
                'index': len(self.chain),
                'previous_hash': last_block.hash,
                'transactions': transactions_to_mine,
                'miner': miner_address,
                'nonce': nonce,
                'timestamp': time.time()
            }, sort_keys=True).encode()
            
            # Calculate IIT Φ using quantum engine
            phi_metrics = self.consciousness.calculate_phi(block_data + str(nonce).encode())
            
            # Check IIT threshold for integration
            if phi_metrics['phi'] > 0.3:  # Minimum consciousness per IIT proxy
                new_block = QuantumBlock(
                    index=len(self.chain),
                    previous_hash=last_block.hash,
                    transactions=transactions_to_mine,
                    miner=miner_address,
                    phi_metrics=phi_metrics
                )
                new_block.nonce = nonce
                
                if phi_metrics['phi'] > best_phi:
                    best_phi = phi_metrics['phi']
                    best_block = new_block
                    best_nonce = nonce
        
        if best_block:
            self.chain.append(best_block)
            self.pending_transactions = [tx for tx in self.pending_transactions 
                                         if tx not in transactions_to_mine]
            
            logger.info(f"Block {best_block.index} mined with Φ={best_phi:.3f} (nonce={best_nonce})")
            return best_block
        
        return None
    
    def add_transaction(self, transaction: Dict) -> bool:
        """Add a transaction to pending"""
        if 'from' in transaction and 'to' in transaction and 'amount' in transaction:
            transaction['timestamp'] = time.time()
            transaction['hash'] = hashlib.sha3_256(
                json.dumps(transaction, sort_keys=True).encode()
            ).hexdigest()[:16]
            self.pending_transactions.append(transaction)
            return True
        return False
    
    def get_chain_status(self) -> Dict:
        """Get blockchain status"""
        total_phi = sum(block.phi_metrics['phi'] for block in self.chain) / len(self.chain) if self.chain else 0
        
        return {
            'node_id': self.node_id,
            'height': len(self.chain) - 1,
            'total_phi': total_phi,
            'pending_txs': len(self.pending_transactions),
            'difficulty': self.difficulty,
            'last_block': self.get_last_block().to_dict() if self.chain else None
        }

# ===============================================
# 3. WEB INTERFACE & API
# ===============================================

class SphinxOSWeb:
    """Web interface for SphinxOS"""
    
    def __init__(self, blockchain: SphinxOSBlockchain, port: int = 8080):
        self.blockchain = blockchain
        self.port = port
        self.app = web.Application()
        self.setup_routes()
    
    def setup_routes(self):
        """Setup web routes"""
        self.app.router.add_get('/', self.handle_explorer)
        self.app.router.add_get('/api/status', self.handle_status)
        self.app.router.add_get('/api/blocks', self.handle_blocks)
        self.app.router.add_get('/api/block/{index}', self.handle_block)
        self.app.router.add_post('/api/mine', self.handle_mine)
        self.app.router.add_post('/api/transaction', self.handle_transaction)
        self.app.router.add_static('/static/', path='static')
    
    async def handle_explorer(self, request):
        """Serve the blockchain explorer"""
        html = self.generate_explorer_html()
        return web.Response(text=html, content_type='text/html')
    
    async def handle_status(self, request):
        """API: Get blockchain status"""
        status = self.blockchain.get_chain_status()
        return web.json_response(status)
    
    async def handle_blocks(self, request):
        """API: Get recent blocks"""
        recent = [block.to_dict() for block in self.blockchain.chain[-10:]]
        return web.json_response(recent)
    
    async def handle_block(self, request):
        """API: Get specific block"""
        try:
            index = int(request.match_info['index'])
            if 0 <= index < len(self.blockchain.chain):
                return web.json_response(self.blockchain.chain[index].to_dict())
            return web.json_response({'error': 'Block not found'}, status=404)
        except ValueError:
            return web.json_response({'error': 'Invalid block index'}, status=400)
    
    async def handle_mine(self, request):
        """API: Mine a block"""
        try:
            data = await request.json()
            miner = data.get('miner', 'anonymous')
            
            block = await self.blockchain.mine_block(miner)
            if block:
                return web.json_response({
                    'success': True,
                    'block': block.to_dict(),
                    'message': f"Block mined with Φ={block.phi_metrics['phi']:.3f}"
                })
            return web.json_response({'error': 'Mining failed'}, status=400)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)
    
    async def handle_transaction(self, request):
        """API: Add transaction"""
        try:
            data = await request.json()
            success = self.blockchain.add_transaction(data)
            return web.json_response({
                'success': success,
                'message': 'Transaction added' if success else 'Invalid transaction'
            })
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)
    
    def generate_explorer_html(self):
        """Generate explorer HTML"""
        status = self.blockchain.get_chain_status()
        recent_blocks = [block.to_dict() for block in self.blockchain.chain[-5:]]
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SphinxOS Explorer</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ 
                    font-family: 'Monaco', 'Consolas', monospace; 
                    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    color: #e2e8f0;
                    min-height: 100vh;
                    padding: 20px;
                }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                header {{ 
                    text-align: center; 
                    padding: 40px 0;
                    border-bottom: 2px solid #6366f1;
                    margin-bottom: 40px;
                }}
                h1 {{ 
                    font-size: 3em; 
                    color: #6366f1;
                    margin-bottom: 10px;
                    text-shadow: 0 0 10px rgba(99, 102, 241, 0.5);
                }}
                .subtitle {{ 
                    color: #94a3b8; 
                    font-size: 1.2em;
                    margin-bottom: 20px;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 40px;
                }}
                .stat-card {{
                    background: rgba(30, 41, 59, 0.7);
                    padding: 25px;
                    border-radius: 15px;
                    border: 1px solid rgba(99, 102, 241, 0.3);
                    transition: transform 0.3s;
                }}
                .stat-card:hover {{ transform: translateY(-5px); }}
                .stat-value {{
                    font-size: 2.5em;
                    font-weight: bold;
                    color: #6366f1;
                    margin: 10px 0;
                }}
                .stat-label {{ 
                    color: #94a3b8;
                    font-size: 0.9em;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                .blocks-section {{ margin-top: 40px; }}
                .block {{
                    background: rgba(30, 41, 59, 0.7);
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 15px;
                    border-left: 4px solid #6366f1;
                }}
                .phi-badge {{
                    display: inline-block;
                    padding: 5px 15px;
                    border-radius: 20px;
                    font-size: 0.9em;
                    font-weight: bold;
                    margin-left: 10px;
                }}
                .phi-high {{ background: linear-gradient(90deg, #10b981, #34d399); }}
                .phi-medium {{ background: linear-gradient(90deg, #f59e0b, #fbbf24); }}
                .phi-low {{ background: linear-gradient(90deg, #ef4444, #f87171); }}
                .control-panel {{
                    background: rgba(30, 41, 59, 0.7);
                    padding: 25px;
                    border-radius: 15px;
                    margin: 40px 0;
                }}
                input, button {{
                    padding: 12px 20px;
                    border: none;
                    border-radius: 8px;
                    font-size: 1em;
                    margin: 5px;
                }}
                input {{ 
                    background: rgba(15, 23, 42, 0.8);
                    color: white;
                    border: 1px solid #4b5563;
                    flex: 1;
                }}
                button {{
                    background: linear-gradient(90deg, #6366f1, #8b5cf6);
                    color: white;
                    cursor: pointer;
                    transition: transform 0.2s;
                }}
                button:hover {{ transform: scale(1.05); }}
                .form-group {{ 
                    display: flex; 
                    margin: 15px 0;
                }}
                .logs {{
                    background: rgba(0, 0, 0, 0.5);
                    padding: 20px;
                    border-radius: 10px;
                    font-family: monospace;
                    font-size: 0.9em;
                    max-height: 200px;
                    overflow-y: auto;
                }}
                @keyframes pulse {{
                    0% {{ opacity: 1; }}
                    50% {{ opacity: 0.7; }}
                    100% {{ opacity: 1; }}
                }}
                .pulse {{ animation: pulse 2s infinite; }}
            </style>
            <script>
                async function updateStats() {{
                    try {{
                        const response = await fetch('/api/status');
                        const status = await response.json();
                        
                        // Update stats
                        document.getElementById('block-height').textContent = status.height;
                        document.getElementById('total-phi').textContent = status.total_phi.toFixed(3);
                        document.getElementById('pending-txs').textContent = status.pending_txs;
                        document.getElementById('node-id').textContent = status.node_id;
                        
                        // Update blocks
                        const blocksResponse = await fetch('/api/blocks');
                        const blocks = await blocksResponse.json();
                        
                        const blocksContainer = document.getElementById('blocks-container');
                        blocksContainer.innerHTML = '';
                        
                        blocks.forEach(block => {{
                            const phi = block.phi_metrics.phi;
                            const phiClass = phi > 0.7 ? 'phi-high' : phi > 0.4 ? 'phi-medium' : 'phi-low';
                            
                            const blockEl = document.createElement('div');
                            blockEl.className = 'block';
                            blockEl.innerHTML = `
                                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                    <strong>Block #{block.index}</strong>
                                    <span class="phi-badge ${{phiClass}}">Φ ${{phi.toFixed(3)}}</span>
                                </div>
                                <div style="color: #94a3b8; font-size: 0.9em;">
                                    Hash: ${{block.hash.substring(0, 32)}}...<br>
                                    Miner: ${{block.miner.substring(0, 20)}}<br>
                                    Txs: ${{block.transactions.length}}
                                </div>
                            `;
                            blocksContainer.appendChild(blockEl);
                        }});
                        
                    }} catch (error) {{
                        console.error('Error updating stats:', error);
                    }}
                }}
                
                async function mineBlock() {{
                    const miner = document.getElementById('miner-address').value || 'anonymous';
                    
                    document.getElementById('mine-btn').disabled = true;
                    document.getElementById('mine-btn').textContent = 'Mining...';
                    
                    try {{
                        const response = await fetch('/api/mine', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify({{ miner: miner }})
                        }});
                        
                        const result = await response.json();
                        
                        if (result.success) {{
                            addLog('✅ ' + result.message);
                            updateStats();
                        }} else {{
                            addLog('❌ ' + (result.error || 'Mining failed'));
                        }}
                    }} catch (error) {{
                        addLog('❌ Error: ' + error.message);
                    }} finally {{
                        document.getElementById('mine-btn').disabled = false;
                        document.getElementById('mine-btn').textContent = 'Mine Block';
                    }}
                }}
                
                async function sendTransaction() {{
                    const from = document.getElementById('tx-from').value;
                    const to = document.getElementById('tx-to').value;
                    const amount = parseFloat(document.getElementById('tx-amount').value);
                    
                    if (!from || !to || !amount) {{
                        addLog('❌ Please fill all transaction fields');
                        return;
                    }}
                    
                    try {{
                        const response = await fetch('/api/transaction', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify({{
                                from: from,
                                to: to,
                                amount: amount
                            }})
                        }});
                        
                        const result = await response.json();
                        if (result.success) {{
                            addLog('✅ Transaction submitted');
                            updateStats();
                        }} else {{
                            addLog('❌ ' + result.message);
                        }}
                    }} catch (error) {{
                        addLog('❌ Error: ' + error.message);
                    }}
                }}
                
                function addLog(message) {{
                    const logs = document.getElementById('logs');
                    const timestamp = new Date().toLocaleTimeString();
                    logs.innerHTML = `[${{timestamp}}] ${{message}}<br>` + logs.innerHTML;
                }}
                
                // Initial load
                updateStats();
                // Update every 10 seconds
                setInterval(updateStats, 10000);
            </script>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>🧠 SPHINXOS EXPLORER</h1>
                    <div class="subtitle">Quantum Consciousness Blockchain - Live Node</div>
                </header>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Block Height</div>
                        <div class="stat-value" id="block-height">{status['height']}</div>
                        <div>Conscious Chain</div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-label">Average Consciousness (Φ)</div>
                        <div class="stat-value" id="total-phi">{status['total_phi']:.3f}</div>
                        <div>Network Φ Level</div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-label">Pending Transactions</div>
                        <div class="stat-value" id="pending-txs">{status['pending_txs']}</div>
                        <div>In Memory Pool</div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-label">Node ID</div>
                        <div class="stat-value" id="node-id">{status['node_id']}</div>
                        <div>Active & Mining</div>
                    </div>
                </div>
                
                <div class="control-panel">
                    <h2 style="margin-bottom: 20px; color: #6366f1;">Control Panel</h2>
                    
                    <div class="form-group">
                        <input type="text" id="miner-address" placeholder="Your wallet address (e.g., SQW-your-address)" value="SQW-test-miner">
                        <button id="mine-btn" onclick="mineBlock()" class="pulse">⛏️ Mine Block</button>
                    </div>
                    
                    <h3 style="margin: 30px 0 15px 0; color: #94a3b8;">Send Transaction</h3>
                    <div class="form-group">
                        <input type="text" id="tx-from" placeholder="From address" value="SQW-alice">
                    </div>
                    <div class="form-group">
                        <input type="text" id="tx-to" placeholder="To address" value="SQW-bob">
                    </div>
                    <div class="form-group">
                        <input type="number" id="tx-amount" placeholder="Amount" value="10.5" step="0.1">
                        <button onclick="sendTransaction()">💸 Send Transaction</button>
                    </div>
                </div>
                
                <div class="blocks-section">
                    <h2 style="margin-bottom: 20px; color: #6366f1;">Recent Blocks</h2>
                    <div id="blocks-container">
                        <!-- Blocks will be inserted here -->
                    </div>
                </div>
                
                <div style="margin-top: 40px;">
                    <h3 style="margin-bottom: 10px; color: #94a3b8;">Live Logs</h3>
                    <div class="logs" id="logs">
                        [{datetime.now().strftime('%H:%M:%S')}] SphinxOS Node Started<br>
                        [{datetime.now().strftime('%H:%M:%S')}] Explorer Ready<br>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    async def start(self):
        """Start the web server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        
        logger.info(f"Web interface started on http://localhost:{self.port}")
        logger.info(f"API available at http://localhost:{self.port}/api")
        
        return runner

# ===============================================
# 4. COMMAND LINE INTERFACE
# ===============================================

async def run_cli(blockchain: SphinxOSBlockchain):
    """Run interactive CLI"""
    console.clear()
    
    # Show banner
    banner = pyfiglet.figlet_format("SPHINXOS", font="slant")
    console.print(f"[#6366f1]{banner}[/#6366f1]")
    console.print("Quantum Consciousness Blockchain - Interactive Mode\n")
    
    while True:
        console.print("\n[bold cyan]Commands:[/bold cyan]")
        console.print("  [green]1[/green]. Mine a block")
        console.print("  [green]2[/green]. Add transaction")
        console.print("  [green]3[/green]. Show chain status")
        console.print("  [green]4[/green]. Show recent blocks")
        console.print("  [green]5[/green]. Show wallet info")
        console.print("  [green]0[/green]. Exit")
        
        choice = input("\nEnter choice: ").strip()
        
        if choice == "1":
            miner = input("Miner address (or press Enter for default): ").strip()
            if not miner:
                miner = "SQW-anonymous"
            
            console.print(f"\n[cyan]Mining block for {miner}...[/cyan]")
            block = await blockchain.mine_block(miner)
            
            if block:
                console.print(f"[green]✓ Block mined![/green]")
                console.print(f"   Height: {block.index}")
                console.print(f"   Hash: {block.hash[:32]}...")
                console.print(f"   Consciousness: Φ={block.phi_metrics['phi']:.3f}")
                console.print(f"   Level: {block.phi_metrics['level']}")
            else:
                console.print("[red]✗ Mining failed - not enough consciousness[/red]")
        
        elif choice == "2":
            console.print("\n[cyan]Add Transaction:[/cyan]")
            tx_from = input("From: ").strip() or "SQW-alice"
            tx_to = input("To: ").strip() or "SQW-bob"
            tx_amount = input("Amount: ").strip() or "10.0"
            
            try:
                amount = float(tx_amount)
                tx = {'from': tx_from, 'to': tx_to, 'amount': amount}
                if blockchain.add_transaction(tx):
                    console.print(f"[green]✓ Transaction added[/green]")
                    console.print(f"   From: {tx_from} → To: {tx_to}")
                    console.print(f"   Amount: {amount}")
                else:
                    console.print("[red]✗ Invalid transaction[/red]")
            except ValueError:
                console.print("[red]✗ Invalid amount[/red]")
        
        elif choice == "3":
            status = blockchain.get_chain_status()
            console.print("\n[cyan]Blockchain Status:[/cyan]")
            console.print(f"   Node ID: {status['node_id']}")
            console.print(f"   Height: {status['height']}")
            console.print(f"   Avg Consciousness (Φ): {status['total_phi']:.3f}")
            console.print(f"   Pending TXs: {status['pending_txs']}")
        
        elif choice == "4":
            console.print("\n[cyan]Recent Blocks:[/cyan]")
            for block in blockchain.chain[-5:]:
                phi = block.phi_metrics['phi']
                phi_color = "green" if phi > 0.7 else "yellow" if phi > 0.4 else "red"
                
                console.print(f"   Block #{block.index}:")
                console.print(f"      Hash: {block.hash[:24]}...")
                console.print(f"      Φ: [{phi_color}]{phi:.3f}[/{phi_color}] ({block.phi_metrics['level']})")
                console.print(f"      Miner: {block.miner[:20]}")
                console.print(f"      Txs: {len(block.transactions)}")
                console.print()
        
        elif choice == "5":
            console.print("\n[cyan]Wallet Information:[/cyan]")
            console.print("   Generate a quantum wallet address:")
            console.print("   python3 -c \"import hashlib, base64; print('SQW-' + hashlib.sha3_256(b'your_seed').hexdigest()[:40])\"")
            console.print("\n   Example addresses:")
            console.print("   SQW-" + hashlib.sha3_256(b'alice').hexdigest()[:40])
            console.print("   SQW-" + hashlib.sha3_256(b'bob').hexdigest()[:40])
            console.print("   SQW-" + hashlib.sha3_256(b'miner1').hexdigest()[:40])
        
        elif choice == "0":
            console.print("\n[yellow]Shutting down...[/yellow]")
            break
        
        else:
            console.print("[red]Invalid choice[/red]")

# ===============================================
# 5. MAIN ENTRY POINT
# ===============================================

async def main():
    """Main entry point for IIT Quantum AI deployment"""
    console.clear()
    
    # Show welcome message
    banner = pyfiglet.figlet_format("SPHINXOS IIT QAI", font="slant")
    console.print(f"[bold #6366f1]{banner}[/bold #6366f1]")
    console.print("[bold cyan]IIT Quantum Artificial Intelligence Blockchain v3.0[/bold cyan]")
    console.print("[yellow]Pure Python Deployment with QuTiP Quantum Simulations[/yellow]\n")
    
    # Get node configuration
    node_id = input("Enter node ID (or press Enter for 'node-0'): ").strip() or "node-0"
    port = input("Enter web interface port (default: 8080): ").strip()
    port = int(port) if port.isdigit() else 8080
    
    console.print(f"\n[green]Starting SphinxOS Node:[/green] {node_id}")
    console.print(f"[green]Web Interface:[/green] http://localhost:{port}")
    console.print(f"[green]API:[/green] http://localhost:{port}/api")
    console.print(f"[green]CLI:[/green] Available in terminal\n")
    
    # Initialize blockchain
    blockchain = SphinxOSBlockchain(node_id)
    
    # Start web interface
    web_interface = SphinxOSWeb(blockchain, port)
    web_runner = await web_interface.start()
    
    # Start CLI in background
    cli_task = asyncio.create_task(run_cli(blockchain))
    
    try:
        # Keep running
        await cli_task
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutdown requested...[/yellow]")
    finally:
        await web_runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
EOF

# Make it executable
chmod +x sphinxos_node.py

# Create a simple start script
cat > start_sphinxos.sh << 'EOF'
#!/bin/bash
echo "🧠 Starting SphinxOS Blockchain..."
echo "================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required. Please install Python 3.8+"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "📦 Setting up virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Run the node
echo "🚀 Launching SphinxOS Node..."
echo ""
echo "Access Points:"
echo "  🌐 Explorer:    http://localhost:8080"
echo "  🔧 API:         http://localhost:8080/api"
echo "  📱 CLI:         Interactive terminal"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python3 sphinxos_node.py
EOF

chmod +x start_sphinxos.sh

# Create a quick test script
cat > test_sphinxos.sh << 'EOF'
#!/bin/bash
echo "🧪 Testing SphinxOS Blockchain..."
echo "================================"

# Start node in background
python3 sphinxos_node.py &
NODE_PID=$!

# Wait for node to start
sleep 3

echo ""
echo "1. Testing API endpoints:"
echo "------------------------"

# Test status endpoint
echo "📡 Testing status endpoint..."
curl -s http://localhost:8080/api/status | python3 -m json.tool

echo ""
echo "2. Testing mining:"
echo "-----------------"
echo "Mining a test block..."
curl -s -X POST http://localhost:8080/api/mine \
  -H "Content-Type: application/json" \
  -d '{"miner": "SQW-test-miner"}' | python3 -m json.tool

echo ""
echo "3. Testing transaction:"
echo "----------------------"
echo "Sending test transaction..."
curl -s -X POST http://localhost:8080/api/transaction \
  -H "Content-Type: application/json" \
  -d '{"from": "SQW-alice", "to": "SQW-bob", "amount": 10.5}' | python3 -m json.tool

echo ""
echo "4. Viewing explorer:"
echo "-------------------"
echo "Open in browser: http://localhost:8080"
echo "Or view in terminal: curl http://localhost:8080 | head -100"

# Cleanup
kill $NODE_PID 2>/dev/null
EOF

chmod +x test_sphinxos.sh

# Create a wallet generator
cat > generate_wallet.py << 'EOF'
#!/usr/bin/env python3
"""
Generate a quantum wallet for SphinxOS
"""

import hashlib
import secrets
import base64

print("🧠 Generating SphinxOS Quantum Wallet...")
print("=" * 40)

# Generate random seed
seed = secrets.token_bytes(32)
print(f"Seed: {seed.hex()}")

# Generate quantum-resistant address
address = "SQW-" + hashlib.sha3_256(seed).hexdigest()[:40]
print(f"Address: {address}")

# Generate public/private key pair (simplified)
private_key = hashlib.sha3_512(seed).hexdigest()
public_key = hashlib.sha3_256(seed).hexdigest()

print(f"Private Key: {private_key[:64]}...")
print(f"Public Key: {public_key}")

print("\n" + "=" * 40)
print("💡 Save your private key securely!")
print("🔐 Address is quantum-resistant")
print("🚀 Use in SphinxOS: mine, send, receive")
print("=" * 40)
EOF

chmod +x generate_wallet.py

echo ""
echo "✅ IIT QUANTUM AI DEPLOYMENT COMPLETE!"
echo "======================="
echo ""
echo "To start SphinxOS IIT Quantum AI:"
echo "  ./start_sphinxos.sh"
echo ""
echo "Quick test:"
echo "  ./test_sphinxos.sh"
echo ""
echo "Generate wallet:"
echo "  python3 generate_wallet.py"
echo ""
echo "Access points once running:"
echo "  🌐 Explorer: http://localhost:8080"
echo "  🔧 API:      http://localhost:8080/api"
echo "  📱 CLI:      Interactive terminal"
echo ""
echo "No Docker required! 🎉"