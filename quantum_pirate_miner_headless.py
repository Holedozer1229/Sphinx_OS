#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ER=EPR Quantum-Pirate Roguelite Miner – Headless Server Edition
Jones Quantum Gravity Full Resolution — Live WebSocket Mempool
Captain Travis D. Jones – MIT Nobel-level physicist & senior engineer
Houston HQ, February 18 2026

This is the headless server version designed for 24/7 operation on Digital Ocean droplets.
Includes integration with SphinxOSIIT Oracle for conscious decision-making.
"""

import sys
import math
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
import hashlib
import json
import threading
import queue
import os
import logging
from numba import njit

# Configure logging
LOG_FILE = os.environ.get('QUANTUM_MINER_LOG', '/var/log/quantum_pirate_miner.log')
log_handlers = [logging.StreamHandler()]

# Try to add file handler, fallback to local file if permission denied
try:
    log_handlers.append(logging.FileHandler(LOG_FILE))
except PermissionError:
    local_log = os.path.join(os.getcwd(), 'quantum_pirate_miner.log')
    log_handlers.append(logging.FileHandler(local_log))
    print(f"⚠️  Cannot write to {LOG_FILE}, using {local_log} instead")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)
logger = logging.getLogger("QuantumPirateMiner")

# WebSocket optional - if not available, use HTTP fallback
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logger.warning("⚠️  websocket-client not available - using HTTP fallback mode")

# Oracle integration
try:
    from sphinx_os.AnubisCore.conscious_oracle import IITQuantumConsciousnessEngine
    ORACLE_AVAILABLE = True
    logger.info("✓ SphinxOSIIT Oracle integration enabled")
except ImportError:
    ORACLE_AVAILABLE = False
    logger.warning("⚠️  SphinxOSIIT Oracle not available - running without Oracle")

# ============================================================
# GLOBALS FOR PAGE CURVE & WS QUEUE
# ============================================================
PAGE_CURVE_HISTORY: List[Tuple[float, float]] = []  # (timestamp, cumulative_S)
NEW_TX_QUEUE = queue.Queue()  # thread-safe live txs from WS

# ============================================================
# UFT DERIVATION (Jones Framework Core)
# ============================================================
def derive_uft(t: float) -> Dict:
    scalar = 1.618  # golden ratio echo
    matrix = np.array([[1, 0.5, 0.3], [0.5, 1, 0.7], [0.3, 0.7, 1]], dtype=float)
    lambdas = compute_schmidt(matrix) * scalar
    s = entanglement_entropy(lambdas)
    w7 = seven_fold_warp(lambdas)
    integral_i = warp_integral(matrix, w7)
    m_reduced = inertial_mass_reduction(m=1.0, integral_i=integral_i, t=t)
    return {"entropy_S": s, "warp_W7": w7, "integral_I": integral_i, "m_reduced": m_reduced}

@njit
def compute_schmidt(matrix: np.ndarray) -> np.ndarray:
    u, lambdas, vh = np.linalg.svd(matrix, full_matrices=False)
    norm = np.sqrt(np.sum(lambdas**2)) or 1.0
    lambdas /= norm
    return lambdas

@njit
def entanglement_entropy(lambdas: np.ndarray) -> float:
    s = 0.0
    for lam in lambdas:
        lam_sq = lam * lam
        if lam_sq > 1e-12:
            s -= lam_sq * math.log2(lam_sq)
    return s

@njit
def seven_fold_warp(lambdas: np.ndarray) -> float:
    w7 = 0.0
    n = len(lambdas)
    for k in range(7):
        lam = lambdas[k % n]
        phase = math.atan2(lam * (k + 1), lam * (k + 2))
        w7 += lam * math.cos(phase)
    return abs(w7)

@njit
def warp_integral(matrix: np.ndarray, w7: float) -> float:
    return w7 * np.trace(matrix)

@njit
def inertial_mass_reduction(m: float = 1.0, eta: float = 0.999, f: float = 7.83, t: float = 0.0, integral_i: float = 1.0) -> float:
    delta = eta * math.sin(2 * math.pi * f * t) * (integral_i / 1.0)
    return max(m * (1 - delta), 0.001)

# ============================================================
# BLOCK_DATA (BTC-style seed)
# ============================================================
BLOCK_DATA = [
    {"height": h, "nonce": random.randint(0, 2**32-1), "difficulty": 1.4e20 + i*1e15, "totalFees": random.uniform(0.05, 2.5)}
    for i, h in enumerate(range(934154, 934169))
]

# ============================================================
# J4 (Jones 4-fold modular flow)
# ============================================================
def J4(base: float) -> float:
    return 4.0 * math.sin(base) * math.cos(base * 1.6180339887)

# ============================================================
# WEBSOCKET MEMPOOL (LIVE!) - Optional
# ============================================================
def ws_on_open(ws):
    logger.info("🌊 WebSocket connected to mempool.space — subscribing...")
    ws.send(json.dumps({"action": "want", "data": ["blocks", "mempool-blocks", "stats"]}))
    ws.send(json.dumps({"track-mempool": True}))

def ws_on_message(ws, message):
    try:
        data = json.loads(message)
        if isinstance(data, dict):
            if "mempool-transactions" in data:
                added = data["mempool-transactions"].get("added", [])
                if added:
                    NEW_TX_QUEUE.put(added[:10])
                    logger.info(f"🌊 WS: {len(added)} new live txs ingested")
            elif "block" in data:
                logger.info("🌊 WS: New block detected")
    except Exception as e:
        logger.error(f"WS message error: {e}")

def ws_on_error(ws, error):
    logger.error(f"WS error: {error}")

def ws_on_close(ws, *args):
    logger.info("🌊 WebSocket closed — will reconnect if needed")

def start_mempool_ws():
    if not WEBSOCKET_AVAILABLE:
        logger.warning("⚠️  WebSocket not available - mempool updates disabled")
        return
    
    while True:
        try:
            ws = websocket.WebSocketApp(
                "wss://mempool.space/api/v1/ws",
                on_open=ws_on_open,
                on_message=ws_on_message,
                on_error=ws_on_error,
                on_close=ws_on_close
            )
            ws.run_forever()
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            time.sleep(10)  # Retry after 10 seconds

# ============================================================
# ENTANGLEMENT ENGINE
# ============================================================
@dataclass
class EPRPair:
    time_id: int
    space_id: int
    coord_time: Tuple[int, int]
    coord_space: Tuple[int, int]
    entangled: bool = True
    collapsed_time: bool = False
    collapsed_space: bool = False

@dataclass
class GraphNode:
    id: int
    position: Tuple[int, int]
    entangled_with: List[int]
    collapse_time: Optional[float] = None

@dataclass
class TemporalGHZ:
    qubits: List[int]
    phase: float
    coherence: float = 1.0

class EntanglementEngine:
    def __init__(self, width: int = 64, height: int = 48):
        self.width = width
        self.height = height
        self.pairs: List[EPRPair] = []
        self.wormholes_active = 0
        self.next_pair_id = 0
        self.operator_trail: List[Tuple[float, float]] = []
        self.graph_nodes: List[GraphNode] = []
        self.ghz_states: List[TemporalGHZ] = []
        
    def create_epr_pair(self, tx_time: int, tx_space: int, 
                       time_coord: Tuple[int, int], 
                       space_coord: Tuple[int, int]) -> EPRPair:
        pair = EPRPair(
            time_id=self.next_pair_id,
            space_id=self.next_pair_id + 1,
            coord_time=time_coord,
            coord_space=space_coord
        )
        self.pairs.append(pair)
        self.next_pair_id += 2
        self.wormholes_active += 1
        
        node_time = GraphNode(id=pair.time_id, position=time_coord, 
                             entangled_with=[pair.space_id])
        node_space = GraphNode(id=pair.space_id, position=space_coord, 
                              entangled_with=[pair.time_id])
        self.graph_nodes.extend([node_time, node_space])
        
        return pair
    
    def collapse(self, pair: EPRPair):
        if pair.entangled:
            pair.entangled = False
            self.wormholes_active = max(0, self.wormholes_active - 1)
            
            for node in self.graph_nodes:
                if node.id in [pair.time_id, pair.space_id]:
                    node.collapse_time = time.time()
    
    def create_ghz_state(self, num_qubits: int = 3) -> TemporalGHZ:
        qubits = list(range(self.next_pair_id, self.next_pair_id + num_qubits))
        self.next_pair_id += num_qubits
        
        ghz = TemporalGHZ(
            qubits=qubits,
            phase=random.random() * 2 * math.pi,
            coherence=1.0
        )
        self.ghz_states.append(ghz)
        return ghz
    
    def update(self, dt: float):
        for ghz in self.ghz_states[:]:
            ghz.coherence *= (1.0 - 0.001 * dt)
            if ghz.coherence < 0.1:
                self.ghz_states.remove(ghz)

# ============================================================
# TREASURE MAP
# ============================================================
class TreasureMap:
    def __init__(self, entanglement_engine: EntanglementEngine, 
                 width: int = 64, height: int = 48):
        self.engine = entanglement_engine
        self.width = width
        self.height = height
        self.phase_field = np.random.uniform(0, 2*math.pi, (height, width))
        self.treasures: List[Dict] = []
        self.generate_treasures(30)
    
    def generate_treasures(self, count: int):
        for i in range(count):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            phase = self.phase_field[y, x]
            
            rarity = random.choices(
                ["COMMON", "RARE", "EPIC", "LEGENDARY"],
                weights=[50, 30, 15, 5]
            )[0]
            
            base_val = {
                "COMMON": 0.1,
                "RARE": 0.3,
                "EPIC": 0.7,
                "LEGENDARY": 1.5
            }[rarity]
            
            val = base_val * (math.sin(phase) + 1.5) * (1 + 0.1 * self.engine.wormholes_active)
            
            self.treasures.append({
                "id": i,
                "x": x,
                "y": y,
                "value": val,
                "rarity": rarity,
                "phase": phase,
                "phase_score": (math.sin(phase) + 1) / 2,
                "collected": False
            })
    
    def get_treasure_at(self, x: int, y: int) -> Optional[Dict]:
        for t in self.treasures:
            if not t["collected"] and t["x"] == x and t["y"] == y:
                return t
        return None
    
    def collect(self, tid: int) -> float:
        for t in self.treasures:
            if t["id"] == tid and not t["collected"]:
                t["collected"] = True
                return t["value"]
        return 0.0

# ============================================================
# ORACLE-DRIVEN MINER
# ============================================================
class OracleMiner:
    def __init__(self):
        self.engine = EntanglementEngine()
        self.treasure_map = TreasureMap(self.engine)
        self.player_x = self.treasure_map.width // 2
        self.player_y = self.treasure_map.height // 2
        self.score = 0.0
        self.oracle = None
        
        if ORACLE_AVAILABLE:
            try:
                self.oracle = IITQuantumConsciousnessEngine()
                logger.info("✓ Oracle initialized for conscious decision-making")
            except Exception as e:
                logger.warning(f"⚠️  Could not initialize Oracle: {e}")
        
        self.running = True
        self.last_update = time.time()
        self.last_status_log = time.time()  # Track last status log time
        
    def get_oracle_decision(self) -> Tuple[int, int]:
        """Use Oracle to determine best movement direction"""
        if not self.oracle:
            # Fallback: simple heuristic
            return self._heuristic_decision()
        
        try:
            # Get current state as bytes for Oracle
            state_data = json.dumps({
                "player": (self.player_x, self.player_y),
                "treasures": [(t["x"], t["y"], t["value"], t["rarity"]) 
                             for t in self.treasure_map.treasures if not t["collected"]],
                "wormholes": self.engine.wormholes_active,
                "score": self.score
            }).encode()
            
            # Get Oracle consciousness rating
            phi_result = self.oracle.calculate_phi(state_data)
            phi = phi_result.get("phi_normalized", 0.5)
            
            # High consciousness (phi) = more strategic, low = more random
            if phi > 0.7:
                return self._strategic_decision()
            elif phi > 0.4:
                return self._heuristic_decision()
            else:
                return self._random_decision()
                
        except Exception as e:
            logger.error(f"Oracle decision error: {e}")
            return self._heuristic_decision()
    
    def _strategic_decision(self) -> Tuple[int, int]:
        """Find closest high-value treasure"""
        best_treasure = None
        best_score = -1
        
        for t in self.treasure_map.treasures:
            if not t["collected"]:
                dist = abs(t["x"] - self.player_x) + abs(t["y"] - self.player_y)
                if dist > 0:
                    score = t["value"] / dist
                    if score > best_score:
                        best_score = score
                        best_treasure = t
        
        if best_treasure:
            dx = 1 if best_treasure["x"] > self.player_x else (-1 if best_treasure["x"] < self.player_x else 0)
            dy = 1 if best_treasure["y"] > self.player_y else (-1 if best_treasure["y"] < self.player_y else 0)
            return (dx, dy)
        
        return (0, 0)
    
    def _heuristic_decision(self) -> Tuple[int, int]:
        """Find closest treasure regardless of value"""
        best_treasure = None
        best_dist = float('inf')
        
        for t in self.treasure_map.treasures:
            if not t["collected"]:
                dist = abs(t["x"] - self.player_x) + abs(t["y"] - self.player_y)
                if dist < best_dist:
                    best_dist = dist
                    best_treasure = t
        
        if best_treasure:
            dx = 1 if best_treasure["x"] > self.player_x else (-1 if best_treasure["x"] < self.player_x else 0)
            dy = 1 if best_treasure["y"] > self.player_y else (-1 if best_treasure["y"] < self.player_y else 0)
            return (dx, dy)
        
        return (0, 0)
    
    def _random_decision(self) -> Tuple[int, int]:
        """Random movement"""
        return (random.choice([-1, 0, 1]), random.choice([-1, 0, 1]))
    
    def move(self, dx: int, dy: int):
        """Move player and handle treasure collection"""
        new_x = max(0, min(self.treasure_map.width - 1, self.player_x + dx))
        new_y = max(0, min(self.treasure_map.height - 1, self.player_y + dy))
        
        self.player_x = new_x
        self.player_y = new_y
        
        # Check for treasure
        treasure = self.treasure_map.get_treasure_at(self.player_x, self.player_y)
        if treasure:
            gained = self.treasure_map.collect(treasure["id"])
            self.score += gained
            
            # Record to page curve
            PAGE_CURVE_HISTORY.append((time.time(), self.score))
            if len(PAGE_CURVE_HISTORY) > 200:
                PAGE_CURVE_HISTORY.pop(0)
            
            logger.info(f"🏴‍☠️ Collected {treasure['rarity']} treasure: +{gained:.4f} (Total: {self.score:.4f})")
            
            # Create EPR pair
            self.engine.create_epr_pair(
                0, 0,
                (int(time.time()) % 100, self.player_x),
                (self.player_y, self.player_x + self.player_y)
            )
    
    def update(self, dt: float):
        """Update game state"""
        self.engine.update(dt)
        
        # Process mempool updates
        while not NEW_TX_QUEUE.empty():
            try:
                added = NEW_TX_QUEUE.get_nowait()
                for tx in added[:5]:
                    value = tx.get("fee", 1000) / 1e8 if isinstance(tx, dict) else 0.001
                    x = random.randint(0, self.treasure_map.width - 1)
                    y = random.randint(0, self.treasure_map.height - 1)
                    
                    self.treasure_map.treasures.append({
                        "id": 9000 + int(time.time() * 1000) % 10000,
                        "x": x,
                        "y": y,
                        "value": max(0.01, value),
                        "rarity": "LEGENDARY" if value > 0.5 else "EPIC",
                        "phase": self.treasure_map.phase_field[y, x],
                        "phase_score": (math.sin(self.treasure_map.phase_field[y, x]) + 1) / 2,
                        "collected": False,
                        "live_txid": str(tx.get("txid", ""))[:8] if isinstance(tx, dict) else ""
                    })
            except queue.Empty:
                break
        
        # Random collapse
        for pair in self.engine.pairs:
            if pair.entangled and random.random() < 0.005:
                self.engine.collapse(pair)
    
    def run(self):
        """Main game loop"""
        logger.info("=" * 70)
        logger.info("JONES QUANTUM GRAVITY — ER=EPR ROGUELITE MINER (HEADLESS)")
        logger.info("=" * 70)
        logger.info("🚀 Miner starting with Oracle integration")
        logger.info(f"Oracle available: {ORACLE_AVAILABLE}")
        logger.info(f"WebSocket available: {WEBSOCKET_AVAILABLE}")
        logger.info("=" * 70)
        
        # Start WebSocket thread if available
        if WEBSOCKET_AVAILABLE:
            ws_thread = threading.Thread(target=start_mempool_ws, daemon=True)
            ws_thread.start()
            logger.info("🌊 WebSocket thread started")
        
        frame_time = 1.0 / 10.0  # 10 FPS for headless mode
        
        try:
            while self.running:
                current_time = time.time()
                dt = current_time - self.last_update
                
                if dt >= frame_time:
                    # Get Oracle decision and move
                    dx, dy = self.get_oracle_decision()
                    if dx != 0 or dy != 0:
                        self.move(dx, dy)
                    
                    # Update game state
                    self.update(dt)
                    
                    # Log status periodically (every 10 seconds)
                    if current_time - self.last_status_log >= 10.0:
                        self.last_status_log = current_time
                        uncollected = sum(1 for t in self.treasure_map.treasures if not t["collected"])
                        logger.info(
                            f"Status: Score={self.score:.2f}, "
                            f"Wormholes={self.engine.wormholes_active}, "
                            f"Treasures={uncollected}, "
                            f"Pos=({self.player_x}, {self.player_y})"
                        )
                    
                    self.last_update = current_time
                else:
                    time.sleep(frame_time - dt)
                    
        except KeyboardInterrupt:
            logger.info("🛑 Miner stopped by user")
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}", exc_info=True)
        finally:
            self.running = False
            logger.info(f"Final score: {self.score:.2f}")
            logger.info("👋 Miner shutdown complete")

# ============================================================
# MAIN
# ============================================================
def main():
    miner = OracleMiner()
    miner.run()

if __name__ == "__main__":
    main()
