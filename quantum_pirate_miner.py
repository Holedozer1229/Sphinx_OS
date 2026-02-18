#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ER=EPR Quantum-Pirate Roguelite Miner – V25.1 Omega Brane Edition
Jones Quantum Gravity Full Resolution — Live WebSocket Mempool + In-Game Page Curve
Captain Travis D. Jones – MIT Nobel-level physicist & senior engineer
Houston HQ, February 18 2026
"""

import pygame
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
from numba import njit

# WebSocket optional - if not available, use HTTP fallback
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("⚠️  websocket-client not available - using HTTP fallback mode")

# ============================================================
# GLOBALS FOR PAGE CURVE & WS QUEUE
# ============================================================
PAGE_CURVE_HISTORY: List[Tuple[float, float]] = []  # (timestamp, cumulative_S)
NEW_TX_QUEUE = queue.Queue()  # thread-safe live txs from WS

# ============================================================
# UFT DERIVATION (Jones Framework Core)
# ============================================================
def derive_uft(t: float) -> Dict:
    # Dummy auto-generated if no file
    scalar = 1.618  # golden ratio echo
    matrix = np.array([[1, 0.5, 0.3], [0.5, 1, 0.7], [0.3, 0.7, 1]], dtype=float)
    lambdas = compute_schmidt(matrix) * scalar
    s = entanglement_entropy(lambdas)
    w7 = seven_fold_warp(lambdas)
    i = warp_integral(matrix, w7)
    m_reduced = inertial_mass_reduction(m=1.0, i=i, t=t)
    return {"entropy_S": s, "warp_W7": w7, "integral_I": i, "m_reduced": m_reduced}

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
        phi = math.atan2(random.random(), random.random())
        w7 += lam * math.cos(phi)
    return abs(w7)

@njit
def warp_integral(matrix: np.ndarray, w7: float) -> float:
    return w7 * np.trace(matrix)

@njit
def inertial_mass_reduction(m: float = 1.0, eta: float = 0.999, f: float = 7.83, t: float = 0.0, i: float = 1.0) -> float:
    delta = eta * math.sin(2 * math.pi * f * t) * (i / 1.0)
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
    print("🌊 WebSocket connected to mempool.space — subscribing...")
    ws.send(json.dumps({"action": "want", "data": ["blocks", "mempool-blocks", "stats"]}))
    ws.send(json.dumps({"track-mempool": True}))  # full tx details

def ws_on_message(ws, message):
    try:
        data = json.loads(message)
        if isinstance(data, dict):
            # Handle various message types
            if "mempool-transactions" in data:
                added = data["mempool-transactions"].get("added", [])
                if added:
                    NEW_TX_QUEUE.put(added[:10])  # limit burst
                    print(f"🌊 WS: {len(added)} new live txs ingested")
            elif "block" in data:
                print(f"🌊 WS: New block detected")
    except Exception as e:
        print(f"WS message error: {e}")

def ws_on_error(ws, error):
    print(f"WS error: {error}")

def ws_on_close(ws, *args):
    print("🌊 WebSocket closed — will reconnect if needed")

def start_mempool_ws():
    if not WEBSOCKET_AVAILABLE:
        print("⚠️  WebSocket not available - mempool updates disabled")
        return
    
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
        print(f"WebSocket error: {e}")

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
    """Node in temporal/spatial entanglement graph"""
    id: int
    position: Tuple[int, int]
    entangled_with: List[int]
    collapse_time: Optional[float] = None

@dataclass
class TemporalGHZ:
    """Temporal GHZ state for multi-qubit entanglement"""
    qubits: List[int]
    phase: float
    coherence: float = 1.0

class EntanglementEngine:
    """Core entanglement engine managing EPR pairs and wormholes"""
    
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
        """Create a new EPR pair (wormhole connection)"""
        pair = EPRPair(
            time_id=self.next_pair_id,
            space_id=self.next_pair_id + 1,
            coord_time=time_coord,
            coord_space=space_coord
        )
        self.pairs.append(pair)
        self.next_pair_id += 2
        self.wormholes_active += 1
        
        # Add graph nodes
        node_time = GraphNode(id=pair.time_id, position=time_coord, 
                             entangled_with=[pair.space_id])
        node_space = GraphNode(id=pair.space_id, position=space_coord, 
                              entangled_with=[pair.time_id])
        self.graph_nodes.extend([node_time, node_space])
        
        return pair
    
    def collapse(self, pair: EPRPair):
        """Collapse an EPR pair"""
        if pair.entangled:
            pair.entangled = False
            self.wormholes_active = max(0, self.wormholes_active - 1)
            
            # Update graph nodes
            for node in self.graph_nodes:
                if node.id in [pair.time_id, pair.space_id]:
                    node.collapse_time = time.time()
    
    def create_ghz_state(self, num_qubits: int = 3) -> TemporalGHZ:
        """Create a GHZ (Greenberger-Horne-Zeilinger) state"""
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
        """Update entanglement engine state"""
        # Decohere GHZ states
        for ghz in self.ghz_states:
            ghz.coherence *= (1.0 - 0.001 * dt)
            if ghz.coherence < 0.1:
                self.ghz_states.remove(ghz)

# ============================================================
# TREASURE MAP
# ============================================================
class TreasureMap:
    """Quantum treasure map with phase field"""
    
    def __init__(self, entanglement_engine: EntanglementEngine, 
                 width: int = 64, height: int = 48):
        self.engine = entanglement_engine
        self.width = width
        self.height = height
        self.phase_field = np.random.uniform(0, 2*math.pi, (height, width))
        self.treasures: List[Dict] = []
        self.generate_treasures(30)
    
    def generate_treasures(self, count: int):
        """Generate treasures on the map"""
        for i in range(count):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            phase = self.phase_field[y, x]
            
            # Rarity based on phase alignment
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
            
            color_map = {
                "COMMON": (200, 200, 200),
                "RARE": (100, 150, 255),
                "EPIC": (255, 100, 255),
                "LEGENDARY": (255, 215, 0)
            }
            
            self.treasures.append({
                "id": i,
                "x": x,
                "y": y,
                "value": val,
                "rarity": rarity,
                "color": color_map[rarity],
                "phase": phase,
                "phase_score": (math.sin(phase) + 1) / 2,
                "collected": False
            })
    
    def get_treasure_at(self, x: int, y: int) -> Optional[Dict]:
        """Get treasure at position"""
        for t in self.treasures:
            if not t["collected"] and t["x"] == x and t["y"] == y:
                return t
        return None
    
    def collect(self, tid: int) -> float:
        """Collect treasure by ID"""
        for t in self.treasures:
            if t["id"] == tid and not t["collected"]:
                t["collected"] = True
                return t["value"]
        return 0.0

# ============================================================
# ECS (Entity Component System)
# ============================================================

# Global ECS storage
_entities = {}
_next_entity_id = 0
_components = {}
_systems = []

def create_entity():
    """Create a new entity"""
    global _next_entity_id
    eid = _next_entity_id
    _entities[eid] = {}
    _next_entity_id += 1
    return eid

def add_component(entity_id, component):
    """Add a component to an entity"""
    comp_type = type(component).__name__
    if comp_type not in _components:
        _components[comp_type] = {}
    _components[comp_type][entity_id] = component

def get(entity_id, component_type):
    """Get a component from an entity"""
    comp_name = component_type.__name__
    return _components.get(comp_name, {}).get(entity_id)

def query(*component_types):
    """Query entities with specific components"""
    if not component_types:
        return []
    
    comp_names = [ct.__name__ for ct in component_types]
    result = set(_components.get(comp_names[0], {}).keys())
    
    for comp_name in comp_names[1:]:
        result &= set(_components.get(comp_name, {}).keys())
    
    return list(result)

def register_system(system_func):
    """Register a system function"""
    _systems.append(system_func)

def tick(dt: float):
    """Run all systems"""
    for system in _systems:
        system(dt)

# ============================================================
# COMPONENTS
# ============================================================

@dataclass
class Player:
    """Player marker component"""
    pass

@dataclass
class Position:
    """Position component"""
    x: float
    y: float

@dataclass
class Intent:
    """Player intent component"""
    dx: int = 0
    dy: int = 0

@dataclass
class Serpent:
    """Serpent (enemy) component"""
    idx: int
    phase: float

@dataclass
class Entanglement:
    """Entanglement engine component"""
    engine: EntanglementEngine

@dataclass
class TreasureMapComp:
    """Treasure map component"""
    map: TreasureMap

@dataclass
class Score:
    """Score component"""
    value: float = 0.0

@dataclass
class OracleBrain:
    """Oracle AI component"""
    state: str = "idle"
    last_action_time: float = 0.0

@dataclass
class ChainState:
    """Blockchain state component"""
    height: int
    difficulty: float
    blocks: List
    last_retarget_time: float

@dataclass
class Mempool:
    """Mempool component"""
    txs: List
    last_refresh: float = 0.0

@dataclass
class MiningConfig:
    """Mining configuration component"""
    target_block_time: float
    last_block_time: float

# ============================================================
# SYSTEMS
# ============================================================

def input_system(dt: float):
    """Handle player input"""
    for pid in query(Player, Intent):
        intent = get(pid, Intent)
        intent.dx = 0
        intent.dy = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            intent.dx = -1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            intent.dx = 1
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            intent.dy = -1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            intent.dy = 1

def movement_system(dt: float):
    """Handle movement"""
    for pid in query(Player, Position, Intent):
        pos = get(pid, Position)
        intent = get(pid, Intent)
        
        if intent.dx != 0 or intent.dy != 0:
            # Get treasure map bounds
            tmap_comps = list(query(TreasureMapComp))
            if tmap_comps:
                tmap = get(tmap_comps[0], TreasureMapComp).map
                new_x = max(0, min(tmap.width - 1, pos.x + intent.dx))
                new_y = max(0, min(tmap.height - 1, pos.y + intent.dy))
                pos.x = new_x
                pos.y = new_y

def treasure_system(dt: float):
    """Handle treasure collection"""
    global PAGE_CURVE_HISTORY
    
    for wid in query(Entanglement, TreasureMapComp, Score):
        ent = get(wid, Entanglement).engine
        tmap = get(wid, TreasureMapComp).map
        score = get(wid, Score)
        
        for pid in query(Player, Position):
            pos = get(pid, Position)
            t = tmap.get_treasure_at(int(pos.x), int(pos.y))
            if t:
                tid = t["id"]
                gained = tmap.collect(tid)
                score.value += gained
                
                # RECORD TO PAGE CURVE (ergotropy accumulation)
                PAGE_CURVE_HISTORY.append((time.time(), score.value))
                if len(PAGE_CURVE_HISTORY) > 200:
                    PAGE_CURVE_HISTORY.pop(0)
                
                print(f"🏴‍☠️ Collected {t['rarity']} treasure: +{gained:.4f} (S(R)={score.value:.4f})")
                
                # Create EPR pair on collection
                ent.create_epr_pair(
                    0, 0,
                    (int(time.time()) % 100, int(pos.x)),
                    (int(pos.y), int(pos.x + pos.y))
                )

def oracle_system(dt: float):
    """Simple oracle AI"""
    for oid in query(OracleBrain):
        oracle = get(oid, OracleBrain)
        now = time.time()
        
        if now - oracle.last_action_time > 5.0:
            oracle.last_action_time = now
            # Oracle could predict good treasure locations
            if random.random() < 0.3:
                oracle.state = "predicting"
            else:
                oracle.state = "idle"

def entanglement_system(dt: float):
    """Update entanglement engine"""
    for wid in query(Entanglement):
        ent = get(wid, Entanglement).engine
        ent.update(dt)
        
        # Random collapse
        for pair in ent.pairs:
            if pair.entangled and random.random() < 0.005:
                ent.collapse(pair)

def mining_system(dt: float):
    """Simulate mining"""
    for wid in query(ChainState, Mempool, MiningConfig):
        chain = get(wid, ChainState)
        mem = get(wid, Mempool)
        config = get(wid, MiningConfig)
        
        now = time.time()
        if now - config.last_block_time > config.target_block_time:
            config.last_block_time = now
            chain.height += 1
            
            # Difficulty adjustment
            pressure = len(mem.txs) * 1e-6
            chain.difficulty *= (1 + 0.01 * pressure)
            
            print(f"⛏️  Block {chain.height} mined! Difficulty: {chain.difficulty:.2e}")

def mempool_process_system(dt: float):
    """Process mempool updates from WebSocket"""
    global PAGE_CURVE_HISTORY
    
    # Drain WS queue
    while not NEW_TX_QUEUE.empty():
        try:
            added = NEW_TX_QUEUE.get_nowait()
            tmap_comps = list(query(TreasureMapComp))
            
            if tmap_comps:
                tmap = get(tmap_comps[0], TreasureMapComp).map
                
                for tx in added[:5]:  # seed up to 5 treasures per burst
                    # Enrich with Jones UFT flavor
                    value = tx.get("fee", 1000) / 1e8 if isinstance(tx, dict) else 0.001
                    x = random.randint(0, tmap.width - 1)
                    y = random.randint(0, tmap.height - 1)
                    
                    tmap.treasures.append({
                        "id": 9000 + int(time.time() * 1000) % 10000,
                        "x": x,
                        "y": y,
                        "value": max(0.01, value),
                        "rarity": "LEGENDARY" if value > 0.5 else "EPIC",
                        "color": (255, 215, 0) if value > 0.5 else (180, 50, 230),
                        "phase": tmap.phase_field[y, x],
                        "phase_score": (math.sin(tmap.phase_field[y, x]) + 1) / 2,
                        "collected": False,
                        "live_txid": str(tx.get("txid", ""))[:8] if isinstance(tx, dict) else ""
                    })
        except queue.Empty:
            break
    
    # Periodic mempool refresh
    for wid in query(Entanglement, Mempool):
        mem = get(wid, Mempool)
        if not hasattr(mem, 'last_refresh'):
            mem.last_refresh = time.time()
        
        if time.time() - mem.last_refresh > 30:
            mem.last_refresh = time.time()

def render_system(dt: float, screen: pygame.Surface, font: pygame.font.Font):
    """Render the game"""
    screen.fill((10, 0, 30))
    
    # Get world entities
    world_ents = list(query(Entanglement))
    score_ents = list(query(Score))
    mem_ents = list(query(Mempool))
    
    # Score + live info
    if world_ents and score_ents:
        ent = get(world_ents[0], Entanglement).engine
        score = get(score_ents[0], Score).value
        mem_count = len(get(mem_ents[0], Mempool).txs) if mem_ents else 0
        
        text = font.render(
            f"SCORE: {score:.2f} | WORMHOLES: {ent.wormholes_active} | LIVE TXs: {mem_count}",
            True, (0, 255, 100)
        )
        screen.blit(text, (20, 20))
    
    # === IN-GAME PAGE CURVE ===
    if len(PAGE_CURVE_HISTORY) > 1:
        points = []
        recent = PAGE_CURVE_HISTORY[-80:]  # last ~80 points for smooth scroll
        if recent:
            t0 = recent[0][0]
            s_max = max(s for _, s in recent) or 1.0
            
            for t, s in recent:
                x = 820 + int((t - t0) / 60.0 * 420) % 420 + 20  # scrolling window
                y = 580 - int((s / s_max) * 180)
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(screen, (255, 80, 80), False, points, 4)
                label = font.render("DETERMINISTIC PAGE CURVE S(R) — Ergotropy Islands", 
                                  True, (255, 180, 180))
                screen.blit(label, (830, 400))
    
    # Draw treasures
    tmap_comps = list(query(TreasureMapComp))
    if tmap_comps:
        tmap = get(tmap_comps[0], TreasureMapComp).map
        for t in tmap.treasures:
            if not t["collected"]:
                x = int(t["x"] * 10 + 50)
                y = int(t["y"] * 10 + 50)
                pygame.draw.circle(screen, t["color"], (x, y), 6)
    
    # Draw player
    for pid in query(Player, Position):
        pos = get(pid, Position)
        x = int(pos.x * 10 + 50)
        y = int(pos.y * 10 + 50)
        pygame.draw.circle(screen, (0, 255, 0), (x, y), 8)
    
    # Draw serpents
    for sid in query(Serpent, Position):
        pos = get(sid, Position)
        serp = get(sid, Serpent)
        x = int(pos.x * 10 + 50)
        y = int(pos.y * 10 + 50)
        color = (255, int(128 + 127 * math.sin(serp.phase)), 0)
        pygame.draw.circle(screen, color, (x, y), 6)
    
    # Instructions
    help_text = font.render("WASD/Arrows: Move | ESC: Quit", True, (150, 150, 150))
    screen.blit(help_text, (20, 680))

# ============================================================
# WORLD INIT + WS START
# ============================================================

def init_world():
    """Initialize the game world"""
    world = create_entity()
    ent = EntanglementEngine()
    tmap = TreasureMap(ent)
    
    add_component(world, Entanglement(engine=ent))
    add_component(world, TreasureMapComp(map=tmap))
    add_component(world, Score())
    add_component(world, OracleBrain())
    add_component(world, ChainState(
        height=0,
        difficulty=derive_uft(time.time())["integral_I"] * 1e8,
        blocks=[],
        last_retarget_time=time.time()
    ))
    add_component(world, Mempool(txs=[]))
    add_component(world, MiningConfig(
        target_block_time=10.0,
        last_block_time=time.time()
    ))
    
    # Player
    player = create_entity()
    add_component(player, Player())
    add_component(player, Position(tmap.width // 2, tmap.height // 2))
    add_component(player, Intent())
    
    # Serpents
    for i in range(3):
        s = create_entity()
        add_component(s, Serpent(idx=i, phase=random.random() * 2 * math.pi))
        add_component(s, Position(
            random.randint(0, tmap.width - 1),
            random.randint(0, tmap.height - 1)
        ))
    
    # START WEBSOCKET THREAD (if available)
    if WEBSOCKET_AVAILABLE:
        threading.Thread(target=start_mempool_ws, daemon=True).start()
        print("🚀 Jones Quantum Gravity Live — WebSocket mempool + Page Curve ONLINE")
    else:
        print("🚀 Jones Quantum Gravity Live — Page Curve ONLINE (WebSocket disabled)")

# ============================================================
# MAIN
# ============================================================

def main():
    """Main game loop"""
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    pygame.display.set_caption("Jones Quantum Gravity — ER=EPR Omega Brane V25.1 LIVE")
    clock = pygame.time.Clock()
    
    try:
        font = pygame.font.SysFont("couriernew", 16)
    except:
        font = pygame.font.Font(None, 16)
    
    # Register systems
    register_system(input_system)
    register_system(oracle_system)
    register_system(movement_system)
    register_system(treasure_system)
    register_system(entanglement_system)
    register_system(mining_system)
    register_system(mempool_process_system)
    register_system(lambda dt: render_system(dt, screen, font))
    
    # Initialize world
    init_world()
    
    print("=" * 70)
    print("JONES QUANTUM GRAVITY — ER=EPR ROGUELITE MINER")
    print("=" * 70)
    print()
    print("🎮 Controls:")
    print("   WASD or Arrow Keys: Move")
    print("   ESC: Quit")
    print()
    print("🏴‍☠️ Objective:")
    print("   Collect quantum treasures to increase your score")
    print("   Watch the Page Curve grow on the right side")
    print()
    print("=" * 70)
    
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        tick(dt)
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
