#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ER=EPR Quantum-Pirate Roguelite Miner – V27.0 Omega Brane Edition (Web Version)
Jones Quantum Gravity Full Resolution — Headless Game Engine for Web
Captain Travis D. Jones – Houston HQ, February 18 2026
"""

import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time
import json
import threading
import queue
from numba import njit

# ============================================================
# GLOBALS FOR PAGE CURVE, WS QUEUE, LEADERBOARD
# ============================================================
PAGE_CURVE_HISTORY: List[Tuple[float, float]] = []
NEW_TX_QUEUE = queue.Queue()
LEADERBOARD_FILE = "jqg_leaderboard.json"

# Shared state from NPTC simulation
NPTC_SPECTRAL_GAP = 0.0
NPTC_LAST_BLOB = None

# ============================================================
# UFT DERIVATION (Jones Framework Core)
# ============================================================
def derive_uft(t: float) -> Dict:
    scalar = 1.618
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
# J4 (Jones 4-fold modular flow)
# ============================================================
def J4(base: float) -> float:
    return 4.0 * math.sin(base) * math.cos(base * 1.6180339887)

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

class EntanglementEngine:
    def __init__(self, width=64, height=48):
        self.width = width
        self.height = height
        self.pairs = []
        self.wormholes_active = 0
        self.next_pair_id = 0
        self.operator_trail = []

    def create_epr_pair(self, tx_time, tx_space, time_coord, space_coord):
        pair = EPRPair(
            time_id=self.next_pair_id,
            space_id=self.next_pair_id+1,
            coord_time=time_coord,
            coord_space=space_coord
        )
        self.pairs.append(pair)
        self.next_pair_id += 2
        self.wormholes_active += 1
        return pair

    def collapse(self, pair):
        if pair.entangled:
            pair.entangled = False
            self.wormholes_active -= 1

# ============================================================
# TREASURE MAP
# ============================================================
class TreasureMap:
    def __init__(self, entanglement_engine, width=64, height=48):
        self.engine = entanglement_engine
        self.width = width
        self.height = height
        self.phase_field = np.random.uniform(0, 2*math.pi, (height, width))
        self.treasures = []
        self.generate_treasures(30)

    def generate_treasures(self, count):
        for i in range(count):
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
            phase = self.phase_field[y, x]
            rarity = random.choices(["COMMON", "RARE", "EPIC", "LEGENDARY"],
                                    weights=[50,30,15,5])[0]
            base_val = {"COMMON":0.1, "RARE":0.3, "EPIC":0.7, "LEGENDARY":1.5}[rarity]
            val = base_val * (math.sin(phase) + 1.5) * (1 + 0.1 * self.engine.wormholes_active)
            color_map = {
                "COMMON": [200,200,200],
                "RARE": [100,150,255],
                "EPIC": [255,100,255],
                "LEGENDARY": [255,215,0]
            }
            self.treasures.append({
                "id": i,
                "x": x,
                "y": y,
                "value": val,
                "rarity": rarity,
                "color": color_map[rarity],
                "phase": phase,
                "phase_score": (math.sin(phase)+1)/2,
                "collected": False
            })

    def get_treasure_at(self, x, y):
        for t in self.treasures:
            if not t["collected"] and t["x"] == x and t["y"] == y:
                return t
        return None

    def collect(self, tid):
        for t in self.treasures:
            if t["id"] == tid and not t["collected"]:
                t["collected"] = True
                return t["value"]
        return 0.0

    def to_dict(self):
        """Convert treasure map to dict for JSON serialization"""
        return {
            "treasures": [
                {
                    "id": t["id"],
                    "x": t["x"],
                    "y": t["y"],
                    "value": t["value"],
                    "rarity": t["rarity"],
                    "color": t["color"],
                    "collected": t["collected"]
                }
                for t in self.treasures
            ]
        }

# ============================================================
# GAME STATE
# ============================================================
class QuantumGameState:
    def __init__(self):
        self.engine = EntanglementEngine()
        self.treasure_map = TreasureMap(self.engine)
        self.player_x = self.treasure_map.width // 2
        self.player_y = self.treasure_map.height // 2
        self.score = 0.0
        self.mempool_txs = []
        self.chain_height = 0
        self.difficulty = derive_uft(time.time())["integral_I"] * 1e8
        self.last_block_time = time.time()
        self.last_update = time.time()
        
    def move_player(self, dx, dy):
        """Move player and handle boundary checks"""
        new_x = self.player_x + dx
        new_y = self.player_y + dy
        self.player_x = max(0, min(self.treasure_map.width-1, new_x))
        self.player_y = max(0, min(self.treasure_map.height-1, new_y))
        
    def check_treasure_collection(self):
        """Check if player is on a treasure and collect it"""
        global PAGE_CURVE_HISTORY, NPTC_SPECTRAL_GAP
        t = self.treasure_map.get_treasure_at(int(self.player_x), int(self.player_y))
        if t:
            gained = self.treasure_map.collect(t["id"])
            gained *= (1.0 + 0.5 * NPTC_SPECTRAL_GAP)
            self.score += gained
            PAGE_CURVE_HISTORY.append((time.time(), self.score))
            if len(PAGE_CURVE_HISTORY) > 200:
                PAGE_CURVE_HISTORY.pop(0)
            print(f"🏴‍☠️ Collected {t['rarity']} treasure: +{gained:.4f} (S={self.score:.4f})")
            self.engine.create_epr_pair(0, 0, (int(time.time())%100, int(self.player_x)), 
                                       (int(self.player_y), int(self.player_x+self.player_y)))
            return True
        return False
        
    def update(self, dt):
        """Update game state"""
        self.last_update = time.time()
        
        # Entanglement collapse
        for pair in self.engine.pairs:
            if pair.entangled and random.random() < 0.005:
                self.engine.collapse(pair)
        
        # Mining simulation
        if time.time() - self.last_block_time > 10.0:
            self.last_block_time = time.time()
            self.chain_height += 1
            pressure = len(self.mempool_txs) * 1e-6
            diff_mod = 1.0 + 0.1 * NPTC_SPECTRAL_GAP
            self.difficulty *= (1 + 0.01 * pressure) * diff_mod
            
    def to_dict(self):
        """Convert game state to dict for JSON serialization"""
        return {
            "player": {
                "x": self.player_x,
                "y": self.player_y
            },
            "score": self.score,
            "wormholes_active": self.engine.wormholes_active,
            "mempool_txs": len(self.mempool_txs),
            "chain_height": self.chain_height,
            "difficulty": self.difficulty,
            "spectral_gap": NPTC_SPECTRAL_GAP,
            "treasure_map": self.treasure_map.to_dict(),
            "page_curve": [{"time": t, "score": s} for t, s in PAGE_CURVE_HISTORY[-80:]]
        }
