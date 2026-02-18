# Jones Quantum Pirate Miner

## ER=EPR Quantum-Pirate Roguelite Miner – V25.1 Omega Brane Edition

A quantum physics-inspired roguelite game combining Jones Quantum Gravity theory with Bitcoin mempool data streaming. Experience real-time entanglement mechanics, deterministic page curves, and quantum treasure hunting!

## Features

### Core Quantum Mechanics
- **UFT (Unified Field Theory) Derivation**: Real-time computation of Schmidt decomposition, entanglement entropy, and seven-fold warp fields
- **ER=EPR Entanglement Engine**: Create and manage EPR pairs representing wormhole connections
- **GHZ States**: Multi-qubit entanglement with temporal coherence
- **Deterministic Page Curve**: Visualize entropy accumulation as you collect treasures

### Game Features
- **Quantum Treasure Map**: 64x48 grid with phase field-modulated treasure values
- **Live Bitcoin Mempool Integration**: WebSocket connection to mempool.space for real-time transaction data
- **Treasure Rarities**: COMMON, RARE, EPIC, and LEGENDARY treasures with varying values
- **Mining Simulation**: Blockchain height tracking with difficulty adjustment
- **Oracle AI**: Predictive AI system for optimal treasure discovery

### Technical Architecture
- **ECS (Entity Component System)**: Efficient game architecture separating data from logic
- **Numba JIT Compilation**: High-performance numerical computations
- **Multi-threaded**: WebSocket runs in separate thread for non-blocking updates
- **Pygame Rendering**: 1280x720 game window with real-time visualizations

## Installation

### Requirements
- Python 3.8+
- pygame
- numpy
- scipy (required for numba linear algebra)
- numba
- websocket-client (optional, for live mempool data)

### Quick Start

```bash
# Install dependencies
pip install pygame numpy scipy numba websocket-client

# Run the game
python quantum_pirate_miner.py

# Or use the launcher script
./run_quantum_pirate_miner.sh
```

### Alternative: Install from requirements.txt

```bash
pip install -r requirements.txt
python quantum_pirate_miner.py
```

## How to Play

### Controls
- **WASD** or **Arrow Keys**: Move your player
- **ESC**: Quit the game

### Objective
1. Navigate the quantum treasure map
2. Collect treasures to increase your score
3. Watch your Page Curve grow on the right side of the screen
4. Create wormholes (EPR pairs) by collecting treasures
5. Maximize your quantum entropy score!

### Game Elements

#### Player (Green Circle)
- Your quantum pirate character
- Moves on a discrete 64x48 grid

#### Treasures (Colored Circles)
- **Gray** (COMMON): Low value, frequent
- **Blue** (RARE): Medium value
- **Purple** (EPIC): High value
- **Gold** (LEGENDARY): Maximum value, extremely rare

#### Serpents (Orange Circles)
- Quantum entities with oscillating phases
- Currently harmless, add to the quantum atmosphere

#### Page Curve (Red Line, Right Side)
- Real-time visualization of entropy accumulation
- Shows deterministic ergotropy islands
- Scrolls as you collect more treasures

## Technical Details

### Jones Quantum Gravity Framework

The game implements key concepts from Jones Quantum Gravity theory:

1. **Schmidt Decomposition**: SVD-based entanglement quantification
2. **Entanglement Entropy**: S = -Σ λ²log₂(λ²)
3. **Seven-Fold Warp**: W7 = Σ λₖ cos(φₖ)
4. **Warp Integral**: I = W7 × Tr(M)
5. **Inertial Mass Reduction**: m' = m(1 - η sin(2πft)(I/I₀))

### WebSocket Integration

When available, the game connects to `wss://mempool.space/api/v1/ws` to receive:
- New transaction notifications
- Block updates
- Mempool statistics

Live transactions are converted into LEGENDARY and EPIC treasures with values based on transaction fees.

### ECS Architecture

The game uses a pure Entity Component System:

**Components:**
- `Player`: Marker for player entity
- `Position`: x, y coordinates
- `Intent`: Movement intention
- `Serpent`: Enemy with phase
- `Entanglement`: Quantum entanglement engine
- `TreasureMapComp`: Treasure map data
- `Score`: Player score
- `OracleBrain`: AI state
- `ChainState`: Blockchain state
- `Mempool`: Transaction pool
- `MiningConfig`: Mining parameters

**Systems:**
- `input_system`: Process keyboard input
- `movement_system`: Update positions
- `treasure_system`: Handle collection and scoring
- `oracle_system`: AI decision making
- `entanglement_system`: Quantum state updates
- `mining_system`: Blockchain simulation
- `mempool_process_system`: WebSocket data processing
- `render_system`: Draw everything

## Performance

- Target: 60 FPS
- Resolution: 1280x720
- JIT-compiled numerical functions for optimal performance
- Multi-threaded WebSocket for non-blocking network operations

## Troubleshooting

### WebSocket Not Available
If you see "websocket-client not available", the game will still work but without live Bitcoin mempool data. To enable:
```bash
pip install websocket-client
```

### Display Issues
If you're running on a headless server, you need an X server or use VNC/RDP for display. The game requires a graphical environment.

### NumPy/SciPy Errors
Ensure you have scipy installed:
```bash
pip install scipy
```

This is required for numba's linear algebra operations (SVD).

## Architecture

```
quantum_pirate_miner.py
├── UFT Derivation Engine
│   ├── Schmidt Decomposition (SVD)
│   ├── Entanglement Entropy
│   ├── Seven-Fold Warp
│   └── Inertial Mass Reduction
├── WebSocket Handler (optional)
│   └── Mempool.space integration
├── Entanglement Engine
│   ├── EPR Pair Management
│   ├── Graph Nodes
│   └── GHZ States
├── Treasure Map
│   ├── Phase Field Generation
│   └── Treasure Spawning
├── ECS Framework
│   ├── Entity Management
│   ├── Component Storage
│   └── System Registry
└── Game Loop
    ├── Input Processing
    ├── Physics Update
    ├── Rendering
    └── Page Curve Visualization
```

## Future Enhancements

Potential features for future versions:
- [ ] Multi-player support
- [ ] Persistent leaderboard
- [ ] Advanced AI serpents
- [ ] Power-ups and special abilities
- [ ] Additional quantum mechanics (teleportation, superposition)
- [ ] Sound effects and music
- [ ] Procedural map generation
- [ ] Achievement system

## Credits

**Author**: Captain Travis D. Jones  
**Organization**: Houston HQ  
**Framework**: Jones Quantum Gravity Resolution  
**Date**: February 18, 2026  
**Version**: V25.1 Omega Brane Edition

## License

See LICENSE file in the root directory.

## References

- Jones Quantum Gravity Framework
- ER=EPR Correspondence (Maldacena & Susskind)
- Page Curve and Black Hole Information Paradox
- Bitcoin Mempool Architecture
- Entity Component System Design Pattern

---

**May your wormholes be stable and your treasures legendary! 🏴‍☠️⚛️**
