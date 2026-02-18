# Jones Quantum Pirate Miner - Integration Guide

## Overview

SphinxOS now includes **two versions** of the Jones Quantum Pirate Miner:

1. **Desktop Version** (`quantum_pirate_miner.py`) - Full Pygame implementation with live WebSocket
2. **Web Version** (`quantum_game_web/`) - Flask-based web server for browser play

Both versions implement the same core quantum mechanics based on the Jones Quantum Gravity framework.

## Desktop Version (Pygame)

### Location
- Main script: `/quantum_pirate_miner.py`
- Launcher: `/run_quantum_pirate_miner.sh`
- Tests: `/test_quantum_pirate_miner.py`
- Documentation: `/QUANTUM_PIRATE_MINER_README.md`

### Features
- Full 1280x720 Pygame window
- Real-time page curve visualization
- Optional WebSocket connection to Bitcoin mempool
- 60 FPS gameplay
- Keyboard controls (WASD/Arrows)

### Running

```bash
# Quick start
python quantum_pirate_miner.py

# Or use launcher
./run_quantum_pirate_miner.sh

# Run tests
python test_quantum_pirate_miner.py
```

### Dependencies
```bash
pip install pygame numpy scipy numba websocket-client
```

### Use Cases
- Local development and testing
- Desktop gaming experience
- High-performance visualization
- Direct mempool integration

## Web Version (Flask)

### Location
- Application: `/quantum_game_web/app.py`
- Game engine: `/quantum_game_web/game_engine.py`
- Templates: `/quantum_game_web/templates/`
- Documentation: `/quantum_game_web/README.md`

### Features
- Browser-based gameplay
- REST API for game state
- Embeddable iframe support
- Headless game engine
- Mobile-friendly controls

### Running

```bash
cd quantum_game_web
python app.py
```

Server starts at `http://localhost:5050`

### Dependencies
```bash
pip install flask flask-cors numpy numba scikit-learn
```

### Use Cases
- Web deployment
- Embedding in websites
- Multi-user access
- Cross-platform compatibility

## Technical Comparison

| Feature | Desktop (Pygame) | Web (Flask) |
|---------|-----------------|-------------|
| **Graphics** | Pygame rendering | HTML5 Canvas |
| **Display** | 1280x720 window | Browser-based |
| **FPS** | 60 | 10 (server) |
| **Input** | Keyboard direct | AJAX/WebSocket |
| **Mempool** | Direct WebSocket | Optional |
| **Deployment** | Local executable | Web server |
| **Dependencies** | Pygame + SDL | Flask + Browser |

## Core Quantum Mechanics (Shared)

Both versions implement:

### 1. UFT Derivation
- Schmidt decomposition via SVD
- Entanglement entropy: S = -Σ λ²log₂(λ²)
- Seven-fold warp: W7 = Σ λₖ cos(φₖ)
- Warp integral: I = W7 × Tr(M)
- Inertial mass reduction

### 2. Entanglement Engine
- EPR pair creation (wormholes)
- Graph node management
- GHZ state generation
- Quantum collapse mechanics

### 3. Treasure System
- Phase field generation
- Rarity tiers (COMMON, RARE, EPIC, LEGENDARY)
- Value modulation
- Collection mechanics

### 4. Page Curve
- Real-time entropy tracking
- Ergotropy island visualization
- Scrolling time window

## Deployment Options

### Option 1: Desktop Only
Best for: Development, testing, local play

```bash
python quantum_pirate_miner.py
```

### Option 2: Web Only
Best for: Production, multi-user, embedding

```bash
cd quantum_game_web
gunicorn -w 4 -b 0.0.0.0:5050 app:app
```

### Option 3: Both
Best for: Complete offering

```bash
# Terminal 1: Desktop version
python quantum_pirate_miner.py

# Terminal 2: Web version
cd quantum_game_web && python app.py
```

## Integration with SphinxOS

Both versions integrate with the SphinxOS ecosystem:

### Desktop Integration
- Can read from SphinxOS blockchain state
- Can emit quantum events to SphinxOS event bus
- Can integrate with NPTC spectral gap data

### Web Integration
- REST API endpoints for SphinxOS services
- WebSocket for real-time updates
- CORS enabled for cross-origin access

## Configuration

### Desktop Configuration
Edit `quantum_pirate_miner.py`:

```python
# Display settings
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

# Game settings
MAP_WIDTH = 64
MAP_HEIGHT = 48
INITIAL_TREASURES = 30

# WebSocket (optional)
WEBSOCKET_AVAILABLE = True  # Set to False to disable
```

### Web Configuration
Edit `quantum_game_web/app.py`:

```python
# Server settings
HOST = '0.0.0.0'
PORT = 5050

# Game settings (in game_engine.py)
WIDTH = 64
HEIGHT = 48
INITIAL_TREASURES = 30
```

## Testing

### Desktop Tests
```bash
python test_quantum_pirate_miner.py
```

Runs 24 unit tests covering:
- UFT derivation functions
- Entanglement engine
- Treasure map
- ECS framework
- Data structures

### Web Tests
```bash
cd quantum_game_web
# Manual testing via browser or:
curl http://localhost:5050/health
curl http://localhost:5050/api/game/state
```

## Troubleshooting

### Desktop Issues

**"No module named pygame"**
```bash
pip install pygame
```

**"SDL_VIDEODRIVER not available"**
- Ensure you have a display (not headless)
- Or set: `export SDL_VIDEODRIVER=dummy` for testing

**"scipy 0.16+ required"**
```bash
pip install scipy
```

### Web Issues

**"Address already in use"**
```bash
# Change port in app.py or:
lsof -ti:5050 | xargs kill -9
```

**"CORS errors"**
- Ensure `flask-cors` is installed
- Check CORS(app) is called in app.py

## Performance Optimization

### Desktop
- Runs at 60 FPS by default
- Numba JIT compilation for hot paths
- Multi-threaded WebSocket

### Web
- Background thread for game updates (10 FPS)
- Client-side rendering (HTML5 Canvas)
- State synchronization (5 Hz)
- Consider using WebSocket for real-time (not just AJAX)

## Security Considerations

### Desktop
- WebSocket to mempool.space is read-only
- No network exposure
- Local file access only

### Web
- Enable HTTPS in production
- Rate limit API endpoints
- Sanitize user input (movement commands)
- Use CORS properly
- Consider authentication for leaderboard

## Future Enhancements

Potential improvements for both versions:

- [ ] Multi-player support
- [ ] Persistent leaderboards
- [ ] Achievement system
- [ ] Sound effects
- [ ] Advanced AI enemies
- [ ] Quantum power-ups
- [ ] Procedural map generation
- [ ] Tournament mode
- [ ] NFT integration for treasures
- [ ] L2 blockchain integration

## API Reference (Web Only)

### GET /api/game/state
Returns complete game state

**Response:**
```json
{
  "player": {"x": 32, "y": 24},
  "score": 12.34,
  "wormholes_active": 5,
  "treasure_map": {...},
  "page_curve": [...]
}
```

### POST /api/game/move
Move player

**Request:**
```json
{"dx": 1, "dy": 0}
```

**Response:**
```json
{
  "success": true,
  "collected": false,
  "player": {"x": 33, "y": 24},
  "score": 12.34
}
```

### POST /api/game/reset
Reset game to initial state

### GET /api/game/leaderboard
Get top scores

### GET /health
Health check

## Contact & Support

For issues, questions, or contributions:

- **Author**: Captain Travis D. Jones
- **Organization**: Houston HQ
- **Framework**: Jones Quantum Gravity Resolution
- **Repository**: SphinxOS

## License

See LICENSE file in repository root.

---

**Choose your quantum reality: Desktop for immersion, Web for accessibility! 🏴‍☠️⚛️**
