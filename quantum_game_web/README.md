# Jones Quantum Gravity Web Game

## ER=EPR Quantum-Pirate Roguelite Miner – V27.0 Omega Brane Edition (Web Version)

A web-based adaptation of the Jones Quantum Gravity game, featuring:

- **Quantum Entanglement Engine**: ER=EPR wormhole mechanics
- **Real-time Page Curve**: Entanglement entropy visualization
- **Treasure Mining**: Collect quantum treasures with phase-modulated values
- **NPTC Spectral Gap Integration**: Dynamic difficulty scaling
- **Blockchain Mining Simulation**: Live chain height and difficulty tracking

## Installation

### Prerequisites

```bash
pip install flask flask-cors numpy numba scikit-learn
```

### Running the Web Server

```bash
cd quantum_game_web
python app.py
```

The server will start on `http://localhost:5050`

## Endpoints

- `/` - Full game interface with stats panel
- `/embed` - Embeddable version for iframes
- `/api/game/state` - Get current game state (JSON)
- `/api/game/move` - Move player (POST with {dx, dy})
- `/api/game/reset` - Reset game (POST)
- `/api/game/leaderboard` - Get leaderboard data
- `/health` - Health check endpoint

## Controls

- **Arrow Keys** or **WASD**: Move player
- **Collect Treasures**: Move onto colored circles
- **Score**: Accumulate quantum treasure values

## Embedding on Your Website

To embed the game on www.mindofthecosmos.com, use an iframe:

```html
<iframe 
    src="http://your-server:5050/embed" 
    width="820" 
    height="620" 
    frameborder="0"
    style="border: 2px solid #00ff66; border-radius: 10px;">
</iframe>
```

### Full Deployment

For production deployment:

1. **Use a production WSGI server** (e.g., Gunicorn):
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5050 app:app
   ```

2. **Set up reverse proxy** (nginx example):
   ```nginx
   server {
       listen 80;
       server_name quantum-game.mindofthecosmos.com;
       
       location / {
           proxy_pass http://localhost:5050;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

3. **Enable HTTPS** with Let's Encrypt:
   ```bash
   certbot --nginx -d quantum-game.mindofthecosmos.com
   ```

## Architecture

- **Backend**: Flask server with threaded game state updates
- **Game Engine**: Headless Python game logic (no Pygame dependency)
- **Frontend**: HTML5 Canvas with vanilla JavaScript
- **Communication**: REST API with JSON state synchronization

## Features

### Treasure Rarity System
- **COMMON** (0.1 BTC base): Gray
- **RARE** (0.3 BTC base): Blue
- **EPIC** (0.7 BTC base): Purple
- **LEGENDARY** (1.5 BTC base): Gold

### Quantum Mechanics
- EPR pair creation on treasure collection
- Wormhole collapse mechanics
- Spectral gap modulation (λ₁)
- Page curve visualization

### Blockchain Integration
- Simulated chain height
- Dynamic difficulty adjustment
- Mempool transaction tracking

## Performance

- Game state updates: 10 FPS (background thread)
- Client rendering: 10 FPS
- State synchronization: 5 Hz
- Movement throttling: 150ms

## Development

To modify the game:

1. **Game Logic**: Edit `game_engine.py`
2. **API Routes**: Edit `app.py`
3. **UI/Rendering**: Edit `templates/game.html`

## Captain Travis D. Jones
Houston HQ | February 18, 2026

🏴‍☠️ **Happy Quantum Mining!** 🏴‍☠️
