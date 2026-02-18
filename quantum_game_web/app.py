#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jones Quantum Gravity Web App
Flask server for ER=EPR Quantum-Pirate Roguelite game
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
import time
import threading
from game_engine import QuantumGameState

app = Flask(__name__)
CORS(app)

# Global game state
game_state = None
game_lock = threading.Lock()

def init_game():
    """Initialize the game state"""
    global game_state
    with game_lock:
        game_state = QuantumGameState()
        print("🚀 Jones Quantum Gravity Game initialized!")

@app.route('/')
def index():
    """Main game page"""
    return render_template('game.html')

@app.route('/embed')
def embed():
    """Embeddable game page (minimal UI for iframe)"""
    return render_template('game_embed.html')

@app.route('/api/game/state')
def get_game_state():
    """Get current game state"""
    with game_lock:
        if game_state is None:
            return jsonify({"error": "Game not initialized"}), 500
        return jsonify(game_state.to_dict())

@app.route('/api/game/move', methods=['POST'])
def move_player():
    """Move player"""
    data = request.json
    dx = int(data.get('dx', 0))
    dy = int(data.get('dy', 0))
    
    with game_lock:
        if game_state is None:
            return jsonify({"error": "Game not initialized"}), 500
        
        game_state.move_player(dx, dy)
        collected = game_state.check_treasure_collection()
        
        return jsonify({
            "success": True,
            "collected": collected,
            "player": {
                "x": game_state.player_x,
                "y": game_state.player_y
            },
            "score": game_state.score
        })

@app.route('/api/game/reset', methods=['POST'])
def reset_game():
    """Reset the game"""
    global game_state
    with game_lock:
        game_state = QuantumGameState()
        return jsonify({"success": True, "message": "Game reset"})

@app.route('/api/game/leaderboard')
def get_leaderboard():
    """Get leaderboard"""
    try:
        with open('jqg_leaderboard.json', 'r') as f:
            data = json.load(f)
            return jsonify(data)
    except FileNotFoundError:
        return jsonify({"top27": []})

def update_loop():
    """Background thread to update game state"""
    global game_state
    while True:
        time.sleep(0.1)  # 10 FPS update
        with game_lock:
            if game_state:
                game_state.update(0.1)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "game_initialized": game_state is not None,
        "timestamp": time.time()
    })

if __name__ == '__main__':
    print("=" * 70)
    print("JONES QUANTUM GRAVITY — ER=EPR ROGUELITE WEB SERVER")
    print("=" * 70)
    print()
    print("🌐 Starting web server...")
    print("📍 URL: http://localhost:5050")
    print("🎮 Game URL: http://localhost:5050")
    print("🔗 Embed URL: http://localhost:5050/embed")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 70)
    
    # Initialize game
    init_game()
    
    # Start background update thread
    update_thread = threading.Thread(target=update_loop, daemon=True)
    update_thread.start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5050, debug=False, threaded=True)
