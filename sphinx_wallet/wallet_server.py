"""
SphinxOS Wallet Web Server
Flask application serving MetaMask-like wallet UI
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.wallet_backend import SecureWallet

app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')
app.secret_key = os.urandom(24)
CORS(app)

wallet = SecureWallet()


@app.route('/')
def index():
    """Main wallet page"""
    if 'session_token' in session:
        return render_template('wallet.html')
    return redirect(url_for('login'))


@app.route('/login')
def login():
    """Login page"""
    return render_template('login.html')


@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """API: Authenticate user"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    result = wallet.authenticate(username, password)
    
    if result['success']:
        session['session_token'] = result['session_token']
        session['user_id'] = result['user_id']
        session['username'] = result['username']
    
    return jsonify(result)


@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    """API: Logout user"""
    session.clear()
    return jsonify({"success": True})


@app.route('/api/auth/register', methods=['POST'])
def api_register():
    """API: Register new user"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    result = wallet.create_user(username, password)
    
    if result['success']:
        # Create default wallet
        wallet_result = wallet.create_wallet(result['user_id'], "Main Wallet", password)
        result['wallet'] = wallet_result
    
    return jsonify(result)


@app.route('/api/wallet/create', methods=['POST'])
def api_create_wallet():
    """API: Create new wallet"""
    if 'user_id' not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    data = request.json
    name = data.get('name', 'New Wallet')
    password = data.get('password')
    
    result = wallet.create_wallet(session['user_id'], name, password)
    return jsonify(result)


@app.route('/api/wallet/list', methods=['GET'])
def api_list_wallets():
    """API: Get all wallets for user"""
    if 'user_id' not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    wallets = wallet.get_wallets(session['user_id'])
    return jsonify({"success": True, "wallets": wallets})


@app.route('/api/wallet/balance/<int:wallet_id>', methods=['GET'])
def api_get_balance(wallet_id):
    """API: Get wallet balance (placeholder)"""
    # In production, this would query blockchain
    return jsonify({
        "success": True,
        "balance": {
            "STX": 1250.5,
            "BTC": 0.0125,
            "USD": 625.25
        }
    })


@app.route('/api/transaction/send', methods=['POST'])
def api_send_transaction():
    """API: Send transaction"""
    if 'user_id' not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    data = request.json
    wallet_id = data.get('wallet_id')
    to_address = data.get('to_address')
    amount = data.get('amount')
    token = data.get('token', 'STX')
    
    result = wallet.add_transaction(wallet_id, to_address, amount, token)
    return jsonify(result)


if __name__ == '__main__':
    print("=" * 70)
    print("SPHINXOS WALLET SERVER")
    print("=" * 70)
    print()
    print("🌐 Starting wallet server...")
    print("📍 URL: http://localhost:5000")
    print("🔒 Secure: HTTPS recommended for production")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
