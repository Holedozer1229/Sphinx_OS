# SphinxOS Secure Admin Wallet - Setup Guide

## 🔐 Overview

SphinxOS Wallet is a secure, MetaMask-like cryptocurrency wallet with enterprise-grade security features including:

- ✅ Password-based encryption (PBKDF2 with 100,000 iterations)
- ✅ Secure session management  
- ✅ Multi-account support
- ✅ Transaction history
- ✅ MetaMask-inspired UI
- ✅ Local-first security (private keys never leave your device)

---

## 📋 Prerequisites

- Python 3.8+
- pip (Python package manager)
- Web browser (Chrome, Firefox, Safari, or Edge)

---

## 🚀 Installation

### Step 1: Install Dependencies

```bash
cd Sphinx_OS
pip install flask flask-cors
```

### Step 2: Verify Installation

```bash
ls sphinx_wallet/
# Should show: backend/ static/ templates/ wallet_server.py
```

---

## 👤 Creating Admin Credentials

### Method 1: Interactive Setup (Recommended)

Run the setup script:

```bash
cd sphinx_wallet/backend
python wallet_backend.py create-admin
```

You will be prompted:

```
======================================================================
SPHINXOS SECURE WALLET - ADMIN SETUP
======================================================================

Creating admin user...
Enter admin username: admin
Enter admin password: ********
Confirm password: ********

✅ Admin user 'admin' created successfully!
User ID: 1
✅ Default wallet created: ST1PQHQKV0RJXZFY1DGX8MNSNYVE3VGZJSRTPGZGM
```

**⚠️ IMPORTANT**: 
- Use a strong password (minimum 8 characters)
- Mix uppercase, lowercase, numbers, and symbols
- Never share your password
- Write it down securely offline

### Method 2: Python Script

Create a setup script `create_admin.py`:

```python
from sphinx_wallet.backend.wallet_backend import SecureWallet

wallet = SecureWallet()

# Create admin user
result = wallet.create_user("admin", "YourSecurePassword123!")

if result["success"]:
    print(f"✅ Admin created: User ID {result['user_id']}")
    
    # Create default wallet
    wallet_result = wallet.create_wallet(
        result['user_id'], 
        "Main Wallet", 
        "YourSecurePassword123!"
    )
    
    if wallet_result["success"]:
        print(f"✅ Wallet created: {wallet_result['address']}")
else:
    print(f"❌ Error: {result['error']}")
```

Run it:

```bash
python create_admin.py
```

---

## 🌐 Starting the Wallet Server

### Start the Server

```bash
cd sphinx_wallet
python wallet_server.py
```

Output:

```
======================================================================
SPHINXOS WALLET SERVER
======================================================================

🌐 Starting wallet server...
📍 URL: http://localhost:5000
🔒 Secure: HTTPS recommended for production

Press Ctrl+C to stop
======================================================================
 * Running on http://0.0.0.0:5000/
```

### Access the Wallet

Open your browser and navigate to:

```
http://localhost:5000
```

You will see the login page:

![Login Page Preview](login-preview.png)

---

## 🔑 First Login

### Step 1: Navigate to Login Page

The URL will redirect you to: `http://localhost:5000/login`

### Step 2: Enter Credentials

- **Username**: Enter the admin username you created (e.g., `admin`)
- **Password**: Enter your secure password
- **Remember me**: Check this to stay logged in for 24 hours

### Step 3: Click "Unlock Wallet"

If credentials are correct, you'll be redirected to the main wallet dashboard.

---

## 💼 Using the Wallet

### Main Dashboard

After login, you'll see:

1. **Balance Card** (top)
   - Current balance in STX
   - USD equivalent
   - Action buttons: Send, Receive, Swap

2. **Assets Tab**
   - List of all tokens (STX, BTC, etc.)
   - Current balances and values

3. **Activity Tab**
   - Transaction history
   - Pending transactions
   - Completed transfers

### Sending Transactions

1. Click **"Send"** button
2. Select asset (STX, BTC, etc.)
3. Enter recipient address
4. Enter amount
5. Select gas fee (Slow, Medium, Fast)
6. Click **"Send"**

### Receiving Assets

1. Click **"Receive"** button
2. Your wallet address is displayed
3. Click **"Copy"** button to copy address
4. Share with sender

### Creating Additional Wallets

```python
from sphinx_wallet.backend.wallet_backend import SecureWallet

wallet = SecureWallet()

# Assuming user_id = 1
result = wallet.create_wallet(1, "Trading Wallet", "YourPassword123!")

print(f"New wallet: {result['address']}")
```

---

## 🔒 Security Best Practices

### Password Security

✅ **DO:**
- Use minimum 12 characters
- Mix uppercase, lowercase, numbers, symbols
- Use unique password (not reused elsewhere)
- Store in password manager
- Enable 2FA when available

❌ **DON'T:**
- Use common words or patterns
- Share with anyone
- Write in plain text files
- Use same password for multiple accounts
- Store in browser autofill

### Session Security

- Sessions expire after 24 hours
- Always logout on shared computers
- Clear browser cache after use
- Use private/incognito mode on public computers

### Private Key Security

- Private keys are encrypted with your password
- Never share encrypted keys
- Backup wallet database separately
- Use hardware wallet for large amounts

### Database Backup

```bash
# Backup wallet database
cp sphinx_wallet/wallet.db sphinx_wallet/wallet_backup_$(date +%Y%m%d).db

# Verify backup
ls -lh sphinx_wallet/wallet_backup_*.db
```

---

## 🛠️ Advanced Configuration

### Change Database Location

Edit `wallet_backend.py`:

```python
wallet = SecureWallet(db_path="/secure/path/wallet.db")
```

### Enable HTTPS (Production)

For production, use HTTPS:

```python
# In wallet_server.py
if __name__ == '__main__':
    app.run(
        ssl_context=('cert.pem', 'key.pem'),
        host='0.0.0.0',
        port=443
    )
```

Generate SSL certificates:

```bash
openssl req -x509 -newkey rsa:4096 -nodes \
  -out cert.pem -keyout key.pem -days 365
```

### Custom Port

```python
app.run(host='0.0.0.0', port=8080)
```

### Production Deployment

For production, use gunicorn:

```bash
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:5000 wallet_server:app
```

---

## 🔍 Troubleshooting

### Issue: "Database locked"

**Solution**: Only one process can access SQLite at a time

```bash
# Stop all wallet servers
pkill -f wallet_server.py

# Restart
python wallet_server.py
```

### Issue: "Invalid credentials"

**Solution**: 
1. Verify username is correct
2. Check password (case-sensitive)
3. Create new user if forgotten

### Issue: "Port already in use"

**Solution**: Change port or kill existing process

```bash
# Find process
lsof -i :5000

# Kill it
kill -9 <PID>
```

### Issue: "Module not found"

**Solution**: Install dependencies

```bash
pip install flask flask-cors
```

---

## 📊 Database Schema

### Users Table

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE,
    password_hash TEXT,
    salt TEXT,
    created_at TIMESTAMP
);
```

### Wallets Table

```sql
CREATE TABLE wallets (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    name TEXT,
    address TEXT,
    encrypted_key TEXT,
    key_salt TEXT,
    chain TEXT,
    balance REAL,
    created_at TIMESTAMP
);
```

### Sessions Table

```sql
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    session_token TEXT UNIQUE,
    expires_at TIMESTAMP,
    created_at TIMESTAMP
);
```

### Transactions Table

```sql
CREATE TABLE transactions (
    id INTEGER PRIMARY KEY,
    wallet_id INTEGER,
    tx_hash TEXT,
    from_address TEXT,
    to_address TEXT,
    amount REAL,
    token TEXT,
    status TEXT,
    created_at TIMESTAMP
);
```

---

## 🔐 API Endpoints

### Authentication

```
POST /api/auth/login
Body: {"username": "admin", "password": "..."}
Response: {"success": true, "session_token": "..."}

POST /api/auth/register
Body: {"username": "newuser", "password": "..."}
Response: {"success": true, "user_id": 2}

POST /api/auth/logout
Response: {"success": true}
```

### Wallet Management

```
GET /api/wallet/list
Response: {"success": true, "wallets": [...]}

POST /api/wallet/create
Body: {"name": "New Wallet", "password": "..."}
Response: {"success": true, "address": "ST..."}

GET /api/wallet/balance/{wallet_id}
Response: {"success": true, "balance": {...}}
```

### Transactions

```
POST /api/transaction/send
Body: {
  "wallet_id": 1,
  "to_address": "ST...",
  "amount": 10.0,
  "token": "STX"
}
Response: {"success": true, "tx_hash": "..."}
```

---

## 📱 Mobile Access

### Option 1: Local Network

On same WiFi network:

```bash
# Find your IP
ifconfig | grep "inet " | grep -v 127.0.0.1

# Start server
python wallet_server.py

# Access from phone
http://192.168.1.x:5000
```

### Option 2: Tunnel (Development Only)

Using ngrok:

```bash
ngrok http 5000
# Access via: https://xyz123.ngrok.io
```

**⚠️ WARNING**: Never use tunnels for production!

---

## 🎨 Customization

### Change Theme Colors

Edit `static/css/wallet.css`:

```css
:root {
    --primary-color: #6366f1;  /* Change this */
    --primary-hover: #4f46e5;
}
```

### Add Custom Logo

Replace logo icon in templates:

```html
<div class="logo-icon">🌌</div>
<!-- Change to your logo -->
<div class="logo-icon">
    <img src="/static/logo.png" alt="Logo">
</div>
```

---

## 🔄 Backup & Recovery

### Backup Wallet

```bash
# Full backup
tar -czf wallet_backup.tar.gz sphinx_wallet/wallet.db

# Encrypted backup
gpg -c sphinx_wallet/wallet.db
```

### Restore Wallet

```bash
# From tar
tar -xzf wallet_backup.tar.gz

# From gpg
gpg -d wallet.db.gpg > sphinx_wallet/wallet.db
```

---

## 📞 Support

For issues or questions:

1. Check troubleshooting section
2. Review error logs in terminal
3. Open GitHub issue: https://github.com/Holedozer1229/Sphinx_OS/issues
4. Email: holedozer@iCloud.com

---

## 📄 License

SphinxOS Commercial License - See LICENSE file

---

## ✅ Quick Start Checklist

- [ ] Install Python 3.8+
- [ ] Install Flask dependencies
- [ ] Create admin credentials
- [ ] Start wallet server
- [ ] Access http://localhost:5000
- [ ] Login with admin credentials
- [ ] Create backup of wallet.db
- [ ] Test sending/receiving
- [ ] Set up HTTPS for production
- [ ] Configure firewall rules
- [ ] Enable regular backups

---

**Built by**: SphinxOS Team  
**Author**: Travis D. Jones  
**Date**: February 2026

🔐 **Secure your quantum future** 🔐
