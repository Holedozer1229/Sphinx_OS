# SphinxSkynet Gasless Blockchain

## 🚀 100% Free, Standalone Blockchain with NO Gas Fees!

SphinxSkynet is a completely gasless blockchain system with built-in wallet, free mining, and monetization features. Start earning TODAY with $0 investment!

---

## ✨ Features

### **Gasless Architecture**
- ✅ **NO Ethereum** - Completely standalone blockchain
- ✅ **NO Gas Fees** - Transaction fees paid in SPHINX tokens (not ETH)
- ✅ **Free Mining** - Mine directly in your browser
- ✅ **Built-in Wallet** - No MetaMask required
- ✅ **SQLite Database** - No external database costs
- ✅ **Pure PoW Consensus** - Fair and decentralized

### **Monetization Features**
- 💰 **Transaction Fees** - Earn 0.001 SPHINX per transaction
- 💰 **Premium Mining** - $5/month for 10x faster mining
- 💰 **Referral Program** - 5% commission on referrals' earnings
- 💰 **Node Hosting** - $10/month per hosted node

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  SphinxSkynet Blockchain                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Blockchain │  │    Wallet    │  │    Mining    │     │
│  │   (PoW/SPHINX)│  │  (Built-in)  │  │  (Free Tiers)│     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │Fee Collector │  │ Subscriptions│  │  Referrals   │     │
│  │  (Revenue)   │  │  ($5-20/mo)  │  │  (5% comm)   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                             │
│  ┌───────────────────────────────────────────────────┐     │
│  │            FastAPI REST API                       │     │
│  └───────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### **1. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **2. Start the API Server**

```bash
# Start locally
python -m sphinx_os.api.main

# Or use uvicorn
uvicorn sphinx_os.api.main:app --reload --host 0.0.0.0 --port 8000
```

### **3. Access the API**

```bash
# Check health
curl http://localhost:8000/health

# Get blockchain info
curl http://localhost:8000/api/blockchain/info
```

---

## 💻 API Usage Examples

### **Create a Wallet (FREE!)**

```bash
curl -X POST http://localhost:8000/api/wallet/create \
  -H "Content-Type: application/json" \
  -d '{"name": "my_wallet"}'
```

Response:
```json
{
  "success": true,
  "wallet": {
    "name": "my_wallet",
    "address": "0xSPHINX...",
    "private_key": "...",
    "mnemonic": "word1 word2 ... word12",
    "warning": "⚠️ Save your private key and mnemonic securely!"
  }
}
```

### **Check Balance**

```bash
curl http://localhost:8000/api/wallet/{address}/balance
```

### **Start Mining (FREE!)**

```bash
curl -X POST http://localhost:8000/api/mining/start \
  -H "Content-Type: application/json" \
  -d '{
    "address": "0xSPHINX...",
    "tier": "free"
  }'
```

### **Mine a Block**

```bash
curl -X POST "http://localhost:8000/api/mining/mine-block?address=0xSPHINX..."
```

### **Send Transaction**

```bash
curl -X POST http://localhost:8000/api/transaction/send \
  -H "Content-Type: application/json" \
  -d '{
    "from_address": "0xSPHINX...",
    "to_address": "0xSPHINX...",
    "amount": 10.0,
    "private_key": "your_private_key"
  }'
```

---

## 💰 Mining Tiers

| Tier | Hashrate | Cost/Month | Daily Limit |
|------|----------|------------|-------------|
| **Free** | 10 MH/s | $0 | 1,000 SPHINX |
| **Premium** | 100 MH/s | $5 | 10,000 SPHINX |
| **Pro** | 1,000 MH/s | $20 | Unlimited |

### **Upgrade to Premium**

```bash
curl -X POST http://localhost:8000/api/subscription/upgrade \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "tier": "premium"
  }'
```

---

## 🎁 Referral Program

Earn 5% commission on your referrals' mining earnings!

### **Get Your Referral Code**

```bash
curl http://localhost:8000/api/referral/{user_id}/code
```

### **Sign Up with Referral Code**

```bash
curl -X POST http://localhost:8000/api/referral/signup \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "new_user",
    "referral_code": "ABC12345"
  }'
```

---

## 📊 Revenue Dashboard (Admin)

### **Get Today's Revenue**

```bash
curl http://localhost:8000/api/admin/revenue/today
```

### **Get Total Revenue**

```bash
curl http://localhost:8000/api/admin/revenue/total
```

### **Get Comprehensive Stats**

```bash
curl http://localhost:8000/api/admin/revenue/stats
```

Response:
```json
{
  "revenue": {
    "today": {
      "transaction_fees": 1.234,
      "subscription_revenue": 25.00,
      "total_revenue": 26.234
    },
    "total": {
      "transaction_fees": 123.45,
      "subscription_revenue": 500.00,
      "total_revenue": 623.45
    }
  },
  "subscriptions": {
    "active_subscriptions": 100,
    "premium_users": 80,
    "pro_users": 20,
    "monthly_revenue": 800.00
  },
  "referrals": {
    "total_referrals": 500,
    "total_commission_paid": 50.00
  }
}
```

---

## 🚀 Deployment (FREE!)

### **Option 1: Railway (FREE API Hosting)**

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Deploy
railway up
```

### **Option 2: Fly.io (FREE VMs)**

```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Login
flyctl auth login

# Deploy
flyctl deploy
```

### **Option 3: Use the Deploy Script**

```bash
chmod +x scripts/deploy/deploy-free.sh
./scripts/deploy/deploy-free.sh
```

---

## 💡 Earnings Projections

### **Week 1**
- 100 free miners → $10/day in tx fees
- 5 premium users → $25/month
- **Total: ~$70-100**

### **Month 1**
- 1,000 free miners → $100/day in tx fees
- 50 premium users → $250/month
- 10 hosted nodes → $100/month
- **Total: ~$3,000-3,500**

### **Month 3**
- 10,000 free miners → $500/day in tx fees
- 200 premium users → $1,000/month
- 50 hosted nodes → $500/month
- **Total: ~$15,000-20,000/month**

---

## 🧪 Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov httpx

# Run tests
pytest tests/test_gasless_blockchain.py -v

# Run with coverage
pytest tests/test_gasless_blockchain.py --cov=sphinx_os --cov-report=html
```

---

## 📁 Project Structure

```
Sphinx_OS/
├── sphinx_os/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py              # FastAPI application
│   ├── blockchain/
│   │   ├── __init__.py
│   │   ├── standalone.py        # Standalone blockchain
│   │   ├── block.py             # Block class
│   │   └── transaction.py       # Transaction class
│   ├── wallet/
│   │   ├── __init__.py
│   │   └── builtin_wallet.py    # Built-in wallet system
│   ├── mining/
│   │   ├── __init__.py
│   │   └── free_miner.py        # Free mining system
│   └── revenue/
│       ├── __init__.py
│       ├── fee_collector.py     # Fee collection
│       ├── subscriptions.py     # Subscription management
│       └── referrals.py         # Referral program
├── tests/
│   └── test_gasless_blockchain.py
├── scripts/
│   └── deploy/
│       └── deploy-free.sh       # Deployment script
├── fly.toml                      # Fly.io config
├── railway.json                  # Railway config
└── requirements.txt
```

---

## 🔒 Security Notes

⚠️ **Important:** This is a demonstration implementation. For production use:

1. Implement proper ECDSA key generation and signing
2. Add authentication and authorization to admin endpoints
3. Use proper encryption for private keys (AES-256)
4. Implement rate limiting to prevent abuse
5. Add input validation and sanitization
6. Use environment variables for sensitive configuration
7. Implement proper Stripe payment processing
8. Add SSL/TLS for all communications

---

## 📝 License

SphinxOS Software License (see LICENSE file)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

---

## 📞 Support

- **Documentation:** This file
- **Issues:** https://github.com/Holedozer1229/Sphinx_OS/issues
- **Discussions:** https://github.com/Holedozer1229/Sphinx_OS/discussions

---

## 🎯 Roadmap

- [ ] Web UI (React/Next.js)
- [ ] WebAssembly mining
- [ ] P2P networking layer
- [ ] Mobile apps (iOS/Android)
- [ ] Token bridges (Ethereum, BSC, Polygon)
- [ ] DEX integration
- [ ] NFT marketplace
- [ ] Staking mechanism
- [ ] Governance system

---

**Made with ❤️ by the SphinxOS Team**

**Start earning TODAY with $0 investment! 🚀**
