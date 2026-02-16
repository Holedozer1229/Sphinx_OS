# SphinxSkynet Cross-Chain Bridge

## Overview

The SphinxSkynet Bridge enables trustless cross-chain transfers between SphinxSkynet and 7+ major blockchains using lock/mint and burn/release mechanisms with 5-of-9 multi-signature validation.

## Supported Chains

| Chain | Symbol | Status | Fee | Min Transfer |
|-------|--------|--------|-----|--------------|
| Bitcoin | BTC | ✅ Active | 0.1% | 0.01 BTC |
| Ethereum | ETH | ✅ Active | 0.1% | 0.1 ETH |
| Ethereum Classic | ETC | ✅ Active | 0.1% | 0.1 ETC |
| Polygon | MATIC | ✅ Active | 0.05% | 1 MATIC |
| Avalanche | AVAX | ✅ Active | 0.05% | 0.5 AVAX |
| BNB Chain | BNB | ✅ Active | 0.05% | 0.1 BNB |
| Stacks | STX | ✅ Active | 0.1% | 10 STX |

## How It Works

### Lock & Mint

1. **User locks tokens** on source chain
2. **Guardians validate** transaction (5-of-9 signatures)
3. **Wrapped tokens minted** on SphinxSkynet
4. **User receives** wrapped SPHINX tokens

### Burn & Release

1. **User burns** wrapped tokens on SphinxSkynet
2. **Guardians validate** burn transaction
3. **Original tokens released** on source chain
4. **User receives** original tokens

## Quick Start

### 1. Deploy Bridge

```bash
./scripts/deploy/deploy-bridge.sh
```

### 2. Lock Tokens (Bridge In)

```bash
curl -X POST http://localhost:8001/api/bridge/lock \
  -H "Content-Type: application/json" \
  -d '{
    "source_chain": "eth",
    "amount": 10.0,
    "sender": "0xYourETHAddress",
    "recipient": "YourSPHINXAddress"
  }'
```

Response:
```json
{
  "status": "locked",
  "tx_hash": "0xabc123...",
  "source_chain": "eth",
  "amount": 9.99,
  "fee": 0.01
}
```

### 3. Burn Tokens (Bridge Out)

```bash
curl -X POST http://localhost:8001/api/bridge/burn \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 10.0,
    "sender": "YourSPHINXAddress",
    "destination_chain": "btc",
    "recipient": "YourBTCAddress"
  }'
```

## Bridge API

### Lock Tokens

**Endpoint:** `POST /api/bridge/lock`

**Request:**
```json
{
  "source_chain": "btc|eth|etc|matic|avax|bnb|stx",
  "amount": float,
  "sender": "address_on_source_chain",
  "recipient": "sphinx_address"
}
```

**Response:**
```json
{
  "status": "locked",
  "tx_hash": "...",
  "source_chain": "...",
  "amount": float,
  "fee": float
}
```

### Mint Tokens (Guardian Only)

**Endpoint:** `POST /api/bridge/mint`

**Request:**
```json
{
  "tx_hash": "bridge_tx_hash",
  "recipient": "sphinx_address",
  "signatures": ["guardian_sig_1", "guardian_sig_2", ...]
}
```

### Burn Tokens

**Endpoint:** `POST /api/bridge/burn`

**Request:**
```json
{
  "amount": float,
  "sender": "sphinx_address",
  "destination_chain": "btc|eth|etc|matic|avax|bnb|stx",
  "recipient": "address_on_destination_chain"
}
```

### Release Tokens (Guardian Only)

**Endpoint:** `POST /api/bridge/release`

**Request:**
```json
{
  "tx_hash": "bridge_tx_hash",
  "recipient": "destination_address",
  "signatures": ["guardian_sig_1", "guardian_sig_2", ...]
}
```

### Get Transaction Status

**Endpoint:** `GET /api/bridge/status/{tx_hash}`

**Response:**
```json
{
  "tx_hash": "...",
  "source_chain": "eth",
  "destination_chain": "sphinx",
  "amount": 9.99,
  "sender": "0x...",
  "recipient": "SPHINX...",
  "status": "minted",
  "created_at": 1704067200,
  "updated_at": 1704067260,
  "signatures": ["guardian_1", ...]
}
```

### Get Supported Chains

**Endpoint:** `GET /api/bridge/supported-chains`

**Response:**
```json
{
  "chains": [
    {
      "name": "Bitcoin",
      "symbol": "BTC",
      "chain_id": "btc"
    },
    ...
  ],
  "count": 7
}
```

### Get Bridge Fees

**Endpoint:** `GET /api/bridge/fees`

**Response:**
```json
{
  "bridge_fee_percent": 0.1,
  "example": {
    "amount": 100,
    "fee": 0.1,
    "net_amount": 99.9
  }
}
```

### Get Balance

**Endpoint:** `GET /api/bridge/balance/{address}?chain=eth`

**Response:**
```json
{
  "address": "...",
  "chain": "eth",
  "locked_balance": 100.0
}
```

### Get Bridge Stats

**Endpoint:** `GET /api/bridge/stats`

**Response:**
```json
{
  "bridge": {
    "total_volume": 1234567.89,
    "total_fees": 1234.56,
    "transactions_count": 5678,
    "supported_chains": 7,
    "bridge_fee_percent": 0.1,
    "multi_sig_threshold": "5-of-9"
  },
  "relayer": {
    "is_running": true,
    "relayed_count": 100,
    "failed_count": 2,
    "pending_mints": 3,
    "pending_releases": 1,
    "uptime_seconds": 3600
  }
}
```

## Security

### Multi-Signature Validation

Requires 5-of-9 guardian signatures for:
- Minting wrapped tokens
- Releasing locked tokens
- Emergency pause

### ZK-Proof Verification

Bridge transactions include zero-knowledge proofs to ensure:
- Transaction validity
- Amount correctness
- No double-spending

### Time Locks

Large transfers (>1000 tokens) have 1-hour time lock:
- Prevents flash attacks
- Allows dispute resolution
- Guardian review time

### Emergency Controls

Guardians can pause the bridge if:
- Security vulnerability detected
- Suspicious activity
- System maintenance

## Bridge Guardians

### Guardian Roles

- **Validate transactions**: Verify cross-chain transfers
- **Sign operations**: Provide multi-sig signatures
- **Monitor security**: Watch for attacks
- **Emergency response**: Pause bridge if needed

### Current Guardians

1. Guardian 1 - Primary validator
2. Guardian 2 - Secondary validator
3. Guardian 3 - Third validator
4. Guardian 4 - Backup validator
5. Guardian 5 - Backup validator
6. Guardian 6 - Emergency responder
7. Guardian 7 - Emergency responder
8. Guardian 8 - Monitoring specialist
9. Guardian 9 - Security analyst

### Becoming a Guardian

Requirements:
- Proven track record in blockchain
- Technical expertise
- 24/7 availability
- Security clearance
- Stake requirement: 100,000 SPHINX

## Smart Contracts

### Ethereum (and EVM chains)

**SphinxBridge.sol** - Main bridge contract

Key functions:
```solidity
function lockTokens(string destinationChain, address recipient) external payable
function mintTokens(bytes32 txHash) external onlyGuardian
function burnTokens(uint256 amount, string destinationChain, address recipient) external
function releaseTokens(bytes32 txHash) external onlyGuardian
```

### Bitcoin

Uses multi-sig P2SH addresses:
```
2-of-3 multi-sig for small amounts (<10 BTC)
5-of-9 multi-sig for large amounts (≥10 BTC)
```

### Stacks

**bridge.clar** - Clarity smart contract

## Fees

| Transfer Amount | Fee |
|----------------|-----|
| < 100 tokens | 0.1% |
| 100-1,000 | 0.08% |
| 1,000-10,000 | 0.05% |
| > 10,000 | 0.03% |

## Limits

### Transfer Limits

| Chain | Min | Max (24h) |
|-------|-----|-----------|
| BTC | 0.01 | 100 |
| ETH | 0.1 | 1,000 |
| ETC | 0.1 | 1,000 |
| MATIC | 1.0 | 10,000 |
| AVAX | 0.5 | 5,000 |
| BNB | 0.1 | 5,000 |
| STX | 10.0 | 100,000 |

### Rate Limits

- **Transactions per address**: 10 per hour
- **Total bridge volume**: 1M tokens per day
- **Guardian signatures**: 5 required, 10 second timeout

## Troubleshooting

### Transaction Stuck

1. Check transaction status: `GET /api/bridge/status/{tx_hash}`
2. Verify guardian signatures collected
3. Wait for confirmation period
4. Contact support if >1 hour

### Insufficient Signatures

1. Guardian availability issue
2. Wait for backup guardians
3. Check bridge relayer status
4. Emergency: Contact guardians directly

### Failed Transfer

1. Check source chain confirmation
2. Verify amount meets minimum
3. Confirm address format correct
4. Review transaction logs

## Best Practices

### For Users

✅ **Do:**
- Double-check addresses before bridging
- Start with small test amounts
- Keep transaction receipts
- Monitor bridge status
- Use supported wallets

❌ **Don't:**
- Bridge to exchange addresses
- Skip address verification
- Ignore minimum amounts
- Use unsupported tokens

### For Integrators

✅ **Do:**
- Implement proper error handling
- Monitor bridge health endpoint
- Cache supported chains
- Validate addresses before submission
- Test on testnet first

❌ **Don't:**
- Skip transaction confirmations
- Ignore rate limits
- Hard-code chain parameters
- Bypass validation

## Monitoring

### Bridge Health

```bash
curl http://localhost:8001/api/bridge/health
```

Response:
```json
{
  "status": "healthy",
  "relayer_running": true,
  "pending_operations": 4
}
```

## Support

- **Bridge Issues**: bridge-support@sphinxskynet.io
- **Guardian Contact**: guardians@sphinxskynet.io
- **Emergency**: emergency@sphinxskynet.io

## License

SphinxOS Software License
