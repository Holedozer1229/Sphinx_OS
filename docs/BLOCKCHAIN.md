# SphinxSkynet Blockchain Architecture

## Overview

SphinxSkynet is a production-ready blockchain with hybrid Proof-of-Work (PoW) and Proof-of-Consciousness (Φ) consensus. It supports multiple mining algorithms, merge mining, and integrates with a cross-chain bridge.

## Core Features

### 1. Multiple PoW Algorithms

- **Spectral PoW**: Quantum-resistant using Riemann zeta function properties
- **SHA-256**: Bitcoin-compatible double SHA-256
- **Ethash**: Ethereum-compatible (simplified DAG)
- **Keccak256**: Ethereum Classic compatible

### 2. Hybrid Consensus

The blockchain uses a hybrid consensus mechanism:

```
Final Difficulty = Base Difficulty × (1 - Φ_reduction)
Block Reward = Base Reward × Φ_boost

Where:
- Φ_reduction: 0-10% based on Φ score (200-1000)
- Φ_boost: 1.0x-2.0x multiplier based on Φ score
```

### 3. Block Structure

```python
{
    "index": int,              # Block height
    "timestamp": int,          # Unix timestamp
    "transactions": [...],     # List of transactions
    "previous_hash": str,      # Previous block hash
    "nonce": int,              # Mining nonce
    "merkle_root": str,        # Merkle root of transactions
    "difficulty": int,         # Mining difficulty
    "miner": str,              # Miner address
    "phi_score": float,        # Φ consciousness score (200-1000)
    "pow_algorithm": str,      # PoW algorithm used
    "merge_mining_headers": {  # Auxiliary chain headers
        "btc": str,
        "eth": str,
        "etc": str
    },
    "hash": str                # Block hash
}
```

### 4. Transaction Model (UTXO)

SphinxSkynet uses the UTXO (Unspent Transaction Output) model:

```python
{
    "txid": str,               # Transaction ID
    "inputs": [{
        "prev_txid": str,      # Previous transaction
        "output_index": int,   # Output index
        "signature": str,      # Digital signature
        "public_key": str      # Public key
    }],
    "outputs": [{
        "address": str,        # Recipient address
        "amount": float        # Amount in SPHINX
    }],
    "fee": float,              # Transaction fee
    "timestamp": int,          # Unix timestamp
    "phi_boost": float         # Φ boost (1.0-2.0)
}
```

## Blockchain Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Block Time | 10 seconds | Target time between blocks |
| Block Size | 2 MB | Maximum block size |
| Max Supply | 21 million | Total SPHINX tokens |
| Initial Reward | 50 SPHINX | Block reward at genesis |
| Halving Interval | 210,000 blocks | ~24 days |
| Difficulty Adjustment | Every 2,016 blocks | ~5.6 hours |

## Consensus Algorithm

### PoW Validation

1. Block hash must meet difficulty target: `hash < 2^(256-difficulty)`
2. Φ score must be in valid range: `200 ≤ Φ ≤ 1000`
3. PoW algorithm must be one of: `[spectral, sha256, ethash, keccak256]`

### Difficulty Adjustment

Every 2,016 blocks:

```python
expected_time = 2016 * 10  # 20,160 seconds
actual_time = last_block_time - first_block_time

new_difficulty = current_difficulty * expected_time / actual_time

# Limit adjustment to 4x change
min_time = expected_time / 4
max_time = expected_time * 4
actual_time = min(max(actual_time, min_time), max_time)
```

### Φ Consciousness Scoring

Φ scores measure the "quality" of mining:

```python
# Spectral distribution calculation
phi_score = 200 + (shannon_entropy / max_entropy) * 800

# Applied as:
mining_boost = 1.0 + ((phi_score - 200) / 800)  # 1.0x to 2.0x
difficulty_reduction = (phi_score - 200) / 800 * 0.1  # 0-10%
```

## Chain Validation

### Block Validation Rules

1. ✅ Block size ≤ 2 MB
2. ✅ Previous hash matches parent block
3. ✅ Index is sequential (parent.index + 1)
4. ✅ All transactions are valid
5. ✅ First transaction is coinbase
6. ✅ Only first transaction is coinbase
7. ✅ Hash meets difficulty target
8. ✅ Φ score is valid (200-1000)
9. ✅ PoW algorithm is supported

### Transaction Validation Rules

1. ✅ Coinbase transactions have no inputs
2. ✅ Regular transactions have ≥1 input
3. ✅ All outputs have positive amounts
4. ✅ Input UTXOs exist and are unspent
5. ✅ Sum(inputs) ≥ Sum(outputs) + fee
6. ✅ Signatures are valid

## API Usage

### Get Chain Stats

```bash
curl http://localhost:8000/api/chain/stats
```

Response:
```json
{
  "chain_length": 1234,
  "total_transactions": 5678,
  "total_supply": 123456.78,
  "max_supply": 21000000,
  "current_difficulty": 1000000,
  "latest_block_hash": "0xabcd...",
  "latest_block_height": 1233,
  "transactions_in_pool": 10,
  "target_block_time": 10
}
```

### Get Recent Blocks

```bash
curl http://localhost:8000/api/blocks?limit=10
```

### Get Specific Block

```bash
curl http://localhost:8000/api/blocks/<block_hash>
```

## Implementation Details

### Merkle Tree

Transactions are organized in a Merkle tree for efficient verification:

```python
merkle_tree = MerkleTree(transaction_hashes)
merkle_root = merkle_tree.get_root()

# Verify transaction inclusion
proof = merkle_tree.get_proof(tx_hash)
is_valid = MerkleTree.verify_proof(tx_hash, merkle_root, proof)
```

### UTXO Management

```python
# Build UTXO set from chain
utxo_set = chain_manager.get_utxo_set(blockchain.chain)

# Get balance
balance = chain_manager.get_balance(address, utxo_set)

# Validate transaction
is_valid = transaction.verify(utxo_set)
```

## Security Considerations

1. **Double-spend Prevention**: UTXO model with strict validation
2. **51% Attack Resistance**: Multiple PoW algorithms distribute hashrate
3. **Quantum Resistance**: Spectral PoW using PSPACE-complete problems
4. **Sybil Resistance**: Φ-based quality scoring
5. **Chain Reorganization**: Longest valid chain rule

## Performance

- **Block Time**: 10 seconds (configurable)
- **Transaction Throughput**: ~100 TPS (with 2MB blocks)
- **Confirmation Time**: ~60 seconds (6 blocks)
- **Storage**: ~1GB per million blocks

## Future Improvements

- [ ] State channels for instant transactions
- [ ] Sharding for scalability
- [ ] Smart contract support
- [ ] Privacy features (confidential transactions)
- [ ] DAG integration for parallel processing
