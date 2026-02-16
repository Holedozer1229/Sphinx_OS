# SphinxSkynet Mining Guide

## Overview

SphinxSkynet supports multiple Proof-of-Work algorithms, merge mining, and Φ-boosted rewards. This guide covers setup, operation, and optimization.

## Mining Algorithms

### 1. Spectral PoW (Recommended)

Quantum-resistant algorithm using Riemann zeta function properties.

**Advantages:**
- Highest Φ boost potential (up to 2.0x rewards)
- PSPACE-complete security
- CPU-friendly

**Usage:**
```bash
./scripts/mining/start-mining.sh YOUR_ADDRESS spectral
```

### 2. SHA-256

Bitcoin-compatible double SHA-256.

**Advantages:**
- Hardware support (ASICs available)
- Battle-tested security
- Compatible with BTC mining infrastructure

**Usage:**
```bash
./scripts/mining/start-mining.sh YOUR_ADDRESS sha256
```

### 3. Ethash

Ethereum-compatible (simplified implementation).

**Advantages:**
- GPU-friendly
- Memory-hard
- Resistant to ASICs

**Usage:**
```bash
./scripts/mining/start-mining.sh YOUR_ADDRESS ethash
```

### 4. Keccak256

Ethereum Classic compatible.

**Advantages:**
- Fast on modern CPUs
- Lower memory requirements
- ETC infrastructure compatible

**Usage:**
```bash
./scripts/mining/start-mining.sh YOUR_ADDRESS keccak256
```

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn numpy scipy
```

### 2. Deploy Blockchain

```bash
./scripts/deploy/deploy-blockchain.sh
```

### 3. Start Mining

Single algorithm:
```bash
./scripts/mining/start-mining.sh YOUR_SPHINX_ADDRESS spectral
```

### 4. Monitor Status

```bash
curl http://localhost:8000/api/mining/status
```

## Configuration

Edit `config/mining.yaml`:

```yaml
mining:
  enabled: true
  algorithms:
    - spectral
    - sha256
    - ethash
    - keccak256
  default_algorithm: spectral
  threads: 8
  gpu_enabled: false
  auto_start: true
  payout_address: "YOUR_ADDRESS_HERE"
```

## Merge Mining

### What is Merge Mining?

Merge mining allows simultaneous mining on multiple blockchains, earning rewards from SphinxSkynet plus auxiliary chains (BTC, ETH, ETC).

### Enable Merge Mining

```bash
./scripts/mining/start-merge-mining.sh YOUR_ADDRESS --chains btc,eth,etc
```

### Reward Distribution

- **Main Chain (SphinxSkynet)**: 70% of rewards
- **Auxiliary Chains**: 30% of rewards (split among enabled chains)
- **Bonus**: +10% per auxiliary chain enabled

Example with 3 chains:
```
Base Reward: 50 SPHINX
Φ Boost: 1.5x = 75 SPHINX
Merge Bonus: 1.3x = 97.5 SPHINX
```

### Supported Auxiliary Chains

| Chain | Algorithm | Status |
|-------|-----------|--------|
| Bitcoin (BTC) | SHA-256 | ✅ Active |
| Ethereum (ETH) | Ethash | ✅ Active |
| Ethereum Classic (ETC) | Keccak256 | ✅ Active |

## Mining API

### Start Mining

```bash
curl -X POST http://localhost:8000/api/mining/start \
  -H "Content-Type: application/json" \
  -d '{
    "miner_address": "YOUR_ADDRESS",
    "algorithm": "spectral",
    "num_threads": 4
  }'
```

### Stop Mining

```bash
curl -X POST http://localhost:8000/api/mining/stop
```

### Get Mining Status

```bash
curl http://localhost:8000/api/mining/status
```

Response:
```json
{
  "is_mining": true,
  "algorithm": "spectral",
  "blocks_mined": 42,
  "total_rewards": 2100.5,
  "hashrate": 1234.56,
  "average_phi_score": 750.3,
  "uptime_seconds": 3600,
  "miner_address": "YOUR_ADDRESS",
  "current_block_height": 1234
}
```

### Get Hashrate

```bash
curl http://localhost:8000/api/mining/hashrate
```

### Get Rewards

```bash
curl http://localhost:8000/api/mining/rewards
```

### Enable Merge Mining

```bash
curl -X POST http://localhost:8000/api/mining/merge/enable \
  -H "Content-Type: application/json" \
  -d '{"chains": ["btc", "eth", "etc"]}'
```

## Φ Score Optimization

### Understanding Φ Scores

Φ (Phi) scores measure mining "consciousness" quality (200-1000):

- **200-400**: Low quality (1.0x-1.25x rewards)
- **400-600**: Average (1.25x-1.5x rewards)
- **600-800**: Good (1.5x-1.75x rewards)
- **800-1000**: Excellent (1.75x-2.0x rewards)

### Improving Φ Scores

1. **Use Spectral PoW**: Inherently produces higher Φ scores
2. **Stable Mining**: Consistent uptime improves scores
3. **Quality Over Quantity**: Focus on valid blocks, not just attempts
4. **Merge Mining**: Coordinated mining boosts Φ

### Monitoring Φ Scores

```bash
# In mining status
curl http://localhost:8000/api/mining/status | grep phi_score
```

## Mining Profitability

### Calculate Profitability

```python
# Block reward
base_reward = 50 SPHINX
phi_boost = 1.5  # Based on Φ score
merge_bonus = 1.3  # 3 auxiliary chains

total_reward = base_reward * phi_boost * merge_bonus
# = 50 * 1.5 * 1.3 = 97.5 SPHINX per block

# Blocks per day
blocks_per_day = (24 * 60 * 60) / 10  # 8,640 blocks
your_hashrate_percent = your_hashrate / network_hashrate

daily_blocks = blocks_per_day * your_hashrate_percent
daily_rewards = daily_blocks * total_reward
```

### Factors Affecting Profitability

1. **Hashrate**: Higher = more blocks found
2. **Network Difficulty**: Lower = easier mining
3. **Φ Score**: Higher = better rewards (up to 2x)
4. **Merge Mining**: +10% per auxiliary chain
5. **Hardware Costs**: Power consumption
6. **SPHINX Price**: Market value

## Hardware Requirements

### Minimum (Solo Mining)

- **CPU**: 4 cores, 2.0 GHz
- **RAM**: 8 GB
- **Storage**: 50 GB SSD
- **Network**: 10 Mbps

### Recommended (Pool Mining)

- **CPU**: 8+ cores, 3.0 GHz
- **RAM**: 16 GB
- **Storage**: 100 GB NVMe
- **Network**: 100 Mbps
- **GPU** (optional): NVIDIA RTX 3060+ or AMD RX 6800+

### Enterprise (Mining Farm)

- **CPU**: 16+ cores, 4.0 GHz
- **RAM**: 32+ GB
- **Storage**: 500 GB NVMe RAID
- **Network**: 1 Gbps+
- **GPU**: Multiple high-end GPUs
- **Cooling**: Industrial cooling system

## Troubleshooting

### Mining Not Starting

```bash
# Check if blockchain is running
curl http://localhost:8000/api/chain/stats

# Check logs
tail -f logs/blockchain-node.log

# Verify Python dependencies
pip list | grep -E "fastapi|uvicorn|numpy|scipy"
```

### Low Hashrate

1. **Check CPU usage**: Should be near 100%
2. **Increase threads**: Edit `config/mining.yaml`
3. **Enable GPU**: Set `gpu_enabled: true`
4. **Optimize algorithm**: Try different algorithms

### No Rewards

1. **Verify address**: Check `payout_address` in config
2. **Check network difficulty**: May need more hashrate
3. **Confirm blocks mined**: Check `/api/mining/status`
4. **Validate chain sync**: Ensure blockchain is synced

### High Reject Rate

1. **Network latency**: Check connection
2. **Out of sync**: Restart blockchain node
3. **Invalid shares**: Review mining configuration

## Best Practices

### Solo Mining

✅ **Do:**
- Use latest mining software
- Monitor Φ scores
- Keep node synced
- Backup wallet regularly
- Use stable internet connection

❌ **Don't:**
- Mine on unstable hardware
- Ignore software updates
- Skip configuration optimization
- Forget to monitor logs

### Pool Mining

✅ **Do:**
- Choose reputable pools
- Diversify across pools
- Monitor pool fees
- Verify payouts
- Keep mining software updated

❌ **Don't:**
- Use unknown pools
- Ignore pool statistics
- Skip payout verification
- Trust without verification

## Advanced Topics

### GPU Mining (Coming Soon)

CUDA/OpenCL support for algorithms:
- Spectral PoW (experimental)
- Ethash (full support)
- Keccak256 (optimized)

### ASIC Resistance

SphinxSkynet uses multiple algorithms to prevent ASIC dominance:
- Spectral PoW: PSPACE-complete (no efficient shortcuts)
- Algorithm switching prevents single ASIC advantage

### Mining Pools

Pool protocol support (Stratum):
```
stratum+tcp://pool.sphinxskynet.io:3333
```

## Support

- **Documentation**: https://docs.sphinxskynet.io
- **Discord**: https://discord.gg/sphinxskynet
- **Forum**: https://forum.sphinxskynet.io
- **GitHub**: https://github.com/Holedozer1229/Sphinx_OS

## License

SphinxOS Software License
