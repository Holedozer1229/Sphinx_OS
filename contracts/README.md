# SphinxOS Smart Contracts

This directory contains Clarity smart contracts for the SphinxOS economic system.

## 📄 Contracts

### pox-automation.clar

**Purpose**: Automated PoX (Proof of Transfer) delegation with BTC yield routing.

**Features:**
- ✅ Non-custodial STX delegation
- ✅ DAO-controlled pool operator
- ✅ User-initiated revocation
- ✅ On-chain audit trail
- ✅ BTC yield routing to treasury

**Key Functions:**

```clarity
;; Governance (DAO only)
(set-pool (new-pool principal)) → Changes pool operator

;; User functions
(delegate (amount uint)) → Delegate STX to current pool
(revoke-delegation) → Revoke your delegation

;; Read-only
(get-pool) → Get current pool operator
(get-total-delegated) → Get total STX delegated
(get-user-delegation (user principal)) → Get user's delegation info
(get-stats) → Get contract statistics
```

**Security Properties:**
1. **Non-Custodial**: STX never transferred, only delegated
2. **Revocable**: Users can withdraw anytime
3. **DAO-Controlled**: Pool operator managed by governance
4. **Immutable Economics**: Core rules cannot be upgraded

## 🚀 Deployment

### Prerequisites

- [Clarinet](https://github.com/hirosystems/clarinet) CLI tool
- Stacks wallet with testnet STX

### Deploy to Testnet

```bash
# Using Clarinet
clarinet integrate
clarinet deploy contracts/pox-automation.clar --testnet

# Or using Stacks CLI
stx deploy_contract pox-automation contracts/pox-automation.clar --testnet
```

### Deploy to Mainnet

```bash
clarinet deploy contracts/pox-automation.clar --mainnet
```

**⚠️ Important**: Update DAO and TREASURY addresses before mainnet deployment!

## 🧪 Testing

### Unit Tests

Create `contracts/pox-automation_test.clar`:

```clarity
;; Test delegation
(define-public (test-delegate)
  (let ((result (contract-call? .pox-automation delegate u10000000000)))
    (asserts! (is-ok result) (err u1))
    (ok true)))

;; Test revocation
(define-public (test-revoke)
  (let ((result (contract-call? .pox-automation revoke-delegation)))
    (asserts! (is-ok result) (err u2))
    (ok true)))
```

Run tests:
```bash
clarinet test
```

### Integration Tests

```bash
# Deploy to local devnet
clarinet integrate

# Test delegation
clarinet console
> (contract-call? .pox-automation delegate u10000000000)

# Check stats
> (contract-call? .pox-automation get-stats)
```

## 📊 Usage Examples

### For Users

```clarity
;; 1. Delegate 10,000 STX
(contract-call? .pox-automation delegate u10000000000)

;; 2. Check your delegation
(contract-call? .pox-automation get-user-delegation tx-sender)

;; 3. Revoke when you want
(contract-call? .pox-automation revoke-delegation)
```

### For DAO

```clarity
;; Rotate pool operator (requires DAO authority)
(contract-call? .pox-automation set-pool 'ST2NEW_POOL_ADDRESS)

;; Check pool history
(contract-call? .pox-automation get-pool-history u12345)
```

## 🔐 Security

### Audit Checklist

- [ ] DAO address verified
- [ ] Treasury address verified
- [ ] No custodial transfers
- [ ] Revocation always allowed
- [ ] Pool history immutable
- [ ] Error codes comprehensive

### Known Limitations

1. **BTC Routing**: Pool operator receives BTC off-chain and must route to treasury
   - **Mitigation**: DAO can rotate dishonest operators
   - **Transparency**: On-chain pool history for auditing

2. **Cycle Timing**: Delegation updates once per cycle
   - **Impact**: Users must wait for next cycle
   - **Standard**: Matches Stacks PoX behavior

## 🔄 Upgrade Path

The contracts are **non-upgradable** by design for security. To upgrade:

1. Deploy new version
2. DAO announces migration
3. Users revoke from old contract
4. Users delegate to new contract
5. Treasury routing updated

**Why non-upgradable?**
- Prevents DAO from modifying economics
- Prevents unauthorized access to funds
- Users trust immutable code

## 📈 Economics

### Yield Distribution

```
R = α · (S / ΣS_i) · R_total
```

Where:
- **R**: BTC reward per user
- **α**: Pool efficiency (0.92-0.98)
- **S**: User's STX delegation
- **ΣS_i**: Total STX in pool
- **R_total**: Total BTC rewards

### Treasury Split

```
R_T = R · min(0.30, 0.05 + Φ/2000)
```

- Base: 5%
- Max: 30%
- Higher Φ = Higher treasury share

See [ECONOMICS.md](../ECONOMICS.md) for full details.

## 🛠️ Development

### Add New Features

1. Create feature branch
2. Write contract code
3. Add tests
4. Deploy to testnet
5. Audit and review
6. Deploy to mainnet

### Code Style

- Use descriptive function names
- Comment all public functions
- Include error codes
- Add usage examples

## 📚 Resources

- [Clarity Language Reference](https://docs.stacks.co/clarity)
- [PoX Documentation](https://docs.stacks.co/stacks-101/proof-of-transfer)
- [Clarinet Guide](https://book.clarity-lang.org/clarinet)

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests
4. Submit a pull request

## 📝 License

SphinxOS Commercial License - See [LICENSE](../LICENSE) file for details.

---

**Built by**: SphinxOS Team  
**Author**: Travis D. Jones  
**Date**: February 2026
