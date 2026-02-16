# Mainnet Deployment Procedures

## Overview

This document provides step-by-step procedures for deploying Sphinx_OS to mainnet production environment.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Pre-Deployment Validation](#pre-deployment-validation)
3. [Smart Contract Deployment](#smart-contract-deployment)
4. [Infrastructure Deployment](#infrastructure-deployment)
5. [Configuration](#configuration)
6. [Verification](#verification)
7. [Rollback Procedures](#rollback-procedures)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools

```bash
# Install required tools
pip install web3 eth-account pyyaml
npm install -g @openzeppelin/hardhat-upgrades
kubectl version
helm version
docker version
```

### Required Credentials

- Deployer private key (secure, funded with ETH)
- RPC endpoints for Ethereum, Polygon, Arbitrum
- Kubernetes cluster access
- Docker registry credentials
- Secrets management access

### Environment Variables

```bash
export DEPLOYER_PRIVATE_KEY="0x..."
export ETH_RPC_URL="https://mainnet.infura.io/v3/YOUR_KEY"
export POLYGON_RPC_URL="https://polygon-rpc.com"
export ARBITRUM_RPC_URL="https://arb1.arbitrum.io/rpc"
export JWT_SECRET="$(openssl rand -hex 32)"
export DATABASE_URL="postgresql://..."
export REDIS_URL="redis://..."
export SENTRY_DSN="https://..."
export ENVIRONMENT="mainnet"
```

---

## Pre-Deployment Validation

### 1. Review Checklist

Ensure all items in `MAINNET_CHECKLIST.md` are completed.

### 2. Test on Testnet

```bash
# Deploy to testnet first
export ENVIRONMENT="testnet"
python scripts/deploy_mainnet.py --network polygon
./scripts/deploy_infrastructure.sh
```

### 3. Verify Contracts

```bash
# Check contract compilation
cd contracts
npm run compile

# Run tests
npm run test

# Check gas estimates
npm run analyze-gas
```

### 4. Database Backup

```bash
# Backup current state (if upgrading)
pg_dump $DATABASE_URL > backup-$(date +%Y%m%d-%H%M%S).sql
```

---

## Smart Contract Deployment

### Step 1: Prepare Constructor Arguments

Edit `scripts/deploy_mainnet.py` and update constructor arguments:

```python
contracts = [
    {
        "name": "SphinxYieldAggregator",
        "args": [
            "0x...",  # Treasury address (multi-sig)
            "0x..."   # ZK Verifier address
        ]
    },
    {
        "name": "SpaceFlightNFT",
        "args": [
            "0x...",  # Sphinx Token address
            "0x...",  # Treasury address (multi-sig)
            "0x..."   # OpenSea Proxy address
        ]
    }
]
```

### Step 2: Deploy to Ethereum

```bash
# Set environment
export ENVIRONMENT="mainnet"

# Deploy to Ethereum mainnet
python scripts/deploy_mainnet.py --network ethereum

# Verify deployment
# Check deployments/mainnet.json for addresses
cat deployments/mainnet.json
```

### Step 3: Deploy to Layer 2s

```bash
# Deploy to Polygon
python scripts/deploy_mainnet.py --network polygon

# Deploy to Arbitrum
python scripts/deploy_mainnet.py --network arbitrum

# Or deploy to all networks
python scripts/deploy_mainnet.py --network all
```

### Step 4: Verify on Explorers

```bash
# Verify on Etherscan
npx hardhat verify --network mainnet CONTRACT_ADDRESS "CONSTRUCTOR_ARG1" "CONSTRUCTOR_ARG2"

# Verify on PolygonScan
npx hardhat verify --network polygon CONTRACT_ADDRESS "CONSTRUCTOR_ARG1" "CONSTRUCTOR_ARG2"

# Verify on Arbiscan
npx hardhat verify --network arbitrum CONTRACT_ADDRESS "CONSTRUCTOR_ARG1" "CONSTRUCTOR_ARG2"
```

### Step 5: Configure Contracts

```bash
# Grant roles to operators
cast send $CONTRACT_ADDRESS "grantRole(bytes32,address)" \
  $(cast keccak "OPERATOR_ROLE") \
  $OPERATOR_ADDRESS \
  --private-key $ADMIN_PRIVATE_KEY

# Add supported tokens
cast send $YIELD_AGGREGATOR "addToken(address)" \
  $TOKEN_ADDRESS \
  --private-key $ADMIN_PRIVATE_KEY

# Add strategies
cast send $YIELD_AGGREGATOR "addStrategy(string,address,uint256,uint256)" \
  "Strategy Name" \
  $STRATEGY_ADDRESS \
  1000 \  # 10% APR in basis points
  50 \    # Risk score
  --private-key $ADMIN_PRIVATE_KEY
```

### Step 6: Transfer Ownership

```bash
# Transfer to multi-sig wallet
cast send $CONTRACT_ADDRESS "transferOwnership(address)" \
  $MULTISIG_ADDRESS \
  --private-key $DEPLOYER_PRIVATE_KEY

# Grant admin role to multi-sig
cast send $CONTRACT_ADDRESS "grantRole(bytes32,address)" \
  $(cast keccak "ADMIN_ROLE") \
  $MULTISIG_ADDRESS \
  --private-key $DEPLOYER_PRIVATE_KEY

# Revoke deployer admin role
cast send $CONTRACT_ADDRESS "revokeRole(bytes32,address)" \
  $(cast keccak "ADMIN_ROLE") \
  $DEPLOYER_ADDRESS \
  --private-key $DEPLOYER_PRIVATE_KEY
```

---

## Infrastructure Deployment

### Step 1: Prepare Kubernetes Cluster

```bash
# Verify cluster access
kubectl cluster-info
kubectl get nodes

# Create namespace
kubectl create namespace sphinxos-prod

# Set context
kubectl config set-context --current --namespace=sphinxos-prod
```

### Step 2: Configure Secrets

```bash
# Create secrets from environment variables
kubectl create secret generic sphinxos-secrets \
  --from-literal=jwt-secret="$JWT_SECRET" \
  --from-literal=database-url="$DATABASE_URL" \
  --from-literal=redis-url="$REDIS_URL" \
  --from-literal=sentry-dsn="$SENTRY_DSN" \
  --namespace=sphinxos-prod

# Verify secrets
kubectl get secrets -n sphinxos-prod
```

### Step 3: Deploy Infrastructure

```bash
# Set domain
export DOMAIN="sphinxos.io"

# Run deployment script
./scripts/deploy_infrastructure.sh

# Monitor deployment
kubectl get pods -n sphinxos-prod -w
```

### Step 4: Update Configuration

```bash
# Update contract addresses in config
kubectl create configmap sphinxos-config \
  --from-file=config/mainnet.yaml \
  --namespace=sphinxos-prod \
  --dry-run=client -o yaml | kubectl apply -f -
```

### Step 5: Verify Deployment

```bash
# Check pod status
kubectl get pods -n sphinxos-prod

# Check logs
kubectl logs -n sphinxos-prod -l app=sphinxos-node --tail=100

# Check services
kubectl get svc -n sphinxos-prod

# Check ingress
kubectl get ingress -n sphinxos-prod
```

---

## Configuration

### Update Config Files

1. Edit `config/mainnet.yaml`
2. Update contract addresses from deployments
3. Verify network settings
4. Set appropriate rate limits

```yaml
contracts:
  yield_aggregator: "0x..."  # From deployments/mainnet.json
  nft_contract: "0x..."      # From deployments/mainnet.json
```

### Apply Configuration

```bash
# Restart pods to pick up new config
kubectl rollout restart deployment/sphinxos-node -n sphinxos-prod

# Verify rollout
kubectl rollout status deployment/sphinxos-node -n sphinxos-prod
```

---

## Verification

### Health Checks

```bash
# Check API health
curl https://api.sphinxos.io/health

# Check metrics
curl https://api.sphinxos.io:8001/metrics

# Check specific endpoint
curl https://api.sphinxos.io/rarity
```

### Contract Verification

```bash
# Verify contract is accessible
cast call $CONTRACT_ADDRESS "owner()(address)"

# Verify contract state
cast call $YIELD_AGGREGATOR "getTotalTVL()(uint256)"

# Test deposit (with test wallet)
cast send $YIELD_AGGREGATOR "deposit(address,uint256,uint256)" \
  $TOKEN_ADDRESS \
  1000000000000000000 \  # 1 token
  500 \                  # Phi score
  --private-key $TEST_WALLET_KEY
```

### Monitoring

```bash
# Access Grafana
kubectl port-forward -n sphinxos-prod svc/sphinxos-monitoring-grafana 3000:80

# Access Prometheus
kubectl port-forward -n sphinxos-prod svc/sphinxos-monitoring-kube-prom-prometheus 9090:9090

# Check logs
kubectl logs -n sphinxos-prod -l app=sphinxos-node -f
```

---

## Rollback Procedures

### Contract Rollback

```bash
# If contract has issue, use emergency pause
cast send $CONTRACT_ADDRESS "pause()" \
  --private-key $ADMIN_PRIVATE_KEY

# If upgradeable, revert to previous version
cast send $PROXY_ADMIN "upgrade(address,address)" \
  $PROXY_ADDRESS \
  $PREVIOUS_IMPLEMENTATION \
  --private-key $ADMIN_PRIVATE_KEY
```

### Infrastructure Rollback

```bash
# Rollback deployment to previous version
kubectl rollout undo deployment/sphinxos-node -n sphinxos-prod

# Verify rollback
kubectl rollout status deployment/sphinxos-node -n sphinxos-prod

# Or rollback to specific revision
kubectl rollout undo deployment/sphinxos-node --to-revision=2 -n sphinxos-prod
```

### Database Rollback

```bash
# Restore from backup
psql $DATABASE_URL < backup-20240101-120000.sql

# Verify restoration
psql $DATABASE_URL -c "SELECT COUNT(*) FROM transactions;"
```

---

## Troubleshooting

### Common Issues

#### Pod Not Starting

```bash
# Check pod events
kubectl describe pod <pod-name> -n sphinxos-prod

# Check logs
kubectl logs <pod-name> -n sphinxos-prod

# Common fixes:
# - Check secret values
# - Verify image exists
# - Check resource limits
```

#### Contract Transaction Failing

```bash
# Check gas price
cast gas-price

# Estimate gas
cast estimate $CONTRACT_ADDRESS "functionName(args)" --from $ADDRESS

# Check balance
cast balance $ADDRESS

# Common fixes:
# - Increase gas limit
# - Wait for lower gas price
# - Verify contract state
```

#### API Not Responding

```bash
# Check pod health
kubectl get pods -n sphinxos-prod

# Check logs
kubectl logs -n sphinxos-prod -l app=sphinxos-node --tail=100

# Check database connection
kubectl exec -it <pod-name> -n sphinxos-prod -- env | grep DATABASE

# Common fixes:
# - Restart pods
# - Check database connectivity
# - Verify configuration
```

### Emergency Contacts

- On-call Engineer: +1-XXX-XXX-XXXX
- CTO: +1-XXX-XXX-XXXX
- DevOps Lead: +1-XXX-XXX-XXXX

### Escalation Procedure

1. Level 1: On-call engineer (0-15 min)
2. Level 2: Lead engineer (15-30 min)
3. Level 3: CTO (30+ min)

---

## Post-Deployment

### Monitoring

- Monitor for 48 hours continuously
- Check metrics every hour
- Review logs for errors
- Track transaction success rate
- Monitor gas costs

### Documentation

- Update deployment log
- Document any issues encountered
- Update runbooks with lessons learned
- Share deployment report with team

### Optimization

- Analyze performance metrics
- Identify bottlenecks
- Plan optimizations for next release
- Gather user feedback

---

## References

- [Mainnet Checklist](../MAINNET_CHECKLIST.md)
- [Configuration Guide](../config/README.md)
- [API Documentation](./API.md)
- [Security Guidelines](./SECURITY.md)

---

**Last Updated:** 2024-02-16
**Version:** 1.0
