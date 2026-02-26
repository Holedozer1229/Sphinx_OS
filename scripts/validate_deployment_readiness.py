#!/usr/bin/env python3
"""
Pre-Deployment Validation Script
Checks all prerequisites before mainnet deployment
"""

import os
import sys
from web3 import Web3

class ValidationError(Exception):
    pass

def validate_environment():
    """Validate environment variables"""
    required_vars = [
        'DEPLOYER_PRIVATE_KEY',
        'ETH_RPC_URL',
        'TREASURY_ADDRESS',
        'ZK_VERIFIER_ADDRESS',
        'SKYNT_TOKEN_ADDRESS',
        'OPENSEA_PROXY_ADDRESS',
        'JWT_SECRET',
        'ETHERSCAN_API_KEY'
    ]
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        raise ValidationError(f"Missing environment variables: {', '.join(missing)}")
    
    print("✅ All environment variables set")

def validate_addresses():
    """Validate constructor argument addresses are not placeholders"""
    addresses = {
        'TREASURY_ADDRESS': os.getenv('TREASURY_ADDRESS'),
        'ZK_VERIFIER_ADDRESS': os.getenv('ZK_VERIFIER_ADDRESS'),
        'SKYNT_TOKEN_ADDRESS': os.getenv('SKYNT_TOKEN_ADDRESS'),
        'OPENSEA_PROXY_ADDRESS': os.getenv('OPENSEA_PROXY_ADDRESS'),
    }
    
    for name, addr in addresses.items():
        if not addr or addr.startswith('0x0000000000000000000000000000000000000'):
            raise ValidationError(f"{name} is a placeholder address. Update .env with real address.")
        if not Web3.is_address(addr):
            raise ValidationError(f"{name} is not a valid Ethereum address: {addr}")
    
    print("✅ All addresses valid")

def validate_wallet_balance():
    """Check deployer wallet has sufficient ETH"""
    networks = {
        'ethereum': os.getenv('ETH_RPC_URL'),
        'polygon': os.getenv('POLYGON_RPC_URL'),
        'arbitrum': os.getenv('ARBITRUM_RPC_URL'),
    }
    
    private_key = os.getenv('DEPLOYER_PRIVATE_KEY')
    from eth_account import Account
    account = Account.from_key(private_key)
    
    for network, rpc in networks.items():
        if not rpc:
            print(f"⚠️  {network} RPC URL not set, skipping balance check")
            continue
            
        try:
            w3 = Web3(Web3.HTTPProvider(rpc))
            if not w3.is_connected():
                raise ValidationError(f"Cannot connect to {network} RPC: {rpc}")
            
            balance = w3.eth.get_balance(account.address)
            balance_eth = w3.from_wei(balance, 'ether')
            
            min_balance = 0.5  # Minimum 0.5 ETH per network
            if balance_eth < min_balance:
                raise ValidationError(
                    f"Insufficient balance on {network}. "
                    f"Has {balance_eth} ETH, need at least {min_balance} ETH"
                )
            
            print(f"✅ {network}: {balance_eth:.4f} ETH")
        except Exception as e:
            if "Cannot connect" in str(e) or "Insufficient balance" in str(e):
                raise
            print(f"⚠️  Warning: Could not check balance for {network}: {e}")

def validate_contract_artifacts():
    """Check contract compilation artifacts exist"""
    contracts = ['SphinxYieldAggregator', 'SpaceFlightNFT', 'SphinxBridge']
    missing = []
    
    for contract in contracts:
        artifact_path = f"contracts/artifacts/{contract}.json"
        if not os.path.exists(artifact_path):
            missing.append(contract)
    
    if missing:
        raise ValidationError(
            f"Missing contract artifacts: {', '.join(missing)}\n"
            f"Run: cd contracts && npm run compile"
        )
    
    print("✅ All contract artifacts found")

def main():
    print("🔍 Validating Deployment Readiness...\n")
    
    try:
        validate_environment()
        validate_addresses()
        validate_wallet_balance()
        validate_contract_artifacts()
        
        print("\n" + "="*60)
        print("✅ ALL VALIDATION CHECKS PASSED")
        print("="*60)
        print("\n⚠️  FINAL CHECKLIST:")
        print("  [ ] Security audits completed")
        print("  [ ] Multi-sig wallet configured")
        print("  [ ] Testnet deployment successful")
        print("  [ ] Team approval obtained")
        print("\nReady to deploy? Run:")
        print("  python scripts/deploy_mainnet.py --network <network>")
        
    except ValidationError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
