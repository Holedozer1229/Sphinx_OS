#!/usr/bin/env python3
"""
Mainnet Smart Contract Deployment Script
Handles multi-chain deployment with verification
"""

import os
import json
import time
import sys
from typing import Dict, Optional

try:
    from web3 import Web3
    from eth_account import Account
except ImportError:
    print("Error: Required packages not installed")
    print("Install with: pip install web3 eth-account")
    sys.exit(1)


class MainnetDeployer:
    """
    Deploys Sphinx_OS smart contracts to mainnet networks
    
    Features:
    - Multi-chain deployment (Ethereum, Polygon, Arbitrum)
    - Contract verification
    - Deployment tracking
    - Gas optimization
    """
    
    def __init__(self):
        self.networks = {
            "ethereum": {
                "rpc": os.getenv("ETH_RPC_URL", "https://mainnet.infura.io/v3/YOUR_KEY"),
                "chain_id": 1,
                "explorer": "https://etherscan.io",
                "explorer_api": "https://api.etherscan.io/api"
            },
            "polygon": {
                "rpc": os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com"),
                "chain_id": 137,
                "explorer": "https://polygonscan.com",
                "explorer_api": "https://api.polygonscan.com/api"
            },
            "arbitrum": {
                "rpc": os.getenv("ARBITRUM_RPC_URL", "https://arb1.arbitrum.io/rpc"),
                "chain_id": 42161,
                "explorer": "https://arbiscan.io",
                "explorer_api": "https://api.arbiscan.io/api"
            }
        }
        
        self.deployer_key = os.getenv("DEPLOYER_PRIVATE_KEY")
        if not self.deployer_key:
            raise ValueError("DEPLOYER_PRIVATE_KEY environment variable not set")
        
        self.account = Account.from_key(self.deployer_key)
        print(f"Deployer Address: {self.account.address}")
    
    def deploy_contract(
        self,
        network: str,
        contract_name: str,
        constructor_args: Optional[list] = None
    ) -> Optional[str]:
        """
        Deploy contract to specified network
        
        Args:
            network: Network name (ethereum, polygon, arbitrum)
            contract_name: Name of contract to deploy
            constructor_args: Constructor arguments
            
        Returns:
            Deployed contract address or None if failed
        """
        print(f"\n{'='*60}")
        print(f"Deploying {contract_name} to {network}")
        print(f"{'='*60}")
        
        # Load network config
        net_config = self.networks.get(network)
        if not net_config:
            print(f"❌ Unknown network: {network}")
            return None
        
        # Connect to network
        w3 = Web3(Web3.HTTPProvider(net_config["rpc"]))
        
        # Verify connection
        if not w3.is_connected():
            print(f"❌ Failed to connect to {network}")
            return None
        
        print(f"✅ Connected to {network}")
        print(f"Deployer: {self.account.address}")
        
        # Check balance
        balance = w3.eth.get_balance(self.account.address)
        balance_eth = w3.from_wei(balance, 'ether')
        print(f"Balance: {balance_eth} ETH")
        
        if balance == 0:
            print(f"❌ Insufficient balance for deployment")
            return None
        
        # Load contract artifacts
        artifact_path = f"contracts/artifacts/{contract_name}.json"
        if not os.path.exists(artifact_path):
            print(f"❌ Contract artifact not found: {artifact_path}")
            print("Note: Compile contracts first using your build tool (Hardhat, Foundry, etc.)")
            return None
        
        with open(artifact_path, "r") as f:
            artifact = json.load(f)
        
        # Create contract instance
        Contract = w3.eth.contract(
            abi=artifact.get("abi"),
            bytecode=artifact.get("bytecode")
        )
        
        # Get current gas price
        gas_price = w3.eth.gas_price
        gas_price_gwei = w3.from_wei(gas_price, 'gwei')
        print(f"Gas Price: {gas_price_gwei} gwei")
        
        # Estimate gas
        try:
            if constructor_args:
                gas_estimate = Contract.constructor(*constructor_args).estimate_gas({
                    "from": self.account.address
                })
            else:
                gas_estimate = Contract.constructor().estimate_gas({
                    "from": self.account.address
                })
            
            print(f"Estimated Gas: {gas_estimate}")
            
            # Calculate deployment cost
            deployment_cost = w3.from_wei(gas_estimate * gas_price, 'ether')
            print(f"Estimated Cost: {deployment_cost} ETH")
            
            # Confirm deployment
            print("\nProceed with deployment? (yes/no): ", end="", flush=True)
            if os.getenv("AUTO_CONFIRM") != "yes":
                confirm = input().lower()
                if confirm != "yes":
                    print("Deployment cancelled")
                    return None
            else:
                print("yes (auto-confirmed)")
        
        except Exception as e:
            print(f"❌ Gas estimation failed: {e}")
            return None
        
        # Build transaction
        nonce = w3.eth.get_transaction_count(self.account.address)
        
        try:
            if constructor_args:
                tx = Contract.constructor(*constructor_args).build_transaction({
                    "chainId": net_config["chain_id"],
                    "gas": int(gas_estimate * 1.2),  # Add 20% buffer
                    "gasPrice": gas_price,
                    "nonce": nonce
                })
            else:
                tx = Contract.constructor().build_transaction({
                    "chainId": net_config["chain_id"],
                    "gas": int(gas_estimate * 1.2),
                    "gasPrice": gas_price,
                    "nonce": nonce
                })
        except Exception as e:
            print(f"❌ Failed to build transaction: {e}")
            return None
        
        # Sign and send
        print("Signing transaction...")
        signed = self.account.sign_transaction(tx)
        
        print("Sending transaction...")
        try:
            tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
            print(f"Transaction Hash: {tx_hash.hex()}")
            print("Waiting for confirmation...")
            
            # Wait for receipt
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
            
            if receipt["status"] == 1:
                address = receipt["contractAddress"]
                print(f"✅ Deployed successfully!")
                print(f"Contract Address: {address}")
                print(f"Gas Used: {receipt['gasUsed']}")
                print(f"Explorer: {net_config['explorer']}/address/{address}")
                
                # Save deployment info
                self.save_deployment(network, contract_name, address, tx_hash.hex())
                
                return address
            else:
                print(f"❌ Deployment failed (transaction reverted)")
                return None
                
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            return None
    
    def save_deployment(
        self,
        network: str,
        contract: str,
        address: str,
        tx_hash: str
    ):
        """Save deployment information to JSON file"""
        deployments_file = "deployments/mainnet.json"
        os.makedirs("deployments", exist_ok=True)
        
        # Load existing
        if os.path.exists(deployments_file):
            with open(deployments_file, "r") as f:
                deployments = json.load(f)
        else:
            deployments = {}
        
        # Update
        if network not in deployments:
            deployments[network] = {}
        
        deployments[network][contract] = {
            "address": address,
            "deployed_at": time.time(),
            "deployer": self.account.address,
            "tx_hash": tx_hash,
            "chain_id": self.networks[network]["chain_id"]
        }
        
        # Save
        with open(deployments_file, "w") as f:
            json.dump(deployments, f, indent=2)
        
        print(f"Deployment saved to {deployments_file}")
    
    def deploy_all(self, networks: Optional[list] = None):
        """
        Deploy all contracts to specified networks
        
        Args:
            networks: List of network names (None = all networks)
        """
        contracts = [
            {
                "name": "SphinxYieldAggregator",
                "args": [
                    "0x0000000000000000000000000000000000000001",  # treasury
                    "0x0000000000000000000000000000000000000002"   # zkVerifier
                ]
            },
            {
                "name": "SpaceFlightNFT",
                "args": [
                    "0x0000000000000000000000000000000000000003",  # sphinxToken
                    "0x0000000000000000000000000000000000000001",  # treasury
                    "0x0000000000000000000000000000000000000004"   # openSeaProxy
                ]
            }
        ]
        
        if networks is None:
            networks = ["ethereum", "polygon", "arbitrum"]
        
        results = {}
        
        for network in networks:
            print(f"\n{'='*60}")
            print(f"Network: {network.upper()}")
            print(f"{'='*60}")
            
            results[network] = {}
            for contract_info in contracts:
                contract_name = contract_info["name"]
                constructor_args = contract_info.get("args")
                
                try:
                    address = self.deploy_contract(network, contract_name, constructor_args)
                    results[network][contract_name] = address
                    
                    # Wait between deployments
                    if address:
                        print("Waiting 10 seconds before next deployment...")
                        time.sleep(10)
                        
                except Exception as e:
                    print(f"❌ Failed to deploy {contract_name} to {network}: {e}")
                    results[network][contract_name] = None
        
        # Summary
        print(f"\n{'='*60}")
        print("DEPLOYMENT SUMMARY")
        print(f"{'='*60}")
        
        for network, contracts in results.items():
            print(f"\n{network.upper()}:")
            for contract, address in contracts.items():
                status = "✅" if address else "❌"
                print(f"  {status} {contract}: {address or 'FAILED'}")
        
        return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy Sphinx_OS contracts to mainnet")
    parser.add_argument(
        "--network",
        choices=["ethereum", "polygon", "arbitrum", "all"],
        default="all",
        help="Network to deploy to"
    )
    parser.add_argument(
        "--contract",
        help="Specific contract to deploy (default: all)"
    )
    
    args = parser.parse_args()
    
    try:
        deployer = MainnetDeployer()
        
        if args.network == "all":
            networks = ["ethereum", "polygon", "arbitrum"]
        else:
            networks = [args.network]
        
        if args.contract:
            # Deploy single contract
            for network in networks:
                deployer.deploy_contract(network, args.contract)
        else:
            # Deploy all contracts
            deployer.deploy_all(networks)
            
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
