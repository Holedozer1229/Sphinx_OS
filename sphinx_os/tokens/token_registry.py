"""
Multi-Chain Token Registry for SphinxSkynet

Supports all major chains compatible with zk-EVM:
- Ethereum (ETH)
- Polygon (MATIC)
- Binance Smart Chain (BNB)
- Avalanche (AVAX)
- Arbitrum (ARB)
- Optimism (OP)
- Stacks (STX)
- zkSync Era
- Polygon zkEVM
- Scroll

Integrates with SphinxSkynet hypercube network for optimal routing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import json


class ChainType(Enum):
    """Supported blockchain types"""
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    BSC = "bsc"
    AVALANCHE = "avalanche"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    STACKS = "stacks"
    ZKSYNC = "zksync"
    POLYGON_ZKEVM = "polygon_zkevm"
    SCROLL = "scroll"


@dataclass
class ChainConfig:
    """Configuration for a blockchain network"""
    chain_id: int
    name: str
    chain_type: ChainType
    rpc_url: str
    explorer_url: str
    native_token: str
    supports_evm: bool
    supports_zk: bool
    gas_token: str
    

@dataclass
class Token:
    """Token information"""
    symbol: str
    name: str
    address: str
    decimals: int
    chain_type: ChainType
    chain_id: int
    is_native: bool = False
    coingecko_id: Optional[str] = None
    yield_apr: float = 0.0
    liquidity_usd: float = 0.0


class TokenRegistry:
    """
    Central registry for all supported tokens across chains.
    
    Integrates with:
    - SphinxSkynet hypercube routing
    - zk-EVM proof generation
    - Enhanced yield engine
    - Cross-chain bridges
    """
    
    # Mainnet chain configurations
    CHAINS: Dict[ChainType, ChainConfig] = {
        ChainType.ETHEREUM: ChainConfig(
            chain_id=1,
            name="Ethereum Mainnet",
            chain_type=ChainType.ETHEREUM,
            rpc_url="https://eth-mainnet.g.alchemy.com/v2/",
            explorer_url="https://etherscan.io",
            native_token="ETH",
            supports_evm=True,
            supports_zk=True,
            gas_token="ETH"
        ),
        ChainType.POLYGON: ChainConfig(
            chain_id=137,
            name="Polygon Mainnet",
            chain_type=ChainType.POLYGON,
            rpc_url="https://polygon-rpc.com",
            explorer_url="https://polygonscan.com",
            native_token="MATIC",
            supports_evm=True,
            supports_zk=True,
            gas_token="MATIC"
        ),
        ChainType.BSC: ChainConfig(
            chain_id=56,
            name="Binance Smart Chain",
            chain_type=ChainType.BSC,
            rpc_url="https://bsc-dataseed.binance.org",
            explorer_url="https://bscscan.com",
            native_token="BNB",
            supports_evm=True,
            supports_zk=False,
            gas_token="BNB"
        ),
        ChainType.AVALANCHE: ChainConfig(
            chain_id=43114,
            name="Avalanche C-Chain",
            chain_type=ChainType.AVALANCHE,
            rpc_url="https://api.avax.network/ext/bc/C/rpc",
            explorer_url="https://snowtrace.io",
            native_token="AVAX",
            supports_evm=True,
            supports_zk=False,
            gas_token="AVAX"
        ),
        ChainType.ARBITRUM: ChainConfig(
            chain_id=42161,
            name="Arbitrum One",
            chain_type=ChainType.ARBITRUM,
            rpc_url="https://arb1.arbitrum.io/rpc",
            explorer_url="https://arbiscan.io",
            native_token="ETH",
            supports_evm=True,
            supports_zk=True,
            gas_token="ETH"
        ),
        ChainType.OPTIMISM: ChainConfig(
            chain_id=10,
            name="Optimism",
            chain_type=ChainType.OPTIMISM,
            rpc_url="https://mainnet.optimism.io",
            explorer_url="https://optimistic.etherscan.io",
            native_token="ETH",
            supports_evm=True,
            supports_zk=True,
            gas_token="ETH"
        ),
        ChainType.ZKSYNC: ChainConfig(
            chain_id=324,
            name="zkSync Era",
            chain_type=ChainType.ZKSYNC,
            rpc_url="https://mainnet.era.zksync.io",
            explorer_url="https://explorer.zksync.io",
            native_token="ETH",
            supports_evm=True,
            supports_zk=True,
            gas_token="ETH"
        ),
        ChainType.POLYGON_ZKEVM: ChainConfig(
            chain_id=1101,
            name="Polygon zkEVM",
            chain_type=ChainType.POLYGON_ZKEVM,
            rpc_url="https://zkevm-rpc.com",
            explorer_url="https://zkevm.polygonscan.com",
            native_token="ETH",
            supports_evm=True,
            supports_zk=True,
            gas_token="ETH"
        ),
        ChainType.SCROLL: ChainConfig(
            chain_id=534352,
            name="Scroll",
            chain_type=ChainType.SCROLL,
            rpc_url="https://rpc.scroll.io",
            explorer_url="https://scrollscan.com",
            native_token="ETH",
            supports_evm=True,
            supports_zk=True,
            gas_token="ETH"
        ),
        ChainType.STACKS: ChainConfig(
            chain_id=0,  # Stacks uses different addressing
            name="Stacks",
            chain_type=ChainType.STACKS,
            rpc_url="https://stacks-node-api.mainnet.stacks.co",
            explorer_url="https://explorer.stacks.co",
            native_token="STX",
            supports_evm=False,
            supports_zk=True,  # Via SphinxSkynet zk-proofs
            gas_token="STX"
        ),
    }
    
    def __init__(self):
        """Initialize token registry with default tokens"""
        self.tokens: Dict[str, Token] = {}
        self._initialize_default_tokens()
    
    def _initialize_default_tokens(self):
        """Load default token set for all supported chains"""
        
        # Native tokens
        native_tokens = [
            Token("ETH", "Ethereum", "0x0000000000000000000000000000000000000000", 
                  18, ChainType.ETHEREUM, 1, True, "ethereum", 3.5, 50_000_000_000),
            Token("MATIC", "Polygon", "0x0000000000000000000000000000000000000001",
                  18, ChainType.POLYGON, 137, True, "matic-network", 5.2, 8_000_000_000),
            Token("BNB", "BNB", "0x0000000000000000000000000000000000000000",
                  18, ChainType.BSC, 56, True, "binancecoin", 1.8, 45_000_000_000),
            Token("AVAX", "Avalanche", "0x0000000000000000000000000000000000000000",
                  18, ChainType.AVALANCHE, 43114, True, "avalanche-2", 7.5, 12_000_000_000),
            Token("STX", "Stacks", "SP000000000000000000002Q6VF78",
                  6, ChainType.STACKS, 0, True, "stacks", 12.3, 2_500_000_000),
        ]
        
        # Major ERC-20 tokens (multi-chain)
        erc20_tokens = [
            # Stablecoins
            Token("USDC", "USD Coin", "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                  6, ChainType.ETHEREUM, 1, False, "usd-coin", 4.5, 25_000_000_000),
            Token("USDT", "Tether", "0xdAC17F958D2ee523a2206206994597C13D831ec7",
                  6, ChainType.ETHEREUM, 1, False, "tether", 4.2, 80_000_000_000),
            Token("DAI", "Dai Stablecoin", "0x6B175474E89094C44Da98b954EedeAC495271d0F",
                  18, ChainType.ETHEREUM, 1, False, "dai", 5.8, 5_000_000_000),
            
            # DeFi tokens
            Token("AAVE", "Aave", "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",
                  18, ChainType.ETHEREUM, 1, False, "aave", 2.1, 2_000_000_000),
            Token("UNI", "Uniswap", "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
                  18, ChainType.ETHEREUM, 1, False, "uniswap", 1.5, 4_500_000_000),
            Token("LINK", "Chainlink", "0x514910771AF9Ca656af840dff83E8264EcF986CA",
                  18, ChainType.ETHEREUM, 1, False, "chainlink", 4.8, 7_000_000_000),
            Token("CRV", "Curve DAO", "0xD533a949740bb3306d119CC777fa900bA034cd52",
                  18, ChainType.ETHEREUM, 1, False, "curve-dao-token", 8.2, 1_200_000_000),
            
            # Polygon tokens
            Token("USDC", "USD Coin", "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
                  6, ChainType.POLYGON, 137, False, "usd-coin", 5.5, 800_000_000),
            Token("WETH", "Wrapped Ether", "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619",
                  18, ChainType.POLYGON, 137, False, "weth", 3.2, 450_000_000),
            Token("WBTC", "Wrapped Bitcoin", "0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6",
                  8, ChainType.POLYGON, 137, False, "wrapped-bitcoin", 0.5, 9_000_000_000),
            
            # BSC tokens
            Token("CAKE", "PancakeSwap", "0x0E09FaBB73Bd3Ade0a17ECC321fD13a19e81cE82",
                  18, ChainType.BSC, 56, False, "pancakeswap-token", 35.5, 600_000_000),
            Token("BUSD", "Binance USD", "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56",
                  18, ChainType.BSC, 56, False, "binance-usd", 3.8, 5_000_000_000),
            
            # Arbitrum tokens
            Token("ARB", "Arbitrum", "0x912CE59144191C1204E64559FE8253a0e49E6548",
                  18, ChainType.ARBITRUM, 42161, False, "arbitrum", 0.0, 1_500_000_000),
            Token("GMX", "GMX", "0xfc5A1A6EB076a2C7aD06eD22C90d7E710E35ad0a",
                  18, ChainType.ARBITRUM, 42161, False, "gmx", 28.5, 400_000_000),
            
            # Optimism tokens
            Token("OP", "Optimism", "0x4200000000000000000000000000000000000042",
                  18, ChainType.OPTIMISM, 10, False, "optimism", 2.8, 800_000_000),
            
            # Avalanche tokens
            Token("JOE", "Joe Token", "0x6e84a6216eA6dACC71eE8E6b0a5B7322EEbC0fDd",
                  18, ChainType.AVALANCHE, 43114, False, "joe", 15.2, 150_000_000),
        ]
        
        # Register all tokens
        for token in native_tokens + erc20_tokens:
            self.register_token(token)
    
    def register_token(self, token: Token):
        """Register a new token"""
        key = f"{token.chain_type.value}:{token.symbol}:{token.address}"
        self.tokens[key] = token
    
    def get_token(self, symbol: str, chain_type: ChainType) -> Optional[Token]:
        """Get token by symbol and chain"""
        for token in self.tokens.values():
            if token.symbol == symbol and token.chain_type == chain_type:
                return token
        return None
    
    def get_tokens_by_chain(self, chain_type: ChainType) -> List[Token]:
        """Get all tokens for a specific chain"""
        return [t for t in self.tokens.values() if t.chain_type == chain_type]
    
    def get_zk_compatible_tokens(self) -> List[Token]:
        """Get all tokens on zk-compatible chains"""
        zk_chains = [ct for ct, config in self.CHAINS.items() if config.supports_zk]
        return [t for t in self.tokens.values() if t.chain_type in zk_chains]
    
    def get_chain_config(self, chain_type: ChainType) -> Optional[ChainConfig]:
        """Get configuration for a chain"""
        return self.CHAINS.get(chain_type)
    
    def get_all_tokens(self) -> List[Token]:
        """Get all registered tokens"""
        return list(self.tokens.values())
    
    def calculate_total_tvl(self) -> float:
        """Calculate total value locked across all tokens"""
        return sum(token.liquidity_usd for token in self.tokens.values())
    
    def get_highest_yield_tokens(self, limit: int = 10) -> List[Token]:
        """Get tokens with highest yield APR"""
        return sorted(self.tokens.values(), key=lambda t: t.yield_apr, reverse=True)[:limit]
    
    def export_to_json(self, filepath: str):
        """Export token registry to JSON"""
        data = {
            "chains": {
                ct.value: {
                    "chain_id": config.chain_id,
                    "name": config.name,
                    "rpc_url": config.rpc_url,
                    "explorer_url": config.explorer_url,
                    "native_token": config.native_token,
                    "supports_evm": config.supports_evm,
                    "supports_zk": config.supports_zk,
                }
                for ct, config in self.CHAINS.items()
            },
            "tokens": [
                {
                    "symbol": t.symbol,
                    "name": t.name,
                    "address": t.address,
                    "decimals": t.decimals,
                    "chain": t.chain_type.value,
                    "chain_id": t.chain_id,
                    "is_native": t.is_native,
                    "yield_apr": t.yield_apr,
                    "liquidity_usd": t.liquidity_usd,
                }
                for t in self.tokens.values()
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def summary(self) -> str:
        """Get summary of token registry"""
        total_tokens = len(self.tokens)
        total_chains = len(self.CHAINS)
        zk_compatible = len(self.get_zk_compatible_tokens())
        total_tvl = self.calculate_total_tvl()
        
        return f"""
SphinxSkynet Token Registry Summary
{'='*50}
Total Chains: {total_chains}
Total Tokens: {total_tokens}
zk-Compatible Tokens: {zk_compatible}
Total TVL: ${total_tvl:,.0f}

Top 5 Yield Tokens:
"""  + "\n".join([
            f"  {t.symbol} ({t.chain_type.value}): {t.yield_apr}% APR"
            for t in self.get_highest_yield_tokens(5)
        ])


if __name__ == "__main__":
    # Demo
    registry = TokenRegistry()
    print(registry.summary())
    
    # Export to JSON
    registry.export_to_json("/tmp/token_registry.json")
    print("\n✅ Token registry exported to /tmp/token_registry.json")
