"""
OpenSea Integration Service for SphinxOS

Automatically lists Legendary Space Flight NFTs on OpenSea
with configurable pricing and 10% royalties.
"""

import requests
import json
import time
from typing import Dict, Optional
from web3 import Web3
from eth_account import Account


class OpenSeaIntegration:
    """
    OpenSea API integration for automatic NFT listing.
    
    Features:
    - Automatic Legendary NFT listing
    - Dynamic pricing based on Φ score
    - 10% royalty configuration
    - Metadata generation and IPFS upload
    """
    
    def __init__(
        self,
        api_key: str,
        contract_address: str,
        treasury_address: str,
        network: str = "ethereum"
    ):
        """
        Initialize OpenSea integration.
        
        Args:
            api_key: OpenSea API key
            contract_address: SpaceFlightNFT contract address
            treasury_address: Address to receive royalties
            network: "ethereum" or "polygon"
        """
        self.api_key = api_key
        self.contract_address = contract_address
        self.treasury_address = treasury_address
        self.network = network
        
        # OpenSea API endpoints
        if network == "ethereum":
            self.api_base = "https://api.opensea.io/v2"
        else:
            self.api_base = "https://api.opensea.io/v2"
        
        self.headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
    
    def create_listing(
        self,
        token_id: int,
        price_eth: float,
        duration_days: int = 30,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Create fixed-price listing on OpenSea.
        
        Args:
            token_id: NFT token ID
            price_eth: Listing price in ETH
            duration_days: Listing duration (default 30 days)
            metadata: NFT metadata dict
        
        Returns:
            Listing response from OpenSea
        """
        print(f"Creating OpenSea listing for Token #{token_id}")
        print(f"Price: {price_eth} ETH")
        
        # Convert price to Wei
        price_wei = Web3.to_wei(price_eth, 'ether')
        
        # Calculate expiration
        expiration = int(time.time()) + (duration_days * 24 * 60 * 60)
        
        # Create listing payload
        payload = {
            "parameters": {
                "offerer": self.treasury_address,
                "offer": [{
                    "itemType": 2,  # ERC721
                    "token": self.contract_address,
                    "identifierOrCriteria": str(token_id),
                    "startAmount": "1",
                    "endAmount": "1"
                }],
                "consideration": [{
                    "itemType": 0,  # ETH
                    "token": "0x0000000000000000000000000000000000000000",
                    "identifierOrCriteria": "0",
                    "startAmount": str(price_wei),
                    "endAmount": str(price_wei),
                    "recipient": self.treasury_address
                }],
                "startTime": str(int(time.time())),
                "endTime": str(expiration),
                "orderType": 0,  # Full open
                "zone": "0x0000000000000000000000000000000000000000",
                "zoneHash": "0x" + "0" * 64,
                "salt": str(int(time.time() * 1000)),
                "conduitKey": "0x" + "0" * 64,
                "totalOriginalConsiderationItems": 1
            }
        }
        
        # Upload metadata to IPFS if provided
        if metadata:
            ipfs_url = self._upload_to_ipfs(metadata)
            print(f"Metadata uploaded to IPFS: {ipfs_url}")
        
        # Post listing
        endpoint = f"{self.api_base}/orders/{self.network}/seaport/listings"
        
        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            print(f"✅ Listed on OpenSea: {self._get_listing_url(token_id)}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to create listing: {e}")
            return {"error": str(e)}
    
    def set_royalty(
        self,
        collection_slug: str,
        royalty_bps: int = 1000  # 10%
    ) -> Dict:
        """
        Set collection-wide royalty.
        
        Args:
            collection_slug: OpenSea collection slug
            royalty_bps: Royalty in basis points (1000 = 10%)
        
        Returns:
            API response
        """
        payload = {
            "fee_recipient": self.treasury_address,
            "fee_basis_points": royalty_bps
        }
        
        endpoint = f"{self.api_base}/collection/{collection_slug}/edit"
        
        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            print(f"✅ Royalty set to {royalty_bps/100}%")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to set royalty: {e}")
            return {"error": str(e)}
    
    def calculate_listing_price(
        self,
        rarity: str,
        phi_score: int,
        base_price: float = 10.0
    ) -> float:
        """
        Calculate optimal listing price.
        
        Formula:
        Price = Base × Rarity_Multiplier × Φ_Multiplier
        
        Args:
            rarity: "LEGENDARY", "EPIC", "RARE", etc.
            phi_score: User's Φ score (200-1000)
            base_price: Base price in ETH (default 10.0)
        
        Returns:
            Calculated price in ETH
        """
        # Rarity multipliers
        rarity_multipliers = {
            "LEGENDARY": 1.0,
            "EPIC": 0.5,
            "RARE": 0.25,
            "UNCOMMON": 0.1,
            "COMMON": 0.05
        }
        
        rarity_mult = rarity_multipliers.get(rarity.upper(), 0.05)
        
        # Φ score multiplier
        # Φ = 950: 1.45x, Φ = 800: 1.3x, Φ = 500: 1.0x
        phi_mult = 1.0 + (phi_score - 500) / 1000
        phi_mult = max(0.7, min(1.8, phi_mult))  # Clamp to 0.7x - 1.8x
        
        price = base_price * rarity_mult * phi_mult
        
        return round(price, 4)
    
    def generate_metadata(
        self,
        token_id: int,
        name: str,
        description: str,
        theme: str,
        rarity: str,
        mission_name: str,
        rocket_type: str,
        launch_timestamp: int,
        phi_score: int,
        image_url: str
    ) -> Dict:
        """
        Generate OpenSea-compatible metadata.
        
        Args:
            token_id: NFT token ID
            name: NFT name
            description: NFT description
            theme: "stranger", "warhammer", "starwars"
            rarity: Rarity tier
            mission_name: Space mission name
            rocket_type: Rocket type
            launch_timestamp: Unix timestamp
            phi_score: Φ score
            image_url: IPFS or hosted image URL
        
        Returns:
            Metadata dict
        """
        metadata = {
            "name": name,
            "description": description,
            "image": image_url,
            "external_url": f"https://www.mindofthecosmos.com/nft/{token_id}",
            "attributes": [
                {"trait_type": "Rarity", "value": rarity},
                {"trait_type": "Theme", "value": theme.title()},
                {"trait_type": "Mission", "value": mission_name},
                {"trait_type": "Rocket", "value": rocket_type},
                {"trait_type": "Φ Score", "value": phi_score},
                {
                    "trait_type": "Launch Date",
                    "display_type": "date",
                    "value": launch_timestamp
                }
            ],
            "properties": {
                "category": "Space Flight Commemorative",
                "creators": [{
                    "address": self.treasury_address,
                    "share": 100
                }]
            }
        }
        
        return metadata
    
    def _upload_to_ipfs(self, metadata: Dict) -> str:
        """
        Upload metadata to IPFS via Pinata.
        
        Args:
            metadata: Metadata dict
        
        Returns:
            IPFS URL
        """
        # In production, use Pinata or another IPFS service
        # For now, return placeholder
        return f"ipfs://QmPLACEHOLDER/{metadata.get('name', 'nft')}.json"
    
    def _get_listing_url(self, token_id: int) -> str:
        """Get OpenSea listing URL"""
        if self.network == "ethereum":
            return f"https://opensea.io/assets/ethereum/{self.contract_address}/{token_id}"
        else:
            return f"https://opensea.io/assets/matic/{self.contract_address}/{token_id}"
    
    def get_collection_stats(self, collection_slug: str) -> Dict:
        """
        Get collection statistics from OpenSea.
        
        Args:
            collection_slug: Collection slug
        
        Returns:
            Stats dict with floor price, volume, etc.
        """
        endpoint = f"{self.api_base}/collection/{collection_slug}/stats"
        
        try:
            response = requests.get(endpoint, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except:
            return {}
    
    def bulk_list(
        self,
        listings: list,
        batch_size: int = 10
    ) -> list:
        """
        Bulk list multiple NFTs.
        
        Args:
            listings: List of (token_id, price_eth) tuples
            batch_size: Listings per batch
        
        Returns:
            List of results
        """
        results = []
        
        for i in range(0, len(listings), batch_size):
            batch = listings[i:i + batch_size]
            
            for token_id, price_eth in batch:
                result = self.create_listing(token_id, price_eth)
                results.append(result)
                
                # Rate limiting
                time.sleep(1)
        
        return results


def example_usage():
    """Example usage of OpenSeaIntegration"""
    
    # Initialize
    opensea = OpenSeaIntegration(
        api_key="YOUR_API_KEY",
        contract_address="0x...",
        treasury_address="0x...",
        network="ethereum"
    )
    
    # Calculate price for Legendary NFT with Φ=950
    price = opensea.calculate_listing_price(
        rarity="LEGENDARY",
        phi_score=950,
        base_price=10.0
    )
    print(f"Calculated price: {price} ETH")
    
    # Generate metadata
    metadata = opensea.generate_metadata(
        token_id=1,
        name="Falcon 9 Launch #42",
        description="Commemorative NFT from historic SpaceX launch",
        theme="starwars",
        rarity="LEGENDARY",
        mission_name="Starlink Group 6-7",
        rocket_type="Falcon 9 Block 5",
        launch_timestamp=int(time.time()),
        phi_score=950,
        image_url="ipfs://..."
    )
    
    # Create listing
    result = opensea.create_listing(
        token_id=1,
        price_eth=price,
        duration_days=30,
        metadata=metadata
    )
    
    print(f"Listing result: {result}")


if __name__ == "__main__":
    example_usage()
