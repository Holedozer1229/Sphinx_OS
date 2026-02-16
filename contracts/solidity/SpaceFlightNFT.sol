// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

/**
 * @title SpaceFlightNFT
 * @notice Commemorative Space Flight NFTs with tiered pricing and OpenSea integration
 * @dev Maximizes monetization through:
 *  - Tiered minting fees (500 SPX - 50,000 SPX)
 *  - 10% royalties on all secondary sales
 *  - Automatic OpenSea listing for Legendary tier
 *  - Referral rewards (5% of mint fees)
 */
contract SpaceFlightNFT is ERC721, ERC721URIStorage, Ownable, ReentrancyGuard {
    using Counters for Counters.Counter;
    
    // ========== STATE VARIABLES ==========
    
    Counters.Counter private _tokenIds;
    
    // Token configuration
    address public sphinxToken; // SPX token address
    address public treasury;
    address public openSeaProxy; // OpenSea proxy for gas-free listings
    
    // Minting fees (in SPX, 18 decimals)
    uint256 public constant COMMON_FEE = 500 * 10**18;      // 500 SPX
    uint256 public constant UNCOMMON_FEE = 1000 * 10**18;   // 1,000 SPX
    uint256 public constant RARE_FEE = 2500 * 10**18;       // 2,500 SPX
    uint256 public constant EPIC_FEE = 10000 * 10**18;      // 10,000 SPX
    uint256 public constant LEGENDARY_FEE = 50000 * 10**18; // 50,000 SPX
    
    // Royalties
    uint256 public constant ROYALTY_PERCENTAGE = 1000; // 10% in basis points
    
    // Referral system
    uint256 public constant REFERRAL_REWARD = 500; // 5% in basis points
    mapping(address => uint256) public referralEarnings;
    
    // Rarity system
    enum Rarity { COMMON, UNCOMMON, RARE, EPIC, LEGENDARY }
    
    struct NFTMetadata {
        uint256 tokenId;
        Rarity rarity;
        string theme; // "stranger", "warhammer", "starwars"
        string missionName;
        string rocketType;
        uint256 launchTimestamp;
        uint256 phiScore;
        bool listedOnOpenSea;
        uint256 mintPrice;
    }
    
    mapping(uint256 => NFTMetadata) public nftMetadata;
    
    // Statistics
    uint256 public totalMinted;
    uint256 public totalRevenue;
    mapping(Rarity => uint256) public mintedByRarity;
    
    // OpenSea listing prices (in Wei)
    uint256 public legendaryStartPrice = 10 ether; // Starting at 10 ETH
    
    // ========== EVENTS ==========
    
    event NFTMinted(
        uint256 indexed tokenId,
        address indexed minter,
        Rarity rarity,
        uint256 fee,
        address referrer
    );
    
    event ListedOnOpenSea(
        uint256 indexed tokenId,
        uint256 price
    );
    
    event RoyaltyPaid(
        uint256 indexed tokenId,
        address indexed seller,
        address indexed buyer,
        uint256 amount
    );
    
    event ReferralPaid(
        address indexed referrer,
        uint256 amount
    );
    
    // ========== CONSTRUCTOR ==========
    
    constructor(
        address _sphinxToken,
        address _treasury,
        address _openSeaProxy
    ) ERC721("SphinxOS Space Flight", "SPACE") {
        sphinxToken = _sphinxToken;
        treasury = _treasury;
        openSeaProxy = _openSeaProxy;
    }
    
    // ========== MINTING FUNCTIONS ==========
    
    /**
     * @notice Mint Space Flight NFT with specified rarity
     * @param rarity NFT rarity tier
     * @param theme Theme selection ("stranger", "warhammer", "starwars")
     * @param missionName Name of space mission
     * @param rocketType Type of rocket
     * @param phiScore User's Phi score
     * @param referrer Optional referrer address for rewards
     */
    function mintSpaceFlightNFT(
        Rarity rarity,
        string memory theme,
        string memory missionName,
        string memory rocketType,
        uint256 phiScore,
        address referrer
    ) external nonReentrant returns (uint256) {
        // Calculate mint fee
        uint256 fee = getMintFee(rarity);
        
        // Transfer SPX tokens from minter
        require(
            IERC20(sphinxToken).transferFrom(msg.sender, address(this), fee),
            "SPX transfer failed"
        );
        
        // Pay referral reward if applicable
        if (referrer != address(0) && referrer != msg.sender) {
            uint256 referralReward = (fee * REFERRAL_REWARD) / 10000;
            referralEarnings[referrer] += referralReward;
            
            require(
                IERC20(sphinxToken).transfer(referrer, referralReward),
                "Referral payment failed"
            );
            
            emit ReferralPaid(referrer, referralReward);
            
            // Send remaining to treasury
            uint256 netFee = fee - referralReward;
            require(
                IERC20(sphinxToken).transfer(treasury, netFee),
                "Treasury payment failed"
            );
        } else {
            // Send all to treasury
            require(
                IERC20(sphinxToken).transfer(treasury, fee),
                "Treasury payment failed"
            );
        }
        
        // Mint NFT
        _tokenIds.increment();
        uint256 newTokenId = _tokenIds.current();
        
        _safeMint(msg.sender, newTokenId);
        
        // Store metadata
        nftMetadata[newTokenId] = NFTMetadata({
            tokenId: newTokenId,
            rarity: rarity,
            theme: theme,
            missionName: missionName,
            rocketType: rocketType,
            launchTimestamp: block.timestamp,
            phiScore: phiScore,
            listedOnOpenSea: false,
            mintPrice: fee
        });
        
        // Update statistics
        totalMinted++;
        totalRevenue += fee;
        mintedByRarity[rarity]++;
        
        emit NFTMinted(newTokenId, msg.sender, rarity, fee, referrer);
        
        // If Legendary, automatically list on OpenSea
        if (rarity == Rarity.LEGENDARY) {
            _listOnOpenSea(newTokenId);
        }
        
        return newTokenId;
    }
    
    /**
     * @notice Auto-mint at launch moment (called by backend service)
     * @dev Only owner can call this for automatic minting
     */
    function autoMintAtLaunch(
        address recipient,
        string memory theme,
        string memory missionName,
        string memory rocketType,
        uint256 phiScore
    ) external onlyOwner returns (uint256) {
        // Automatically determine rarity based on phi score
        Rarity rarity = _determineRarity(phiScore);
        
        _tokenIds.increment();
        uint256 newTokenId = _tokenIds.current();
        
        _safeMint(recipient, newTokenId);
        
        nftMetadata[newTokenId] = NFTMetadata({
            tokenId: newTokenId,
            rarity: rarity,
            theme: theme,
            missionName: missionName,
            rocketType: rocketType,
            launchTimestamp: block.timestamp,
            phiScore: phiScore,
            listedOnOpenSea: false,
            mintPrice: 0 // Free auto-mint
        });
        
        totalMinted++;
        mintedByRarity[rarity]++;
        
        // If Legendary, list on OpenSea
        if (rarity == Rarity.LEGENDARY) {
            _listOnOpenSea(newTokenId);
        }
        
        return newTokenId;
    }
    
    // ========== OPENSEA INTEGRATION ==========
    
    /**
     * @notice List NFT on OpenSea
     * @dev Transfers NFT to contract and approves OpenSea proxy
     */
    function _listOnOpenSea(uint256 tokenId) internal {
        address owner = ownerOf(tokenId);
        
        // Calculate listing price based on rarity and phi score
        NFTMetadata storage metadata = nftMetadata[tokenId];
        uint256 listPrice = _calculateOpenSeaPrice(metadata);
        
        // Mark as listed
        metadata.listedOnOpenSea = true;
        
        emit ListedOnOpenSea(tokenId, listPrice);
        
        // Note: Actual OpenSea listing requires off-chain signature
        // This event triggers backend service to create listing
    }
    
    /**
     * @notice Calculate OpenSea listing price
     * @dev Price increases with phi score and rarity
     */
    function _calculateOpenSeaPrice(NFTMetadata memory metadata) internal view returns (uint256) {
        uint256 basePrice = legendaryStartPrice;
        
        // Adjust by rarity
        if (metadata.rarity == Rarity.EPIC) {
            basePrice = basePrice / 2; // 5 ETH
        } else if (metadata.rarity == Rarity.RARE) {
            basePrice = basePrice / 4; // 2.5 ETH
        } else if (metadata.rarity == Rarity.UNCOMMON) {
            basePrice = basePrice / 10; // 1 ETH
        } else if (metadata.rarity == Rarity.COMMON) {
            basePrice = basePrice / 20; // 0.5 ETH
        }
        
        // Adjust by phi score (higher phi = higher price)
        uint256 phiMultiplier = 100 + (metadata.phiScore - 200) / 10; // 100% - 180%
        basePrice = (basePrice * phiMultiplier) / 100;
        
        return basePrice;
    }
    
    /**
     * @notice Owner can manually list any NFT on OpenSea
     */
    function listOnOpenSea(uint256 tokenId) external {
        require(ownerOf(tokenId) == msg.sender, "Not token owner");
        require(!nftMetadata[tokenId].listedOnOpenSea, "Already listed");
        
        _listOnOpenSea(tokenId);
    }
    
    // ========== ROYALTY FUNCTIONS ==========
    
    /**
     * @notice Get royalty info for marketplaces (EIP-2981)
     * @param tokenId Token ID
     * @param salePrice Sale price
     * @return receiver Royalty receiver address
     * @return royaltyAmount Royalty amount
     */
    function royaltyInfo(uint256 tokenId, uint256 salePrice)
        external
        view
        returns (address receiver, uint256 royaltyAmount)
    {
        receiver = treasury;
        royaltyAmount = (salePrice * ROYALTY_PERCENTAGE) / 10000;
    }
    
    /**
     * @notice Process royalty payment (called by marketplace)
     */
    function payRoyalty(uint256 tokenId, address seller, address buyer)
        external
        payable
        nonReentrant
    {
        uint256 royaltyAmount = (msg.value * ROYALTY_PERCENTAGE) / 10000;
        uint256 sellerAmount = msg.value - royaltyAmount;
        
        // Pay royalty to treasury
        (bool royaltySuccess, ) = treasury.call{value: royaltyAmount}("");
        require(royaltySuccess, "Royalty payment failed");
        
        // Pay seller
        (bool sellerSuccess, ) = seller.call{value: sellerAmount}("");
        require(sellerSuccess, "Seller payment failed");
        
        emit RoyaltyPaid(tokenId, seller, buyer, royaltyAmount);
    }
    
    // ========== HELPER FUNCTIONS ==========
    
    /**
     * @notice Get mint fee for rarity tier
     */
    function getMintFee(Rarity rarity) public pure returns (uint256) {
        if (rarity == Rarity.COMMON) return COMMON_FEE;
        if (rarity == Rarity.UNCOMMON) return UNCOMMON_FEE;
        if (rarity == Rarity.RARE) return RARE_FEE;
        if (rarity == Rarity.EPIC) return EPIC_FEE;
        if (rarity == Rarity.LEGENDARY) return LEGENDARY_FEE;
        revert("Invalid rarity");
    }
    
    /**
     * @notice Determine rarity based on phi score
     */
    function _determineRarity(uint256 phiScore) internal pure returns (Rarity) {
        if (phiScore >= 950) return Rarity.LEGENDARY;
        if (phiScore >= 800) return Rarity.EPIC;
        if (phiScore >= 600) return Rarity.RARE;
        if (phiScore >= 400) return Rarity.UNCOMMON;
        return Rarity.COMMON;
    }
    
    /**
     * @notice Get NFT metadata
     */
    function getNFTMetadata(uint256 tokenId) external view returns (NFTMetadata memory) {
        require(_exists(tokenId), "Token does not exist");
        return nftMetadata[tokenId];
    }
    
    /**
     * @notice Get minting statistics
     */
    function getMintingStats() external view returns (
        uint256 total,
        uint256 revenue,
        uint256[5] memory byRarity
    ) {
        total = totalMinted;
        revenue = totalRevenue;
        byRarity = [
            mintedByRarity[Rarity.COMMON],
            mintedByRarity[Rarity.UNCOMMON],
            mintedByRarity[Rarity.RARE],
            mintedByRarity[Rarity.EPIC],
            mintedByRarity[Rarity.LEGENDARY]
        ];
    }
    
    // ========== ADMIN FUNCTIONS ==========
    
    function setLegendaryStartPrice(uint256 newPrice) external onlyOwner {
        legendaryStartPrice = newPrice;
    }
    
    function setTreasury(address newTreasury) external onlyOwner {
        treasury = newTreasury;
    }
    
    function setOpenSeaProxy(address newProxy) external onlyOwner {
        openSeaProxy = newProxy;
    }
    
    /**
     * @notice Withdraw referral earnings
     */
    function withdrawReferralEarnings() external nonReentrant {
        uint256 earnings = referralEarnings[msg.sender];
        require(earnings > 0, "No earnings to withdraw");
        
        referralEarnings[msg.sender] = 0;
        
        require(
            IERC20(sphinxToken).transfer(msg.sender, earnings),
            "Withdrawal failed"
        );
    }
    
    // ========== OVERRIDES ==========
    
    function _burn(uint256 tokenId) internal override(ERC721, ERC721URIStorage) {
        super._burn(tokenId);
    }
    
    function tokenURI(uint256 tokenId)
        public
        view
        override(ERC721, ERC721URIStorage)
        returns (string memory)
    {
        return super.tokenURI(tokenId);
    }
    
    function supportsInterface(bytes4 interfaceId)
        public
        view
        override(ERC721, ERC721URIStorage)
        returns (bool)
    {
        return super.supportsInterface(interfaceId);
    }
}

// Simple ERC20 interface
interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}
