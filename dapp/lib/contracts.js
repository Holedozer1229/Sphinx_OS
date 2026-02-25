/**
 * contracts.js
 * Minimal ABIs and address map for the SphinxOS DApp.
 * Replace the placeholder addresses with real deployed addresses.
 */

// ── Contract Addresses ─────────────────────────────────────────────────────
// Populated from environment variables at build time (or hardcoded after deploy)
export const ADDRESSES = {
  // Ethereum Mainnet
  1: {
    SKYNTExcalibur:       process.env.NEXT_PUBLIC_SKYNT_MAINNET        || "",
    SphinxBridge:         process.env.NEXT_PUBLIC_BRIDGE_MAINNET       || "",
    SphinxYieldAggregator:process.env.NEXT_PUBLIC_AGGREGATOR_MAINNET   || "",
    SpaceFlightNFT:       process.env.NEXT_PUBLIC_NFT_MAINNET          || "",
  },
  // Polygon
  137: {
    SKYNTExcalibur:       process.env.NEXT_PUBLIC_SKYNT_POLYGON        || "",
    SphinxBridge:         process.env.NEXT_PUBLIC_BRIDGE_POLYGON       || "",
    SphinxYieldAggregator:process.env.NEXT_PUBLIC_AGGREGATOR_POLYGON   || "",
    SpaceFlightNFT:       process.env.NEXT_PUBLIC_NFT_POLYGON          || "",
  },
  // Arbitrum One
  42161: {
    SKYNTExcalibur:       process.env.NEXT_PUBLIC_SKYNT_ARBITRUM       || "",
    SphinxBridge:         process.env.NEXT_PUBLIC_BRIDGE_ARBITRUM      || "",
    SphinxYieldAggregator:process.env.NEXT_PUBLIC_AGGREGATOR_ARBITRUM  || "",
    SpaceFlightNFT:       process.env.NEXT_PUBLIC_NFT_ARBITRUM         || "",
  },
  // Hardhat / localhost
  31337: {
    SKYNTExcalibur:       process.env.NEXT_PUBLIC_SKYNT_LOCAL          || "",
    SphinxBridge:         process.env.NEXT_PUBLIC_BRIDGE_LOCAL         || "",
    SphinxYieldAggregator:process.env.NEXT_PUBLIC_AGGREGATOR_LOCAL     || "",
    SpaceFlightNFT:       process.env.NEXT_PUBLIC_NFT_LOCAL            || "",
  },
};

// ── Minimal ABIs ───────────────────────────────────────────────────────────

export const SKYNT_ABI = [
  // ERC-20
  "function name() view returns (string)",
  "function symbol() view returns (string)",
  "function decimals() view returns (uint8)",
  "function totalSupply() view returns (uint256)",
  "function balanceOf(address) view returns (uint256)",
  "function approve(address spender, uint256 amount) returns (bool)",
  "function allowance(address owner, address spender) view returns (uint256)",
  // Excalibur yield engine
  "function excaliburAPR() view returns (uint256)",
  "function totalStaked() view returns (uint256)",
  "function positions(address) view returns (uint256 stakedAmount, uint256 phiScore, uint256 lastClaimTime, uint256 pendingYield)",
  "function pendingYield(address) view returns (uint256)",
  "function stake(uint256 amount, uint256 phiScore)",
  "function unstake(uint256 amount)",
  "function claimYield()",
  // Events
  "event Staked(address indexed user, uint256 amount, uint256 phiScore)",
  "event Unstaked(address indexed user, uint256 amount, uint256 yieldClaimed)",
  "event YieldClaimed(address indexed user, uint256 yieldAmount, uint256 treasuryFee)",
];

export const AGGREGATOR_ABI = [
  "function deposit(address token, uint256 amount, uint256 phiScore)",
  "function withdraw(address token, uint256 amount)",
  "function claimYield(address token)",
  "function getPendingYield(address user, address token) view returns (uint256)",
  "function getUserPosition(address user, address token) view returns (tuple(uint256 depositedAmount, uint256 phiScore, uint256 lastClaimTime, uint256 accumulatedYield, uint256[] strategyAllocations))",
  "function getTotalTVL() view returns (uint256)",
  "function getStrategyCount() view returns (uint256)",
];

export const BRIDGE_ABI = [
  "function lockTokens(string destinationChain, address recipient) payable returns (bytes32)",
  "function burnTokens(uint256 amount, string destinationChain, address recipient) returns (bytes32)",
  "function getWrappedBalance(address account) view returns (uint256)",
  "function getLockedBalance(address account) view returns (uint256)",
  "function getTransactionStatus(bytes32 txHash) view returns (address sender, address recipient, uint256 amount, uint8 status, uint8 signatures)",
  "event TokensLocked(bytes32 indexed txHash, address indexed sender, uint256 amount, string destinationChain)",
  "event TokensReleased(bytes32 indexed txHash, address indexed recipient, uint256 amount)",
];

export const NFT_ABI = [
  "function mintSpaceFlightNFT(uint8 rarity, string theme, string missionName, string rocketType, uint256 phiScore, address referrer) returns (uint256)",
  "function getMintFee(uint8 rarity) pure returns (uint256)",
  "function getMintingStats() view returns (uint256 total, uint256 revenue, uint256[5] byRarity)",
  "function getNFTMetadata(uint256 tokenId) view returns (tuple(uint256 tokenId, uint8 rarity, string theme, string missionName, string rocketType, uint256 launchTimestamp, uint256 phiScore, bool listedOnOpenSea, uint256 mintPrice))",
  "function balanceOf(address owner) view returns (uint256)",
  "event NFTMinted(uint256 indexed tokenId, address indexed minter, uint8 rarity, uint256 fee, address referrer)",
];
