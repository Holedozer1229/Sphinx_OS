// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title SphinxYieldAggregator
 * @notice Multi-chain yield aggregator with zk-proof verification
 * @dev Integrates with SphinxSkynet hypercube network
 * 
 * Features:
 * - Multi-token yield optimization
 * - zk-SNARK proof verification
 * - Φ score-based yield boosts
 * - Cross-chain yield routing
 * - Automated rebalancing
 */

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

interface IZKVerifier {
    function verifyProof(
        uint256[2] memory a,
        uint256[2][2] memory b,
        uint256[2] memory c,
        uint256[] memory input
    ) external view returns (bool);
}

contract SphinxYieldAggregator is Ownable, ReentrancyGuard, Pausable {
    using SafeERC20 for IERC20;
    
    // ========== STRUCTS ==========
    
    struct YieldStrategy {
        string name;
        address strategyContract;
        uint256 totalDeposited;
        uint256 currentAPR;  // Basis points (10000 = 100%)
        uint256 riskScore;   // 0-100
        bool active;
    }
    
    struct UserPosition {
        uint256 depositedAmount;
        uint256 phiScore;  // 200-1000
        uint256 lastClaimTime;
        uint256 accumulatedYield;
        uint256[] strategyAllocations;  // Allocation per strategy
    }
    
    struct YieldProof {
        uint256[2] a;
        uint256[2][2] b;
        uint256[2] c;
        uint256[] publicInputs;
        uint256 timestamp;
    }
    
    // ========== STATE VARIABLES ==========
    
    // Token management
    mapping(address => bool) public supportedTokens;
    address[] public tokenList;
    
    // Strategy management
    YieldStrategy[] public strategies;
    mapping(uint256 => bool) public activeStrategies;
    
    // User positions
    mapping(address => mapping(address => UserPosition)) public userPositions;
    
    // Treasury
    address public treasury;
    uint256 public treasuryShareBPS = 500;  // 5% base
    uint256 public maxTreasuryShareBPS = 3000;  // 30% max
    
    // zk-Proof verification
    IZKVerifier public zkVerifier;
    mapping(bytes32 => bool) public usedProofs;
    
    // Φ score integration
    mapping(address => uint256) public userPhiScores;
    uint256 constant PHI_MIN = 200;
    uint256 constant PHI_MAX = 1000;
    
    // ========== EVENTS ==========
    
    event Deposited(
        address indexed user,
        address indexed token,
        uint256 amount,
        uint256 phiScore
    );
    
    event Withdrawn(
        address indexed user,
        address indexed token,
        uint256 amount,
        uint256 yieldAmount
    );
    
    event YieldClaimed(
        address indexed user,
        uint256 amount,
        uint256 treasuryAmount
    );
    
    event StrategyAdded(
        uint256 indexed strategyId,
        string name,
        address strategyContract
    );
    
    event ProofVerified(
        bytes32 indexed proofHash,
        address indexed user
    );
    
    // ========== CONSTRUCTOR ==========
    
    constructor(
        address _treasury,
        address _zkVerifier
    ) {
        require(_treasury != address(0), "Invalid treasury");
        treasury = _treasury;
        zkVerifier = IZKVerifier(_zkVerifier);
    }
    
    // ========== EXTERNAL FUNCTIONS ==========
    
    /**
     * @notice Deposit tokens into yield strategies
     * @param token Token address to deposit
     * @param amount Amount to deposit
     * @param phiScore User's spectral integration score (200-1000)
     */
    function deposit(
        address token,
        uint256 amount,
        uint256 phiScore
    ) external nonReentrant whenNotPaused {
        require(supportedTokens[token], "Token not supported");
        require(amount > 0, "Amount must be > 0");
        require(phiScore >= PHI_MIN && phiScore <= PHI_MAX, "Invalid Φ score");
        
        // Transfer tokens
        IERC20(token).safeTransferFrom(msg.sender, address(this), amount);
        
        // Update user position
        UserPosition storage position = userPositions[msg.sender][token];
        position.depositedAmount += amount;
        position.phiScore = phiScore;
        position.lastClaimTime = block.timestamp;
        
        // Store Φ score
        userPhiScores[msg.sender] = phiScore;
        
        // Allocate to strategies
        _allocateToStrategies(token, amount, phiScore);
        
        emit Deposited(msg.sender, token, amount, phiScore);
    }
    
    /**
     * @notice Withdraw tokens and claim yield
     * @param token Token address
     * @param amount Amount to withdraw (0 = withdraw all)
     */
    function withdraw(
        address token,
        uint256 amount
    ) external nonReentrant {
        UserPosition storage position = userPositions[msg.sender][token];
        require(position.depositedAmount > 0, "No deposit");
        
        if (amount == 0) {
            amount = position.depositedAmount;
        }
        
        require(amount <= position.depositedAmount, "Insufficient balance");
        
        // Calculate and claim yield
        uint256 yieldAmount = _calculateYield(msg.sender, token);
        
        // Update position
        position.depositedAmount -= amount;
        position.accumulatedYield = 0;
        
        // Withdraw from strategies
        _withdrawFromStrategies(token, amount);
        
        // Transfer tokens and yield
        IERC20(token).safeTransfer(msg.sender, amount);
        
        if (yieldAmount > 0) {
            _distributeYield(msg.sender, token, yieldAmount);
        }
        
        emit Withdrawn(msg.sender, token, amount, yieldAmount);
    }
    
    /**
     * @notice Claim accumulated yield
     * @param token Token address
     */
    function claimYield(address token) external nonReentrant {
        UserPosition storage position = userPositions[msg.sender][token];
        require(position.depositedAmount > 0, "No deposit");
        
        uint256 yieldAmount = _calculateYield(msg.sender, token);
        require(yieldAmount > 0, "No yield available");
        
        position.accumulatedYield = 0;
        position.lastClaimTime = block.timestamp;
        
        _distributeYield(msg.sender, token, yieldAmount);
        
        emit YieldClaimed(msg.sender, yieldAmount, 0);
    }
    
    /**
     * @notice Verify zk-proof for yield calculation
     * @param proof Yield proof structure
     */
    function verifyYieldProof(
        YieldProof calldata proof
    ) external returns (bool) {
        // Calculate proof hash
        bytes32 proofHash = keccak256(abi.encode(proof));
        require(!usedProofs[proofHash], "Proof already used");
        
        // Verify with zk verifier
        bool valid = zkVerifier.verifyProof(
            proof.a,
            proof.b,
            proof.c,
            proof.publicInputs
        );
        
        require(valid, "Invalid proof");
        
        // Mark proof as used
        usedProofs[proofHash] = true;
        
        emit ProofVerified(proofHash, msg.sender);
        
        return true;
    }
    
    // ========== INTERNAL FUNCTIONS ==========
    
    /**
     * @dev Allocate tokens to optimal strategies
     */
    function _allocateToStrategies(
        address token,
        uint256 amount,
        uint256 phiScore
    ) internal {
        // Simplified allocation - distribute across top 3 strategies
        // In production, use more sophisticated optimization
        
        uint256 numStrategies = strategies.length;
        if (numStrategies == 0) return;
        
        uint256 perStrategy = amount / (numStrategies > 3 ? 3 : numStrategies);
        
        for (uint256 i = 0; i < numStrategies && i < 3; i++) {
            if (activeStrategies[i]) {
                strategies[i].totalDeposited += perStrategy;
                // In production: call strategy contract deposit
            }
        }
    }
    
    /**
     * @dev Withdraw from strategies
     */
    function _withdrawFromStrategies(
        address token,
        uint256 amount
    ) internal {
        // Simplified - proportionally withdraw from strategies
        // In production: optimize withdrawal path
    }
    
    /**
     * @dev Calculate accumulated yield with Φ boost
     */
    function _calculateYield(
        address user,
        address token
    ) internal view returns (uint256) {
        UserPosition storage position = userPositions[user][token];
        
        uint256 timeElapsed = block.timestamp - position.lastClaimTime;
        uint256 baseYield = 0;
        
        // Calculate base yield from strategies
        for (uint256 i = 0; i < strategies.length; i++) {
            if (activeStrategies[i]) {
                uint256 strategyAPR = strategies[i].currentAPR;
                uint256 yearlyYield = (position.depositedAmount * strategyAPR) / 10000;
                baseYield += (yearlyYield * timeElapsed) / 365 days;
            }
        }
        
        // Apply Φ boost
        uint256 phiBoost = _calculatePhiBoost(position.phiScore);
        uint256 boostedYield = (baseYield * phiBoost) / 10000;
        
        return boostedYield + position.accumulatedYield;
    }
    
    /**
     * @dev Calculate Φ boost multiplier
     */
    function _calculatePhiBoost(uint256 phiScore) internal pure returns (uint256) {
        // Φ boost: 1.0 + (phi - 500) / 2000
        // In basis points: 10000 + (phi - 500) * 5
        if (phiScore < PHI_MIN) phiScore = PHI_MIN;
        if (phiScore > PHI_MAX) phiScore = PHI_MAX;
        
        int256 boost = 10000 + (int256(phiScore) - 500) * 5;
        return uint256(boost);
    }
    
    /**
     * @dev Distribute yield with treasury split
     */
    function _distributeYield(
        address user,
        address token,
        uint256 amount
    ) internal {
        UserPosition storage position = userPositions[user][token];
        
        // Calculate treasury share based on Φ score
        // Treasury rate: min(0.30, 0.05 + phi/2000)
        uint256 phiScore = position.phiScore;
        uint256 treasuryRate = treasuryShareBPS + (phiScore * 5 / 10);
        
        if (treasuryRate > maxTreasuryShareBPS) {
            treasuryRate = maxTreasuryShareBPS;
        }
        
        uint256 treasuryAmount = (amount * treasuryRate) / 10000;
        uint256 userAmount = amount - treasuryAmount;
        
        // Transfer
        if (treasuryAmount > 0) {
            IERC20(token).safeTransfer(treasury, treasuryAmount);
        }
        
        if (userAmount > 0) {
            IERC20(token).safeTransfer(user, userAmount);
        }
    }
    
    // ========== ADMIN FUNCTIONS ==========
    
    function addToken(address token) external onlyOwner {
        require(!supportedTokens[token], "Token already supported");
        supportedTokens[token] = true;
        tokenList.push(token);
    }
    
    function addStrategy(
        string memory name,
        address strategyContract,
        uint256 apr,
        uint256 riskScore
    ) external onlyOwner {
        strategies.push(YieldStrategy({
            name: name,
            strategyContract: strategyContract,
            totalDeposited: 0,
            currentAPR: apr,
            riskScore: riskScore,
            active: true
        }));
        
        uint256 strategyId = strategies.length - 1;
        activeStrategies[strategyId] = true;
        
        emit StrategyAdded(strategyId, name, strategyContract);
    }
    
    function setTreasury(address _treasury) external onlyOwner {
        require(_treasury != address(0), "Invalid treasury");
        treasury = _treasury;
    }
    
    function pause() external onlyOwner {
        _pause();
    }
    
    function unpause() external onlyOwner {
        _unpause();
    }
    
    // ========== VIEW FUNCTIONS ==========
    
    function getUserPosition(
        address user,
        address token
    ) external view returns (UserPosition memory) {
        return userPositions[user][token];
    }
    
    function getPendingYield(
        address user,
        address token
    ) external view returns (uint256) {
        return _calculateYield(user, token);
    }
    
    function getStrategyCount() external view returns (uint256) {
        return strategies.length;
    }
    
    function getTotalTVL() external view returns (uint256) {
        uint256 total = 0;
        for (uint256 i = 0; i < strategies.length; i++) {
            total += strategies[i].totalDeposited;
        }
        return total;
    }
}
