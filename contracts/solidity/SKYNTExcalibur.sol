// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title SKYNTExcalibur
 * @notice SKYNT token with integrated Excalibur yield engine for SphinxOS
 * @dev ERC-20 token with staking, yield distribution, and Φ-score boosts.
 *
 * Excalibur Yield Engine features:
 *  - Stake SKYNT to earn yield at a configurable APR
 *  - Φ-score multiplier (200-1000) boosts yield up to 2×
 *  - Treasury fee on every yield claim
 *  - Emergency shutdown & rate-limiting guards
 *  - Owner-controlled APR management
 */

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

contract SKYNTExcalibur is ERC20, ERC20Burnable, Ownable, AccessControl, ReentrancyGuard, Pausable {

    // ========== ROLES ==========

    bytes32 public constant ADMIN_ROLE    = keccak256("ADMIN_ROLE");
    bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");

    // ========== CONSTANTS ==========

    uint256 public constant PHI_MIN = 200;
    uint256 public constant PHI_MAX = 1000;

    /// @dev 1-minute cooldown between user actions to deter flash-loan abuse
    uint256 public constant ACTION_COOLDOWN = 1 minutes;

    // ========== STRUCTS ==========

    struct StakePosition {
        uint256 stakedAmount;
        uint256 phiScore;       // 200-1000
        uint256 lastClaimTime;
        uint256 pendingYield;
    }

    // ========== STATE ==========

    /// @notice Yield APR in basis points (10 000 = 100 %).  Default 2 000 = 20 % APR.
    uint256 public excaliburAPR = 2000;

    /// @notice Maximum single-position stake (anti-whale guard)
    uint256 public maxStakePerUser = 10_000_000 * 1e18;

    /// @notice Treasury address receives a fee on every yield claim
    address public treasury;

    /// @notice Treasury fee in basis points.  Default 500 = 5 %.
    uint256 public treasuryFeeBPS = 500;

    bool public emergencyShutdown;

    mapping(address => StakePosition) public positions;
    mapping(address => uint256)       public lastActionTime;

    uint256 public totalStaked;

    // ========== EVENTS ==========

    event Staked(address indexed user, uint256 amount, uint256 phiScore);
    event Unstaked(address indexed user, uint256 amount, uint256 yieldClaimed);
    event YieldClaimed(address indexed user, uint256 yieldAmount, uint256 treasuryFee);
    event APRUpdated(uint256 oldAPR, uint256 newAPR);
    event TreasuryUpdated(address oldTreasury, address newTreasury);
    event EmergencyShutdownToggled(bool active);

    // ========== CONSTRUCTOR ==========

    /**
     * @param _treasury  Address that receives treasury fees
     * @param _mintSupply Initial SKYNT supply minted to deployer (18 decimals)
     */
    constructor(address _treasury, uint256 _mintSupply) ERC20("SKYNT Excalibur", "SKYNT") {
        require(_treasury != address(0), "SKYNT: zero treasury");

        treasury = _treasury;

        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
        _grantRole(OPERATOR_ROLE, msg.sender);

        if (_mintSupply > 0) {
            _mint(msg.sender, _mintSupply);
        }
    }

    // ========== MODIFIERS ==========

    modifier notShutdown() {
        require(!emergencyShutdown, "SKYNT: emergency shutdown");
        _;
    }

    modifier rateLimited() {
        require(
            block.timestamp >= lastActionTime[msg.sender] + ACTION_COOLDOWN,
            "SKYNT: rate limit"
        );
        lastActionTime[msg.sender] = block.timestamp;
        _;
    }

    // ========== EXCALIBUR YIELD ENGINE ==========

    /**
     * @notice Stake SKYNT tokens to start earning Excalibur yield.
     * @param amount   Amount of SKYNT to stake (wei)
     * @param phiScore Φ-score of the caller (200-1000)
     */
    function stake(uint256 amount, uint256 phiScore)
        external
        nonReentrant
        whenNotPaused
        notShutdown
        rateLimited
    {
        require(amount > 0, "SKYNT: zero amount");
        require(phiScore >= PHI_MIN && phiScore <= PHI_MAX, "SKYNT: invalid phi");

        StakePosition storage pos = positions[msg.sender];

        // Settle any pending yield before modifying the position
        if (pos.stakedAmount > 0) {
            pos.pendingYield += _accrued(msg.sender);
        }

        require(
            pos.stakedAmount + amount <= maxStakePerUser,
            "SKYNT: exceeds max stake"
        );

        _transfer(msg.sender, address(this), amount);

        pos.stakedAmount   += amount;
        pos.phiScore        = phiScore;
        pos.lastClaimTime   = block.timestamp;

        totalStaked += amount;

        emit Staked(msg.sender, amount, phiScore);
    }

    /**
     * @notice Unstake SKYNT tokens and claim all pending yield in one call.
     * @param amount Amount to unstake (0 = unstake everything)
     */
    function unstake(uint256 amount)
        external
        nonReentrant
        notShutdown
        rateLimited
    {
        StakePosition storage pos = positions[msg.sender];
        require(pos.stakedAmount > 0, "SKYNT: no stake");

        if (amount == 0) amount = pos.stakedAmount;
        require(amount <= pos.stakedAmount, "SKYNT: insufficient stake");

        // Accrue yield before changing balance
        uint256 yield = _accrued(msg.sender) + pos.pendingYield;

        pos.stakedAmount  -= amount;
        pos.pendingYield   = 0;
        pos.lastClaimTime  = block.timestamp;

        totalStaked -= amount;

        // Return principal
        _transfer(address(this), msg.sender, amount);

        // Distribute yield
        uint256 fee = _distributeYield(msg.sender, yield);

        emit Unstaked(msg.sender, amount, yield - fee);
    }

    /**
     * @notice Claim accumulated Excalibur yield without unstaking.
     */
    function claimYield()
        external
        nonReentrant
        notShutdown
        rateLimited
    {
        StakePosition storage pos = positions[msg.sender];
        require(pos.stakedAmount > 0, "SKYNT: no stake");

        uint256 yield = _accrued(msg.sender) + pos.pendingYield;
        require(yield > 0, "SKYNT: no yield");

        pos.pendingYield  = 0;
        pos.lastClaimTime = block.timestamp;

        uint256 fee = _distributeYield(msg.sender, yield);

        emit YieldClaimed(msg.sender, yield - fee, fee);
    }

    // ========== VIEW FUNCTIONS ==========

    /**
     * @notice Current pending yield for a user (without claiming).
     */
    function pendingYield(address user) external view returns (uint256) {
        StakePosition storage pos = positions[user];
        return _accrued(user) + pos.pendingYield;
    }

    // ========== INTERNAL ==========

    /**
     * @dev Yield accrued since last claim, boosted by Φ-score.
     *
     *  base  = stakedAmount × APR × Δt / 365d
     *  boost = 1 + (phi - 500) / 1000   (0.7× – 1.5×)
     */
    function _accrued(address user) internal view returns (uint256) {
        StakePosition storage pos = positions[user];
        if (pos.stakedAmount == 0) return 0;

        uint256 dt          = block.timestamp - pos.lastClaimTime;
        uint256 yearlyYield = (pos.stakedAmount * excaliburAPR) / 10_000;
        uint256 baseYield   = (yearlyYield * dt) / 365 days;

        // Φ boost: multiply by (10_000 + (phi - 500) * 7) / 10_000
        // phi=500 → ×1.00 ; phi=1000 → ×1.35 ; phi=200 → ×0.79
        int256 phiAdj = int256(pos.phiScore) - 500;
        int256 boostBPS = 10_000 + phiAdj * 7;
        if (boostBPS < 7_000) boostBPS = 7_000;   // floor ×0.7
        if (boostBPS > 17_000) boostBPS = 17_000; // cap  ×1.7

        return (baseYield * uint256(boostBPS)) / 10_000;
    }

    /**
     * @dev Mint yield tokens and split between user and treasury.
     * @return fee Amount sent to treasury
     */
    function _distributeYield(address user, uint256 amount) internal returns (uint256 fee) {
        if (amount == 0) return 0;

        fee = (amount * treasuryFeeBPS) / 10_000;
        uint256 userAmount = amount - fee;

        if (fee > 0)        _mint(treasury, fee);
        if (userAmount > 0) _mint(user, userAmount);
    }

    // ========== ADMIN ==========

    /**
     * @notice Update Excalibur APR (basis points).
     */
    function setAPR(uint256 newAPR) external onlyRole(ADMIN_ROLE) {
        require(newAPR <= 50_000, "SKYNT: APR too high"); // max 500 %
        emit APRUpdated(excaliburAPR, newAPR);
        excaliburAPR = newAPR;
    }

    /**
     * @notice Update treasury address.
     */
    function setTreasury(address newTreasury) external onlyRole(ADMIN_ROLE) {
        require(newTreasury != address(0), "SKYNT: zero address");
        emit TreasuryUpdated(treasury, newTreasury);
        treasury = newTreasury;
    }

    /**
     * @notice Update treasury fee (max 30 %).
     */
    function setTreasuryFee(uint256 bps) external onlyRole(ADMIN_ROLE) {
        require(bps <= 3000, "SKYNT: fee too high");
        treasuryFeeBPS = bps;
    }

    /**
     * @notice Update per-user stake cap.
     */
    function setMaxStakePerUser(uint256 cap) external onlyRole(ADMIN_ROLE) {
        maxStakePerUser = cap;
    }

    function activateEmergencyShutdown() external onlyRole(ADMIN_ROLE) {
        emergencyShutdown = true;
        _pause();
        emit EmergencyShutdownToggled(true);
    }

    function deactivateEmergencyShutdown() external onlyRole(ADMIN_ROLE) {
        emergencyShutdown = false;
        _unpause();
        emit EmergencyShutdownToggled(false);
    }

    function pause()   external onlyRole(ADMIN_ROLE) { _pause(); }
    function unpause() external onlyRole(ADMIN_ROLE) { _unpause(); }

    // ========== OVERRIDES ==========

    function supportsInterface(bytes4 interfaceId)
        public
        view
        override(AccessControl)
        returns (bool)
    {
        return super.supportsInterface(interfaceId);
    }
}
