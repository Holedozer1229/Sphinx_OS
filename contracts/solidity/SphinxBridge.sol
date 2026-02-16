// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title SphinxBridge
 * @dev Cross-chain bridge for SphinxSkynet Blockchain
 * Supports lock/mint and burn/release mechanisms with multi-sig validation
 */
contract SphinxBridge {
    // Bridge status
    enum Status { Pending, Locked, Minted, Burned, Released, Failed }
    
    // Bridge transaction
    struct BridgeTransaction {
        bytes32 txHash;
        address sender;
        address recipient;
        uint256 amount;
        string sourceChain;
        string destinationChain;
        Status status;
        uint256 timestamp;
        uint8 signatures;
    }
    
    // State variables
    mapping(bytes32 => BridgeTransaction) public transactions;
    mapping(address => uint256) public lockedBalances;
    mapping(address => uint256) public wrappedBalances;
    mapping(address => bool) public guardians;
    
    address[] public guardianList;
    uint8 public constant REQUIRED_SIGNATURES = 5;
    uint8 public constant TOTAL_GUARDIANS = 9;
    uint256 public constant BRIDGE_FEE = 1; // 0.1% (in basis points: 1/1000)
    
    // Events
    event TokensLocked(bytes32 indexed txHash, address indexed sender, uint256 amount, string destinationChain);
    event TokensMinted(bytes32 indexed txHash, address indexed recipient, uint256 amount);
    event TokensBurned(bytes32 indexed txHash, address indexed sender, uint256 amount, string destinationChain);
    event TokensReleased(bytes32 indexed txHash, address indexed recipient, uint256 amount);
    event GuardianAdded(address indexed guardian);
    event GuardianRemoved(address indexed guardian);
    
    // Modifiers
    modifier onlyGuardian() {
        require(guardians[msg.sender], "Not a guardian");
        _;
    }
    
    modifier validAmount(uint256 amount) {
        require(amount > 0, "Amount must be positive");
        _;
    }
    
    /**
     * @dev Constructor - initialize guardians
     * @param _guardians Array of guardian addresses
     */
    constructor(address[] memory _guardians) {
        require(_guardians.length == TOTAL_GUARDIANS, "Must have exactly 9 guardians");
        
        for (uint i = 0; i < _guardians.length; i++) {
            require(_guardians[i] != address(0), "Invalid guardian address");
            guardians[_guardians[i]] = true;
            guardianList.push(_guardians[i]);
        }
    }
    
    /**
     * @dev Lock tokens for bridging
     * @param destinationChain Target blockchain
     * @param recipient Recipient address on destination chain
     */
    function lockTokens(
        string memory destinationChain,
        address recipient
    ) external payable validAmount(msg.value) returns (bytes32) {
        // Calculate fee
        uint256 fee = (msg.value * BRIDGE_FEE) / 1000;
        uint256 netAmount = msg.value - fee;
        
        // Create transaction hash
        bytes32 txHash = keccak256(abi.encodePacked(
            msg.sender,
            recipient,
            msg.value,
            destinationChain,
            block.timestamp
        ));
        
        // Create bridge transaction
        transactions[txHash] = BridgeTransaction({
            txHash: txHash,
            sender: msg.sender,
            recipient: recipient,
            amount: netAmount,
            sourceChain: "ethereum",
            destinationChain: destinationChain,
            status: Status.Locked,
            timestamp: block.timestamp,
            signatures: 0
        });
        
        // Update locked balance
        lockedBalances[msg.sender] += netAmount;
        
        emit TokensLocked(txHash, msg.sender, netAmount, destinationChain);
        
        return txHash;
    }
    
    /**
     * @dev Mint wrapped tokens (guardian only)
     * @param txHash Bridge transaction hash
     */
    function mintTokens(bytes32 txHash) external onlyGuardian {
        BridgeTransaction storage tx = transactions[txHash];
        
        require(tx.status == Status.Locked, "Invalid status");
        require(tx.signatures < REQUIRED_SIGNATURES, "Already processed");
        
        tx.signatures++;
        
        if (tx.signatures >= REQUIRED_SIGNATURES) {
            tx.status = Status.Minted;
            wrappedBalances[tx.recipient] += tx.amount;
            
            emit TokensMinted(txHash, tx.recipient, tx.amount);
        }
    }
    
    /**
     * @dev Burn wrapped tokens
     * @param amount Amount to burn
     * @param destinationChain Target blockchain
     * @param recipient Recipient address
     */
    function burnTokens(
        uint256 amount,
        string memory destinationChain,
        address recipient
    ) external validAmount(amount) returns (bytes32) {
        require(wrappedBalances[msg.sender] >= amount, "Insufficient balance");
        
        // Calculate fee
        uint256 fee = (amount * BRIDGE_FEE) / 1000;
        uint256 netAmount = amount - fee;
        
        // Burn tokens
        wrappedBalances[msg.sender] -= amount;
        
        // Create transaction hash
        bytes32 txHash = keccak256(abi.encodePacked(
            msg.sender,
            recipient,
            amount,
            destinationChain,
            block.timestamp
        ));
        
        // Create bridge transaction
        transactions[txHash] = BridgeTransaction({
            txHash: txHash,
            sender: msg.sender,
            recipient: recipient,
            amount: netAmount,
            sourceChain: "sphinx",
            destinationChain: destinationChain,
            status: Status.Burned,
            timestamp: block.timestamp,
            signatures: 0
        });
        
        emit TokensBurned(txHash, msg.sender, netAmount, destinationChain);
        
        return txHash;
    }
    
    /**
     * @dev Release locked tokens (guardian only)
     * @param txHash Bridge transaction hash
     */
    function releaseTokens(bytes32 txHash) external onlyGuardian {
        BridgeTransaction storage tx = transactions[txHash];
        
        require(tx.status == Status.Burned, "Invalid status");
        require(tx.signatures < REQUIRED_SIGNATURES, "Already processed");
        
        tx.signatures++;
        
        if (tx.signatures >= REQUIRED_SIGNATURES) {
            tx.status = Status.Released;
            
            // Release tokens
            require(lockedBalances[tx.sender] >= tx.amount, "Insufficient locked balance");
            lockedBalances[tx.sender] -= tx.amount;
            
            payable(tx.recipient).transfer(tx.amount);
            
            emit TokensReleased(txHash, tx.recipient, tx.amount);
        }
    }
    
    /**
     * @dev Get transaction status
     * @param txHash Transaction hash
     */
    function getTransactionStatus(bytes32 txHash) external view returns (
        address sender,
        address recipient,
        uint256 amount,
        Status status,
        uint8 signatures
    ) {
        BridgeTransaction memory tx = transactions[txHash];
        return (tx.sender, tx.recipient, tx.amount, tx.status, tx.signatures);
    }
    
    /**
     * @dev Get wrapped balance
     * @param account Account address
     */
    function getWrappedBalance(address account) external view returns (uint256) {
        return wrappedBalances[account];
    }
    
    /**
     * @dev Get locked balance
     * @param account Account address
     */
    function getLockedBalance(address account) external view returns (uint256) {
        return lockedBalances[account];
    }
}
