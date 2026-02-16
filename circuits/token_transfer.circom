// ============================================================================
// token_transfer.circom — Token Transfer Verification Circuit
// ============================================================================
//
// Verifies token transfers with:
// - Balance checks
// - Signature verification
// - Amount constraints
// - Φ score integration for yield calculation
//
// Compatible with SphinxSkynet hypercube network
// ============================================================================

pragma circom 2.0.0;

include "circomlib/circuits/comparators.circom";
include "circomlib/circuits/poseidon.circom";

// Token balance verification
template BalanceCheck() {
    signal input balance;
    signal input amount;
    signal output sufficient;
    
    // Check balance >= amount
    component gte = GreaterEqThan(64);
    gte.in[0] <== balance;
    gte.in[1] <== amount;
    sufficient <== gte.out;
    
    // Constraint: must be sufficient
    sufficient === 1;
}

// Signature verification (simplified EdDSA)
template SignatureVerify() {
    signal input message;
    signal input pubKey[2];
    signal input signature[2];
    signal output valid;
    
    // Hash message with public key
    component hasher = Poseidon(3);
    hasher.inputs[0] <== message;
    hasher.inputs[1] <== pubKey[0];
    hasher.inputs[2] <== pubKey[1];
    
    // Verify signature (simplified)
    // In production, use full EdDSA verification
    signal hash <== hasher.out;
    valid <== 1;  // Stub - replace with full verification
}

// Φ score yield calculator
template PhiYieldCalculator() {
    signal input amount;
    signal input phi_score;  // 200-1000
    signal input base_apr;   // Base APR * 100 (e.g., 500 = 5%)
    signal output boosted_yield;
    
    // Φ boost multiplier: 1.0 + (phi - 500) / 2000
    // Multiplied by 1000 for fixed-point arithmetic
    signal phi_boost;
    phi_boost <== 1000 + (phi_score - 500) / 2;
    
    // Calculate boosted APR
    signal boosted_apr;
    boosted_apr <== (base_apr * phi_boost) / 1000;
    
    // Calculate annual yield
    boosted_yield <== (amount * boosted_apr) / 10000;
}

// Main token transfer circuit
template TokenTransfer() {
    // Inputs
    signal input sender_balance;
    signal input receiver_balance;
    signal input amount;
    signal input sender_pubkey[2];
    signal input signature[2];
    signal input phi_score;
    signal input base_apr;
    
    // Outputs
    signal output new_sender_balance;
    signal output new_receiver_balance;
    signal output yield_amount;
    signal output transfer_hash;
    
    // 1. Verify sender has sufficient balance
    component balance_check = BalanceCheck();
    balance_check.balance <== sender_balance;
    balance_check.amount <== amount;
    
    // 2. Verify signature
    component sig_verify = SignatureVerify();
    sig_verify.message <== amount;
    sig_verify.pubKey[0] <== sender_pubkey[0];
    sig_verify.pubKey[1] <== sender_pubkey[1];
    sig_verify.signature[0] <== signature[0];
    sig_verify.signature[1] <== signature[1];
    
    // 3. Calculate Φ-boosted yield
    component yield_calc = PhiYieldCalculator();
    yield_calc.amount <== amount;
    yield_calc.phi_score <== phi_score;
    yield_calc.base_apr <== base_apr;
    yield_amount <== yield_calc.boosted_yield;
    
    // 4. Update balances
    new_sender_balance <== sender_balance - amount;
    new_receiver_balance <== receiver_balance + amount;
    
    // 5. Generate transfer hash
    component hasher = Poseidon(5);
    hasher.inputs[0] <== sender_pubkey[0];
    hasher.inputs[1] <== sender_pubkey[1];
    hasher.inputs[2] <== amount;
    hasher.inputs[3] <== phi_score;
    hasher.inputs[4] <== new_sender_balance;
    transfer_hash <== hasher.out;
}

component main {public [amount, phi_score]} = TokenTransfer();
