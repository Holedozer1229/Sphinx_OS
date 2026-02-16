// ============================================================================
// yield_proof.circom — Yield Calculation Verification Circuit
// ============================================================================
//
// Verifies yield calculations for multi-token portfolios
// Integrates with SphinxSkynet Φ scores
// ============================================================================

pragma circom 2.0.0;

include "circomlib/circuits/comparators.circom";
include "circomlib/circuits/poseidon.circom";

// Multi-token yield calculation
template MultiTokenYield(num_tokens) {
    // Inputs
    signal input token_amounts[num_tokens];
    signal input token_aprs[num_tokens];      // APR * 100
    signal input phi_scores[num_tokens];       // Φ scores
    signal input phi_boosts[num_tokens];       // Φ boost multipliers * 1000
    
    // Outputs
    signal output total_yield;
    signal output weighted_apr;
    signal output yield_hash;
    
    // Calculate individual yields
    signal yields[num_tokens];
    signal total_capital;
    total_capital <== 0;
    
    var yield_sum = 0;
    var capital_sum = 0;
    var weighted_sum = 0;
    
    for (var i = 0; i < num_tokens; i++) {
        // Boosted APR
        signal boosted_apr;
        boosted_apr <== (token_aprs[i] * phi_boosts[i]) / 1000;
        
        // Individual yield
        yields[i] <== (token_amounts[i] * boosted_apr) / 10000;
        
        // Accumulate
        yield_sum += yields[i];
        capital_sum += token_amounts[i];
        weighted_sum += boosted_apr * token_amounts[i];
    }
    
    total_yield <== yield_sum;
    
    // Weighted average APR
    if (capital_sum > 0) {
        weighted_apr <== weighted_sum / capital_sum;
    } else {
        weighted_apr <== 0;
    }
    
    // Generate yield hash for verification
    component hasher = Poseidon(num_tokens + 2);
    for (var i = 0; i < num_tokens; i++) {
        hasher.inputs[i] <== yields[i];
    }
    hasher.inputs[num_tokens] <== total_yield;
    hasher.inputs[num_tokens + 1] <== weighted_apr;
    yield_hash <== hasher.out;
}

// Treasury split calculation (from economic model)
template TreasurySplit() {
    signal input reward;
    signal input phi_score;  // 200-1000
    signal output treasury_share;
    signal output user_payout;
    
    // Treasury rate: min(0.30, 0.05 + phi/2000)
    // Multiply by 1000 for fixed-point
    signal phi_factor;
    phi_factor <== phi_score / 2;  // phi/2000 * 1000
    
    signal treasury_rate;
    treasury_rate <== 50 + phi_factor;  // (0.05 + phi/2000) * 1000
    
    // Cap at 300 (0.30)
    component capper = LessThan(16);
    capper.in[0] <== treasury_rate;
    capper.in[1] <== 300;
    
    signal capped_rate;
    capped_rate <== capper.out * treasury_rate + (1 - capper.out) * 300;
    
    // Calculate split
    treasury_share <== (reward * capped_rate) / 1000;
    user_payout <== reward - treasury_share;
}

// Compound APY calculator
template CompoundAPY() {
    signal input apr;          // APR * 100
    signal input compounds;    // Number of compounds per year
    signal output apy;         // APY * 100
    
    // APY = (1 + APR/compounds)^compounds - 1
    // Simplified for circuit (approximation)
    // APY ≈ APR + (APR^2 / (2 * compounds))
    
    signal apr_squared;
    apr_squared <== apr * apr;
    
    signal compound_bonus;
    compound_bonus <== apr_squared / (2 * compounds * 100);
    
    apy <== apr + compound_bonus;
}

// Main yield proof circuit
template YieldProof(num_tokens) {
    // Inputs
    signal input token_amounts[num_tokens];
    signal input token_aprs[num_tokens];
    signal input phi_scores[num_tokens];
    signal input phi_boosts[num_tokens];
    signal input user_phi;  // User's overall Φ score
    
    // Outputs
    signal output total_yield;
    signal output treasury_amount;
    signal output user_amount;
    signal output compound_apy;
    signal output proof_hash;
    
    // 1. Calculate multi-token yield
    component yield_calc = MultiTokenYield(num_tokens);
    for (var i = 0; i < num_tokens; i++) {
        yield_calc.token_amounts[i] <== token_amounts[i];
        yield_calc.token_aprs[i] <== token_aprs[i];
        yield_calc.phi_scores[i] <== phi_scores[i];
        yield_calc.phi_boosts[i] <== phi_boosts[i];
    }
    
    total_yield <== yield_calc.total_yield;
    
    // 2. Calculate treasury split
    component split = TreasurySplit();
    split.reward <== total_yield;
    split.phi_score <== user_phi;
    
    treasury_amount <== split.treasury_share;
    user_amount <== split.user_payout;
    
    // 3. Calculate compound APY
    component apy_calc = CompoundAPY();
    apy_calc.apr <== yield_calc.weighted_apr;
    apy_calc.compounds <== 12;  // Monthly compounding
    
    compound_apy <== apy_calc.apy;
    
    // 4. Generate proof hash
    component hasher = Poseidon(6);
    hasher.inputs[0] <== total_yield;
    hasher.inputs[1] <== treasury_amount;
    hasher.inputs[2] <== user_amount;
    hasher.inputs[3] <== compound_apy;
    hasher.inputs[4] <== user_phi;
    hasher.inputs[5] <== yield_calc.yield_hash;
    proof_hash <== hasher.out;
}

component main {public [user_phi]} = YieldProof(5);
