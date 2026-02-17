#!/usr/bin/env python3
"""
Test script for UnifiedZKProver basic functionality
"""

from sphinx_os.zkevm.unified_zk_prover import UnifiedZKProver, UnifiedProof

def test_basic_initialization():
    """Test that UnifiedZKProver can be initialized"""
    print("Testing UnifiedZKProver initialization...")
    prover = UnifiedZKProver()
    print(f"✓ UnifiedZKProver initialized successfully")
    print(f"  - Circuit directory: {prover.circuit_dir}")
    print(f"  - SECP256K1_N: {hex(prover.SECP256K1_N)}")
    print(f"  - KAPPA_FP: {prover.KAPPA_FP}")
    print(f"  - Weights: {prover.WEIGHTS}")
    return prover

def test_spectral_bins(prover):
    """Test spectral bins generation from Riemann zeta zeros"""
    print("\nTesting spectral bins generation...")
    bins = prover.generate_spectral_bins(n_bins=5)
    print(f"✓ Generated {len(bins)} spectral bins")
    print(f"  - First 5 bins: {bins[:5]}")
    return bins

def test_entropy_computation(prover, bins):
    """Test entropy computation"""
    print("\nTesting entropy computation...")
    entropy, raw_sum = prover.compute_entropy(bins)
    print(f"✓ Computed entropy: {entropy}")
    print(f"  - Raw sum: {raw_sum}")
    return entropy, raw_sum

def test_poseidon_hash(prover, bins):
    """Test Poseidon hash (placeholder)"""
    print("\nTesting Poseidon hash...")
    commitment = prover.poseidon_hash(bins)
    print(f"✓ Computed commitment: {commitment}")
    return commitment

def test_elpr_threshold(prover):
    """Test ELPR threshold computation"""
    print("\nTesting ELPR threshold computation...")
    operator_norm = 123456
    distance = 5
    threshold = prover.compute_elpr_threshold(operator_norm, distance)
    print(f"✓ ELPR threshold computed: {threshold}")
    print(f"  - Operator norm: {operator_norm}")
    print(f"  - Distance: {distance}")
    return threshold

def test_witness_generation(prover):
    """Test witness generation"""
    print("\nTesting witness generation...")
    witness = prover.generate_witness(
        request_id=1,
        operator_norm=123456,
        projected_norm=7890,
        distance=5,
        tetraroot_entropy=10000,
        nft_rarity=5000
    )
    print(f"✓ Witness generated successfully")
    print(f"  - Request ID: {witness['requestId']}")
    print(f"  - Entropy: {witness['entropyOut']}")
    print(f"  - Tetraroot Entropy: {witness['tetrarootEntropy']}")
    print(f"  - NFT Rarity: {witness['nftRarity']}")
    print(f"  - Number of spectral bins: {len(witness['spectralBins'])}")
    return witness

def main():
    """Run all tests"""
    print("="*60)
    print("UnifiedZKProver Test Suite")
    print("="*60)
    
    try:
        # Test initialization
        prover = test_basic_initialization()
        
        # Test spectral bins generation
        bins = test_spectral_bins(prover)
        
        # Test entropy computation
        entropy, raw_sum = test_entropy_computation(prover, bins)
        
        # Test Poseidon hash
        commitment = test_poseidon_hash(prover, bins)
        
        # Test ELPR threshold
        threshold = test_elpr_threshold(prover)
        
        # Test witness generation
        witness = test_witness_generation(prover)
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
        print("\nNote: Full proof generation requires circom circuits and snarkjs,")
        print("which are not tested here but are part of the complete workflow.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
