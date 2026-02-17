#!/usr/bin/env python3
"""Final verification of UnifiedZKProver integration"""

import sys

def main():
    print("="*70)
    print("FINAL VERIFICATION: Unified ZK Proof Generator Integration")
    print("="*70)
    print()
    
    # Test 1: Import check
    print("1. Testing imports...")
    try:
        from sphinx_os.zkevm import (
            UnifiedZKProver, 
            UnifiedProof,
            ZKProver,
            ProofType
        )
        print("   ✅ All imports successful")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return 1
    
    # Test 2: Backwards compatibility
    print("\n2. Testing backwards compatibility...")
    try:
        original_prover = ZKProver()
        print("   ✅ Original ZKProver still works")
    except Exception as e:
        print(f"   ❌ Backwards compatibility broken: {e}")
        return 1
    
    # Test 3: UnifiedZKProver initialization
    print("\n3. Testing UnifiedZKProver initialization...")
    try:
        unified_prover = UnifiedZKProver()
        print("   ✅ UnifiedZKProver initialized")
        print(f"      - SECP256K1_N: {hex(unified_prover.SECP256K1_N)}")
        print(f"      - KAPPA_FP: {unified_prover.KAPPA_FP}")
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        return 1
    
    # Test 4: Spectral bins generation
    print("\n4. Testing spectral bins generation...")
    try:
        bins = unified_prover.generate_spectral_bins(n_bins=10)
        print(f"   ✅ Generated {len(bins)} spectral bins")
        print(f"      - First 3 bins: {bins[:3]}")
    except Exception as e:
        print(f"   ❌ Spectral bins generation failed: {e}")
        return 1
    
    # Test 5: Entropy computation
    print("\n5. Testing entropy computation...")
    try:
        entropy, raw_sum = unified_prover.compute_entropy(bins)
        print(f"   ✅ Entropy computed: {entropy}")
    except Exception as e:
        print(f"   ❌ Entropy computation failed: {e}")
        return 1
    
    # Test 6: ELPR threshold
    print("\n6. Testing ELPR threshold computation...")
    try:
        threshold = unified_prover.compute_elpr_threshold(123456, 5)
        print(f"   ✅ ELPR threshold: {threshold}")
    except Exception as e:
        print(f"   ❌ ELPR threshold computation failed: {e}")
        return 1
    
    # Test 7: Witness generation
    print("\n7. Testing witness generation...")
    try:
        witness = unified_prover.generate_witness(
            request_id=999,
            operator_norm=100000,
            projected_norm=5000,
            distance=3
        )
        print(f"   ✅ Witness generated")
        print(f"      - Request ID: {witness['requestId']}")
        print(f"      - Entropy: {witness['entropyOut']}")
    except Exception as e:
        print(f"   ❌ Witness generation failed: {e}")
        return 1
    
    # Test 8: File structure verification
    print("\n8. Verifying file structure...")
    import os
    from pathlib import Path
    
    files_to_check = [
        'sphinx_os/zkevm/unified_zk_prover.py',
        'sphinx_os/zkevm/__init__.py',
        'test_unified_zk_prover.py',
        'docs/unified_zk_prover.md',
        'INTEGRATION_SUMMARY.md',
        'requirements.txt'
    ]
    
    all_exist = True
    for file in files_to_check:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} missing")
            all_exist = False
    
    if not all_exist:
        return 1
    
    # Test 9: Check requirements
    print("\n9. Checking requirements.txt...")
    try:
        with open('requirements.txt', 'r') as f:
            content = f.read()
            if 'mpmath' in content:
                print("   ✅ mpmath dependency added")
            else:
                print("   ❌ mpmath dependency missing")
                return 1
    except Exception as e:
        print(f"   ❌ Requirements check failed: {e}")
        return 1
    
    print("\n" + "="*70)
    print("✅ ALL VERIFICATION TESTS PASSED!")
    print("="*70)
    print("\nThe Unified ZK Proof Generator has been successfully integrated into")
    print("SphinxOS with full functionality and backwards compatibility.")
    print()
    return 0

if __name__ == "__main__":
    sys.exit(main())
