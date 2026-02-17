# UnifiedZKProver Documentation

## Overview

The `UnifiedZKProver` is a specialized zero-knowledge proof generator that combines multiple cryptographic and mathematical components into unified Groth16 proofs:

- **Spectral Entropy**: Generated from Riemann zeta function zeros
- **ELPR (Extended Lieb-Robinson bound)**: Quantum information propagation constraints
- **Tetraroot + STX**: Tetraroot entropy calculations with STX commitments
- **NFT Rarity**: Rarity score proofs for NFTs

## Key Features

1. **High-Precision Riemann Zeta Zeros**: Uses `mpmath` library with 100 decimal places precision to compute spectral bins from the imaginary parts of Riemann zeta zeros.

2. **ELPR Enforcement**: Validates that projected operator norms satisfy the Extended Lieb-Robinson bound based on distance and operator norms.

3. **Cryptographic Commitments**: Generates Poseidon-style hash commitments for spectral bins.

4. **Modular Witness Generation**: Supports flexible witness generation for various proof scenarios.

## Installation

The `UnifiedZKProver` requires the following dependencies (already added to `requirements.txt`):

```bash
pip install mpmath numpy
```

## Usage

### Basic Initialization

```python
from sphinx_os.zkevm import UnifiedZKProver, UnifiedProof

# Initialize with default circuit directory
prover = UnifiedZKProver()

# Or specify a custom circuit directory
prover = UnifiedZKProver(circuit_dir=Path("/path/to/circuits"))
```

### Generate Spectral Bins

```python
# Generate 20 spectral bins from Riemann zeta zeros
bins = prover.generate_spectral_bins(n_bins=20)
print(f"Generated bins: {bins}")
```

### Compute Entropy

```python
# Compute spectral entropy from bins
entropy, raw_sum = prover.compute_entropy(bins)
print(f"Entropy: {entropy}")
print(f"Raw sum: {raw_sum}")
```

### Generate Witness

```python
# Generate a complete witness for proof generation
witness = prover.generate_witness(
    request_id=1,
    operator_norm=123456,
    projected_norm=7890,
    distance=5,
    tetraroot_entropy=10000,
    nft_rarity=5000,
    domain_separator=1
)
```

### Generate Complete Proof

**Note**: Full proof generation requires:
- Compiled Circom circuits (`.wasm` files)
- Trusted setup keys (`.zkey` files)
- SnarkJS and Node.js installed

```python
# Generate a complete Groth16 proof
proof = prover.generate_proof(
    request_id=1,
    operator_norm=123456,
    projected_norm=7890,
    distance=5,
    tetraroot_entropy=10000,
    nft_rarity=5000
)

# Access proof components
print(f"Request ID: {proof.request_id}")
print(f"Entropy: {proof.entropy}")
print(f"ELPR Satisfied: {proof.elpr_satisfied}")
print(f"Commitment: {proof.spectral_commitment}")
```

## Architecture

### Constants

- **SECP256K1_N**: The order of the secp256k1 elliptic curve
- **WEIGHTS**: Prime-based weights `[1,2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67]`
- **KAPPA_FP**: Fixed-point constant (1.059 * 1000) for ELPR threshold calculations

### Methods

#### `generate_spectral_bins(n_bins: int = 20) -> List[int]`
Generates spectral bins from the imaginary parts of Riemann zeta zeros.

#### `poseidon_hash(inputs: List[int]) -> int`
Computes a Poseidon-style hash commitment (currently using SHA256 as placeholder).

#### `compute_entropy(bins: List[int]) -> Tuple[int, int]`
Computes spectral entropy using weighted bins modulo SECP256K1_N.

#### `compute_elpr_threshold(operator_norm: int, distance: int) -> int`
Computes the ELPR (Extended Lieb-Robinson) threshold based on operator norm and distance.

#### `generate_witness(request_id: int, **kwargs) -> Dict`
Generates a complete witness including all components needed for proof generation.

#### `generate_proof(request_id: int, **kwargs) -> UnifiedProof`
Generates a complete Groth16 proof using the witness and circuit infrastructure.

## Data Classes

### `UnifiedProof`

A dataclass containing the complete proof result:

```python
@dataclass
class UnifiedProof:
    request_id: int              # Request identifier
    entropy: int                 # Computed spectral entropy
    tetraroot_entropy: int       # Tetraroot entropy value
    nft_rarity: int             # NFT rarity score
    elpr_satisfied: bool         # ELPR constraint satisfaction
    spectral_commitment: str     # Cryptographic commitment
    proof: Dict                  # Groth16 proof object
    public_inputs: List[int]     # Public inputs for verification
```

## Circuit Requirements

For full proof generation, you need:

1. **Circuit Files**: 
   - `circuits/build/spectral_entropy_elpr_js/spectral_entropy_elpr.wasm`
   - `circuits/build/spectral_entropy_elpr_js/generate_witness.js`

2. **Setup Keys**:
   - `circuits/build/spectral_entropy_elpr.zkey`

3. **External Tools**:
   - Node.js for witness generation
   - SnarkJS for proof generation

## Testing

Run the test suite to verify functionality:

```bash
python3 test_unified_zk_prover.py
```

This tests:
- Initialization
- Spectral bins generation
- Entropy computation
- Poseidon hash
- ELPR threshold calculation
- Witness generation

## Integration with SphinxOS

The `UnifiedZKProver` is integrated into the `sphinx_os.zkevm` module and can be imported alongside other ZK components:

```python
from sphinx_os.zkevm import (
    UnifiedZKProver,
    UnifiedProof,
    ZKProver,
    ProofType
)
```

## Security Considerations

1. **Poseidon Hash**: The current implementation uses SHA256 as a placeholder. For production use, implement a proper Poseidon hash function.

2. **Circuit Security**: Ensure circuits are properly audited before production use.

3. **Trusted Setup**: Use a secure multi-party computation ceremony for generating trusted setup parameters.

4. **Input Validation**: Always validate inputs before generating proofs to prevent malicious data.

## Future Enhancements

- Implement native Poseidon hash function
- Add proof verification capabilities
- Support for additional proof systems (PLONK, STARKs)
- Batch proof generation
- Recursive proof composition
- Integration with on-chain verifiers

## License

Part of SphinxOS - see main repository LICENSE file.
