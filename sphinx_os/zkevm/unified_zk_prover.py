#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZK Proof Generator for:
- Spectral Entropy
- ELPR (Lieb-Robinson bound)
- Tetraroot + STX
- NFT Rarity
"""

import json
import subprocess
import tempfile
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from mpmath import mp, zetazero

mp.dps = 100  # high precision for zeta zeros

@dataclass
class UnifiedProof:
    request_id: int
    entropy: int
    tetraroot_entropy: int
    nft_rarity: int
    elpr_satisfied: bool
    spectral_commitment: str
    proof: Dict
    public_inputs: List[int]

class UnifiedZKProver:
    """Generates full unified Groth16 proofs with ELPR enforcement."""

    def __init__(self, circuit_dir: Path = Path("circuits/build")):
        self.circuit_dir = circuit_dir
        self.wasm_path = circuit_dir / "spectral_entropy_elpr_js" / "spectral_entropy_elpr.wasm"
        self.zkey_path = circuit_dir / "spectral_entropy_elpr.zkey"

        self.SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        self.WEIGHTS = [1,2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67]

        self.KAPPA_FP = 1095  # fixed-point 1.059 * 1000

    def generate_spectral_bins(self, n_bins: int = 20) -> List[int]:
        """Generate spectral bins from Riemann zeta zeros"""
        bins = []
        for k in range(1, n_bins + 1):
            zero = zetazero(k)
            imag_part = float(zero.imag)
            scaled = imag_part * 1e12
            quantized = int(scaled) % (1 << 32)
            bins.append(quantized)
        return bins

    def poseidon_hash(self, inputs: List[int]) -> int:
        """Poseidon hash placeholder"""
        import hashlib
        data = b''.join(i.to_bytes(32, 'big') for i in inputs)
        return int.from_bytes(hashlib.sha256(data).digest(), 'big')

    def compute_entropy(self, bins: List[int]) -> Tuple[int, int]:
        """Compute spectral entropy"""
        raw_sum = sum(bin_val * weight for bin_val, weight in zip(bins, self.WEIGHTS[:len(bins)]))
        entropy = raw_sum % self.SECP256K1_N
        return entropy, raw_sum

    def compute_elpr_threshold(self, operator_norm: int, distance: int) -> int:
        """Compute ELPR threshold"""
        return operator_norm * 1000 // (self.KAPPA_FP ** distance)

    def generate_witness(self, request_id: int,
                         operator_norm: int = 123456,
                         projected_norm: int = 7890,
                         distance: int = 5,
                         tetraroot_entropy: int = 10000,
                         nft_rarity: int = 5000,
                         domain_separator: int = 1) -> Dict:
        """Generate full witness for the ELPR + Spectral + Tetraroot + NFT proof"""
        bins = self.generate_spectral_bins()
        entropy, raw_sum = self.compute_entropy(bins)
        commitment = self.poseidon_hash(bins)
        elpr_satisfied = projected_norm <= self.compute_elpr_threshold(operator_norm, distance)

        return {
            "requestId": request_id,
            "spectralCommitment": str(commitment),
            "entropyOut": str(entropy),
            "domainSeparator": str(domain_separator),
            "curveOrder": str(self.SECP256K1_N),
            "spectralBins": [str(b) for b in bins],
            "rawSum": str(raw_sum),
            "operatorNorm": operator_norm,
            "projectedNorm": projected_norm,
            "distance": distance,
            "tetrarootEntropy": tetraroot_entropy,
            "stxCommitment": tetraroot_entropy + 1,  # placeholder
            "nftRarity": nft_rarity
        }

    def generate_proof(self, request_id: int, **kwargs) -> UnifiedProof:
        """Generate Groth16 proof"""
        witness_data = self.generate_witness(request_id, **kwargs)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(witness_data, f)
            input_path = f.name

        try:
            # Generate witness
            witness_cmd = [
                "node", str(self.wasm_path.parent / "generate_witness.js"),
                str(self.wasm_path),
                input_path,
                "witness.json"
            ]
            subprocess.run(witness_cmd, check=True, capture_output=True)

            # Generate proof
            proof_cmd = [
                "snarkjs", "groth16", "prove",
                str(self.zkey_path),
                "witness.json",
                "proof.json",
                "public.json"
            ]
            subprocess.run(proof_cmd, check=True, capture_output=True)

            # Load proof & public inputs
            with open("proof.json", "r") as f:
                proof = json.load(f)
            with open("public.json", "r") as f:
                public_inputs = json.load(f)

            return UnifiedProof(
                request_id=request_id,
                entropy=int(witness_data["entropyOut"]),
                tetraroot_entropy=int(witness_data["tetrarootEntropy"]),
                nft_rarity=int(witness_data["nftRarity"]),
                elpr_satisfied=kwargs.get("projected_norm", 0) <= self.compute_elpr_threshold(
                    kwargs.get("operator_norm", 0), kwargs.get("distance", 0)),
                spectral_commitment=witness_data["spectralCommitment"],
                proof=proof,
                public_inputs=public_inputs
            )
        finally:
            for f in [input_path, "witness.json", "proof.json", "public.json"]:
                if os.path.exists(f):
                    os.remove(f)
