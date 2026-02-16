"""
Tests for Bridge Auto-Deploy System
"""
import pytest
import os
import json
from sphinx_os.bridge.auto_deploy import deploy_bridge, save_deployment_info, _simulate_deployment


def test_save_deployment_info(tmp_path):
    """Test saving deployment information"""
    # Change to temp directory for testing
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Save deployment info
        save_deployment_info("polygon", "0xabcdef123456", "0xtxhash123456")
        
        # Check file was created
        assert os.path.exists("deployments.json")
        
        # Check content
        with open("deployments.json", "r") as f:
            data = json.load(f)
        
        assert "polygon" in data
        assert data["polygon"]["address"] == "0xabcdef123456"
        assert data["polygon"]["tx_hash"] == "0xtxhash123456"
        assert "deployed_at" in data["polygon"]
    finally:
        os.chdir(original_dir)


def test_save_multiple_deployments(tmp_path):
    """Test saving multiple deployment records"""
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Save multiple deployments
        save_deployment_info("polygon", "0xpoly123", "0xtxpoly")
        save_deployment_info("avalanche", "0xavax456", "0xtxavax")
        save_deployment_info("bnb", "0xbnb789", "0xtxbnb")
        
        # Check all were saved
        with open("deployments.json", "r") as f:
            data = json.load(f)
        
        assert len(data) == 3
        assert "polygon" in data
        assert "avalanche" in data
        assert "bnb" in data
    finally:
        os.chdir(original_dir)


def test_simulate_deployment(tmp_path):
    """Test simulated deployment"""
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Simulate deployment
        result = _simulate_deployment("polygon")
        
        assert result is True
        
        # Check deployment info was saved
        assert os.path.exists("deployments.json")
        
        with open("deployments.json", "r") as f:
            data = json.load(f)
        
        assert "polygon" in data
        assert data["polygon"]["address"].startswith("0x")
        assert data["polygon"]["tx_hash"].startswith("0x")
    finally:
        os.chdir(original_dir)


def test_deploy_bridge_without_dependencies(tmp_path):
    """Test bridge deployment without web3 dependencies (should simulate)"""
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Deploy should work even without dependencies (falls back to simulation)
        result = deploy_bridge("polygon", 0.1)
        
        # Should succeed via simulation
        assert result is True
        
        # Check deployment was recorded
        assert os.path.exists("deployments.json")
    finally:
        os.chdir(original_dir)


def test_deploy_bridge_invalid_chain(tmp_path):
    """Test deployment with invalid chain"""
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        result = deploy_bridge("invalid_chain", 0.1)
        assert result is False
    finally:
        os.chdir(original_dir)


def test_deploy_bridge_all_chains(tmp_path):
    """Test deployment to all supported chains"""
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        chains = ["polygon", "avalanche", "bnb", "ethereum"]
        
        for chain in chains:
            result = deploy_bridge(chain, 0.1)
            assert result is True
        
        # Check all were recorded
        with open("deployments.json", "r") as f:
            data = json.load(f)
        
        for chain in chains:
            assert chain in data
    finally:
        os.chdir(original_dir)


def test_deployment_info_persistence(tmp_path):
    """Test deployment info persists across saves"""
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Save first deployment
        save_deployment_info("polygon", "0xpoly", "0xtxpoly")
        
        # Save second deployment
        save_deployment_info("avalanche", "0xavax", "0xtxavax")
        
        # First deployment should still exist
        with open("deployments.json", "r") as f:
            data = json.load(f)
        
        assert "polygon" in data
        assert "avalanche" in data
        assert data["polygon"]["address"] == "0xpoly"
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
