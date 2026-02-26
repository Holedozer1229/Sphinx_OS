"""
Tests for Treasury API
"""
import pytest
from fastapi.testclient import TestClient
from sphinx_os.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "features" in data


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_get_treasury_stats():
    """Test getting treasury stats"""
    response = client.get("/api/treasury/stats")
    assert response.status_code == 200
    
    data = response.json()
    assert "balance_sphinx" in data
    assert "balance_usd" in data
    assert "deployments" in data
    
    # Check deployment chains
    assert "polygon" in data["deployments"]
    assert "avalanche" in data["deployments"]
    assert "bnb" in data["deployments"]
    assert "ethereum" in data["deployments"]


def test_get_deployment_status():
    """Test getting deployment status"""
    response = client.get("/api/treasury/deployments")
    assert response.status_code == 200
    
    data = response.json()
    assert "deployments" in data
    assert "treasury_balance" in data


def test_collect_nft_mint_fee():
    """Test collecting NFT mint fee"""
    response = client.post("/api/treasury/collect/nft_mint?amount=0.1")
    assert response.status_code == 200
    
    data = response.json()
    assert data["success"] is True
    assert "distribution" in data
    assert "new_balance" in data
    
    # Check fee distribution
    distribution = data["distribution"]
    assert distribution["treasury"] == pytest.approx(0.07)
    assert distribution["operator"] == pytest.approx(0.02)
    assert distribution["rewards"] == pytest.approx(0.01)


def test_collect_rarity_proof_fee():
    """Test collecting rarity proof fee"""
    response = client.post("/api/treasury/collect/rarity_proof?amount=0.05")
    assert response.status_code == 200
    
    data = response.json()
    assert data["success"] is True
    assert "distribution" in data
    assert "new_balance" in data
    
    # Check fee distribution
    distribution = data["distribution"]
    assert distribution["treasury"] == pytest.approx(0.04)
    assert distribution["operator"] == pytest.approx(0.0075)
    assert distribution["miners"] == pytest.approx(0.0025)


def test_manual_deploy_invalid_chain():
    """Test manual deployment with invalid chain"""
    response = client.post("/api/treasury/deploy/invalid_chain")
    assert response.status_code == 400
    assert "Invalid chain" in response.json()["detail"]


def test_manual_deploy_insufficient_funds():
    """Test manual deployment with insufficient funds"""
    # Try to deploy polygon (needs $50) without enough funds
    response = client.post("/api/treasury/deploy/polygon")
    assert response.status_code == 400
    assert "Insufficient treasury funds" in response.json()["detail"]


def test_treasury_accumulation_via_api():
    """Test treasury accumulates fees through API"""
    # Get initial balance
    response = client.get("/api/treasury/stats")
    initial_balance = response.json()["balance_sphinx"]
    
    # Collect some fees (just 2 calls to test accumulation)
    client.post("/api/treasury/collect/nft_mint?amount=0.1")
    client.post("/api/treasury/collect/nft_mint?amount=0.1")
    
    # Check balance increased
    response = client.get("/api/treasury/stats")
    new_balance = response.json()["balance_sphinx"]
    
    # Should have increased by 2 * 0.1 * 0.7 = 0.14
    assert new_balance == pytest.approx(initial_balance + 0.14)


def test_deployment_readiness_check():
    """Test deployment readiness is reflected in stats"""
    # Collect enough fees for avalanche (threshold: 30)
    # 430 * 0.1 * 0.7 = 30.1 SKYNT
    for _ in range(430):
        client.post("/api/treasury/collect/nft_mint?amount=0.1")
    
    response = client.get("/api/treasury/stats")
    data = response.json()
    
    # Avalanche should be deployed now (auto-deployed when threshold met)
    avalanche_status = data["deployments"]["avalanche"]
    assert avalanche_status["deployed"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
