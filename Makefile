# Makefile

.PHONY: help install compile test validate deploy-testnet deploy-mainnet

help:
	@echo "Sphinx_OS Deployment Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install dependencies"
	@echo "  make compile          Compile smart contracts"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run contract tests"
	@echo "  make validate         Validate deployment readiness"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy-testnet   Deploy to testnet"
	@echo "  make deploy-mainnet   Deploy to mainnet"

install:
	pip install -r requirements.txt
	cd contracts && npm install

compile:
	@echo "Compiling contracts..."
	cd contracts && npx hardhat compile
	@echo "✅ Contracts compiled successfully"

test:
	cd contracts && npx hardhat test

validate:
	@echo "Validating deployment readiness..."
	python scripts/validate_deployment_readiness.py

deploy-testnet:
	@echo "Deploying to testnet..."
	ENVIRONMENT=testnet python scripts/deploy_mainnet.py --network polygon

deploy-mainnet: validate
	@echo "⚠️  DEPLOYING TO MAINNET"
	@read -p "Are you sure? (yes/no): " confirm && [ "$$confirm" = "yes" ]
	python scripts/deploy_mainnet.py --network all
