"""
Configuration Manager for Sphinx_OS
Loads environment-specific configuration from YAML files
"""

import os
from typing import Any, Dict, Optional
import yaml
import re


class ConfigManager:
    """
    Manages environment-specific configuration
    
    Features:
    - Load configuration from YAML files
    - Environment variable substitution
    - Configuration validation
    - Safe access with defaults
    """
    
    def __init__(self, environment: Optional[str] = None, config_dir: str = "config"):
        """
        Initialize ConfigManager
        
        Args:
            environment: Environment name (mainnet, testnet, local)
            config_dir: Directory containing config files
        """
        self.environment = environment or os.getenv("SPHINXOS_ENV", "local")
        self.config_dir = config_dir
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_file = os.path.join(self.config_dir, f"{self.environment}.yaml")
        
        if not os.path.exists(config_file):
            raise FileNotFoundError(
                f"Configuration file not found: {config_file}\n"
                f"Available environments: mainnet, testnet, local"
            )
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Substitute environment variables
        config = self._substitute_env_vars(config)
        
        return config
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """
        Recursively substitute environment variables in config
        
        Supports ${VAR_NAME} syntax
        """
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Find all ${VAR} patterns
            pattern = re.compile(r'\$\{([^}]+)\}')
            matches = pattern.findall(obj)
            
            for var_name in matches:
                env_value = os.getenv(var_name, "")
                obj = obj.replace(f"${{{var_name}}}", env_value)
            
            return obj
        else:
            return obj
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path
        
        Args:
            key_path: Dot-separated path (e.g., "api.port")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            config.get("api.port")  # Returns 8000
            config.get("api.workers", 1)  # Returns 1 if not set
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_network_config(self, network: str) -> Dict[str, Any]:
        """Get network-specific configuration"""
        return self.get(f"network.{network}", {})
    
    def get_contract_address(self, contract_name: str) -> str:
        """Get deployed contract address"""
        return self.get(f"contracts.{contract_name}", "")
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self.get("api", {})
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return self.get("security", {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self.get("monitoring", {})
    
    def get_blockchain_config(self) -> Dict[str, Any]:
        """Get blockchain configuration"""
        return self.get("blockchain", {})
    
    def get_hypercube_config(self) -> Dict[str, Any]:
        """Get hypercube configuration"""
        return self.get("hypercube", {})
    
    def get_zkp_config(self) -> Dict[str, Any]:
        """Get ZKP configuration"""
        return self.get("zkp", {})
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == "mainnet"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == "local"
    
    def validate(self) -> bool:
        """
        Validate configuration
        
        Returns:
            True if valid, raises exception otherwise
        """
        required_keys = [
            "environment",
            "api.host",
            "api.port",
        ]
        
        for key in required_keys:
            value = self.get(key)
            if value is None:
                raise ValueError(f"Required configuration key missing: {key}")
        
        # Validate production settings
        if self.is_production():
            jwt_secret = self.get("security.jwt_secret")
            if not jwt_secret or jwt_secret == "CHANGE_ME_IN_PRODUCTION":
                raise ValueError("JWT_SECRET must be set in production")
            
            if not self.get("security.api_keys_required"):
                raise ValueError("API keys must be required in production")
        
        return True


# Global config instance
_config = None


def get_config(environment: Optional[str] = None) -> ConfigManager:
    """
    Get or create global ConfigManager instance
    
    Args:
        environment: Environment name (optional, uses env var if not provided)
        
    Returns:
        ConfigManager instance
    """
    global _config
    if _config is None:
        _config = ConfigManager(environment)
        _config.validate()
    return _config
