"""
Input Validator for Sphinx_OS APIs
Provides sanitization and validation for user inputs
"""

import re
from typing import Any, List, Optional

from fastapi import HTTPException, status


class InputValidator:
    """
    Validates and sanitizes user inputs to prevent injection attacks
    
    Features:
    - SQL injection prevention
    - XSS prevention
    - Path traversal prevention
    - Type validation
    - Range validation
    """
    
    # Patterns for detecting malicious input
    SQL_INJECTION_PATTERNS = [
        r"(\bOR\b|\bAND\b).*=.*",
        r"(\bUNION\b|\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b)",
        r"--",
        r"/\*.*\*/",
        r";\s*(\bDROP\b|\bDELETE\b)",
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe",
        r"<object",
        r"<embed",
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.",
        r"%2e%2e",
        r"%252e%252e",
    ]
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """
        Sanitize string input
        
        Args:
            value: Input string
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
            
        Raises:
            HTTPException: If input is malicious or invalid
        """
        if not isinstance(value, str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input must be a string"
            )
        
        # Check length
        if len(value) > max_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input exceeds maximum length of {max_length}"
            )
        
        # Check for SQL injection
        for pattern in InputValidator.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Potentially malicious input detected"
                )
        
        # Check for XSS
        for pattern in InputValidator.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Potentially malicious input detected"
                )
        
        # Check for path traversal
        for pattern in InputValidator.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Potentially malicious input detected"
                )
        
        # Strip whitespace
        return value.strip()
    
    @staticmethod
    def validate_eth_address(address: str) -> str:
        """
        Validate Ethereum address format
        
        Args:
            address: Ethereum address
            
        Returns:
            Validated address
            
        Raises:
            HTTPException: If address is invalid
        """
        if not re.match(r"^0x[a-fA-F0-9]{40}$", address):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Ethereum address format"
            )
        return address.lower()
    
    @staticmethod
    def validate_number_range(
        value: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        field_name: str = "value"
    ) -> float:
        """
        Validate number is within range
        
        Args:
            value: Number to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            field_name: Field name for error messages
            
        Returns:
            Validated number
            
        Raises:
            HTTPException: If value is out of range
        """
        if min_value is not None and value < min_value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field_name} must be at least {min_value}"
            )
        
        if max_value is not None and value > max_value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field_name} must be at most {max_value}"
            )
        
        return value
    
    @staticmethod
    def validate_phi_score(score: float) -> float:
        """
        Validate Φ score is within valid range (200-1000)
        
        Args:
            score: Φ score value
            
        Returns:
            Validated score
        """
        return InputValidator.validate_number_range(
            score,
            min_value=200,
            max_value=1000,
            field_name="Phi score"
        )
    
    @staticmethod
    def validate_node_id(node_id: int, max_nodes: int = 1000) -> int:
        """
        Validate node ID
        
        Args:
            node_id: Node identifier
            max_nodes: Maximum number of nodes
            
        Returns:
            Validated node ID
        """
        return int(InputValidator.validate_number_range(
            node_id,
            min_value=0,
            max_value=max_nodes - 1,
            field_name="Node ID"
        ))
    
    @staticmethod
    def validate_list(
        value: List[Any],
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        field_name: str = "list"
    ) -> List[Any]:
        """
        Validate list length
        
        Args:
            value: List to validate
            min_length: Minimum list length
            max_length: Maximum list length
            field_name: Field name for error messages
            
        Returns:
            Validated list
            
        Raises:
            HTTPException: If list length is invalid
        """
        if not isinstance(value, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field_name} must be a list"
            )
        
        if min_length is not None and len(value) < min_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field_name} must contain at least {min_length} items"
            )
        
        if max_length is not None and len(value) > max_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field_name} must contain at most {max_length} items"
            )
        
        return value
