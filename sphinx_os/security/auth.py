"""
JWT-based Authentication Manager for Sphinx_OS APIs
Provides token generation, verification, and user authentication
"""

import os
from datetime import datetime, timedelta
from typing import Optional

import jwt
from fastapi import Security, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()


class AuthManager:
    """
    Manages JWT authentication for API endpoints
    
    Features:
    - Token generation with configurable expiration
    - Token verification and validation
    - Role-based access control
    - Secure secret key management
    """
    
    def __init__(self, secret_key: Optional[str] = None, token_expiry_hours: int = 24):
        """
        Initialize AuthManager
        
        Args:
            secret_key: JWT secret key (defaults to env var JWT_SECRET)
            token_expiry_hours: Token expiration time in hours
        """
        self.secret_key = secret_key or os.getenv("JWT_SECRET", "CHANGE_ME_IN_PRODUCTION")
        self.token_expiry_hours = token_expiry_hours
        self.algorithm = "HS256"
        
        if self.secret_key == "CHANGE_ME_IN_PRODUCTION":
            import warnings
            warnings.warn(
                "Using default JWT secret. Set JWT_SECRET environment variable in production!",
                UserWarning
            )
    
    def create_token(self, user_id: str, roles: Optional[list] = None) -> str:
        """
        Create JWT token for user
        
        Args:
            user_id: Unique user identifier
            roles: Optional list of user roles
            
        Returns:
            Encoded JWT token string
        """
        payload = {
            "user_id": user_id,
            "roles": roles or ["user"],
            "exp": datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            "iat": datetime.utcnow(),
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> dict:
        """
        Verify and decode JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload
            
        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def get_current_user(
        self,
        credentials: HTTPAuthorizationCredentials = Security(security)
    ) -> dict:
        """
        Dependency for FastAPI endpoints to get current authenticated user
        
        Args:
            credentials: HTTP Bearer credentials
            
        Returns:
            User information from token payload
        """
        return self.verify_token(credentials.credentials)
    
    def require_role(self, required_role: str):
        """
        Dependency factory for role-based access control
        
        Args:
            required_role: Required role for access
            
        Returns:
            Dependency function for FastAPI
        """
        def role_checker(user: dict = Depends(self.get_current_user)) -> dict:
            user_roles = user.get("roles", [])
            if required_role not in user_roles and "admin" not in user_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role '{required_role}' required"
                )
            return user
        return role_checker


# Global auth manager instance
_auth_manager = None


def get_auth_manager() -> AuthManager:
    """Get or create global AuthManager instance"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> dict:
    """
    FastAPI dependency for getting current authenticated user
    
    Usage:
        @app.get("/protected")
        def protected_route(user: dict = Depends(get_current_user)):
            return {"user_id": user["user_id"]}
    """
    return get_auth_manager().get_current_user(credentials)
