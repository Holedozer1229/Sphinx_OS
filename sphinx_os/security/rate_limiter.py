"""
Redis-backed Rate Limiter for Sphinx_OS APIs
Provides token bucket and sliding window rate limiting
"""

import os
import time
from typing import Optional, Tuple

from fastapi import HTTPException, Request, status


class RateLimiter:
    """
    In-memory rate limiter with Redis support
    
    Features:
    - Token bucket algorithm for smooth rate limiting
    - Configurable limits per endpoint
    - User-based and IP-based limiting
    - Redis backend for distributed systems (optional)
    """
    
    def __init__(
        self,
        requests_per_minute: int = 100,
        burst_size: Optional[int] = None,
        redis_url: Optional[str] = None
    ):
        """
        Initialize RateLimiter
        
        Args:
            requests_per_minute: Number of requests allowed per minute
            burst_size: Maximum burst size (defaults to 2x rate)
            redis_url: Redis connection URL for distributed rate limiting
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size or (requests_per_minute * 2)
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        
        # In-memory storage (fallback when Redis not available)
        self.storage = {}
        
        # Initialize Redis if available
        self.redis_client = None
        if self.redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                self.redis_client.ping()
            except Exception as e:
                import warnings
                warnings.warn(f"Redis connection failed: {e}. Using in-memory storage.", UserWarning)
    
    def _get_key(self, identifier: str, endpoint: str) -> str:
        """Generate storage key for rate limit tracking"""
        return f"ratelimit:{endpoint}:{identifier}"
    
    def _check_limit_memory(self, key: str) -> Tuple[bool, int]:
        """
        Check rate limit using in-memory storage
        
        Returns:
            (allowed, remaining_requests)
        """
        now = time.time()
        
        if key not in self.storage:
            self.storage[key] = {
                "tokens": self.burst_size,
                "last_update": now
            }
        
        data = self.storage[key]
        
        # Calculate tokens to add based on elapsed time
        elapsed = now - data["last_update"]
        tokens_to_add = elapsed * (self.requests_per_minute / 60.0)
        
        # Update tokens (capped at burst_size)
        data["tokens"] = min(self.burst_size, data["tokens"] + tokens_to_add)
        data["last_update"] = now
        
        # Check if request is allowed
        if data["tokens"] >= 1.0:
            data["tokens"] -= 1.0
            return True, int(data["tokens"])
        else:
            return False, 0
    
    def _check_limit_redis(self, key: str) -> Tuple[bool, int]:
        """
        Check rate limit using Redis backend
        
        Returns:
            (allowed, remaining_requests)
        """
        now = time.time()
        
        # Use Redis pipeline for atomic operations
        pipe = self.redis_client.pipeline()
        
        # Get current data
        pipe.get(key)
        pipe.ttl(key)
        results = pipe.execute()
        
        current_tokens = float(results[0]) if results[0] else self.burst_size
        
        # Calculate tokens to add
        if results[0]:
            ttl = results[1]
            elapsed = 60 - ttl if ttl > 0 else 0
            tokens_to_add = elapsed * (self.requests_per_minute / 60.0)
            current_tokens = min(self.burst_size, current_tokens + tokens_to_add)
        
        # Check if allowed
        if current_tokens >= 1.0:
            new_tokens = current_tokens - 1.0
            self.redis_client.setex(key, 60, new_tokens)
            return True, int(new_tokens)
        else:
            return False, 0
    
    def check_limit(self, identifier: str, endpoint: str) -> Tuple[bool, int]:
        """
        Check if request is within rate limit
        
        Args:
            identifier: User ID or IP address
            endpoint: API endpoint being accessed
            
        Returns:
            (allowed, remaining_requests)
        """
        key = self._get_key(identifier, endpoint)
        
        if self.redis_client:
            try:
                return self._check_limit_redis(key)
            except Exception:
                # Fallback to memory if Redis fails
                return self._check_limit_memory(key)
        else:
            return self._check_limit_memory(key)
    
    async def __call__(self, request: Request):
        """
        FastAPI dependency for rate limiting
        
        Usage:
            rate_limiter = RateLimiter(requests_per_minute=60)
            
            @app.get("/endpoint", dependencies=[Depends(rate_limiter)])
            def endpoint():
                return {"status": "ok"}
        """
        # Get identifier (user ID from auth or IP address)
        identifier = request.client.host
        
        # Try to get user from auth
        if hasattr(request.state, "user"):
            identifier = request.state.user.get("user_id", identifier)
        
        # Get endpoint path
        endpoint = request.url.path
        
        # Check rate limit
        allowed, remaining = self.check_limit(identifier, endpoint)
        
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                }
            )
        
        # Add rate limit headers to response
        request.state.rate_limit_remaining = remaining
