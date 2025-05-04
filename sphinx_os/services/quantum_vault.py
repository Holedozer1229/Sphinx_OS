# sphinx_os/services/quantum_vault.py
"""
QuantumVault: Security vault for authentication.
"""
import logging
import contextlib
from typing import Iterator

logger = logging.getLogger("SphinxOS.QuantumVault")

class QuantumVault:
    """Security vault for authentication."""
    
    def __init__(self):
        """Initialize QuantumVault."""
        self.authenticated_users = set()
        logger.info("QuantumVault initialized")

    @contextlib.contextmanager
    def authenticate(self, user: str) -> Iterator[None]:
        """
        Authenticate a user within a context.

        Args:
            user (str): User identifier.
        """
        logger.debug("Authenticating user %s", user)
        self.authenticated_users.add(user)
        try:
            yield
        finally:
            if user in self.authenticated_users:
                self.authenticated_users.remove(user)
            logger.debug("User %s authentication ended", user)
