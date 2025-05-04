# sphinx_os/quantum/entanglement_cache.py
"""
EntanglementCache: Cache for entanglement states, including Rydberg gate effects.
"""
import logging
from typing import Any, Optional

logger = logging.getLogger("SphinxOS.EntanglementCache")

class EntanglementCache:
    """Cache for entanglement states."""
    
    def __init__(self):
        """Initialize EntanglementCache."""
        self.cache = {}
        self.rydberg_cache = {}  # Separate cache for Rydberg-induced entanglement
        logger.info("EntanglementCache initialized")

    def store(self, key: Any, entanglement_data: Any) -> None:
        """
        Store entanglement data in the cache.

        Args:
            key (Any): Cache key.
            entanglement_data (Any): Entanglement data to store.
        """
        self.cache[key] = entanglement_data
        logger.debug("Entanglement data stored with key %s", str(key))

    def store_rydberg(self, key: Any, rydberg_data: Any) -> None:
        """
        Store Rydberg-induced entanglement data.

        Args:
            key (Any): Cache key.
            rydberg_data (Any): Rydberg entanglement data to store.
        """
        self.rydberg_cache[key] = rydberg_data
        logger.debug("Rydberg entanglement data stored with key %s", str(key))

    def retrieve(self, key: Any) -> Optional[Any]:
        """
        Retrieve entanglement data from the cache.

        Args:
            key (Any): Cache key.

        Returns:
            Optional[Any]: Entanglement data, or None if not found.
        """
        data = self.cache.get(key, None)
        logger.debug("Entanglement data retrieved with key %s: %s", str(key), str(data))
        return data

    def retrieve_rydberg(self, key: Any) -> Optional[Any]:
        """
        Retrieve Rydberg-induced entanglement data.

        Args:
            key (Any): Cache key.

        Returns:
            Optional[Any]: Rydberg entanglement data, or None if not found.
        """
        data = self.rydberg_cache.get(key, None)
        logger.debug("Rydberg entanglement data retrieved with key %s: %s", str(key), str(data))
        return data
