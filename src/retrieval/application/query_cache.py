"""Bounded query vector cache and shared search capacity."""

import threading
import time
from collections import OrderedDict

from src.indexing.domain.entities import stable_hash


class QueryCache:
    """Keep vectors isolated by actor, query and embedding contract.

    Attributes:
        search_slots: Nonblocking semaphore limiting concurrent searches.
    """

    def __init__(self, ttl_seconds, max_entries, max_concurrent_searches):
        """Create an empty cache and capacity limiter without external resources.

        Args:
            ttl_seconds: Lifetime of an entry, measured on a monotonic clock.
            max_entries: Maximum number of stored entries.
            max_concurrent_searches: Maximum simultaneous search operations.
        """
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self._lock = threading.RLock()
        self.search_slots = threading.BoundedSemaphore(max_concurrent_searches)

    @staticmethod
    def key(normalized, actor, contract):
        """Return the original actor/query/contract/runtime cache hash.

        Args:
            normalized: Query text after Unicode and whitespace normalization.
            actor: Complete authorization context.
            contract: Encoder contract containing its hash and runtime settings.

        Returns:
            Stable cache key covering all four inputs.
        """
        return stable_hash(
            {
                "actor": actor.model_dump(mode="json"),
                "query": normalized,
                "contract": contract.hash,
                "runtime": contract.runtime,
            }
        )

    def get(self, key):
        """Return a live vector and mark it recently used, or return None.

        Args:
            key: Hash returned by key().

        Returns:
            Cached vector, or None when absent or expired.
        """
        with self._lock:
            cached = self._entries.get(key)
            if cached and cached[0] > time.monotonic():
                self._entries.move_to_end(key)
                return cached[1]
        return None

    def put(self, key, vector):
        """Store a vector with a fresh expiry and evict oldest entries as needed.

        Args:
            key: Hash returned by key().
            vector: Validated embedding vector retained by reference.
        """
        with self._lock:
            self._entries[key] = (time.monotonic() + self.ttl_seconds, vector)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
