"""Redis read/write helpers for the real-time feature store.

Provides get/set operations for per-account feature blobs
stored in Redis. All features are serialised as JSON strings
with a 24-hour TTL.

Key pattern: feat:{account_id} → JSON blob of windowed features.
"""

import json
import os
from typing import Any, Dict, List, Optional

import redis

# --------------- Configuration ---------------
REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
FEATURE_TTL: int = 86_400  # 24 hours in seconds
KEY_PREFIX: str = "feat"

# --------------- Connection Pool ---------------
_pool: redis.ConnectionPool = redis.ConnectionPool(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True,
    max_connections=20,
    socket_timeout=2,
    socket_connect_timeout=2,
)


def _get_client() -> redis.Redis:
    """Return a Redis client from the shared connection pool.

    Returns:
        redis.Redis: A Redis client instance.
    """
    return redis.Redis(connection_pool=_pool)


def _make_key(account_id: str) -> str:
    """Build the Redis key for a given account.

    Args:
        account_id: The origin account identifier (nameOrig).

    Returns:
        str: The formatted Redis key, e.g. 'feat:C1231006815'.
    """
    return f"{KEY_PREFIX}:{account_id}"


def set_features(account_id: str, features: Dict[str, Any]) -> None:
    """Write a feature dict to Redis with a 24-hour TTL.

    Args:
        account_id: The origin account identifier.
        features: Dictionary of computed feature name → value pairs.
    """
    client = _get_client()
    key = _make_key(account_id)
    client.setex(key, FEATURE_TTL, json.dumps(features))


def get_features(account_id: str) -> Optional[Dict[str, Any]]:
    """Read the feature dict for an account from Redis.

    Args:
        account_id: The origin account identifier.

    Returns:
        Optional[Dict[str, Any]]: The feature dict, or None if not found.
    """
    client = _get_client()
    key = _make_key(account_id)
    raw = client.get(key)
    if raw is None:
        return None
    return json.loads(raw)


def batch_get_features(account_ids: List[str]) -> List[Optional[Dict[str, Any]]]:
    """Read feature dicts for multiple accounts in one round-trip.

    Args:
        account_ids: List of account identifiers.

    Returns:
        List[Optional[Dict[str, Any]]]: List of feature dicts (None for missing keys).
    """
    if not account_ids:
        return []
    client = _get_client()
    keys = [_make_key(aid) for aid in account_ids]
    raw_values = client.mget(keys)
    return [json.loads(v) if v is not None else None for v in raw_values]


def delete_features(account_id: str) -> None:
    """Remove the feature entry for an account.

    Args:
        account_id: The origin account identifier.
    """
    client = _get_client()
    client.delete(_make_key(account_id))
