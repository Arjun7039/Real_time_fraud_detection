"""Feature writer — writes computed features to Redis.

Called by the Faust agent after computing windowed features.
Serialises feature dicts to JSON and writes to Redis with
a 24-hour TTL.
"""

import logging
from typing import Any, Dict

from features.redis_store import set_features

logger = logging.getLogger(__name__)


def write_features(account_id: str, features: Dict[str, Any]) -> None:
    """Persist computed windowed features for an account to Redis.

    This function is the single write-path used by the Faust stream
    processor. It delegates to the Redis store module and logs the
    operation for observability.

    Args:
        account_id: The origin account identifier (nameOrig).
        features: Dictionary of computed windowed feature name → value pairs.
    """
    try:
        set_features(account_id, features)
        logger.debug("Wrote features for %s: %s", account_id, list(features.keys()))
    except Exception:
        logger.exception("Failed to write features for account %s", account_id)
        raise
