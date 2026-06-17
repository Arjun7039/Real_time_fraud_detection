"""Faust stream processor — windowed feature computation.

Consumes the 'raw-transactions' Kafka topic, computes per-account
windowed features (txn_count_5m, avg_amount_1h, etc.), and writes
the computed features to Redis via the feature_writer module.

The windowed aggregations are maintained using in-memory dictionaries
keyed by account ID, with timestamps used to expire old entries from
each window.

Usage:
    faust -A streaming.faust_app worker -l info
"""

import os
import time
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import faust

from streaming.feature_writer import write_features

logger = logging.getLogger(__name__)

# --------------- Configuration ---------------
KAFKA_BROKER: str = os.getenv("KAFKA_BROKER", "kafka:9092")
KAFKA_TOPIC: str = "raw-transactions"

# Window sizes in seconds
WINDOW_5M: int = 5 * 60        # 300 seconds
WINDOW_1H: int = 60 * 60       # 3600 seconds
WINDOW_24H: int = 24 * 60 * 60 # 86400 seconds


# --------------- Faust App ---------------
app = faust.App(
    "realguard-streaming",
    broker=f"kafka://{KAFKA_BROKER}",
    value_serializer="json",
    topic_partitions=1,
    processing_guarantee="at_least_once",
)


class Transaction(faust.Record, serializer="json"):
    """Faust record mapping the Kafka transaction message schema."""

    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float
    isFraud: int = 0
    isFlaggedFraud: int = 0


# Source topic
raw_topic = app.topic(KAFKA_TOPIC, value_type=Transaction)


# --------------- In-Memory Window State ---------------
# Each account maps to a list of (timestamp, amount, dest_id) tuples.
# We prune entries older than 24 hours on every event.
_account_events: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)


def _prune_window(events: List[Tuple[float, float, str]], now: float) -> List[Tuple[float, float, str]]:
    """Remove events older than 24 hours from the list.

    Args:
        events: List of (timestamp, amount, dest_id) tuples.
        now: Current wall-clock time in seconds.

    Returns:
        List with only events within the last 24 hours.
    """
    cutoff = now - WINDOW_24H
    return [e for e in events if e[0] >= cutoff]


def _compute_features(
    events: List[Tuple[float, float, str]],
    now: float,
    current_txn: Transaction,
) -> Dict[str, Any]:
    """Compute all 7 windowed features from the event history.

    Args:
        events: The pruned event list for this account.
        now: Current timestamp in seconds.
        current_txn: The transaction being processed.

    Returns:
        Dict with all windowed feature values.
    """
    # Time boundaries
    cutoff_5m = now - WINDOW_5M
    cutoff_1h = now - WINDOW_1H

    # Partition events into windows
    events_5m = [e for e in events if e[0] >= cutoff_5m]
    events_1h = [e for e in events if e[0] >= cutoff_1h]
    events_24h = events  # already pruned to 24h

    # -- txn_count_5m --
    txn_count_5m: int = len(events_5m)

    # -- txn_count_1h --
    txn_count_1h: int = len(events_1h)

    # -- avg_amount_1h --
    amounts_1h = [e[1] for e in events_1h]
    avg_amount_1h: float = (
        sum(amounts_1h) / len(amounts_1h) if amounts_1h else 0.0
    )

    # -- max_amount_1h --
    max_amount_1h: float = max(amounts_1h) if amounts_1h else 0.0

    # -- unique_dest_1h --
    unique_dest_1h: int = len({e[2] for e in events_1h})

    # -- balance_drop_pct (per-event) --
    old_bal = current_txn.oldbalanceOrg
    new_bal = current_txn.newbalanceOrig
    balance_drop_pct: float = (old_bal - new_bal) / (old_bal + 1.0)

    # -- txn_count_24h --
    txn_count_24h: int = len(events_24h)

    return {
        "txn_count_5m": txn_count_5m,
        "txn_count_1h": txn_count_1h,
        "avg_amount_1h": round(avg_amount_1h, 4),
        "max_amount_1h": round(max_amount_1h, 4),
        "unique_dest_1h": unique_dest_1h,
        "balance_drop_pct": round(balance_drop_pct, 6),
        "txn_count_24h": txn_count_24h,
    }


# --------------- Faust Agent ---------------
@app.agent(raw_topic)
async def process_transactions(stream) -> None:
    """Consume transactions, compute windowed features, write to Redis.

    For every incoming transaction:
    1. Record the event in the in-memory window state.
    2. Prune events older than 24 hours.
    3. Compute all 7 windowed features.
    4. Write the feature dict to Redis via the feature_writer.
    """
    event_count: int = 0

    async for txn in stream:
        now = time.time()
        account_id: str = txn.nameOrig

        # 1. Append current event
        _account_events[account_id].append((now, txn.amount, txn.nameDest))

        # 2. Prune old events
        _account_events[account_id] = _prune_window(
            _account_events[account_id], now
        )

        # 3. Compute features
        features = _compute_features(
            _account_events[account_id], now, txn
        )

        # 4. Write to Redis
        write_features(account_id, features)

        event_count += 1
        if event_count % 5_000 == 0:
            logger.info(
                "Processed %d events — %d accounts tracked",
                event_count,
                len(_account_events),
            )
