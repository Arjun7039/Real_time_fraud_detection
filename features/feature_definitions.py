"""Feature definitions — all feature names, windows, and types documented.

Central registry of all features used across the pipeline.
Used by both the streaming layer (Faust) and the serving layer (FastAPI)
to ensure consistency.
"""

# ============================================================
# WINDOWED FEATURES (computed by Faust in real-time)
# ============================================================
WINDOWED_FEATURES = {
    "txn_count_5m": {
        "description": "Number of transactions in last 5 minutes",
        "window": "5m",
        "type": "int",
    },
    "txn_count_1h": {
        "description": "Number of transactions in last 1 hour",
        "window": "1h",
        "type": "int",
    },
    "avg_amount_1h": {
        "description": "Average transaction amount in last 1 hour",
        "window": "1h",
        "type": "float",
    },
    "max_amount_1h": {
        "description": "Maximum transaction amount in last 1 hour",
        "window": "1h",
        "type": "float",
    },
    "unique_dest_1h": {
        "description": "Unique destination accounts in last 1 hour",
        "window": "1h",
        "type": "int",
    },
    "balance_drop_pct": {
        "description": "Percentage drop in origin balance this transaction",
        "window": "per_event",
        "type": "float",
    },
    "txn_count_24h": {
        "description": "Total transactions in last 24 hours",
        "window": "24h",
        "type": "int",
    },
}

# ============================================================
# BATCH-ENGINEERED FEATURES (computed in Phase 4 for training)
# ============================================================
BATCH_FEATURES = {
    "amount_to_balance_ratio": {
        "description": "amount / (oldbalanceOrg + 1)",
        "type": "float",
    },
    "is_large_transfer": {
        "description": "1 if type == TRANSFER and amount > 200000",
        "type": "int",
    },
    "dest_balance_increased": {
        "description": "1 if newbalanceDest > oldbalanceDest",
        "type": "int",
    },
    "hour_of_day": {
        "description": "step % 24",
        "type": "int",
    },
    "day_of_month": {
        "description": "step // 24",
        "type": "int",
    },
}

# ============================================================
# ALL FEATURE NAMES (used for model training column ordering)
# ============================================================
ALL_FEATURE_NAMES: list[str] = (
    list(WINDOWED_FEATURES.keys())
    + list(BATCH_FEATURES.keys())
    + [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER",
    ]
)
