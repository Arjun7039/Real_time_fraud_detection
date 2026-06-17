"""Manual retraining script — run when drift is detected.

Retrains all base models and the stacking ensemble on fresh data.
Logs to MLflow and promotes the best ensemble to Production.

Usage:
    python models/retrain.py --data data/raw/paysim_dataset.csv
"""

# TODO: Implement in Phase 5
