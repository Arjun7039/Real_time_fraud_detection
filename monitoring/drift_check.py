"""Evidently drift report generator.

Compares recent production data against the training distribution
and generates an HTML drift report. Prints a warning if PSI > 0.2
on any key feature.

Usage:
    python monitoring/drift_check.py \
        --reference data/processed/train.parquet \
        --current data/processed/recent_production.parquet
"""

# TODO: Implement in Phase 8
