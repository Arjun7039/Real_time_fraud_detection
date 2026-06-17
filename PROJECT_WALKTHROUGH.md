# RealGuard — Complete Project Walkthrough (Phase 1–6)

> **Purpose**: Interview-ready deep-dive into every file, every design decision, and every technical trade-off so you can explain this project flawlessly.

---

## System Architecture (Big Picture)

```
PaySim CSV (6.3M rows)
    │
    ├──► Kafka Producer (ingestion/producer.py)
    │         │
    │         ▼
    │    Apache Kafka (raw-transactions topic)
    │         │
    │         ▼
    │    Faust Stream Processor (streaming/faust_app.py)
    │         │
    │         ▼
    │    Redis Feature Store (features/redis_store.py)
    │         │
    │         ▼
    │    FastAPI /predict (api/main.py) ──► SHAP Explainer (Reason Codes)
    │
    └──► Feature Engineering (notebooks/02_*)
              │
              ▼
         features.parquet (6.3M rows × 22 cols)
              │
         ┌────┴────┐
         ▼         ▼
      XGBoost   LightGBM
         │         │
         └────┬────┘
              ▼
         Meta-Learner (Logistic Regression)
              │
              ▼
         FastAPI /predict
```

**Interview one-liner**: *"RealGuard is an end-to-end real-time fraud detection system. Raw transactions flow through Kafka, get enriched with windowed features in Faust+Redis, and are scored by a stacking ensemble of XGBoost and LightGBM. The API returns a fraud probability plus SHAP-based reason codes, all in under 200ms."*

---

## Phase 1 — Infrastructure (Docker)

### docker-compose.yml

Orchestrates **7 containers** that form the production backbone:

| Service | Image | Port | Role |
|---------|-------|------|------|
| **Zookeeper** | `cp-zookeeper:7.5.3` | 2181 | Kafka coordination |
| **Kafka** | `cp-kafka:7.5.3` | 9092, 29092 | Event streaming backbone |
| **Redis** | `redis:7-alpine` | 6379 | Real-time feature store (append-only persistence) |
| **MLflow** | `python:3.11-slim` | 5000 | Experiment tracking & model registry (SQLite backend) |
| **Prometheus** | `prom/prometheus:v2.49.1` | 9090 | Metrics scraper |
| **Grafana** | `grafana/grafana:10.3.1` | 3000 | Live monitoring dashboards |
| **API** | Custom `Dockerfile.api` | 8000 | FastAPI inference endpoint |

**Interview Q: "Why Kafka instead of a REST queue?"**
> Kafka provides durable, ordered, replayable event logs. If our Faust consumer crashes, it resumes from the last committed offset — zero data loss. REST queues like RabbitMQ delete messages after consumption.

**Interview Q: "Why Redis for features instead of a database?"**
> Redis operates entirely in-memory with sub-millisecond latency. Our SLA is <200ms per prediction. A PostgreSQL round-trip would add 5-15ms; Redis adds <1ms. We use a 24-hour TTL so stale features auto-expire.

---

## Phase 2 — Data Ingestion

### ingestion/schemas.py

A **Pydantic model** defining the `Transaction` schema with 11 fields (step, type, amount, nameOrig, balances, isFraud, etc.). This is the **single source of truth** for data validation across the entire pipeline — the producer, the Faust consumer, and the API all reference compatible schemas.

### ingestion/producer.py

Reads the PaySim CSV row-by-row and publishes each row as a JSON message to Kafka topic `raw-transactions`.

**Key design decisions**:
- **Rate limiting**: Configurable TPS (`--rate 100`) with micro-batch sleeping to simulate real-time flow
- **LZ4 compression**: Reduces network bandwidth by ~60% with minimal CPU overhead
- **Graceful shutdown**: Catches `SIGINT`/`SIGTERM`, flushes remaining messages, then exits cleanly
- **Keyed messages**: Uses `nameOrig` (account ID) as the Kafka key — guarantees all transactions from the same account land on the same partition — preserves ordering for windowed aggregations

**Interview Q: "Why key by nameOrig?"**
> Kafka guarantees ordering within a partition. By keying on the sender's account ID, all of that account's transactions go to the same partition. This means our Faust consumer sees them in chronological order, which is critical for computing accurate rolling windows (e.g., "3 transactions in 5 minutes").

---

## Phase 3 — Streaming Feature Engineering

### streaming/faust_app.py

The **real-time stream processor**. Consumes from `raw-transactions`, computes 7 windowed features per transaction, and writes them to Redis.

**The 7 windowed features**:

| Feature | Window | What it captures |
|---------|--------|-----------------|
| `txn_count_5m` | 5 min | Velocity — rapid-fire transactions signal account takeover |
| `txn_count_1h` | 1 hour | Medium-term activity burst |
| `avg_amount_1h` | 1 hour | Spending pattern baseline |
| `max_amount_1h` | 1 hour | Spike detection — one huge transfer among small ones |
| `unique_dest_1h` | 1 hour | Fan-out — fraudsters scatter money to many accounts |
| `balance_drop_pct` | Per-event | `(old - new) / (old + 1)` — draining an account to zero |
| `txn_count_24h` | 24 hours | Daily volume baseline |

**How the windowing works**: We maintain an in-memory dictionary `_account_events` that maps each account ID to a list of `(timestamp, amount, dest_id)` tuples. On every new event:
1. Append the new event
2. Prune anything older than 24 hours
3. Filter the list into 5m / 1h / 24h sub-windows
4. Compute aggregates (count, avg, max, unique)

**Interview Q: "This is in-memory — what happens if Faust crashes?"**
> The window state is lost, but Kafka retains all raw events. On restart, Faust replays from the last committed offset and rebuilds the window state. For production, we could use Faust's RocksDB-backed tables for persistent state, or checkpoint to Redis periodically.

### streaming/feature_writer.py

A thin **adapter layer** between Faust and Redis. It calls `redis_store.set_features()` and adds structured logging. This separation follows the **Single Responsibility Principle** — if we switch from Redis to DynamoDB, we only change `redis_store.py`, not the Faust agent.

### features/redis_store.py

The **feature store client**. Key design:
- **Connection pooling**: Up to 20 reusable connections (avoids TCP handshake per request)
- **Key pattern**: `feat:{account_id}` → JSON blob
- **24-hour TTL**: Features auto-expire (if an account hasn't transacted in 24h, there's nothing to aggregate)
- **Batch reads**: `mget` for multi-account lookups in a single Redis round-trip

### features/feature_definitions.py

The **central feature registry** — the single source of truth for all 22 feature names used across the entire pipeline. Contains:
- `WINDOWED_FEATURES` (7 features computed by Faust in real-time)
- `BATCH_FEATURES` (5 features computed in Phase 4 for training)
- `ALL_FEATURE_NAMES` (ordered list of all 22 features — this exact ordering is what the model expects at inference time)

**IMPORTANT**: This file prevents **training-serving skew**. Both the training script and the API import `ALL_FEATURE_NAMES` from here, guaranteeing the feature vector is always in the same column order.

---

## Phase 4 — Batch Feature Engineering

### notebooks/01_eda.ipynb

Exploratory data analysis on the raw PaySim dataset. Key findings:
- **6.3 million** transactions, only **8,213 fraudulent** (0.13% — extreme class imbalance)
- Fraud only occurs in `TRANSFER` and `CASH_OUT` transaction types
- Fraudulent transactions almost always drain the sender's balance to exactly zero
- The `isFlaggedFraud` column from the simulator catches <1% of actual fraud — useless as a feature

### notebooks/02_feature_engineering.ipynb

Transforms the raw CSV into the training-ready `features.parquet`. Steps:
1. **One-hot encode** the `type` column → 5 binary columns (`type_CASH_IN`, `type_CASH_OUT`, etc.)
2. **Approximate windowed features** using Pandas `groupby` + `rolling` (since we don't have Kafka in batch mode)
3. **Engineer 5 batch features**: `amount_to_balance_ratio`, `is_large_transfer`, `dest_balance_increased`, `hour_of_day`, `day_of_month`
4. **Export**: Full 6.3M rows → `features.parquet` (345 MB); human-readable 10K sample → `features_inspection.csv`

**Interview Q: "Why Parquet instead of CSV?"**
> Parquet uses columnar compression — our 493MB CSV shrinks to 345MB Parquet. More importantly, Pandas reads Parquet ~5x faster because it can skip columns and use binary encoding instead of parsing text.

---

## Phase 5 — Model Training

### models/evaluate.py

**Shared evaluation module** used by all training scripts. Computes:
- **ROC-AUC**: Area under the ROC curve (how well the model separates fraud from legitimate)
- **PR-AUC**: Area under the Precision-Recall curve (critical for imbalanced datasets — ROC-AUC can be misleadingly high)
- **Optimal threshold**: The probability cutoff that maximizes F1 score (not the default 0.5!)
- **Confusion matrix**: TP, FP, FN, TN counts
- **PR curve plot**: Saved as a PNG to `reports/`

**Interview Q: "Why not just use accuracy?"**
> With 99.87% legitimate transactions, a model that predicts "not fraud" for everything gets 99.87% accuracy but catches zero fraud. PR-AUC and F1 at the optimal threshold are the correct metrics for extreme class imbalance.

### models/train_xgboost.py

**XGBoost training pipeline**. Three major stages:

**Stage 1 — Optuna HPO** (10 trials on a 500K stratified subset):
- Searches over: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`
- Uses a 500K-row subsample to prevent memory crashes on 5M+ rows
- Optimizes ROC-AUC via 80/20 holdout validation

**Stage 2 — Final training** on all 5M rows with best hyperparameters

**Stage 3 — Artifacts**:
- `xgboost_model.json` — serialized model
- `xgboost_oof.npy` — out-of-fold predictions on training set (for meta-learner)
- `xgboost_test_preds.npy` — test set predictions
- Logs everything to MLflow

**How class imbalance is handled**: `scale_pos_weight = neg_count / pos_count ≈ 774`. This tells XGBoost that every positive (fraud) sample is worth 774 negative samples during gradient computation. This is mathematically equivalent to oversampling but uses **zero extra memory**.

**Interview Q: "Why not use SMOTE?"**
> We initially tried SMOTE but it caused `MemoryError`. SMOTE generates synthetic minority samples by computing k-nearest-neighbors on 5M rows — this requires ~32GB RAM. `scale_pos_weight` achieves the same mathematical effect by re-weighting the loss function, using zero additional memory.

**Results**: ROC-AUC **0.999**, F1 **0.954**, Precision **0.960**, Recall **0.949**

### models/train_lightgbm.py

Nearly identical pipeline to XGBoost but uses LightGBM's `is_unbalance=True` parameter and has an additional `num_leaves` hyperparameter. LightGBM uses **leaf-wise tree growth** (vs XGBoost's level-wise), which often converges faster on large datasets.

**Results**: ROC-AUC **0.999**, trained in ~1 minute (faster than XGBoost)

### models/train_meta.py — THE META MODEL (Stacking Ensemble)

This is the **stacking ensemble** — the crown jewel of the model architecture.

**What it does**:
1. Loads the **out-of-fold predictions** from XGBoost and LightGBM (saved as `.npy` files)
2. Stacks them into a 2-column matrix: `[xgb_probability, lgb_probability]`
3. Trains a **Logistic Regression** on these 2 features to learn the optimal blend
4. The Logistic Regression learned coefficients: `XGB=9.85, LGB=8.50` — meaning it trusts XGBoost slightly more than LightGBM

**Why stacking works**: XGBoost and LightGBM make different types of errors because they use different tree-building strategies (level-wise vs leaf-wise). The meta-learner learns "when XGBoost says fraud but LightGBM doesn't, trust XGBoost" and vice versa. This corrects individual model weaknesses.

**Why Logistic Regression as the meta-learner**: It's intentionally simple — a complex meta-learner would overfit to the base models' predictions. Logistic Regression just learns a weighted average, which is exactly what we want.

**What `meta_model.pkl` contains**: A scikit-learn `LogisticRegression` object with 2 coefficients (one per base model) and an intercept. It's only **879 bytes** — tiny because it's just learning weights, not tree structures.

**Results**: ROC-AUC **0.999**, F1 **0.945**, Precision **0.971**, Recall **0.921**

**Interview Q: "Why did the ensemble's recall drop slightly vs standalone XGBoost?"**
> The ensemble optimized for a different precision-recall trade-off. Its precision jumped to 97.1% (from 96.0%), meaning fewer false alarms, at the cost of 2.7% fewer catches. In production, this trade-off is tunable by adjusting the classification threshold.

---

## Phase 6 — Explainability

### explainability/shap_explainer.py

**SHAP (SHapley Additive exPlanations)** module for generating human-readable fraud reason codes.

**How it works**:
1. Loads the XGBoost model and creates a `shap.TreeExplainer`
2. For each transaction, computes **SHAP values** — the marginal contribution of each feature to the final prediction
3. Sorts features by contribution and returns the **Top-3 reason codes**

**What is a SHAP value?** Based on cooperative game theory (Shapley values). For a prediction like "95% fraud probability", SHAP decomposes it as: *"The base rate is 0.13%. The high amount added +40%, the zero remaining balance added +35%, the destination account pattern added +20%..."* — every feature gets a signed contribution that sums to the final prediction.

**Why TreeExplainer?** SHAP offers multiple explainers. `TreeExplainer` is specifically optimized for tree-based models — it computes exact Shapley values in O(TLD²) time instead of the generic O(2^N) exponential complexity. For our 22-feature XGBoost model, this means **<10ms per explanation**.

**Example output**:
```
Feature: oldbalanceOrg             | Value: 0.74       | SHAP: 1.5961
Feature: hour_of_day               | Value: 0.82       | SHAP: 1.2830
Feature: amount_to_balance_ratio   | Value: 0.63       | SHAP: 1.1929
```

### notebooks/04_shap_analysis.ipynb

Interactive notebook generating 3 visualizations:
1. **Global Summary Plot**: Beeswarm showing which features matter most across 5,000 sampled transactions
2. **Fraud Waterfall**: Deep-dive into one flagged transaction showing how each feature pushed the score toward fraud
3. **Legitimate Waterfall**: Same deep-dive for a clean transaction

---

## Data Files Summary

### data/raw/
| File | Size | Description |
|------|------|-------------|
| `paysim_dataset.csv` | 493 MB | Raw PaySim synthetic dataset (6.3M rows × 11 columns) |

### data/processed/
| File | Size | Description |
|------|------|-------------|
| `features.parquet` | 345 MB | Feature-engineered dataset (6.3M rows × 22 features + label) |
| `features_inspection.csv` | 1.5 MB | Human-readable 10K-row sample for manual verification |

### data/models/
| File | Size | Description |
|------|------|-------------|
| `xgboost_model.json` | 790 KB | Serialized XGBoost model (418 trees, max_depth=4) |
| `lightgbm_model.txt` | 2.0 MB | Serialized LightGBM model |
| `meta_model.pkl` | 879 B | Logistic Regression stacking ensemble (2 coefficients) |
| `xgboost_oof.npy` | 40.7 MB | XGBoost training set predictions (for meta-learner input) |
| `lightgbm_oof.npy` | 40.7 MB | LightGBM training set predictions |
| `xgboost_test_preds.npy` | 5.1 MB | XGBoost test set predictions |
| `lightgbm_test_preds.npy` | 10.2 MB | LightGBM test set predictions |
| `y_train.npy` | 40.7 MB | Training labels |
| `y_test.npy` | 10.2 MB | Test labels |

### reports/
| File | Description |
|------|-------------|
| `XGBoost_pr_curve.png` | Precision-Recall curve for XGBoost |
| `LightGBM_pr_curve.png` | Precision-Recall curve for LightGBM |
| `Ensemble_pr_curve.png` | Precision-Recall curve for the stacking ensemble |

---

## Final Model Performance

| Metric | XGBoost | LightGBM | **Ensemble** |
|--------|---------|----------|-------------|
| ROC-AUC | 0.999 | 0.999 | **0.999** |
| PR-AUC | 0.990 | — | **0.987** |
| F1 Score | 0.954 | — | **0.945** |
| Precision | 0.960 | — | **0.971** |
| Recall | 0.949 | — | **0.921** |
| False Positives | 65 | — | **46** |

---

## Common Interview Questions & Answers

**Q: "Walk me through a transaction's journey from raw data to fraud score."**
> A raw CSV row enters via the Kafka producer, lands on the `raw-transactions` topic keyed by account ID. The Faust stream processor consumes it, computes 7 windowed features (txn velocity, avg amount, unique destinations, etc.) using in-memory state, and writes them to Redis with a 24-hour TTL. When the FastAPI endpoint receives a `/predict` request, it grabs the features from Redis, runs them through XGBoost and LightGBM in parallel, stacks the two probabilities into the Logistic Regression meta-learner, and returns the final fraud score plus 3 SHAP reason codes — all in under 200ms.

**Q: "Why two models instead of one?"**
> XGBoost uses level-wise tree growth; LightGBM uses leaf-wise. They make systematically different errors. The stacking ensemble's Logistic Regression learns the optimal blend — when to trust which model — reducing false positives from 65 to 46 while maintaining 92% recall.

**Q: "How do you handle the 0.13% fraud rate?"**
> Three layers: (1) `scale_pos_weight` in XGBoost and `is_unbalance` in LightGBM mathematically up-weight minority samples during gradient computation — no extra memory. (2) We optimize for PR-AUC, not accuracy. (3) The optimal threshold is found by maximizing F1 on the precision-recall curve, not using the naive 0.5 cutoff.

**Q: "What would you do differently in production?"**
> (1) Replace Faust's in-memory state with RocksDB-backed tables for crash recovery. (2) Add A/B testing infrastructure to compare model versions. (3) Implement data drift detection using PSI (Population Stability Index) on feature distributions. (4) Add circuit breakers — if model latency exceeds 200ms, fall back to a rule-based system.
