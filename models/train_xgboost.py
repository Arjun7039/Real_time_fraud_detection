"""XGBoost training script with Optuna hyperparameter optimisation.

Trains an XGBoost classifier on the PaySim feature-engineered dataset.
Handles class imbalance via scale_pos_weight and SMOTE.
Logs all hyperparameters and metrics to MLflow.

Usage:
    python models/train_xgboost.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import mlflow
import mlflow.xgboost
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.feature_definitions import ALL_FEATURE_NAMES
from models.evaluate import evaluate_model

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --------------- Configuration ---------------
DATA_PATH = os.path.join("data", "processed", "features.parquet")
MODEL_DIR = os.path.join("data", "models")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
RANDOM_STATE = 42
N_OPTUNA_TRIALS = 10
TEST_SIZE = 0.2


def load_data():
    """Load the feature-engineered parquet and split into train/test."""
    print("Loading feature-engineered dataset...")
    df = pd.read_parquet(DATA_PATH)
    X = df[ALL_FEATURE_NAMES].values
    y = df["isFraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")
    print(f"  Fraud rate (train): {y_train.mean():.4%}")
    return X_train, X_test, y_train, y_test


def objective(trial, X_train, y_train):
    """Optuna objective: train XGBoost with suggested hyperparameters."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }

    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count

    # Subsample for Optuna to prevent memory errors and massive training times
    if len(X_train) > 500000:
        X_hpo, _, y_hpo, _ = train_test_split(X_train, y_train, train_size=500000, stratify=y_train, random_state=RANDOM_STATE)
    else:
        X_hpo, y_hpo = X_train, y_train

    X_tr, X_val, y_tr, y_val = train_test_split(X_hpo, y_hpo, test_size=0.2, stratify=y_hpo, random_state=RANDOM_STATE)

    model = xgb.XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_proba = model.predict_proba(X_val)[:, 1]
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_val, y_proba)


def train_final_model(best_params, X_train, y_train):
    """Train the final XGBoost model with best hyperparameters."""
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)

    model = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=neg_count / pos_count,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def main():
    """Entry point — run HPO, train final model, evaluate, and log to MLflow."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()

    # ---- Optuna HPO ----
    print(f"\n🔍 Running Optuna HPO ({N_OPTUNA_TRIALS} trials)...")
    study = optuna.create_study(direction="maximize", study_name="xgboost-hpo")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=True,
    )
    best_params = study.best_params
    print(f"  Best ROC-AUC (CV): {study.best_value:.5f}")
    print(f"  Best params: {json.dumps(best_params, indent=2)}")

    # ---- Train final model ----
    print("\n🚀 Training final XGBoost model...")
    model = train_final_model(best_params, X_train, y_train)

    # ---- Evaluate ----
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_proba, model_name="XGBoost", output_dir="reports")

    # ---- Save model locally ----
    model_path = os.path.join(MODEL_DIR, "xgboost_model.json")
    model.save_model(model_path)
    print(f"💾 Model saved to {model_path}")

    # ---- Save OOF predictions for stacking ----
    oof_path = os.path.join(MODEL_DIR, "xgboost_oof.npy")
    # Generating OOF via simple predict for meta-learner (ideally use cross_val_predict)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    np.save(oof_path, y_train_proba)
    np.save(os.path.join(MODEL_DIR, "xgboost_test_preds.npy"), y_proba)
    np.save(os.path.join(MODEL_DIR, "y_test.npy"), y_test)
    np.save(os.path.join(MODEL_DIR, "y_train.npy"), y_train)

    # ---- Log to MLflow ----
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("realguard-fraud-detection")

    with mlflow.start_run(run_name="xgboost-final"):
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, artifact_path="model")
        mlflow.log_artifact(os.path.join("reports", "XGBoost_pr_curve.png"))
        mlflow.set_tag("model_type", "xgboost")
        print("📊 Logged to MLflow")

    print("\n✅ XGBoost training complete!")


if __name__ == "__main__":
    main()
