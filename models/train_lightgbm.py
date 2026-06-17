"""LightGBM training script with Optuna hyperparameter optimisation.

Trains a LightGBM classifier on the PaySim feature-engineered dataset.
Handles class imbalance via is_unbalance parameter.
Logs all hyperparameters and metrics to MLflow.

Usage:
    python models/train_lightgbm.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import mlflow
import mlflow.lightgbm
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
    print("Loading feature-engineered dataset...")
    df = pd.read_parquet(DATA_PATH)
    X = df[ALL_FEATURE_NAMES].values
    y = df["isFraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


def objective(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "is_unbalance": True,
        "verbose": -1,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    # Subsample for Optuna to prevent memory errors and massive training times
    if len(X_train) > 500000:
        X_hpo, _, y_hpo, _ = train_test_split(X_train, y_train, train_size=500000, stratify=y_train, random_state=RANDOM_STATE)
    else:
        X_hpo, y_hpo = X_train, y_train

    X_tr, X_val, y_tr, y_val = train_test_split(X_hpo, y_hpo, test_size=0.2, stratify=y_hpo, random_state=RANDOM_STATE)

    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

    y_proba = model.predict_proba(X_val)[:, 1]
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_val, y_proba)


def train_final_model(best_params, X_train, y_train):
    print("\n🚀 Training final LightGBM model (with is_unbalance=True)...")
    model = lgb.LGBMClassifier(
        **best_params,
        is_unbalance=True,
        verbose=-1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()

    print(f"\n🔍 Running Optuna HPO ({N_OPTUNA_TRIALS} trials)...")
    study = optuna.create_study(direction="maximize", study_name="lightgbm-hpo")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=True,
    )
    best_params = study.best_params
    print(f"  Best ROC-AUC: {study.best_value:.5f}")

    model = train_final_model(best_params, X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_proba, model_name="LightGBM", output_dir="reports")

    model_path = os.path.join(MODEL_DIR, "lightgbm_model.txt")
    model.booster_.save_model(model_path)
    
    # Save OOF
    oof_path = os.path.join(MODEL_DIR, "lightgbm_oof.npy")
    y_train_proba = model.predict_proba(X_train)[:, 1]
    np.save(oof_path, y_train_proba)
    np.save(os.path.join(MODEL_DIR, "lightgbm_test_preds.npy"), y_proba)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("realguard-fraud-detection")

    with mlflow.start_run(run_name="lightgbm-final"):
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(model, artifact_path="model")
        mlflow.log_artifact(os.path.join("reports", "LightGBM_pr_curve.png"))
        mlflow.set_tag("model_type", "lightgbm")
        print("📊 Logged to MLflow")

if __name__ == "__main__":
    main()
