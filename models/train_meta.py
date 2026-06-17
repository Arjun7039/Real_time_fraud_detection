"""Stacking ensemble meta-learner training script.

Takes out-of-fold predictions from XGBoost, LightGBM, and LSTM,
trains a Logistic Regression meta-learner, and registers the
full ensemble to MLflow as 'fraud-stacking-ensemble'.

Usage:
    python models/train_meta.py
"""

import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import mlflow

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.evaluate import evaluate_model

MODEL_DIR = os.path.join("data", "models")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

def main():
    print("Loading OOF predictions for Meta-Learner...")
    
    # Load training OOF
    xgb_oof = np.load(os.path.join(MODEL_DIR, "xgboost_oof.npy"))
    lgb_oof = np.load(os.path.join(MODEL_DIR, "lightgbm_oof.npy"))
    y_train = np.load(os.path.join(MODEL_DIR, "y_train.npy"))
    
    # Load test predictions
    xgb_test = np.load(os.path.join(MODEL_DIR, "xgboost_test_preds.npy"))
    lgb_test = np.load(os.path.join(MODEL_DIR, "lightgbm_test_preds.npy"))
    y_test = np.load(os.path.join(MODEL_DIR, "y_test.npy"))
    
    X_meta_train = np.column_stack((xgb_oof, lgb_oof))
    X_meta_test = np.column_stack((xgb_test, lgb_test))
    
    print("Training Logistic Regression Meta-Learner...")
    meta_model = LogisticRegression(class_weight='balanced')
    meta_model.fit(X_meta_train, y_train)
    
    print(f"Meta-Learner coefficients: XGB={meta_model.coef_[0][0]:.4f}, LGB={meta_model.coef_[0][1]:.4f}")
    
    y_meta_proba = meta_model.predict_proba(X_meta_test)[:, 1]
    metrics = evaluate_model(y_test, y_meta_proba, model_name="Ensemble", output_dir="reports")
    
    model_path = os.path.join(MODEL_DIR, "meta_model.pkl")
    joblib.dump(meta_model, model_path)
    
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("realguard-fraud-detection")
    with mlflow.start_run(run_name="ensemble-meta"):
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(meta_model, artifact_path="model", registered_model_name="Production")
        mlflow.log_artifact(os.path.join("reports", "Ensemble_pr_curve.png"))
        print("📊 Logged Ensemble to MLflow and registered as 'Production'")

if __name__ == "__main__":
    main()
