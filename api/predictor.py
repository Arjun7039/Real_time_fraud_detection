import os
import sys
import json
import logging
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import joblib

# Ensure we can import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import PredictRequest, PredictResponse, ReasonCode
from features.redis_store import get_features
from features.feature_definitions import ALL_FEATURE_NAMES
from explainability.shap_explainer import FraudExplainer

logger = logging.getLogger(__name__)

# The optimal threshold found during Meta-Learner evaluation
# (We saw 0.9998 in the logs, but setting slightly lower for safety in prod)
OPTIMAL_THRESHOLD = 0.9998

class FraudPredictor:
    """Singleton prediction engine for the real-time API.
    
    Loads all models into memory at startup and orchestrates the
    full inference pipeline: Feature Engineering -> Base Models ->
    Meta Learner -> SHAP Explainer.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FraudPredictor, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
        
    def _initialize(self):
        logger.info("Initializing FraudPredictor... Loading models into memory.")
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models")
        
        # 1. Load XGBoost
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(os.path.join(model_dir, "xgboost_model.json"))
        
        # 2. Load LightGBM
        self.lgb_model = lgb.Booster(model_file=os.path.join(model_dir, "lightgbm_model.txt"))
        
        # 3. Load Meta-Learner
        self.meta_model = joblib.load(os.path.join(model_dir, "meta_model.pkl"))
        
        # 4. Load SHAP Explainer
        self.explainer = FraudExplainer(os.path.join(model_dir, "xgboost_model.json"))
        
        logger.info("✅ All models loaded successfully.")

    def _build_feature_vector(self, tx: PredictRequest) -> dict:
        """Assembles the full 22-feature vector for inference."""
        
        # 1. Fetch real-time windowed features from Redis
        windowed_features = get_features(tx.nameOrig)
        if not windowed_features:
            # Fallback if account has no history in Redis
            windowed_features = {
                "txn_count_5m": 0,
                "txn_count_1h": 0,
                "avg_amount_1h": 0.0,
                "max_amount_1h": 0.0,
                "unique_dest_1h": 0,
                "balance_drop_pct": (tx.oldbalanceOrg - tx.newbalanceOrig) / (tx.oldbalanceOrg + 1.0),
                "txn_count_24h": 0
            }
            
        # 2. Compute batch/derived features
        derived = {
            "amount_to_balance_ratio": tx.amount / (tx.oldbalanceOrg + 1.0),
            "is_large_transfer": 1 if (tx.type == "TRANSFER" and tx.amount > 200000) else 0,
            "dest_balance_increased": 1 if (tx.newbalanceDest > tx.oldbalanceDest) else 0,
            "hour_of_day": tx.step % 24,
            "day_of_month": tx.step // 24,
            
            # Raw passthrough
            "amount": tx.amount,
            "oldbalanceOrg": tx.oldbalanceOrg,
            "newbalanceOrig": tx.newbalanceOrig,
            "oldbalanceDest": tx.oldbalanceDest,
            "newbalanceDest": tx.newbalanceDest,
            
            # One-hot encoded types
            "type_CASH_IN": 1 if tx.type == "CASH_IN" else 0,
            "type_CASH_OUT": 1 if tx.type == "CASH_OUT" else 0,
            "type_DEBIT": 1 if tx.type == "DEBIT" else 0,
            "type_PAYMENT": 1 if tx.type == "PAYMENT" else 0,
            "type_TRANSFER": 1 if tx.type == "TRANSFER" else 0,
        }
        
        # Combine everything
        return {**windowed_features, **derived}

    def predict(self, tx: PredictRequest) -> PredictResponse:
        """Executes the full inference pipeline."""
        
        # 1. Assemble features
        feature_dict = self._build_feature_vector(tx)
        
        # Format as 2D numpy array in the exact order the models expect
        X_array = np.array([[feature_dict.get(f, 0.0) for f in ALL_FEATURE_NAMES]])
        
        # 2. Base Model Predictions
        # XGBoost predict_proba returns [prob_class_0, prob_class_1]
        xgb_prob = self.xgb_model.predict_proba(X_array)[0][1]
        
        # LightGBM booster predict returns prob_class_1 directly
        lgb_prob = self.lgb_model.predict(X_array)[0]
        
        # 3. Meta-Learner Prediction
        X_meta = np.column_stack(([xgb_prob], [lgb_prob]))
        final_prob = self.meta_model.predict_proba(X_meta)[0][1]
        
        is_fraud = bool(final_prob >= OPTIMAL_THRESHOLD)
        
        # 4. Explainability (only compute SHAP if it's flagged as fraud to save time)
        reasons = []
        if is_fraud:
            raw_reasons = self.explainer.get_reason_codes(feature_dict, top_k=3)
            reasons = [
                ReasonCode(
                    feature=r["feature"],
                    value=r["value"],
                    contribution=r["contribution"]
                ) for r in raw_reasons
            ]
            
        return PredictResponse(
            transaction_id=tx.nameOrig,
            is_fraud=is_fraud,
            probability=float(final_prob),
            threshold_used=OPTIMAL_THRESHOLD,
            reasons=reasons
        )
