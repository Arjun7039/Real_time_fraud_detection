import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

# Ensure we can import from the features module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.feature_definitions import ALL_FEATURE_NAMES

class FraudExplainer:
    """
    Explainability module using SHAP for the XGBoost fraud detection model.
    """
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "data", "models", "xgboost_model.json"
            )
            
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        
        # We use TreeExplainer for fast SHAP values on tree-based models
        self.explainer = shap.TreeExplainer(self.model)
        
    def get_reason_codes(self, features: dict, top_k: int = 3) -> list[dict]:
        """
        Calculates SHAP values for a single transaction and returns the top K
        features that pushed the fraud probability higher.
        
        Args:
            features: Dictionary mapping feature names to their values.
            top_k: Number of top reason codes to return.
            
        Returns:
            List of dictionaries containing feature name, value, and SHAP contribution.
        """
        # Ensure features are in the exact order expected by the model
        feature_array = np.array([[features.get(f, 0.0) for f in ALL_FEATURE_NAMES]])
        
        # Calculate SHAP values (returns log-odds margin contributions)
        shap_values = self.explainer.shap_values(feature_array)
        
        # For a single prediction, shap_values is a 1D array (num_features,)
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
            
        # Pair feature names with their SHAP values
        feature_contributions = []
        for i, feature_name in enumerate(ALL_FEATURE_NAMES):
            feature_contributions.append({
                "feature": feature_name,
                "value": feature_array[0][i],
                "contribution": float(shap_values[i])
            })
            
        # We are interested in explaining *fraud* risk, so we sort by highest positive contribution
        feature_contributions.sort(key=lambda x: x["contribution"], reverse=True)
        
        return feature_contributions[:top_k]

if __name__ == "__main__":
    # Simple local test
    explainer = FraudExplainer()
    
    # Dummy transaction
    dummy_tx = {f: np.random.random() for f in ALL_FEATURE_NAMES}
    dummy_tx["amount"] = 50000.0  # high amount to test
    
    print("Testing FraudExplainer...")
    reasons = explainer.get_reason_codes(dummy_tx)
    for r in reasons:
        print(f"Feature: {r['feature']:<25} | Value: {r['value']:<10.2f} | SHAP: {r['contribution']:.4f}")
