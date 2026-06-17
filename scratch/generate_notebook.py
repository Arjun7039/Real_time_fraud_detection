import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("""# Phase 6: Explainability and SHAP Analysis

In this notebook, we load the powerful XGBoost model trained in Phase 5 and use **SHAP** (SHapley Additive exPlanations) to interpret its decisions.

We will generate:
1. **Global Summary Plot**: Which features drive fraud overall?
2. **Fraud Waterfall**: A deep dive into a single transaction flagged as fraud.
3. **Legitimate Waterfall**: A deep dive into a legitimate transaction."""))

cells.append(nbf.v4.new_code_cell("""import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# Make sure we can import from our project root
sys.path.insert(0, os.path.dirname(os.getcwd()))
from features.feature_definitions import ALL_FEATURE_NAMES

# Initialize JS for SHAP plots
shap.initjs()"""))

cells.append(nbf.v4.new_markdown_cell("### 1. Load Data and Model"))

cells.append(nbf.v4.new_code_cell("""# Load the trained XGBoost model
model_path = os.path.join(os.path.dirname(os.getcwd()), "data", "models", "xgboost_model.json")
model = xgb.XGBClassifier()
model.load_model(model_path)

# Load a sample of the processed features to avoid memory explosion
data_path = os.path.join(os.path.dirname(os.getcwd()), "data", "processed", "features.parquet")
df = pd.read_parquet(data_path)

# We will take a random sample of 5,000 rows for global explanation
df_sample = df.sample(n=5000, random_state=42)
X_sample = df_sample[ALL_FEATURE_NAMES]
y_sample = df_sample["isFraud"]

print(f"Sample loaded: {X_sample.shape}")
print(f"Fraud cases in sample: {y_sample.sum()}")"""))

cells.append(nbf.v4.new_markdown_cell("### 2. Global Explainability (Summary Plot)"))

cells.append(nbf.v4.new_code_cell("""# Create the TreeExplainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for the sample
shap_values = explainer.shap_values(X_sample)

# Generate Summary Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("SHAP Global Summary Plot")
plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("### 3. Local Explainability (Waterfall Plots)"))

cells.append(nbf.v4.new_code_cell("""# Find one actual fraud transaction and one legitimate transaction
fraud_indices = df_sample[df_sample["isFraud"] == 1].index
legit_indices = df_sample[df_sample["isFraud"] == 0].index

if len(fraud_indices) > 0:
    fraud_idx = fraud_indices[0]
    fraud_idx_iloc = np.where(X_sample.index == fraud_idx)[0][0]
    
    print("========================================")
    print(" WATERFALL PLOT FOR FRAUDULENT TRANSACTION")
    print("========================================")
    
    # We use explainer.shap_values() but formatted for waterfall
    # For waterfall, we need an Explanation object
    explanation = shap.Explanation(
        values=shap_values[fraud_idx_iloc], 
        base_values=explainer.expected_value, 
        data=X_sample.iloc[fraud_idx_iloc],
        feature_names=ALL_FEATURE_NAMES
    )
    shap.waterfall_plot(explanation)
else:
    print("No fraud cases in this random sample.")"""))

cells.append(nbf.v4.new_code_cell("""if len(legit_indices) > 0:
    legit_idx = legit_indices[0]
    legit_idx_iloc = np.where(X_sample.index == legit_idx)[0][0]
    
    print("========================================")
    print(" WATERFALL PLOT FOR LEGITIMATE TRANSACTION")
    print("========================================")
    
    explanation = shap.Explanation(
        values=shap_values[legit_idx_iloc], 
        base_values=explainer.expected_value, 
        data=X_sample.iloc[legit_idx_iloc],
        feature_names=ALL_FEATURE_NAMES
    )
    shap.waterfall_plot(explanation)"""))

nb['cells'] = cells

with open('notebooks/04_shap_analysis.ipynb', 'w') as f:
    nbf.write(nb, f)
    
print("Successfully generated notebooks/04_shap_analysis.ipynb")
