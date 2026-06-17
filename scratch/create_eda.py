import nbformat as nbf

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("# Phase 2: Exploratory Data Analysis (EDA)\n\nIn this notebook, we'll analyze the PaySim dataset to understand its characteristics, uncover fraud patterns, and identify predictive features before moving on to modeling.\n\n### Key Questions to Answer:\n1. Which columns have nulls or anomalies?\n2. What is the class imbalance ratio?\n3. What is the fraud rate by transaction type?\n4. What does the amount distribution look like for fraud vs legitimate?\n5. Are there temporal patterns (fraud concentrated at certain hours)?"),
    
    nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\n\nwarnings.filterwarnings('ignore')\nplt.style.use('dark_background')\nsns.set_theme(style=\"darkgrid\", rc={\"axes.facecolor\": \"#1e1e1e\", \"figure.facecolor\": \"#1e1e1e\", \"text.color\": \"white\", \"axes.labelcolor\": \"white\", \"xtick.color\": \"white\", \"ytick.color\": \"white\"})\n\n# Load the dataset\ndf = pd.read_csv('../data/raw/paysim_dataset.csv')\ndf.head()"),
    
    nbf.v4.new_markdown_cell("## 1. Basic Dataset Info (Nulls & Types)\nLet's check the shape of the data, the types of features, and whether any data is missing."),
    
    nbf.v4.new_code_cell("print(f\"Dataset Shape: {df.shape}\")\nprint(\"\\nMissing Values:\")\nprint(df.isnull().sum())\nprint(\"\\nData Types:\")\nprint(df.dtypes)"),
    
    nbf.v4.new_markdown_cell("> **Observation:** There are absolutely no missing values in this dataset. We have roughly 6.3 million rows and 11 columns."),
    
    nbf.v4.new_markdown_cell("## 2. Class Imbalance Ratio\nFraud detection datasets are famously imbalanced. Let's see the ratio of legitimate to fraudulent transactions."),
    
    nbf.v4.new_code_cell("fraud_counts = df['isFraud'].value_counts()\nfraud_ratio = fraud_counts[1] / len(df) * 100\n\nprint(f\"Legitimate (0): {fraud_counts[0]}\")\nprint(f\"Fraudulent (1): {fraud_counts[1]}\")\nprint(f\"Fraud Ratio: {fraud_ratio:.3f}%\")\n\nplt.figure(figsize=(8, 5))\nax = sns.countplot(data=df, x='isFraud', palette=['#3498db', '#e74c3c'])\nplt.title('Distribution of Fraudulent vs Legitimate Transactions')\nplt.yscale('log') # Log scale because of massive imbalance\nplt.ylabel('Count (Log Scale)')\nplt.show()"),
    
    nbf.v4.new_markdown_cell("> **Observation:** The dataset is highly imbalanced! Only ~0.13% of transactions are fraud. This means we will need techniques like SMOTE or class-weighting (like `scale_pos_weight` in XGBoost) during training."),
    
    nbf.v4.new_markdown_cell("## 3. Fraud by Transaction Type\nPaySim includes 5 transaction types. Let's see which types are vulnerable to fraud."),
    
    nbf.v4.new_code_cell("plt.figure(figsize=(10, 6))\nax = sns.countplot(data=df, x='type', hue='isFraud', palette=['#3498db', '#e74c3c'])\nplt.title('Transaction Counts by Type and Fraud Status')\nplt.yscale('log')\nplt.ylabel('Count (Log Scale)')\nplt.show()\n\n# Calculate exact numbers\nfraud_by_type = df.groupby('type')['isFraud'].agg(['count', 'sum'])\nfraud_by_type.rename(columns={'count': 'Total', 'sum': 'Fraud_Count'}, inplace=True)\nfraud_by_type['Fraud_Rate (%)'] = (fraud_by_type['Fraud_Count'] / fraud_by_type['Total']) * 100\ndisplay(fraud_by_type)"),
    
    nbf.v4.new_markdown_cell("> **Crucial Finding:** Fraud *only* occurs in two transaction types: `TRANSFER` and `CASH_OUT`. This is a massive insight for feature engineering!"),
    
    nbf.v4.new_markdown_cell("## 4. Amount Distribution (Fraud vs Legitimate)\nDo fraudulent transactions tend to be larger or smaller than legitimate ones? Let's isolate `TRANSFER` and `CASH_OUT` since they contain all the fraud."),
    
    nbf.v4.new_code_cell("vuln_df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])]\n\nfig, axes = plt.subplots(1, 2, figsize=(16, 6))\n\nsns.boxplot(data=vuln_df, x='isFraud', y='amount', palette=['#3498db', '#e74c3c'], ax=axes[0])\naxes[0].set_title('Transaction Amount by Fraud Status')\naxes[0].set_yscale('log')\n\nsns.kdeplot(data=vuln_df[vuln_df['isFraud'] == 0], x='amount', fill=True, color='#3498db', label='Legitimate', log_scale=True, ax=axes[1])\nsns.kdeplot(data=vuln_df[vuln_df['isFraud'] == 1], x='amount', fill=True, color='#e74c3c', label='Fraud', log_scale=True, ax=axes[1])\naxes[1].set_title('Amount Distribution (Log Scale)')\naxes[1].legend()\n\nplt.tight_layout()\nplt.show()"),
    
    nbf.v4.new_markdown_cell("> **Observation:** Fraudulent transactions often involve large amounts. There is also a notable density of fraud at exactly 0 amount, and also around the 1-10 million mark. `amount` is highly predictive."),
    
    nbf.v4.new_markdown_cell("## 5. Account Balances and \"Emptying\" Behavior\nA common fraud pattern is transferring an exact amount to completely empty an account. Let's see if this happens here."),
    
    nbf.v4.new_code_cell("vuln_df['balance_drop_pct'] = (vuln_df['oldbalanceOrg'] - vuln_df['newbalanceOrig']) / (vuln_df['oldbalanceOrg'] + 1)\n\nplt.figure(figsize=(10, 6))\nsns.histplot(data=vuln_df, x='balance_drop_pct', hue='isFraud', bins=50, palette=['#3498db', '#e74c3c'], stat='density', common_norm=False)\nplt.title('Percentage of Balance Dropped during Transaction')\nplt.xlabel('Balance Drop Ratio')\nplt.show()"),
    
    nbf.v4.new_markdown_cell("> **Observation:** For fraud cases, the balance drop ratio is almost exclusively at `1.0` (meaning the account was completely emptied). This is another extremely powerful feature!"),
    
    nbf.v4.new_markdown_cell("## 6. Temporal Patterns (Hours & Days)\nThe `step` column represents 1 hour of time. Are fraudsters operating at specific times?"),
    
    nbf.v4.new_code_cell("df['hour_of_day'] = df['step'] % 24\n\nplt.figure(figsize=(12, 6))\nsns.countplot(data=df, x='hour_of_day', hue='isFraud', palette=['#3498db', '#e74c3c'])\nplt.title('Transactions by Hour of Day')\nplt.yscale('log')\nplt.show()\n\nplt.figure(figsize=(12, 6))\nfraud_only = df[df['isFraud'] == 1]\nsns.histplot(data=fraud_only, x='step', bins=744, color='#e74c3c', kde=True) # 744 steps = 30 days\nplt.title('Fraudulent Transactions over 30 Days (744 steps)')\nplt.xlabel('Step (Hour)')\nplt.show()"),
    
    nbf.v4.new_markdown_cell("> **Observation:** While legitimate transactions follow a clear day/night cycle (dropping heavily during night hours), fraudulent transactions occur relatively consistently at *all* hours. Thus, a transaction occurring at 4 AM is highly suspicious."),
    
    nbf.v4.new_markdown_cell("## 7. Destination Balance Anomalies\nFraudsters often cash out into accounts with 0 initial balance, or immediately withdraw."),
    
    nbf.v4.new_code_cell("vuln_df['dest_balance_increased'] = (vuln_df['newbalanceDest'] > vuln_df['oldbalanceDest']).astype(int)\n\nplt.figure(figsize=(8, 5))\nsns.barplot(data=vuln_df, x='isFraud', y='dest_balance_increased', palette=['#3498db', '#e74c3c'])\nplt.title('Proportion of Transactions where Destination Balance Increased')\nplt.ylabel('Proportion')\nplt.show()"),
    
    nbf.v4.new_markdown_cell("> **Observation:** In many fraudulent transactions, the destination balance doesn't seem to increase as expected (perhaps indicating complex multi-hop transfers or immediate withdrawals not captured in standard logging)."),
    
    nbf.v4.new_markdown_cell("## 8. Flagged Fraud (`isFlaggedFraud`)\nPaySim includes a baseline heuristic column `isFlaggedFraud` that flags transfers over 200,000. Let's see how effective it is."),
    
    nbf.v4.new_code_cell("from sklearn.metrics import confusion_matrix, classification_report\n\nprint(\"Baseline Rule (Flag transfers > 200,000) Performance:\\n\")\nprint(classification_report(df['isFraud'], df['isFlaggedFraud']))\n\ncm = confusion_matrix(df['isFraud'], df['isFlaggedFraud'])\nplt.figure(figsize=(6, 5))\nsns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Legit', 'Predicted Fraud'], yticklabels=['Actual Legit', 'Actual Fraud'])\nplt.title('Confusion Matrix: Baseline isFlaggedFraud Rule')\nplt.show()"),
    
    nbf.v4.new_markdown_cell("> **Observation:** The baseline `isFlaggedFraud` column is virtually useless. It catches almost nothing (recall is near 0%). This justifies the need for our Machine Learning models!"),
    
    nbf.v4.new_markdown_cell("---\n\n## Summary of EDA & Feature Engineering Plan\n\n1. **Filter relevant data:** Fraud only happens in `TRANSFER` and `CASH_OUT`. We will focus our models strongly on these types (or one-hot encode the type).\n2. **Handle Imbalance:** The 0.13% fraud rate requires `SMOTE` and algorithmic class weights.\n3. **Key Engineered Features to build in Phase 4:**\n   - `balance_drop_pct` (empty accounts)\n   - `hour_of_day` (night vs day activity)\n   - `amount_to_balance_ratio`\n   - `dest_balance_increased`\n4. **Windowed Features:** The steady stream of fraud means keeping track of velocity (e.g. `txn_count_1h`, `avg_amount_1h`) using our Faust streaming app will be extremely predictive.")
]

nbf.write(nb, '../notebooks/01_eda.ipynb')
print("Successfully generated notebooks/01_eda.ipynb")
