from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data_loader.featured import load_featured_data

# 1. Load data containing features and the assigned cluster
df = load_featured_data()

# 2. Encode binary target: 'Yes' -> 1, 'No' -> 0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

X = df.drop(columns=['Churn'])
y = df['Churn']

# Ensure 'cluster' is treated as a categorical column
if 'cluster' in X.columns:
  X['cluster'] = X['cluster'].astype(str)

# 3. Train/Test split with stratification to preserve 73/27 balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Define ColumnTransformer
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'active_services']
cat_cols = [c for c in X.columns if c not in num_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        (
            'cat',
            OneHotEncoder(drop='first', handle_unknown='ignore'),
            cat_cols,
        ),
    ]
)

# 5. Define candidate models handling class imbalance
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=5, class_weight='balanced', random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42
    ),
    'SVM': SVC(probability=True, class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
}

# 6. Train, Predict, and Evaluate
results = []
trained_pipelines = {}

for name, clf in models.items():
  pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])

  # Fit strictly on train data
  pipeline.fit(X_train, y_train)
  trained_pipelines[name] = pipeline

  # Predictions
  y_pred = pipeline.predict(X_test)
  y_proba = pipeline.predict_proba(X_test)[:, 1]

  # Compute core metrics
  results.append({
      'Model': name,
      'Precision': round(precision_score(y_test, y_pred), 4),
      'Recall': round(recall_score(y_test, y_pred), 4),
      'F1-Score': round(f1_score(y_test, y_pred), 4),
      'ROC-AUC': round(roc_auc_score(y_test, y_proba), 4),
  })

# Display comparison table
metrics_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
print(metrics_df.to_string(index=False))

# 7. Plot Confusion Matrix for Best Model (Logistic Regression)
best_model = trained_pipelines['Logistic Regression']
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Retained (0)', 'Churn (1)'],
    yticklabels=['Retained (0)', 'Churn (1)'],
)
plt.title(
    'Confusion Matrix - Logistic Regression (Balanced)',
    fontweight='bold',
    fontsize=11,
)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()