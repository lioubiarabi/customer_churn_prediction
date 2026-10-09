from pathlib import Path
import warnings
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loader.featured import load_featured_data

warnings.filterwarnings('ignore')

# 1. Chargement des données nettoyées / feature engineering
df = load_featured_data()

# Encodage de la cible
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

X = df.drop(columns=['Churn'])
y = df['Churn']

# 2. Séparation Train / Test AVANT tout clustering (Anti Data-Leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Préparateurs pour les données initiales
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'active_services']
cat_cols = [c for c in X.columns if c not in num_cols]

preprocessor_initial = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
    ]
)

# 4. Apprentissage du K-Means (k=3) UNIQUEMENT sur le Train Set
X_train_scaled = preprocessor_initial.fit_transform(X_train)
X_test_scaled = preprocessor_initial.transform(X_test)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
X_train_clusters = kmeans.fit_predict(X_train_scaled)
X_test_clusters = kmeans.predict(X_test_scaled)

# Ajout de la variable 'cluster' comme variable catégorielle
X_train_clustered = X_train.copy()
X_train_clustered['cluster'] = X_train_clusters.astype(str)

X_test_clustered = X_test.copy()
X_test_clustered['cluster'] = X_test_clusters.astype(str)

preprocessor_with_cluster = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        (
            'cat',
            OneHotEncoder(drop='first', handle_unknown='ignore'),
            cat_cols + ['cluster'],
        ),
    ]
)

# 5. Définition des modèles à tester
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
}

results = []

for name, clf in models.items():
  # Modèle 1 : Variables initiales (Sans Cluster)
  pipe_base = Pipeline([('prep', preprocessor_initial), ('clf', clf)])
  pipe_base.fit(X_train, y_train)
  y_pred_b = pipe_base.predict(X_test)
  y_proba_b = pipe_base.predict_proba(X_test)[:, 1]

  results.append({
      'Modèle': name,
      'Configuration': 'Variables Initiales',
      'Precision': round(precision_score(y_test, y_pred_b), 4),
      'Recall': round(recall_score(y_test, y_pred_b), 4),
      'F1-Score': round(f1_score(y_test, y_pred_b), 4),
      'ROC-AUC': round(roc_auc_score(y_test, y_proba_b), 4),
  })

  # Modèle 2 : Variables initiales + cluster
  pipe_clust = Pipeline([('prep', preprocessor_with_cluster), ('clf', clf)])
  pipe_clust.fit(X_train_clustered, y_train)
  y_pred_c = pipe_clust.predict(X_test_clustered)
  y_proba_c = pipe_clust.predict_proba(X_test_clustered)[:, 1]

  results.append({
      'Modèle': name,
      'Configuration': 'Initiales + Cluster',
      'Precision': round(precision_score(y_test, y_pred_c), 4),
      'Recall': round(recall_score(y_test, y_pred_c), 4),
      'F1-Score': round(f1_score(y_test, y_pred_c), 4),
      'ROC-AUC': round(roc_auc_score(y_test, y_proba_c), 4),
  })

# 6. Affichage du tableau comparatif
comparison_table = pd.DataFrame(results)
print(comparison_table.to_string(index=False))