from pathlib import Path
import warnings
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loader.featured import load_featured_data

warnings.filterwarnings('ignore')

# 1. Chargement des données
df = load_featured_data()

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
X = df.drop(columns=['Churn'])
y = df['Churn']

# 2. Séparation Train / Test stratifiée (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Définition des groupes de variables
num_cols_raw = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = [
    c for c in X.columns if c not in num_cols_raw and c != 'active_services'
]
num_cols_fe = ['tenure', 'MonthlyCharges', 'TotalCharges', 'active_services']

# Préprocesseurs distincts pour isoler les apports
prep_raw = ColumnTransformer([
    ('num', StandardScaler(), num_cols_raw),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
])

prep_fe = ColumnTransformer([
    ('num', StandardScaler(), num_cols_fe),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
])

# 4. Génération du Cluster sur Train (Anti Data-Leakage)
X_train_scaled = prep_fe.fit_transform(X_train)
X_test_scaled = prep_fe.transform(X_test)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
X_train_clustered = X_train.copy()
X_test_clustered = X_test.copy()
X_train_clustered['cluster'] = kmeans.fit_predict(X_train_scaled).astype(str)
X_test_clustered['cluster'] = kmeans.predict(X_test_scaled).astype(str)

prep_cluster = ColumnTransformer([
    ('num', StandardScaler(), num_cols_fe),
    (
        'cat',
        OneHotEncoder(drop='first', handle_unknown='ignore'),
        cat_cols + ['cluster'],
    ),
])

# 5. Modèle Baseline (Avant Optimisation)
pipe_baseline = Pipeline([
    ('prep', prep_fe),
    (
        'clf',
        LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42
        ),
    ),
])
pipe_baseline.fit(X_train, y_train)

# 6. Optimisation des Hyperparamètres (GridSearchCV + StratifiedKFold)
param_grid = {
    'clf__C': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    'clf__penalty': ['l1', 'l2'],
    'clf__solver': ['liblinear', 'saga'],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipe_baseline,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

print(f'Meilleurs hyperparamètres : {grid_search.best_params_}')
print(f'Meilleur score ROC-AUC (CV 5-folds) : {grid_search.best_score_:.4f}')

# 7. Évaluation des Configurations pour Comparaison
configs = {
    '1. Avant Optimisation (Défaut)': (pipe_baseline, X_train, X_test),
    '2. Après Optimisation (GridSearch)': (
        grid_search.best_estimator_,
        X_train,
        X_test,
    ),
    '3. Données Brutes (Sans FE, Sans Cluster)': (
        Pipeline([
            ('prep', prep_raw),
            (
                'clf',
                LogisticRegression(
                    C=0.5,
                    penalty='l2',
                    solver='saga',
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]),
        X_train,
        X_test,
    ),
    '4. Complet (FE + Cluster Optimisé)': (
        Pipeline([
            ('prep', prep_cluster),
            (
                'clf',
                LogisticRegression(
                    C=0.5,
                    penalty='l2',
                    solver='saga',
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]),
        X_train_clustered,
        X_test_clustered,
    ),
}

results = []
for name, (pipeline, xtr, xte) in configs.items():
  pipeline.fit(xtr, y_train)
  y_pred = pipeline.predict(xte)
  y_proba = pipeline.predict_proba(xte)[:, 1]

  results.append({
      'Configuration': name,
      'Precision': round(precision_score(y_test, y_pred), 4),
      'Recall': round(recall_score(y_test, y_pred), 4),
      'F1-Score': round(f1_score(y_test, y_pred), 4),
      'ROC-AUC': round(roc_auc_score(y_test, y_proba), 4),
  })

comp_df = pd.DataFrame(results)
print('\n' + comp_df.to_string(index=False))