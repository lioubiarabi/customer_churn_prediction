from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.data_loader.featured import load_featured_data

df = load_featured_data()

X_cluster = df.drop(columns=['Churn'])
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'active_services']
cat_cols = [c for c in X_cluster.columns if c not in num_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
    ]
)

X_scaled = preprocessor.fit_transform(X_cluster)
print(f"Features ready for clustering: {X_scaled.shape}")

k_range = range(2, 15)
inertias = []
silhouettes = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)
    sil = silhouette_score(X_scaled, labels)
    silhouettes.append(sil)
    print(f"k={k} -> Inertia: {kmeans.inertia_:.2f} | Silhouette Score: {sil:.4f}")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))


ax1.plot(k_range, inertias, marker='o', color='#1f77b4', linewidth=1)
ax1.set_title("Elbow Method (Inertia)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia")
ax1.grid(True, alpha=0.3)

ax2.plot(k_range, silhouettes, marker='s', color='#d62728', linewidth=1, linestyle='--')
ax2.set_title("Silhouette Score by k", fontsize=12, fontweight='bold')
ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Silhouette Score")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()