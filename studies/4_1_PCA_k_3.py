from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader.featured import load_featured_data

# 1. Load the featured dataset
df = load_featured_data()

# 2. Exclude target 'Churn' from clustering input
X = df.drop(columns=['Churn'])

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'active_services']
cat_cols = [c for c in X.columns if c not in num_cols]

# 3. Pipeline: scaling and one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols),
    ]
)
X_scaled = preprocessor.fit_transform(X)

# 4. K-Means clustering (k=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 5. Fit PCA to project into 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

var_ratio = pca.explained_variance_ratio_

# 6. Map readable personas to clusters
cluster_names = {
    0: 'Cluster 0',
    1: 'Cluster 1',
    2: 'Cluster 2',
}
df['cluster_label'] = df['cluster'].map(cluster_names)

# 7. Plot 2D PCA projection
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='PCA1',
    y='PCA2',
    hue='cluster_label',
    palette=['#1f77b4', '#ff7f0e', '#2ca02c'],
    alpha=0.6,
    s=40,
)

# Project and plot centroids
centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(
    centroids_pca[:, 0],
    centroids_pca[:, 1],
    marker='X',
    s=220,
    c='red',
    edgecolor='black',
    linewidth=1.5,
    label='Centroids',
)

plt.title(
    f"Customer Clusters Visualized with PCA (Total Variance: {np.sum(var_ratio)*100:.1f}%)",
    fontsize=13,
    fontweight='bold',
)
plt.xlabel(
    f"Principal Component 1 ({var_ratio[0]*100:.1f}% variance)", fontsize=11
)
plt.ylabel(
    f"Principal Component 2 ({var_ratio[1]*100:.1f}% variance)", fontsize=11
)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
