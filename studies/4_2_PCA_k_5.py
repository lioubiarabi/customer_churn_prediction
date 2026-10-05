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

# 1. Load cleaned/featured data
df = load_featured_data()

# 2. Exclude target 'Churn'
X = df.drop(columns=['Churn'])

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'active_services']
cat_cols = [c for c in X.columns if c not in num_cols]

# 3. Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols),
    ]
)
X_scaled = preprocessor.fit_transform(X)

# 4. K-Means clustering with k=5
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster_5'] = kmeans.fit_predict(X_scaled)

# 5. Fit PCA to project into 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

var_ratio = pca.explained_variance_ratio_

# 6. Plot 2D PCA projection for k=5
plt.figure(figsize=(10, 6))
palette = sns.color_palette("tab10", 5)

sns.scatterplot(
    data=df,
    x='PCA1',
    y='PCA2',
    hue='cluster_5',
    palette=palette,
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
    f"Customer Clusters Visualized with PCA (k=5, Total Variance: {np.sum(var_ratio)*100:.1f}%)",
    fontsize=13,
    fontweight='bold',
)
plt.xlabel(
    f"Principal Component 1 ({var_ratio[0]*100:.1f}% variance)", fontsize=11
)
plt.ylabel(
    f"Principal Component 2 ({var_ratio[1]*100:.1f}% variance)", fontsize=11
)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, title="Cluster")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()