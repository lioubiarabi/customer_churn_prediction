from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate 300 points divided into 4 clusters
X, y_true = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=0.60,
    random_state=42
)


from sklearn.cluster import KMeans

# Initialize KMeans with 4 clusters
kmeans = KMeans(
    n_clusters=4,
    init='k-means++',   # Smart centroid initialization to speed up convergence
    n_init=10,          # Runs the algorithm 10 times with different seeds
    max_iter=300,       # Maximum iterations per run
    random_state=42
)

# Fit model to data
kmeans.fit(X)


# Cluster label for each data point (0, 1, 2, or 3)
labels = kmeans.labels_

# Coordinates of the cluster centers
centroids = kmeans.cluster_centers_

print("First 5 labels:", labels[:5])
print("Centroid coordinates:\n", centroids)

# Plot clustered points colored by their assigned label
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)

# Overlay the cluster centroids as red crosses
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c='red',
    marker='X',
    s=200,
    label='Centroids'
)

plt.title("K-Means Clustering Result")
plt.legend()
plt.show()

inertias = []
k_range = range(1, 10)

for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

# Plot inertia vs k
plt.plot(k_range, inertias, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method For Optimal k")
plt.show()

from sklearn.metrics import silhouette_score
silhouette_scores = []
k_range = range(2, 10)  # Must start at 2

for k in k_range:
    # Fit KMeans and get cluster labels
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)

    # Calculate the mean silhouette coefficient across all points
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

# Plot the scores
plt.figure(figsize=(8, 4))
plt.plot(k_range, silhouette_scores, marker='o', color='teal')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis For Optimal k")
plt.grid(True)
plt.show()