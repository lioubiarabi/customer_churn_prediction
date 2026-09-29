from sklearn.datasets import make_blobs

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