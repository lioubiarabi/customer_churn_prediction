from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate 300 points divided into 4 clusters
X, y_true = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=0.60,
    random_state=42
)

# Plot raw points
plt.scatter(X[:, 0], X[:, 1], s=30, color='gray')
plt.title("Raw Unlabeled Data")
plt.show()