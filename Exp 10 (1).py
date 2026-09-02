import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris

# Load built-in Iris dataset
data = load_iris()

# Use two features
X = data.data[:, [0, 1]]

# Create Gaussian Mixture Model
gmm = GaussianMixture(
    n_components=3,
    random_state=42
)

# Train the model using EM algorithm
gmm.fit(X)

# Predict clusters
clusters = gmm.predict(X)

# Display results
print("Expectation Maximization Algorithm")
print("-----------------------------------")

print("Number of Clusters:", 3)

print("\nCluster Labels:")
print(clusters)

print("\nCluster Means:")
print(gmm.means_)

print("\nCluster Covariances:")
print(gmm.covariances_)

print("\nLog Likelihood:")
print(round(gmm.score(X) * len(X), 2))
