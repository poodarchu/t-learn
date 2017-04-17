from sklearn import datasets
import numpy as np
from utils.data_manipulate import normalize
from utils.data_operate import euclidean_distance
from unsupervised_learning.PCA import PCA

"""
A simple clustering method to form k clusters by iteratively reassigning sampels to the closest centroids 
and after that moves the centroids to the center of the new formed clusters.
"""
class kMeans():
    def __init__(self, k=2, max_iterations=200):
        self.k = k
        self.max_iter = max_iterations

    # Initialise the centroids of train data set X as random samples
    def _init_random_centroids(self, X):
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid

        return centroids

    # Return the index of the closest centroid to the sample
    def _find_closest_centroid(self, sample, centroids):
        closest_i = None
        closest_distance = float("inf")
        for i, centroid in enumerate(centroids):
            distance = euclidean_distance(sample, centroid)
            if distance < closest_distance:
                closest_distance = distance;
                closest_i = i

        return closest_i

    # Assign the samples to the closest centroids to create clusters
    def _create_clusters(self, centroids, X):
        # n_samples = np.shape(X)[0]
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(X):
            centroid_i = self._find_closest_centroid(sample, centroids)
            clusters[centroid_i].append(idx)

        return clusters

    # Calculate new centroids as mean of the samples in each cluster
    def _calculate_new_centroids(self, clusters, X):
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid

        return centroids

    # Classify samples as the index of their clusters
    def _get_cluster_labels(self, clusters, X):
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i

        return y_pred

    # Do k-means clustering and return cluster indices
    def predict(self, X):
        # Initialise centroids
        centroids = self._init_random_centroids(X)

        # Iterate util convergence or get to max iterations
        for _ in range(self.max_iter):
            # Assign samples to closest centroids (create clusters)
            clusters = self._create_clusters(centroids, X)
            prev_centroids = centroids
            # Calculate new centroids from the clusters
            centroids = self._calculate_new_centroids(clusters, X)

            # If no centroids have changed => convergence
            diff = centroids - prev_centroids
            if not diff.any():
                break

        return self._get_cluster_labels(clusters, X)

if __name__ == "__main__":
    # Load the dataset
    X, y = datasets.make_blobs()

    # Clustering using k_means
    clf = kMeans(k=3)
    y_pred = clf.predict(X)

    # Using PCA to plot
