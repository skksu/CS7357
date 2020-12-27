import random
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Normalized dataset X
def normalize(X, axis=-1, p=2):
    lp_norm = np.atleast_1d(np.linalg.norm(X, p, axis))
    lp_norm[lp_norm == 0] = 1
    return X / np.expand_dims(lp_norm, axis)


# Calculate the square of the Euclidean distance between a sample and all samples in the data set
def euclidean_distance(one_sample, X):
    one_sample = one_sample.reshape(1, -1)
    X = X.reshape(X.shape[0], -1)
    distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
    return distances



class Kmeans():
    """Kmeans clustering algorithm.

     Parameters:
     -----------
     k: int
         The number of clusters.
     max_iterations: int
         The maximum number of iterations.
     varepsilon: float
         Determine whether to converge, if the difference between all k cluster centers of the previous time and all k cluster centers of this time is less than varepsilon,
         Then the algorithm has converged
     """
    def __init__(self, k=2, max_iterations=500, varepsilon=0.0001):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon

# Randomly select self.k samples from all samples as the initial cluster center
    def init_random_centroids(self, X):
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

# Return to the nearest center index of the sample [0, self.k)
    def _closest_centroid(self, sample, centroids):
        distances = euclidean_distance(sample, centroids)
        closest_i = np.argmin(distances)
        return closest_i

# Classify all samples, the classification rule is to classify the sample to the nearest center
    def create_clusters(self, centroids, X):
        n_samples = np.shape(X)[0]
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self._closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

# Update the center
    def update_centroids(self, clusters, X):
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

# Categorize all samples, the index of their category is their category label
    def get_cluster_labels(self, clusters, X):
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

# Perform Kmeans clustering on the entire data set X and return its cluster label
    def predict(self, X):
        # Randomly select self.k samples from all samples as the initial cluster center
        centroids = self.init_random_centroids(X)

        # Iterate until the algorithm converges (the cluster center of the previous time and 
        #the cluster center of this time almost coincide) or the maximum number of iterations is reached
        for _ in range(self.max_iterations):
            # All are classified, the classification rule is to classify the sample to the nearest center
            clusters = self.create_clusters(centroids, X)
            former_centroids = centroids

            # Calculate new cluster centers
            centroids = self.update_centroids(clusters, X)

            # If there is almost no change in the cluster center, the algorithm has converged, and the iteration is exited
            diff = centroids - former_centroids
            if diff.any() < self.varepsilon:
                break

        return self.get_cluster_labels(clusters, X)


def main():
    # Load the dataset
    X, y = datasets.make_blobs(n_samples=10000,
                               n_features=3,
                               centers=[[3,3, 3], [0,0,0], [1,1,1], [2,2,2]],
                               cluster_std=[0.2, 0.1, 0.2, 0.2],
                               random_state =9)

    # Clustering with Kmeans algorithm
    clf = Kmeans(k=4)
    y_pred = clf.predict(X)


    # Visual clustering effect
    fig = plt.figure(figsize=(12, 8))
    ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
    plt.scatter(X[y==0][:, 0], X[y==0][:, 1], X[y==0][:, 2])
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], X[y==1][:, 2])
    plt.scatter(X[y==2][:, 0], X[y==2][:, 1], X[y==2][:, 2])
    plt.scatter(X[y==3][:, 0], X[y==3][:, 1], X[y==3][:, 2])
    plt.show()


if __name__ == "__main__":
    main()
