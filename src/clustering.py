# clustering.py

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class KMeansClustering:
    def __init__(self):
        pass

    def cosine_similarity(self, x, y):
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        similarity = dot_product / (norm_x * norm_y)
        return similarity

    def initialize_clusters(self, data, k):
        """Randomly initialize k cluster means as k distinct data points."""
        np.random.seed(42)
        indices = np.random.choice(len(data), k, replace=False)
        centroids = data.iloc[indices]
        return centroids

    def assign_clusters(self, data, centroids):
        """Assign each data point to the nearest centroid."""
        distances = np.array([[self.cosine_similarity(data_point, centroid) for centroid in centroids.values] for data_point in data.values])
        labels = np.argmax(distances, axis=1)
        return labels

    def update_centroids(self, data, labels, k, centroids):
        """Update centroids based on the mean of data points in each cluster."""
        new_centroids = []
        for i in range(k):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                new_centroid = cluster_points.mean(axis=0)
            else:
                # If a cluster has no points assigned, keep the centroid unchanged
                new_centroid = centroids.iloc[i]
            new_centroids.append(new_centroid)
        return pd.DataFrame(new_centroids, columns=data.columns)

    def fit(self, data, k, iterations=20):
        """Perform K-means clustering."""
        # Initialize centroids
        centroids = self.initialize_clusters(data, k)

        # Iterate for the specified number of iterations
        for _ in range(iterations):
            # Assign data points to clusters
            labels = self.assign_clusters(data, centroids)

            # Update centroids based on the mean of data points in each cluster
            centroids = self.update_centroids(data, labels, k, centroids)

        return labels, centroids
    
    def plot(self, data, k, labels, centroids, file_name):
        """Plot K-means clusters with PCA and save the plot."""

        output_file = "../output/plots/" + file_name
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(data)

        # Plot the data points with their assigned clusters
        plt.figure(figsize=(10, 6))
        for cluster_label in range(k):
            cluster_points = pca_data[labels == cluster_label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_label}')

        # Plot the centroids of each cluster
        centroid_points = pca.transform(centroids)
        plt.scatter(centroid_points[:, 0], centroid_points[:, 1], marker='x', s=100, c='black', label='Centroids')

        plt.title('K-means Clustering with PCA (2D)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.savefig(output_file)
    
    def save_clusters(self, labels, file_name):
        """Save cluster information to a file."""
        output_file = "../output/" + file_name

        clusters = [[] for _ in range(len(set(labels)))]

        # Assign data points to clusters based on K-means labels
        for i, label in enumerate(labels):
            clusters[label].append(i)

        # Sort clusters by the minimum index of the data points present in each cluster
        clusters = sorted(clusters, key=lambda x: min(x))

        # Write cluster information to the specified file
        with open(output_file, "w") as file:
            index = 0
            for cluster in clusters:
                # file.write(f"cluster {index}: ")
                index = index + 1
                file.write(",".join(map(str, sorted(cluster))) + "\n")

class CompleteLinkageClustering:
    def __init__(self):
        pass

    def fit(self, data, k):
        """Complete Linkage Agglomerative Clustering."""
        # Initialize each data point as its own cluster
        clusters = [[i] for i in range(len(data))]
        
        # Calculate pairwise distances between data points
        distances = self.pairwise_distances(data)
        
        # Merge clusters until the number of clusters equals k
        while len(clusters) > k:
            min_distance = float('inf')
            merge_indices = None
            
            # Find the pair of clusters with the smallest distance
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    distance = np.max([distances[m][n] for m in clusters[i] for n in clusters[j]])
                    if distance < min_distance:
                        min_distance = distance
                        merge_indices = (i, j)
            
            # Merge the closest pair of clusters
            clusters[merge_indices[0]] += clusters[merge_indices[1]]
            del clusters[merge_indices[1]]
        
        return clusters

    def euclidean_distance(self, x, y):
        """Compute the Euclidean distance between two vectors."""
        return np.sqrt(np.sum((x - y) ** 2))

    def pairwise_distances(self, data):
        """Compute pairwise distances between all data points."""
        n = len(data)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                distances[i, j] = self.euclidean_distance(data[i], data[j])
                distances[j, i] = distances[i, j]  # Since distance matrix is symmetric
        return distances
    
    def save_clusters(self, clusters, file_name):
        """Save cluster information to a file."""
        output_file = "../output/" + file_name
        # Sort clusters by the minimum index of the data points present in each cluster
        clusters = sorted(clusters, key=lambda x: min(x))

        # Write cluster information to the specified file
        with open(output_file, "w") as file:
            index = 0
            for cluster in clusters:
                # file.write(f"cluster {index}: ")
                index = index + 1
                file.write(",".join(map(str, sorted(cluster))) + "\n")

