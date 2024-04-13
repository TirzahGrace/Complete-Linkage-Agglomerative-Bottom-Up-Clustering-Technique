# evaluation.py

import numpy as np

class JaccardSimilarity:
    def __init__(self):
        pass

    def fit(self, set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    def permutations(self, arr):
        if len(arr) == 1:
            return [arr]
        else:
            result = []
            for i in range(len(arr)):
                current = arr[i]
                rest = arr[:i] + arr[i+1:]
                rest_permutations = self.permutations(rest)
                for perm in rest_permutations:
                    result.append([current] + perm)
            return result


class SilhouetteCoefficient:
    def __init__(self):
        pass

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

    def fit(self, data, labels):
        # Compute pairwise distances between all data points
        distances = self.pairwise_distances(data)

        silhouette_scores = []
        for i in range(len(data)):
            # Get the cluster label of the current data point
            label = labels[i]
            
            # Calculate mean distance to all other points in the same cluster (a)
            cluster_distances = distances[i][labels == label]
            a = np.mean(cluster_distances)
            
            # Calculate mean distance to all points in the nearest neighboring cluster (b)
            other_cluster_labels = set(labels) - {label}
            nearest_cluster_distances = [np.mean(distances[i][labels == other_label]) for other_label in other_cluster_labels]
            if nearest_cluster_distances:
                b = np.min(nearest_cluster_distances)
            else:
                b = 0  # If there's only one cluster, set b to 0
            
            # Compute Silhouette coefficient for the current data point
            silhouette = (b - a) / max(a, b)
            silhouette_scores.append(silhouette)
        
        # Compute the mean Silhouette coefficient across all data points
        return np.mean(silhouette_scores)
