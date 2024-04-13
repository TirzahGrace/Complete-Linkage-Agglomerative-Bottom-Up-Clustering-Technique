# Group Number: 51
# Roll Number: 21CS10071
# Project Number : 3
# Project Code: PWHC-AC
# Project Title: Portugal Weather using Complete Linkage Agglomerative (Bottom-Up) Clustering Technique

import numpy as np
import pandas as pd
from pre_process import PreProcess
from clustering import KMeansClustering , CompleteLinkageClustering
from evaluation import SilhouetteCoefficient , JaccardSimilarity

def main():
    try:
        file_path = '../data/weather.csv'  # path of the data source weather.csv
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File not found. Please make sure the file path is correct.")
        return

    data = PreProcess(df=df)

    # Perform K-means clustering for k=3 .
    k = 3  # Number of clusters
    k_means_clustering = KMeansClustering()
    labels, centroids = k_means_clustering.fit(data, k)

    # Save clustering information
    clustering_info = pd.DataFrame({'Cluster': labels})
    clustering_info.to_csv(f"../output/csv/k_{k}_cluster.csv", index=False)

    # Print the centroids of each cluster
    # print(f"Centroids of each cluster for k: {k}:")
    # print(centroids)

    #Plot the cluster, and save it.
    k_means_clustering.plot(data,k,labels,centroids,f"k_{k}_cluster.png")

    # Compute Silhouette coefficient for the clustering result
    silhouette_coefficient = SilhouetteCoefficient()
    silhouette_score = silhouette_coefficient.fit(data.values, labels)
    print(f"k: {k} -> Silhouette Coefficient: {silhouette_score:.6f}")

    # List to store Silhouette coefficients for different values of k
    silhouette_scores = []

    # Iterate over different values of k
    for k in range(3, 7):
        # Perform K-means clustering
        k_means_clustering = KMeansClustering()
        labels, centroids = k_means_clustering.fit(data, k)

        # Save clustering information
        clustering_info = pd.DataFrame({'Cluster': labels})
        clustering_info.to_csv(f"../output/csv/k_{k}_cluster.csv", index=False)

        #Plot the cluster, and save it.
        k_means_clustering.plot(data,k,labels,centroids,f"k_{k}_cluster.png")

        # Compute Silhouette coefficient for the clustering result
        silhouette_coefficient = SilhouetteCoefficient()
        silhouette_score = silhouette_coefficient.fit(data.values, labels)
        silhouette_scores.append(silhouette_score)
        print(f"Silhouette Coefficient for k={k}: {silhouette_score:.6f}")

        # Find the value of k with the highest Silhouette coefficient
        optimal_k = np.argmax(silhouette_scores) + 3  # Add 3 because we started from k=3
        optimal_silhouette_score = silhouette_scores[optimal_k - 3]

    print("\nOptimal value of k:", optimal_k)
    print(f"Highest Silhouette Coefficient: {optimal_silhouette_score:.6f}")

    # Perform K-means clustering for optimal_k
    k_means_clustering = KMeansClustering()
    labels, _ = k_means_clustering.fit(data, optimal_k)
    k_means_clustering.save_clusters(labels, "kmeans.txt")

    # Assign data points to clusters based on K-means labels
    k_means_clusters = [[] for _ in range(optimal_k)]
    for i, label in enumerate(labels):
        k_means_clusters[label].append(i)

    # Perform Complete Linkage Agglomerative Clustering
    complete_linkage_clustering = CompleteLinkageClustering()
    hierarchical_clusters = complete_linkage_clustering.fit(data.values, optimal_k)
    complete_linkage_clustering.save_clusters(hierarchical_clusters, "agglomerative.txt")

    # Initialize variables to store the best mapping and its Jaccard Similarity score
    best_mapping = None
    best_similarity = 0

    # Generate all possible permutations of hierarchical clusters
    jaccard_similarity = JaccardSimilarity()
    hierarchical_permutations = jaccard_similarity.permutations(list(range(optimal_k)))

    # Iterate through each permutation of hierarchical clusters
    for hierarchical_permutation in hierarchical_permutations:
        similarity_sum = 0
        
        # Compute the total similarity for the current permutation
        for i, j in enumerate(hierarchical_permutation):
            similarity_sum += jaccard_similarity.fit(set(k_means_clusters[i]), set(hierarchical_clusters[j]))
        
        # Update best mapping if the current permutation has a higher overall average similarity
        if similarity_sum / optimal_k > best_similarity:
            best_mapping = hierarchical_permutation
            best_similarity = similarity_sum / optimal_k

    # Print the best mapping and its overall average similarity
    for i, j in enumerate(best_mapping):
        print(f"Jaccard Similarity score for K-means cluster {i+1} and best matching Hierarchical cluster {j+1}: {jaccard_similarity.fit(set(k_means_clusters[i]), set(hierarchical_clusters[j])):.6f}")

    print(f"Overall average similarity among K-means cluster and Hierarchical cluster: {best_similarity:.6f}")

if __name__ == '__main__':
    main()