# Clustering Analysis

This repository contains code for performing clustering analysis on weather data using K-means clustering and Complete Linkage Agglomerative Clustering.

## Instructions to Run
  - Clone this repository to your local machine.
  - Ensure you have Python installed on your system.
  - Install the required dependencies :
    ```bash
    cd Complete-Linkage-Agglomerative-Bottom-Up-Clustering-Technique/
    pip install -r requirements.txt
    ```
  - Run the `main.py` script:
    ```bash
    cd src/
    python main.py
    ```
  - The script will generate output files and plots in the `output` directory.

## Classes and Functions

### `pre_process.py`

This module contains the `PreProcess` function, which preprocesses the weather data. It performs data cleaning, feature selection, and standardization.

### `clustering.py`

#### `KMeansClustering`

This class implements the K-means clustering algorithm. It contains methods for initializing clusters, assigning data points to clusters, updating centroids, and plotting the clusters.

#### `CompleteLinkageClustering`

This class implements the Complete Linkage Agglomerative Clustering algorithm. It contains a method for fitting the data and merging clusters until the desired number of clusters is reached.

### `evaluation.py`

#### `JaccardSimilarity`

This class computes the Jaccard similarity between two sets. It contains a method for calculating the Jaccard similarity coefficient.

#### `SilhouetteCoefficient`

This class computes the Silhouette coefficient for a clustering result. It contains a method for calculating the Silhouette coefficient.

### `main.py`

- It collects and preprocesses the data
- Performs K-means clustering for different values of k, selects the optimal value of k based on the Silhouette coefficient
- Performs K-means and hierarchical clustering with the optimal k
- Computes the Jaccard similarity scores between K-means clusters and hierarchical clusters.

---
