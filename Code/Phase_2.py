import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from Phase_1 import get_normalizer, show_ranked_ranges, show_correlation_heatmap
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the data
data = pd.read_csv("Most Streamed Spotify Songs 2024.csv", encoding='latin1')

# Select the relevant columns
columns_of_interest = [
    "Spotify Streams", "YouTube Views", "TikTok Views",
    "YouTube Playlist Reach", "Pandora Streams",
    "Shazam Counts", "Soundcloud Streams", "Deezer Playlist Reach"
]

# Ensure the columns are numeric by removing commas and converting to float
data_clean = data[columns_of_interest].copy()
for col in data_clean.columns:
    data_clean[col] = data_clean[col].astype(str).str.replace(',', '').astype(float)

# Handle missing values
data_clean = data_clean.dropna()

# Normalize the data using your get_normalizer function
df_normalized = get_normalizer(data_clean)

show_correlation_heatmap(df_normalized)

# Elbow Method for KMeans
inertia = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_normalized)
    inertia.append(kmeans.inertia_)

# Silhouette Score for KMeans
silhouette_scores = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_normalized)
    score = silhouette_score(df_normalized, kmeans.labels_)
    silhouette_scores.append(score)

# Plot the Elbow Method and Silhouette Score
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, 10), inertia, marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for KMeans")

plt.subplot(1, 2, 2)
plt.plot(range(2, 10), silhouette_scores, marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for KMeans")
plt.tight_layout()
plt.show()

# Apply KMeans with the optimal number of clusters (e.g., 3)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_clusters = kmeans.fit_predict(df_normalized)
kmeans_centers = kmeans.cluster_centers_

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_clusters = dbscan.fit_predict(df_normalized)

# Agglomerative Clustering with Dendrogram
# Perform hierarchical clustering
Z = linkage(df_normalized, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode='lastp', p=20, show_leaf_counts=True, leaf_rotation=90., leaf_font_size=12., show_contracted=True)
plt.title("Dendrogram (Agglomerative Clustering)")
plt.xlabel("Cluster Size")
plt.ylabel("Distance")
plt.show()

# Apply Agglomerative Clustering with 3 clusters
agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_clusters = agg_clustering.fit_predict(df_normalized)

# Reduce to 2 dimensions for visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_normalized)

# Visualize the clusters for each model
models = {
    "KMeans": kmeans_clusters,
    "DBSCAN": dbscan_clusters,
    "Agglomerative Clustering": agg_clusters
}

for name, clusters in models.items():
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.6)
    if name == "KMeans":
        centers_pca = pca.transform(kmeans_centers)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"Cluster Visualization ({name})")
    plt.colorbar()
    plt.legend()
    plt.show()

    # Print the Silhouette Score (for DBSCAN, handle noise points)
    if name == "DBSCAN":
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        print(f"\n{name}:")
        print(f"Estimated number of clusters: {n_clusters}")
        print(f"Estimated number of noise points: {n_noise}")
    else:
        print(f"\n{name}:")
        print(f"Silhouette Score: {silhouette_score(df_normalized, clusters):.4f}")

    # Add clusters to the DataFrame for analysis
    data_clean[f"{name}_Cluster"] = clusters

    # Calculate the mean values for each cluster (using original data for interpretability)
    if name == "DBSCAN":
        cluster_means = data_clean[data_clean[f"{name}_Cluster"] != -1].groupby(f"{name}_Cluster").mean()
    else:
        cluster_means = data_clean.groupby(f"{name}_Cluster").mean()
    print(cluster_means)
