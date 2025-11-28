import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from Phase_1 import get_normalizer, show_ranked_ranges, show_correlation_heatmap

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
clusters = kmeans.fit_predict(df_normalized)

# Add clusters to the DataFrame for analysis
data_clean["KMeans_Cluster"] = clusters

# Calculate the mean values for each cluster (using original data for interpretability)
cluster_means = data_clean.groupby("KMeans_Cluster").mean()

# Reduce to 2 dimensions for visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_normalized)

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Cluster Visualization (KMeans)")
plt.colorbar()
plt.show()

# Print the mean values for each cluster
print(cluster_means)
