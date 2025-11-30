import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("Most Streamed Spotify Songs 2024.csv", encoding="latin-1")

# Convert the numbers higher than 1000 from strings 
for col in df.columns:
    if df[col].dtype == "object" and col not in ["Track", "Album Name", "Artist"]:
        df[col] = df[col].astype(str).str.replace(",", "")
        numeric_series = pd.to_numeric(df[col], errors="coerce")

        # If at least one value converted successfully, replace column
        if numeric_series.notna().any():
            df[col] = numeric_series

#print(df.head())

#fill in the Nan values with 0
df = df.fillna(0)

def show_summary_statistic(df):
    for col in df.columns:
        if df[col].dtype == "float64":
            range = df[col].max() - df[col].min()
            print(f"\n{col} \n- Mean: {df[col].mean()} \n- Median: {df[col].median()} \n- Standard deviation: {df[col].std()} \n- Variance: {df[col].var()} \n- Maximum: {df[col].max()} \n- Minimum: {df[col].min()} \n- Range: {range}")

def show_ranked_ranges(df):
    col_ranges = {}

    for col in df.columns:
        if df[col].dtype == "float64":
            col_range = df[col].max() - df[col].min()
            col_ranges[col] = col_range

    # Sort columns by range (largest first)
    ranked_by_range = sorted(col_ranges.items(), key=lambda x: x[1], reverse=True)

    print("Column ranking by range:\n")
    for rank, (col, r) in enumerate(ranked_by_range, start=1):
        print(f"{rank}. {col} — Range: {r}")

    # Separate names and values for plotting
    columns = [item[0] for item in ranked_by_range]
    ranges = [item[1] for item in ranked_by_range]

    plt.figure(figsize=(10, 6))

    plt.bar(columns, ranges)

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Range (max - min)")
    plt.title("Range of Values for Each Normalized Numeric Column")

    plt.tight_layout()
    plt.show()

def show_ranked_standard_deviation(df):
    col_std = {}

    for col in df.columns:
        if df[col].dtype == "float64":
            col_std[col] = df[col].std()

    # Sort columns by range (largest first)
    ranked_by_range = sorted(col_std.items(), key=lambda x: x[1], reverse=True)

    print("Column ranking by range:\n")
    for rank, (col, r) in enumerate(ranked_by_range, start=1):
        print(f"{rank}. {col} — Range: {r}")

    # Separate names and values for plotting
    columns = [item[0] for item in ranked_by_range]
    STD = [item[1] for item in ranked_by_range]

    plt.figure(figsize=(10, 6))

    plt.bar(columns, STD)

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Standard Deviation")
    plt.title("The standard deviation for Each Normalized Numeric Column")

    plt.tight_layout()
    plt.show()


def get_normalizer(df):
    scaler = Normalizer()
    df_normalized = df.copy()
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df_normalized

def show_correlation_heatmap(df):
    # Compute the correlation matrix
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(14, 10))

    # Draw the heatmap
    sns.heatmap(
        corr,
        annot=False,        
        cmap="coolwarm",    
        linewidths=0.5,     
    )
    plt.title("Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    plt.show()

def show_PCA(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(
        data=pca_components,
        columns=['PC1', 'PC2']
    )
    plt.figure(figsize=(10, 7))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.6)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)")
    plt.title("PCA Visualization (2 Components)")
    plt.grid(True)
    plt.show()

def show_PCA_3D(df):
    # Keep only numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Scale the data
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)
    
    # PCA with 3 components
    pca = PCA(n_components=3)
    components = pca.fit_transform(scaled)

    # Create a DataFrame for plotting
    pca_df = pd.DataFrame(
        components,
        columns=['PC1', 'PC2', 'PC3']
    )
    
    # 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        pca_df['PC1'], pca_df['PC2'], pca_df['PC3'],
        alpha=0.6
    )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.2f}% variance)")
    ax.set_title("3D PCA Visualization (3 Components)")
    plt.show()

show_ranked_ranges(df)
show_ranked_standard_deviation(df)
df_normalized = get_normalizer(df)
show_ranked_ranges(df_normalized)
show_ranked_standard_deviation(df_normalized)