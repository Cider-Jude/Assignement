import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
import seaborn as sns

df = pd.read_csv("Most Streamed Spotify Songs 2024.csv", encoding="latin-1")

# Convert the numbers higher than 1000 from strings 
for col in df.columns:
    if df[col].dtype == "object" and col not in ["Track", "Album Name", "Artist"]:
        df[col] = df[col].astype(str).str.replace(",", "")
        numeric_series = pd.to_numeric(df[col], errors="coerce")

        # If at least one value converted successfully, replace column
        if numeric_series.notna().any():
            df[col] = numeric_series

print(df.head())

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
    plt.title("Range of Values for Each Numeric Column")

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
        annot=False,        # Set to True if you want numbers inside boxes
        cmap="coolwarm",    # Color palette
        linewidths=0.5,     # Line width between cells
    )
    plt.title("Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    plt.show()

df_normalized = get_normalizer(df)
show_ranked_ranges(df_normalized)
show_correlation_heatmap(df)
show_correlation_heatmap(df_normalized)
