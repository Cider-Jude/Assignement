import pandas as pd

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


for col in df.columns:
    if df[col].dtype == "float64":
        range = df[col].max() - df[col].min()
        print(f"\n{col} \n- Mean: {df[col].mean()} \n- Median: {df[col].median()} \n- Standard deviation: {df[col].std()} \n- Variance: {df[col].var()} \n- Maximum: {df[col].max()} \n- Minimum: {df[col].min()} \n- Range: {range}")

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
