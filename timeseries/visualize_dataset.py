#
## visualize_dataset.py
#
#  This script loads the processed WVC time-series dataset and generates visualizations to explore spatial and temporal patterns in the data.
#
#


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("processed_wvc_timeseries.csv", parse_dates=['Date'], index_col='Date')

# Calculate total crashes for each spatial segment and sort them
spatial_totals = df.sum().sort_values(ascending=False)

# Calculate the overall sparsity percentage
total_cells = df.shape[0] * df.shape[1]

zero_cells = (df == 0).sum().sum()

sparsity_pct = (zero_cells / total_cells) * 100

print(f"[*] Matrix Sparsity: {sparsity_pct:.2f}% of the dataset consists of 0s.")
print(f"[*] Total Wildlife-Vehicle Collisions recorded in 2021: {df.sum().sum()}")


# Spatial Hotspots: top 20 most dangerous highway segments based on total annual crash counts

top_segments = spatial_totals.head(20)

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=top_segments.index, y=top_segments.values, hue=top_segments.index, palette='viridis', legend=False, ax=ax)

ax.set_title('Top 20 Wildlife-Vehicle Collision Hotspots (5-Mile Segments) in 2021', fontsize=14, fontweight='bold')
ax.set_xlabel('Highway Segment (Starting Milepost Number)', fontsize=12)
ax.set_ylabel('Total Annual Crash Count', fontsize=12)
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('wvc_spatial_hotspots.png', dpi=300)
plt.close()

# Temporal Trends: total daily crashes across the entire corridor, with a 7-day rolling average to show week-by-week trends
daily_corridor_totals = df.sum(axis=1)

fig, ax = plt.subplots(figsize=(14, 5))
daily_corridor_totals.plot(ax=ax, color='crimson', alpha=0.4, label='Daily Crash Count', linewidth=1)

# Overlay a 7-day rolling average to smooth out the daily noise and show week-by-week trends
daily_corridor_totals.rolling(window=7, center=True).mean().plot(
    ax=ax, color='black', linestyle='-', linewidth=2, label='7-Day Moving Average'
)

ax.set_title('Daily Wildlife-Vehicle Collisions Across the Entire Corridor (2021)', fontsize=14, fontweight='bold')
ax.set_xlabel('Timeline', fontsize=12)
ax.set_ylabel('Total Collisions Per Day', fontsize=12)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('wvc_temporal_trends.png', dpi=300)
plt.close()

print("[+] Done! Saved 'wvc_spatial_hotspots.png' and 'wvc_temporal_trends.png' to your directory.")