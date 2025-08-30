import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up output directory
output_dir = 'network_output'
os.makedirs(output_dir, exist_ok=True)

# Load JSON file
with open('centrality_top10_with_names.json') as f:
    data = json.load(f)

# Extract DegreeCentrality data (adjust if there are other keys)
degree_data = data.get('DegreeCentrality', [])
df = pd.DataFrame(degree_data)

# Debug: Check df
print("DataFrame preview:")
print(df.head())

# Print summary
print("\nTop 10 Artists by PageRank:")
for _, row in df.iterrows():
    print(f"{row['ArtistName']}: {row['PageRank']:.6f}")

# 1. Bar Chart of Top PageRank Scores
plt.figure(figsize=(10, 6))
sns.barplot(x='ArtistName', y='PageRank', data=df, palette='viridis')
plt.title('Top 10 Artists by PageRank Score')
plt.xlabel('Artist')
plt.ylabel('PageRank Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top_pagerank_bar.png'))
plt.show()

# 2. Heatmap of Centrality Metrics
centrality_df = df[['ArtistName', 'PageRank', 'InDegreeCentrality', 'OutDegreeCentrality', 'EigenvectorCentrality']].set_index('ArtistName')
plt.figure(figsize=(8, 6))
sns.heatmap(centrality_df, annot=True, cmap='YlOrRd', fmt='.2f')
plt.title('Centrality Metrics for Top 10 Artists')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'centrality_heatmap.png'))
plt.show()

# 3. Scatter Plot: PageRank vs. In-Degree
plt.figure(figsize=(8, 6))
sns.scatterplot(x='InDegreeCentrality', y='PageRank', data=df, hue='ArtistName', size='OutDegreeCentrality', sizes=(50, 500))
for i, row in df.iterrows():
    plt.text(row['InDegreeCentrality'] + 0.005, row['PageRank'], row['ArtistName'], fontsize=9)
plt.title('PageRank vs. In-Degree for Top 10 Artists')
plt.xlabel('In-Degree (Number of Covers Received)')
plt.ylabel('PageRank Score')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pagerank_vs_indegree_scatter_02.png'))
plt.show()

print("\nVisualizations saved in 'network_output':")
print("- top_pagerank_bar.png")
print("- centrality_heatmap.png")
print("- pagerank_vs_indegree_scatter_02.png")