import json
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load network
with open("network_output/music_cover_network.pickle", "rb") as f:
    G = pickle.load(f)

# Load PageRank scores
with open("topology_analysis.json", "r") as f:
    topology_data = json.load(f)
pagerank_raw = topology_data['topology_statistics']['centralities']['pagerank']
pagerank = {int(k): v for k, v in pagerank_raw.items()}

# Load artist names and degree hubs
with open("basic_network_analysis.json", "r") as f:
    basic_data = json.load(f)
node_info = {entry['node']: entry['name'] for entry in basic_data['network_hubs']['total_degree_hubs']}

# Build DataFrame for top 10 PageRank artists
top_10 = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
df_top10 = pd.DataFrame({
    "Node": [node for node, _ in top_10],
    "Name": [node_info.get(node, str(node)) for node, _ in top_10],
    "PageRank": [score for _, score in top_10],
    "InDegree": [G.in_degree(node) for node, _ in top_10],
    "OutDegree": [G.out_degree(node) for node, _ in top_10]
})
 
# --- 1. Bar Chart of Top PageRank Scores ---
plt.figure(figsize=(10, 6))
sns.barplot(data=df_top10, x="PageRank", y="Name", palette="viridis")
plt.title("Top 10 Artists by PageRank")
plt.xlabel("PageRank Score")
plt.ylabel("Artist")
plt.tight_layout()
plt.savefig("top_10_pagerank_bar_chart.png", dpi=300)
plt.close()

# --- 2. Heatmap of Centrality Metrics ---
heatmap_data = df_top10.set_index("Name")[["PageRank", "InDegree", "OutDegree"]]
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap="magma", fmt=".2f")
plt.title("Centrality Metrics for Top 10 Artists")
plt.tight_layout()
plt.savefig("top_10_metrics_heatmap.png", dpi=300)
plt.close()

# --- 3. Scatter Plot: PageRank vs. In-Degree ---
all_nodes = list(G.nodes())
df_all = pd.DataFrame({
    "Node": all_nodes,
    "Name": [node_info.get(node, str(node)) for node in all_nodes],
    "PageRank": [pagerank.get(node, 0) for node in all_nodes],
    "InDegree": [G.in_degree(node) for node in all_nodes],
    "OutDegree": [G.out_degree(node) for node in all_nodes]
})

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_all, x="InDegree", y="PageRank", hue="OutDegree", palette="coolwarm", legend=False)
plt.title("PageRank vs. In-Degree for All Artists")
plt.xlabel("In-Degree (Times Covered)")
plt.ylabel("PageRank (Influence Score)")
plt.tight_layout()
plt.savefig("pagerank_vs_indegree_scatter.png", dpi=300)
plt.close()