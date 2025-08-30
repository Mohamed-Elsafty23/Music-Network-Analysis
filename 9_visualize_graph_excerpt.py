import networkx as nx
import pickle
import matplotlib.pyplot as plt
import json
import pandas as pd 
from community import community_louvain
import os  

# Check if files exist
network_file = 'network_output/music_cover_network.pickle'
topology_file = 'topology_analysis.json'
basic_file = 'basic_network_analysis.json'

if not os.path.exists(network_file):
    print(f"Error: {network_file} not found. Please ensure the pickle file is in 'network_output/'.")
    exit(1)
if not os.path.exists(topology_file):
    print(f"Error: {topology_file} not found. Please ensure the JSON file is in the project directory.")
    exit(1)
if not os.path.exists(basic_file):
    print(f"Error: {basic_file} not found. Please ensure the JSON file is in the project directory.")
    exit(1)

# Load network
try:
    with open("network_output/music_cover_network.pickle", "rb") as f:
        G = pickle.load(f)

    # G = nx.read_gpickle(network_file)
    # G = nx.read_gpickle("network_output/music_cover_network.pickle")
    print("Network loaded successfully.")
except Exception as e:
    print(f"Error loading {network_file}: {e}")
    exit(1)

# Load topology analysis
try:
    with open(topology_file, 'r') as f:
        topology_data = json.load(f)
except Exception as e:
    print(f"Error loading {topology_file}: {e}")
    exit(1)

# Load basic analysis for artist names
try:
    # with open(basic_file, 'r') as f:
    #     basic_data = json.load(f)
    # node_info = {item['node']: item['name'] for item in basic_data['top_nodes']}
    with open('basic_network_analysis.json') as f:
        data = json.load(f)
    top_nodes = [entry['node'] for entry in data['network_hubs']['total_degree_hubs']]
except Exception as e:
    print(f"Error loading {basic_file}: {e}")
    exit(1)

# --- Visualization 1: Subgraph of Top 10 PageRank Artists ---
# Get top 10 PageRank artists
pagerank = topology_data['topology_statistics']['centralities']['pagerank']
top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
top_nodes = [int(node) for node, _ in top_pagerank]

# Create subgraph of top 10 artists and their immediate neighbors
subgraph_nodes = set(top_nodes)
for node in top_nodes:
    subgraph_nodes.update(G.successors(node))  # Artists they cover
    subgraph_nodes.update(G.predecessors(node))  # Artists covering them
subgraph_nodes = list(subgraph_nodes)[:50]  # Limit to 50 nodes for clarity
subG = G.subgraph(subgraph_nodes)

with open('basic_network_analysis.json') as f:
    data = json.load(f)
top_nodes = [entry['node'] for entry in data['network_hubs']['total_degree_hubs']]
node_info = {entry['node']: entry['name'] for entry in data['network_hubs']['total_degree_hubs']}

# Map node IDs to names (use ID if name missing)
labels = {node: node_info.get(node, str(node)) for node in subG.nodes()}
# Get PageRank for node sizes (scale for visualization)
node_sizes = [pagerank.get(str(node), 0.001) * 5000 for node in subG.nodes()]

# Community detection on subgraph (undirected for Louvain)
undirected_subG = subG.to_undirected()
partition = community_louvain.best_partition(undirected_subG)
colors = [partition[node] for node in subG.nodes()]

# Visualize
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(subG, k=0.5, iterations=50)
nx.draw(subG, pos, with_labels=True, labels=labels, node_size=node_sizes,
        node_color=colors, cmap=plt.cm.Set3, font_size=8, font_weight='bold',
        arrows=True, edge_color='gray', alpha=0.7)
plt.title('Subgraph of Top 10 PageRank Artists and Neighbors')
plt.savefig('top_pagerank_subgraph.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Visualization 2: Subgraph of a Single Community ---
community_dict = topology_data['topology_statistics']['communities']
# Pick a medium-sized community (10-50 nodes)
comm_id = next((cid for cid, nodes in community_dict.items() if 10 <= len(nodes) <= 50), None)
if comm_id is None:
    print("No community found with 10-50 nodes. Using largest community.")
    comm_id = max(community_dict.items(), key=lambda x: len(x[1]))[0]
comm_nodes = [int(n) for n in community_dict[comm_id]]
comm_subG = G.subgraph(comm_nodes)
comm_labels = {node: node_info.get(node, str(node)) for node in comm_subG.nodes()}
comm_sizes = [pagerank.get(str(node), 0.001) * 5000 for node in comm_subG.nodes()]

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(comm_subG, k=0.5, iterations=50)
nx.draw(comm_subG, pos, with_labels=True, labels=comm_labels, node_size=comm_sizes,
        node_color='lightblue', font_size=8, font_weight='bold',
        arrows=True, edge_color='gray', alpha=0.7)
plt.title(f'Community {comm_id} Subgraph')
plt.savefig('community_subgraph.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Summarize Results for Report ---
print("Top 10 Influential Artists (PageRank):")
for node, score in top_pagerank:
    print(f"{node_info.get(int(node), node)}: {score:.6f}")

print("\nCommunity Sizes (Top 5):")
community_sizes = {cid: len(nodes) for cid, nodes in community_dict.items()}
for cid, size in sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"Community {cid}: {size} artists")
    print(f"Sample artists: {[node_info.get(int(n), n) for n in community_dict[cid][:5]]}")