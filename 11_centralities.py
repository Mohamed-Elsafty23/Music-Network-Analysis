import networkx as nx
import pickle
import json
import pandas as pd

# Load network
with open("network_output/music_cover_network.pickle", "rb") as f:
    G = pickle.load(f)

# Load artist data
artist_df = pd.read_csv("cleaned_data/artist_lookup_cleaned.csv")
artist_map = pd.Series(artist_df.common_name.values, index=artist_df.artist_id).to_dict()

# Compute centralities
degree_centrality = nx.degree_centrality(G)
in_degree_centrality = nx.in_degree_centrality(G)
out_degree_centrality = nx.out_degree_centrality(G)
try:
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
except nx.PowerIterationFailedConvergence:
    eigenvector_centrality = {node: 0 for node in G.nodes()}
pagerank_centrality = nx.pagerank(G, alpha=0.85)

# Prepare data with names and round to 5 decimals
data = []
for node in G.nodes():
    data.append({
        "ArtistID": node,
        "ArtistName": artist_map.get(node, str(node)),
        "DegreeCentrality": round(degree_centrality.get(node, 0), 5),
        "InDegreeCentrality": round(in_degree_centrality.get(node, 0), 5),
        "OutDegreeCentrality": round(out_degree_centrality.get(node, 0), 5),
        "EigenvectorCentrality": round(eigenvector_centrality.get(node, 0), 5),
        "PageRank": round(pagerank_centrality.get(node, 0), 5)
    })

# Print all results
print(json.dumps(data, indent=2))

# Save top 10 per centrality measure in JSON
top10 = {
    "DegreeCentrality": sorted(data, key=lambda x: x["DegreeCentrality"], reverse=True)[:10],
    "InDegreeCentrality": sorted(data, key=lambda x: x["InDegreeCentrality"], reverse=True)[:10],
    "OutDegreeCentrality": sorted(data, key=lambda x: x["OutDegreeCentrality"], reverse=True)[:10],
    "EigenvectorCentrality": sorted(data, key=lambda x: x["EigenvectorCentrality"], reverse=True)[:10],
    "PageRank": sorted(data, key=lambda x: x["PageRank"], reverse=True)[:10]
}

with open("centrality_top10_with_names.json", "w") as f:
    json.dump(top10, f, indent=2)

# Save all results in JSON
with open("centrality_all_with_names.json", "w") as f:
    json.dump(data, f, indent=2)
