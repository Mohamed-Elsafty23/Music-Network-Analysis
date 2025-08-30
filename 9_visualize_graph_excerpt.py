import networkx as nx
import pickle
import matplotlib.pyplot as plt
import json
import pandas as pd 
import os  

try:
    from community import community_louvain
    _COMMUNITY_AVAILABLE = True
except ImportError:
    _COMMUNITY_AVAILABLE = False

class GraphVisualizationGenerator:
    
    def __init__(self, network_file='network_output/music_cover_network.pickle',
                 topology_file='topology_analysis.json',
                 basic_file='basic_network_analysis.json'):
        self.network_file = network_file
        self.topology_file = topology_file
        self.basic_file = basic_file
        self.G = None
        self.topology_data = None
        self.node_info = {}
    
    def load_data(self):
        """Load network and analysis data"""
        # Check if files exist
        if not os.path.exists(self.network_file):
            raise FileNotFoundError(f"Error: {self.network_file} not found. Please ensure the pickle file is in 'network_output/'.")
        if not os.path.exists(self.topology_file):
            raise FileNotFoundError(f"Error: {self.topology_file} not found. Please ensure the JSON file is in the project directory.")
        if not os.path.exists(self.basic_file):
            raise FileNotFoundError(f"Error: {self.basic_file} not found. Please ensure the JSON file is in the project directory.")

        # Load network
        try:
            with open(self.network_file, "rb") as f:
                self.G = pickle.load(f)
            print("Network loaded successfully.")
        except Exception as e:
            raise Exception(f"Error loading {self.network_file}: {e}")

        # Load topology analysis
        try:
            with open(self.topology_file, 'r') as f:
                self.topology_data = json.load(f)
        except Exception as e:
            raise Exception(f"Error loading {self.topology_file}: {e}")

        # Load basic analysis for artist names
        try:
            with open(self.basic_file) as f:
                data = json.load(f)
            self.node_info = {entry['node']: entry['name'] for entry in data['network_hubs']['total_degree_hubs']}
        except Exception as e:
            raise Exception(f"Error loading {self.basic_file}: {e}")
    
    def create_pagerank_subgraph(self):
        """Create subgraph visualization of top PageRank artists"""
        # Get top 10 PageRank artists
        pagerank = self.topology_data['topology_statistics']['centralities']['pagerank']
        top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
        top_nodes = [int(node) for node, _ in top_pagerank]

        # Create subgraph of top 10 artists and their immediate neighbors
        subgraph_nodes = set(top_nodes)
        for node in top_nodes:
            subgraph_nodes.update(self.G.successors(node))  # Artists they cover
            subgraph_nodes.update(self.G.predecessors(node))  # Artists covering them
        subgraph_nodes = list(subgraph_nodes)[:50]  # Limit to 50 nodes for clarity
        subG = self.G.subgraph(subgraph_nodes)

        # Map node IDs to names (use ID if name missing)
        labels = {node: self.node_info.get(node, str(node)) for node in subG.nodes()}
        # Get PageRank for node sizes (scale for visualization)
        node_sizes = [pagerank.get(str(node), 0.001) * 5000 for node in subG.nodes()]

        # Community detection on subgraph (undirected for Louvain)
        if _COMMUNITY_AVAILABLE:
            undirected_subG = subG.to_undirected()
            partition = community_louvain.best_partition(undirected_subG)
            colors = [partition[node] for node in subG.nodes()]
        else:
            colors = ['lightblue'] * len(subG.nodes())

        # Visualize
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(subG, k=0.5, iterations=50)
        nx.draw(subG, pos, with_labels=True, labels=labels, node_size=node_sizes,
                node_color=colors, cmap=plt.cm.Set3, font_size=8, font_weight='bold',
                arrows=True, edge_color='gray', alpha=0.7)
        plt.title('Subgraph of Top 10 PageRank Artists and Neighbors')
        plt.savefig('top_pagerank_subgraph.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return top_pagerank
    
    def create_community_subgraph(self):
        """Create community subgraph visualization"""
        pagerank = self.topology_data['topology_statistics']['centralities']['pagerank']
        community_dict = self.topology_data['topology_statistics']['communities']
        
        # Pick a medium-sized community (10-50 nodes)
        comm_id = next((cid for cid, nodes in community_dict.items() if 10 <= len(nodes) <= 50), None)
        if comm_id is None:
            print("No community found with 10-50 nodes. Using largest community.")
            comm_id = max(community_dict.items(), key=lambda x: len(x[1]))[0]
        
        comm_nodes = [int(n) for n in community_dict[comm_id]]
        comm_subG = self.G.subgraph(comm_nodes)
        comm_labels = {node: self.node_info.get(node, str(node)) for node in comm_subG.nodes()}
        comm_sizes = [pagerank.get(str(node), 0.001) * 5000 for node in comm_subG.nodes()]

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(comm_subG, k=0.5, iterations=50)
        nx.draw(comm_subG, pos, with_labels=True, labels=comm_labels, node_size=comm_sizes,
                node_color='lightblue', font_size=8, font_weight='bold',
                arrows=True, edge_color='gray', alpha=0.7)
        plt.title(f'Community {comm_id} Subgraph')
        plt.savefig('community_subgraph.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return comm_id, len(comm_nodes)
    
    def print_summary(self):
        """Print summary results"""
        pagerank = self.topology_data['topology_statistics']['centralities']['pagerank']
        top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
        community_dict = self.topology_data['topology_statistics']['communities']
        
        print("Top 10 Influential Artists (PageRank):")
        for node, score in top_pagerank:
            print(f"{self.node_info.get(int(node), node)}: {score:.6f}")

        print("\nCommunity Sizes (Top 5):")
        community_sizes = {cid: len(nodes) for cid, nodes in community_dict.items()}
        for cid, size in sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"Community {cid}: {size} artists")
            sample_artists = [self.node_info.get(int(n), n) for n in community_dict[cid][:5]]
            print(f"Sample artists: {sample_artists}")
    
    def generate_graph_visualizations(self):
        """Main method to generate all graph visualizations"""
        print("Generating graph excerpt visualizations...")
        
        # Load all required data
        self.load_data()
        
        # Create visualizations
        top_pagerank = self.create_pagerank_subgraph()
        comm_id, comm_size = self.create_community_subgraph()
        
        # Print summary
        self.print_summary()
        
        results = {
            "status": "completed",
            "files_generated": [
                "top_pagerank_subgraph.png",
                "community_subgraph.png"
            ],
            "top_pagerank_count": len(top_pagerank),
            "community_analyzed": {
                "id": comm_id,
                "size": comm_size
            }
        }
        
        print("Graph visualizations completed successfully.")
        return results

if __name__ == "__main__":
    generator = GraphVisualizationGenerator()
    generator.generate_graph_visualizations()