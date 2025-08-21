import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
from collections import defaultdict, Counter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

try:
    from networkx.algorithms import community as nx_community
    HAS_NX_COMMUNITY = True
except ImportError:
    HAS_NX_COMMUNITY = False

COMMUNITY_AVAILABLE = HAS_LOUVAIN or HAS_NX_COMMUNITY

class NetworkTopologyAnalyzer:
    
    def __init__(self, network_path="network_output"):
        self.network_path = Path(network_path)
        self.network = None
        self.topology_stats = {}
        
    def load_network(self):
        print("Loading Network for Topology Analysis...")
        
        pickle_file = self.network_path / "music_cover_network.pickle"
        if pickle_file.exists():
            try:
                with open(pickle_file, 'rb') as f:
                    self.network = pickle.load(f)
                print(f"Network loaded: {self.network.number_of_nodes()} nodes, {self.network.number_of_edges()} edges")
                return True
            except Exception as e:
                print(f"Warning: Error loading pickle: {e}")
        
        graphml_file = self.network_path / "music_cover_network.graphml"
        if graphml_file.exists():
            try:
                self.network = nx.read_graphml(graphml_file)
                mapping = {node: int(node) for node in self.network.nodes()}
                self.network = nx.relabel_nodes(self.network, mapping)
                print(f"Network loaded: {self.network.number_of_nodes()} nodes, {self.network.number_of_edges()} edges")
                return True
            except Exception as e:
                print(f"Error loading GraphML: {e}")
        
        print("Error: Could not load network. Please run network construction first.")
        return False
    
    def analyze_advanced_centralities(self):
        print("\nAnalyzing Advanced Centrality Measures...")
        
        if self.network is None:
            print("Error: Network not loaded")
            return None
        
        centrality_analysis = {}
        
        weak_components = list(nx.weakly_connected_components(self.network))
        largest_component = max(weak_components, key=len)
        subgraph = self.network.subgraph(largest_component)
        
        print(f"  Analyzing centralities on largest component: {len(largest_component)} nodes")
        
        print("  Computing degree centralities...")
        centrality_analysis['degree_centrality'] = nx.degree_centrality(subgraph)
        centrality_analysis['in_degree_centrality'] = nx.in_degree_centrality(subgraph)
        centrality_analysis['out_degree_centrality'] = nx.out_degree_centrality(subgraph)
        
        print("  Computing betweenness centrality...")
        if len(largest_component) > 1000:
            k_sample = min(1000, len(largest_component))
            centrality_analysis['betweenness_centrality'] = nx.betweenness_centrality(subgraph, k=k_sample)
        else:
            centrality_analysis['betweenness_centrality'] = nx.betweenness_centrality(subgraph)
        
        print("  Computing closeness centrality...")
        centrality_analysis['closeness_centrality'] = nx.closeness_centrality(subgraph)
        
        print("  Computing PageRank...")
        centrality_analysis['pagerank'] = nx.pagerank(subgraph, alpha=0.85)
        
        print("  Computing eigenvector centrality...")
        try:
            centrality_analysis['eigenvector_centrality'] = nx.eigenvector_centrality(subgraph, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            print("    Warning: Eigenvector centrality failed to converge")
            centrality_analysis['eigenvector_centrality'] = {}
        
        print("Centrality analysis completed")
        
        return centrality_analysis
    
    def detect_communities(self):
        print("\nDetecting Communities...")
        
        if self.network is None:
            print("Error: Network not loaded")
            return None
        
        if not COMMUNITY_AVAILABLE:
            print("Warning: No community detection libraries available")
            return None
        
        community_analysis = {}
        
        weak_components = list(nx.weakly_connected_components(self.network))
        largest_component = max(weak_components, key=len)
        subgraph = self.network.subgraph(largest_component)
        
        print(f"  Detecting communities on largest component: {len(largest_component)} nodes")
        
        if HAS_LOUVAIN:
            print("  Using Louvain community detection...")
            try:
                undirected_subgraph = subgraph.to_undirected()
                partition = community_louvain.best_partition(undirected_subgraph)
                communities = defaultdict(list)
                for node, community_id in partition.items():
                    communities[community_id].append(node)
                
                community_analysis['louvain_communities'] = dict(communities)
                community_analysis['louvain_modularity'] = community_louvain.modularity(partition, undirected_subgraph)
                print(f"    Found {len(communities)} communities with modularity {community_analysis['louvain_modularity']:.4f}")
            except Exception as e:
                print(f"    Error in Louvain community detection: {e}")
        
        if HAS_NX_COMMUNITY:
            print("  Using NetworkX community detection...")
            try:
                undirected_subgraph = subgraph.to_undirected()
                communities = list(nx_community.greedy_modularity_communities(undirected_subgraph))
                community_analysis['nx_communities'] = {i: list(comm) for i, comm in enumerate(communities)}
                print(f"    Found {len(communities)} communities")
            except Exception as e:
                print(f"    Error in NetworkX community detection: {e}")
        
        return community_analysis
    
    def analyze_network_motifs(self):
        print("\nAnalyzing Network Motifs...")
        
        if self.network is None:
            print("Error: Network not loaded")
            return None
        
        motif_analysis = {}
        
        weak_components = list(nx.weakly_connected_components(self.network))
        largest_component = max(weak_components, key=len)
        subgraph = self.network.subgraph(largest_component)
        
        print(f"  Analyzing motifs on largest component: {len(largest_component)} nodes")
        
        try:
            triangles = sum(nx.triangles(subgraph.to_undirected()).values()) // 3
            motif_analysis['triangles'] = triangles
            print(f"    Triangles: {triangles}")
        except Exception as e:
            print(f"    Error counting triangles: {e}")
            motif_analysis['triangles'] = 0
        
        try:
            squares = 0
            for node in list(subgraph.nodes())[:1000]:
                neighbors = set(subgraph.neighbors(node))
                for neighbor in neighbors:
                    neighbor_neighbors = set(subgraph.neighbors(neighbor))
                    squares += len(neighbors & neighbor_neighbors)
            squares = squares // 4
            motif_analysis['squares'] = squares
            print(f"    Squares (sampled): {squares}")
        except Exception as e:
            print(f"    Error counting squares: {e}")
            motif_analysis['squares'] = 0
        
        return motif_analysis
    
    def create_topology_visualizations(self):
        print("\nCreating Topology Visualizations...")
        
        if self.network is None:
            print("Error: Network not loaded")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Network Topology Analysis', fontsize=16, fontweight='bold')
        
        weak_components = list(nx.weakly_connected_components(self.network))
        largest_component = max(weak_components, key=len)
        subgraph = self.network.subgraph(largest_component)
        
        centrality_analysis = self.analyze_advanced_centralities()
        
        if centrality_analysis and 'pagerank' in centrality_analysis:
            pagerank_values = list(centrality_analysis['pagerank'].values())
            axes[0, 0].hist(pagerank_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('PageRank Distribution')
            axes[0, 0].set_xlabel('PageRank')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_yscale('log')
        
        if centrality_analysis and 'betweenness_centrality' in centrality_analysis:
            betweenness_values = list(centrality_analysis['betweenness_centrality'].values())
            axes[0, 1].hist(betweenness_values, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0, 1].set_title('Betweenness Centrality Distribution')
            axes[0, 1].set_xlabel('Betweenness Centrality')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_yscale('log')
        
        community_analysis = self.detect_communities()
        
        if community_analysis and 'louvain_communities' in community_analysis:
            community_sizes = [len(comm) for comm in community_analysis['louvain_communities'].values()]
            axes[1, 0].hist(community_sizes, bins=30, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].set_title('Community Size Distribution')
            axes[1, 0].set_xlabel('Community Size')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_yscale('log')
        
        if centrality_analysis and 'degree_centrality' in centrality_analysis:
            degree_centrality_values = list(centrality_analysis['degree_centrality'].values())
            axes[1, 1].hist(degree_centrality_values, bins=50, alpha=0.7, color='red', edgecolor='black')
            axes[1, 1].set_title('Degree Centrality Distribution')
            axes[1, 1].set_xlabel('Degree Centrality')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('network_topology_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Topology visualizations saved as 'network_topology_analysis.png'")
    
    def save_topology_results(self):
        print("\nSaving Topology Analysis Results...")
        
        results = {
            'centrality_analysis': self.analyze_advanced_centralities(),
            'community_analysis': self.detect_communities(),
            'motif_analysis': self.analyze_network_motifs()
        }
        
        with open('topology_analysis.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("Topology analysis results saved as 'topology_analysis.json'")
        
        return results
    
    def generate_topology_analysis_report(self):
        print("NETWORK TOPOLOGY CHARACTERISTICS ANALYSIS REPORT")
        print("="*80)
        
        if not self.load_network():
            return None
        
        self.create_topology_visualizations()
        results = self.save_topology_results()
        
        print("\nNetwork topology analysis completed successfully!")
        print("Ready for export visualizations")
        
        return results

def main():
    analyzer = NetworkTopologyAnalyzer()
    results = analyzer.generate_topology_analysis_report()
    print("\nNetwork topology analysis completed successfully!")

if __name__ == "__main__":
    main()