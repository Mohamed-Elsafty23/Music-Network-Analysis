import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class BasicNetworkAnalyzer:
    
    def __init__(self, network_path="network_output"):
        self.network_path = Path(network_path)
        self.network = None
        self.basic_stats = {}
        
    def load_network(self):
        print("Loading Constructed Network...")
        
        pickle_file = self.network_path / "music_cover_network.pickle"
        if pickle_file.exists():
            try:
                with open(pickle_file, 'rb') as f:
                    self.network = pickle.load(f)
                print(f"Network loaded from pickle: {self.network.number_of_nodes()} nodes, {self.network.number_of_edges()} edges")
                return True
            except Exception as e:
                print(f"Warning: Error loading pickle: {e}")
        
        graphml_file = self.network_path / "music_cover_network.graphml"
        if graphml_file.exists():
            try:
                self.network = nx.read_graphml(graphml_file)
                mapping = {node: int(node) for node in self.network.nodes()}
                self.network = nx.relabel_nodes(self.network, mapping)
                print(f"Network loaded from GraphML: {self.network.number_of_nodes()} nodes, {self.network.number_of_edges()} edges")
                return True
            except Exception as e:
                print(f"Warning: Error loading GraphML: {e}")
        
        try:
            self.network = self.load_from_csv_files()
            if self.network:
                print(f"Network reconstructed from CSV: {self.network.number_of_nodes()} nodes, {self.network.number_of_edges()} edges")
                return True
        except Exception as e:
            print(f"Error loading from CSV: {e}")
        
        print("Error: Could not load network. Please run network construction first.")
        return False
    
    def load_from_csv_files(self):
        nodes_file = self.network_path / "music_cover_network_nodes.csv"
        edges_file = self.network_path / "music_cover_network_edges.csv"
        
        if not (nodes_file.exists() and edges_file.exists()):
            return None
        
        nodes_df = pd.read_csv(nodes_file)
        edges_df = pd.read_csv(edges_file)
        
        G = nx.DiGraph()
        
        for _, node in nodes_df.iterrows():
            node_attrs = {k: v for k, v in node.to_dict().items() if k != 'node_id'}
            G.add_node(node['node_id'], **node_attrs)
        
        for _, edge in edges_df.iterrows():
            edge_attrs = {k: v for k, v in edge.to_dict().items() if k not in ['source', 'target']}
            G.add_edge(edge['source'], edge['target'], **edge_attrs)
        
        return G
    
    def calculate_basic_statistics(self):
        print("\nCalculating Basic Network Statistics...")
        
        if self.network is None:
            print("Error: Network not loaded")
            return None
        
        num_nodes = self.network.number_of_nodes()
        num_edges = self.network.number_of_edges()
        
        density = nx.density(self.network)
        
        degrees = dict(self.network.degree())
        avg_degree = sum(degrees.values()) / num_nodes
        max_degree = max(degrees.values())
        min_degree = min(degrees.values())
        
        in_degrees = dict(self.network.in_degree())
        out_degrees = dict(self.network.out_degree())
        
        avg_in_degree = sum(in_degrees.values()) / num_nodes
        avg_out_degree = sum(out_degrees.values()) / num_nodes
        
        self.basic_stats = {
            'nodes': num_nodes,
            'edges': num_edges,
            'density': density,
            'average_degree': avg_degree,
            'max_degree': max_degree,
            'min_degree': min_degree,
            'average_in_degree': avg_in_degree,
            'average_out_degree': avg_out_degree,
            'is_directed': self.network.is_directed(),
            'is_weighted': nx.is_weighted(self.network)
        }
        
        print("Basic Network Statistics:")
        print(f"  Nodes: {num_nodes:,}")
        print(f"  Edges: {num_edges:,}")
        print(f"  Density: {density:.6f}")
        print(f"  Average degree: {avg_degree:.2f}")
        print(f"  Max degree: {max_degree}")
        print(f"  Min degree: {min_degree}")
        print(f"  Average in-degree: {avg_in_degree:.2f}")
        print(f"  Average out-degree: {avg_out_degree:.2f}")
        
        return self.basic_stats
    
    def analyze_degree_distribution(self):
        print("\nAnalyzing Degree Distribution...")
        
        if self.network is None:
            print("Error: Network not loaded")
            return None
        
        degrees = list(dict(self.network.degree()).values())
        in_degrees = list(dict(self.network.in_degree()).values())
        out_degrees = list(dict(self.network.out_degree()).values())
        
        degree_counts = Counter(degrees)
        in_degree_counts = Counter(in_degrees)
        out_degree_counts = Counter(out_degrees)
        
        degree_distribution = {
            'total_degrees': degrees,
            'in_degrees': in_degrees,
            'out_degrees': out_degrees,
            'degree_counts': dict(degree_counts),
            'in_degree_counts': dict(in_degree_counts),
            'out_degree_counts': dict(out_degree_counts)
        }
        
        print("Degree Distribution Summary:")
        print(f"  Total degree range: {min(degrees)} to {max(degrees)}")
        print(f"  In-degree range: {min(in_degrees)} to {max(in_degrees)}")
        print(f"  Out-degree range: {min(out_degrees)} to {max(out_degrees)}")
        
        return degree_distribution
    
    def create_basic_visualizations(self):
        print("\nCreating Basic Network Visualizations...")
        
        if self.network is None:
            print("Error: Network not loaded")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Basic Network Properties Analysis', fontsize=16, fontweight='bold')
        
        degrees = list(dict(self.network.degree()).values())
        in_degrees = list(dict(self.network.in_degree()).values())
        out_degrees = list(dict(self.network.out_degree()).values())
        
        axes[0, 0].hist(degrees, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Total Degree Distribution')
        axes[0, 0].set_xlabel('Degree')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_yscale('log')
        
        axes[0, 1].hist(in_degrees, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('In-Degree Distribution')
        axes[0, 1].set_xlabel('In-Degree')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_yscale('log')
        
        axes[1, 0].hist(out_degrees, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title('Out-Degree Distribution')
        axes[1, 0].set_xlabel('Out-Degree')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_yscale('log')
        
        degree_counts = Counter(degrees)
        degree_values = sorted(degree_counts.keys())
        degree_frequencies = [degree_counts[d] for d in degree_values]
        
        axes[1, 1].loglog(degree_values, degree_frequencies, 'o-', markersize=4, linewidth=1)
        axes[1, 1].set_title('Degree Distribution (Log-Log Scale)')
        axes[1, 1].set_xlabel('Degree')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('basic_network_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Basic network visualizations saved as 'basic_network_analysis.png'")
    
    def save_analysis_results(self):
        print("\nSaving Analysis Results...")
        
        results = {
            'basic_statistics': self.basic_stats,
            'degree_distribution': self.analyze_degree_distribution()
        }
        
        with open('basic_network_analysis.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("Analysis results saved as 'basic_network_analysis.json'")
        
        return results
    
    def generate_basic_analysis_report(self):
        print("BASIC NETWORK PROPERTIES ANALYSIS REPORT")
        print("="*80)
        
        if not self.load_network():
            return None
        
        self.calculate_basic_statistics()
        self.create_basic_visualizations()
        results = self.save_analysis_results()
        
        print("\nBasic network analysis completed successfully!")
        print("Ready for connectivity analysis")
        
        return results

def main():
    analyzer = BasicNetworkAnalyzer()
    results = analyzer.generate_basic_analysis_report()
    print("\nBasic network analysis completed successfully!")

if __name__ == "__main__":
    main()