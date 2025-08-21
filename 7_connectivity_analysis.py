import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class ConnectivityAnalyzer:
    
    def __init__(self, network_path="network_output"):
        self.network_path = Path(network_path)
        self.network = None
        self.connectivity_stats = {}
        
    def load_network(self):
        print("Loading Network for Connectivity Analysis...")
        
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
    
    def analyze_connected_components(self):
        print("\nAnalyzing Connected Components...")
        
        if self.network is None:
            print("Error: Network not loaded")
            return None
        
        components_analysis = {}
        
        weak_components = list(nx.weakly_connected_components(self.network))
        weak_component_sizes = [len(comp) for comp in weak_components]
        
        components_analysis['weak_components'] = {
            'count': len(weak_components),
            'sizes': weak_component_sizes,
            'largest_size': max(weak_component_sizes) if weak_component_sizes else 0,
            'largest_percentage': max(weak_component_sizes) / self.network.number_of_nodes() * 100 if weak_component_sizes else 0,
            'size_distribution': dict(pd.Series(weak_component_sizes).value_counts().sort_index())
        }
        
        strong_components = list(nx.strongly_connected_components(self.network))
        strong_component_sizes = [len(comp) for comp in strong_components]
        
        components_analysis['strong_components'] = {
            'count': len(strong_components),
            'sizes': strong_component_sizes,
            'largest_size': max(strong_component_sizes) if strong_component_sizes else 0,
            'largest_percentage': max(strong_component_sizes) / self.network.number_of_nodes() * 100 if strong_component_sizes else 0,
            'size_distribution': dict(pd.Series(strong_component_sizes).value_counts().sort_index())
        }
        
        if weak_component_sizes:
            largest_weak_component = max(weak_components, key=len)
            giant_component = self.network.subgraph(largest_weak_component)
            
            components_analysis['giant_component'] = {
                'size': len(largest_weak_component),
                'percentage': len(largest_weak_component) / self.network.number_of_nodes() * 100,
                'density': nx.density(giant_component),
                'edges': giant_component.number_of_edges()
            }
        
        print("Connected Components Analysis:")
        print(f"  Weakly connected components: {len(weak_components)}")
        print(f"  Strongly connected components: {len(strong_components)}")
        if weak_component_sizes:
            print(f"  Largest weak component: {max(weak_component_sizes)} nodes ({max(weak_component_sizes)/self.network.number_of_nodes()*100:.1f}%)")
        if strong_component_sizes:
            print(f"  Largest strong component: {max(strong_component_sizes)} nodes ({max(strong_component_sizes)/self.network.number_of_nodes()*100:.1f}%)")
        
        return components_analysis
    
    def calculate_average_path_length(self):
        print("\nCalculating Average Path Length...")
        
        if self.network is None:
            print("Error: Network not loaded")
            return None
        
        path_analysis = {}
        
        largest_weak_component = max(nx.weakly_connected_components(self.network), key=len)
        giant_component = self.network.subgraph(largest_weak_component)
        
        try:
            avg_path_length = nx.average_shortest_path_length(giant_component)
            path_analysis['average_path_length'] = avg_path_length
            print(f"  Average path length (giant component): {avg_path_length:.3f}")
        except Exception as e:
            print(f"  Error calculating average path length: {e}")
            path_analysis['average_path_length'] = None
        
        try:
            diameter = nx.diameter(giant_component)
            path_analysis['diameter'] = diameter
            print(f"  Network diameter: {diameter}")
        except Exception as e:
            print(f"  Error calculating diameter: {e}")
            path_analysis['diameter'] = None
        
        try:
            radius = nx.radius(giant_component)
            path_analysis['radius'] = radius
            print(f"  Network radius: {radius}")
        except Exception as e:
            print(f"  Error calculating radius: {e}")
            path_analysis['radius'] = None
        
        return path_analysis
    
    def calculate_clustering_coefficient(self):
        print("\nCalculating Clustering Coefficient...")
        
        if self.network is None:
            print("Error: Network not loaded")
            return None
        
        clustering_analysis = {}
        
        try:
            avg_clustering = nx.average_clustering(self.network.to_undirected())
            clustering_analysis['average_clustering'] = avg_clustering
            print(f"  Average clustering coefficient: {avg_clustering:.4f}")
        except Exception as e:
            print(f"  Error calculating average clustering: {e}")
            clustering_analysis['average_clustering'] = None
        
        try:
            transitivity = nx.transitivity(self.network.to_undirected())
            clustering_analysis['transitivity'] = transitivity
            print(f"  Transitivity: {transitivity:.4f}")
        except Exception as e:
            print(f"  Error calculating transitivity: {e}")
            clustering_analysis['transitivity'] = None
        
        return clustering_analysis
    
    def create_connectivity_visualizations(self):
        print("\nCreating Connectivity Visualizations...")
        
        if self.network is None:
            print("Error: Network not loaded")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Network Connectivity Analysis', fontsize=16, fontweight='bold')
        
        weak_components = list(nx.weakly_connected_components(self.network))
        weak_component_sizes = [len(comp) for comp in weak_components]
        
        axes[0, 0].hist(weak_component_sizes, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Weak Component Size Distribution')
        axes[0, 0].set_xlabel('Component Size')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_yscale('log')
        
        strong_components = list(nx.strongly_connected_components(self.network))
        strong_component_sizes = [len(comp) for comp in strong_components]
        
        axes[0, 1].hist(strong_component_sizes, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Strong Component Size Distribution')
        axes[0, 1].set_xlabel('Component Size')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_yscale('log')
        
        largest_weak_component = max(weak_components, key=len)
        giant_component = self.network.subgraph(largest_weak_component)
        
        try:
            path_lengths = []
            for source in list(giant_component.nodes())[:100]:
                lengths = nx.single_source_shortest_path_length(giant_component, source)
                path_lengths.extend(lengths.values())
            
            axes[1, 0].hist(path_lengths, bins=30, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].set_title('Shortest Path Length Distribution')
            axes[1, 0].set_xlabel('Path Length')
            axes[1, 0].set_ylabel('Frequency')
        except Exception as e:
            axes[1, 0].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Shortest Path Length Distribution')
        
        try:
            clustering_coeffs = nx.clustering(giant_component.to_undirected())
            coeff_values = list(clustering_coeffs.values())
            
            axes[1, 1].hist(coeff_values, bins=30, alpha=0.7, color='red', edgecolor='black')
            axes[1, 1].set_title('Clustering Coefficient Distribution')
            axes[1, 1].set_xlabel('Clustering Coefficient')
            axes[1, 1].set_ylabel('Frequency')
        except Exception as e:
            axes[1, 1].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Clustering Coefficient Distribution')
        
        plt.tight_layout()
        plt.savefig('connectivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Connectivity visualizations saved as 'connectivity_analysis.png'")
    
    def save_connectivity_results(self):
        print("\nSaving Connectivity Analysis Results...")
        
        results = {
            'connected_components': self.analyze_connected_components(),
            'path_analysis': self.calculate_average_path_length(),
            'clustering_analysis': self.calculate_clustering_coefficient()
        }
        
        with open('connectivity_analysis.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("Connectivity analysis results saved as 'connectivity_analysis.json'")
        
        return results
    
    def generate_connectivity_analysis_report(self):
        print("CONNECTIVITY ANALYSIS REPORT")
        print("="*80)
        
        if not self.load_network():
            return None
        
        self.create_connectivity_visualizations()
        results = self.save_connectivity_results()
        
        print("\nConnectivity analysis completed successfully!")
        print("Ready for network topology analysis")
        
        return results

def main():
    analyzer = ConnectivityAnalyzer()
    results = analyzer.generate_connectivity_analysis_report()
    print("\nConnectivity analysis completed successfully!")

if __name__ == "__main__":
    main()