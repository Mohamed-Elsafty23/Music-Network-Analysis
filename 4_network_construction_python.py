import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class MusicNetworkConstructor:
    
    def __init__(self, cleaned_data_path="cleaned_data"):
        self.data_path = Path(cleaned_data_path)
        self.artist_data = None
        self.edge_data = None
        self.network = None
        self.network_stats = {}
        
    def load_preprocessed_data(self):
        print("Loading Preprocessed Data...")
        
        try:
            artist_file = self.data_path / "artist_lookup_cleaned.csv"
            self.artist_data = pd.read_csv(artist_file)
            print(f"Artist data loaded: {len(self.artist_data)} artists")
            
            edges_file = self.data_path / "network_edges_cleaned.csv"
            self.edge_data = pd.read_csv(edges_file)
            print(f"Edge data loaded: {len(self.edge_data)} relationships")
            
            return True
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please run data preprocessing first.")
            return False
    
    def calculate_advanced_node_attributes(self):
        print("\nCalculating Advanced Node Attributes...")
        
        source_counts = self.edge_data['source_artist'].value_counts()
        target_counts = self.edge_data['target_artist'].value_counts()
        
        enriched_artists = self.artist_data.copy()
        
        enriched_artists['out_degree'] = enriched_artists['artist_id'].map(source_counts).fillna(0).astype(int)
        enriched_artists['in_degree'] = enriched_artists['artist_id'].map(target_counts).fillna(0).astype(int)
        enriched_artists['total_degree'] = enriched_artists['out_degree'] + enriched_artists['in_degree']
        
        max_in_degree = enriched_artists['in_degree'].max() if enriched_artists['in_degree'].max() > 0 else 1
        enriched_artists['influence_score'] = enriched_artists['in_degree'] / max_in_degree
        
        max_out_degree = enriched_artists['out_degree'].max() if enriched_artists['out_degree'].max() > 0 else 1
        enriched_artists['diversity_score'] = enriched_artists['out_degree'] / max_out_degree
        
        enriched_artists['node_type'] = 'isolated'
        enriched_artists.loc[enriched_artists['out_degree'] > 0, 'node_type'] = 'source'
        enriched_artists.loc[enriched_artists['in_degree'] > 0, 'node_type'] = 'target'
        enriched_artists.loc[(enriched_artists['out_degree'] > 0) & (enriched_artists['in_degree'] > 0), 'node_type'] = 'hub'
        
        if 'birth_year' in enriched_artists.columns:
            current_year = 2024
            enriched_artists['age'] = current_year - enriched_artists['birth_year']
            enriched_artists.loc[enriched_artists['age'] < 0, 'age'] = np.nan
            
            enriched_artists['career_stage'] = 'unknown'
            enriched_artists.loc[enriched_artists['age'] <= 30, 'career_stage'] = 'emerging'
            enriched_artists.loc[(enriched_artists['age'] > 30) & (enriched_artists['age'] <= 50), 'career_stage'] = 'established'
            enriched_artists.loc[enriched_artists['age'] > 50, 'career_stage'] = 'veteran'
        
        artist_temporal_stats = self.edge_data.groupby('source_artist').agg({
            'cover_year': ['min', 'max', 'count'],
            'time_gap': ['mean', 'std']
        }).round(2)
        
        artist_temporal_stats.columns = ['first_cover_year', 'last_cover_year', 'cover_count', 'avg_time_gap', 'time_gap_std']
        artist_temporal_stats = artist_temporal_stats.reset_index()
        
        enriched_artists = enriched_artists.merge(artist_temporal_stats, 
                                                left_on='artist_id', 
                                                right_on='source_artist', 
                                                how='left')
        
        enriched_artists['activity_span'] = enriched_artists['last_cover_year'] - enriched_artists['first_cover_year']
        enriched_artists['activity_span'] = enriched_artists['activity_span'].fillna(0)
        
        self.artist_data = enriched_artists
        print(f"Advanced attributes calculated for {len(enriched_artists)} artists")
        
        return enriched_artists
    
    def calculate_edge_weights(self):
        print("\nCalculating Edge Weights...")
        
        enriched_edges = self.edge_data.copy()
        
        max_year = enriched_edges['cover_year'].max()
        enriched_edges['temporal_weight'] = 1.0 - (max_year - enriched_edges['cover_year']) / (max_year - enriched_edges['cover_year'].min())
        
        optimal_gap = 10
        enriched_edges['time_gap_weight'] = 1.0 / (1.0 + abs(enriched_edges['time_gap'] - optimal_gap) / optimal_gap)
        
        original_decade = (enriched_edges['original_year'] // 10) * 10
        cover_decade = (enriched_edges['cover_year'] // 10) * 10
        enriched_edges['cross_decade'] = (original_decade != cover_decade).astype(int)
        
        enriched_edges['edge_weight'] = (
            0.4 * enriched_edges['temporal_weight'] +
            0.3 * enriched_edges['time_gap_weight'] +
            0.2 * 1.0 +
            0.1 * enriched_edges['cross_decade']
        )
        
        enriched_edges['edge_weight_normalized'] = enriched_edges['edge_weight'] / enriched_edges['edge_weight'].max()
        
        self.edge_data = enriched_edges
        print(f"Edge weights calculated for {len(enriched_edges)} relationships")
        
        return enriched_edges
    
    def construct_network(self):
        print("\nConstructing Music Cover Network...")
        
        G = nx.DiGraph()
        
        for _, artist in self.artist_data.iterrows():
            node_attrs = {
                'name': artist['name'],
                'artist_type': artist['artist_type'],
                'home_country': artist['home_country'],
                'birth_year': artist['birth_year'],
                'in_degree': artist['in_degree'],
                'out_degree': artist['out_degree'],
                'total_degree': artist['total_degree'],
                'influence_score': artist['influence_score'],
                'diversity_score': artist['diversity_score'],
                'node_type': artist['node_type'],
                'career_stage': artist['career_stage'],
                'activity_span': artist['activity_span'],
                'age': artist['age']
            }
            G.add_node(artist['artist_id'], **node_attrs)
        
        for _, edge in self.edge_data.iterrows():
            edge_attrs = {
                'song_title': edge['song_title'],
                'original_year': edge['original_year'],
                'cover_year': edge['cover_year'],
                'time_gap': edge['time_gap'],
                'weight': edge['edge_weight_normalized'],
                'edge_category': 'standard',
                'cross_decade': edge['cross_decade']
            }
            G.add_edge(edge['source_artist'], edge['target_artist'], **edge_attrs)
        
        self.network = G
        print(f"Network constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def calculate_network_statistics(self):
        print("\nCalculating Network Statistics...")
        
        G = self.network
        
        stats = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_directed': G.is_directed(),
            'is_weighted': nx.is_weighted(G),
            'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
            'max_degree': max(dict(G.degree()).values()),
            'min_degree': min(dict(G.degree()).values())
        }
        
        if nx.is_strongly_connected(G):
            stats['strongly_connected_components'] = 1
        else:
            stats['strongly_connected_components'] = nx.number_strongly_connected_components(G)
        
        if nx.is_weakly_connected(G):
            stats['weakly_connected_components'] = 1
        else:
            stats['weakly_connected_components'] = nx.number_weakly_connected_components(G)
        
        try:
            stats['average_clustering'] = nx.average_clustering(G.to_undirected())
        except:
            stats['average_clustering'] = 0.0
        
        try:
            stats['average_shortest_path'] = nx.average_shortest_path_length(G)
        except:
            stats['average_shortest_path'] = float('inf')
        
        self.network_stats = stats
        
        print("Network Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return stats
    
    def save_network(self):
        print("\nSaving Network...")
        
        output_dir = Path("network_output")
        output_dir.mkdir(exist_ok=True)
        
        G = self.network
        
        pickle_path = output_dir / "music_cover_network.pickle"
        with open(pickle_path, 'wb') as f:
            pickle.dump(G, f)
        print(f"Network saved as pickle: {pickle_path}")
        
        graphml_path = output_dir / "music_cover_network.graphml"
        nx.write_graphml(G, graphml_path)
        print(f"Network saved as GraphML: {graphml_path}")
        
        gml_path = output_dir / "music_cover_network.gml"
        nx.write_gml(G, gml_path)
        print(f"Network saved as GML: {gml_path}")
        
        nodes_df = pd.DataFrame([(node, data) for node, data in G.nodes(data=True)])
        if not nodes_df.empty:
            nodes_df.columns = ['node_id'] + list(nodes_df.columns[1:])
            nodes_path = output_dir / "music_cover_network_nodes.csv"
            nodes_df.to_csv(nodes_path, index=False)
            print(f"Nodes saved as CSV: {nodes_path}")
        
        edges_df = pd.DataFrame([(u, v, data) for u, v, data in G.edges(data=True)])
        if not edges_df.empty:
            edges_df.columns = ['source', 'target'] + list(edges_df.columns[2:])
            edges_path = output_dir / "music_cover_network_edges.csv"
            edges_df.to_csv(edges_path, index=False)
            print(f"Edges saved as CSV: {edges_path}")
        
        stats_path = output_dir / "network_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(self.network_stats, f, indent=2)
        print(f"Statistics saved as JSON: {stats_path}")
    
    def generate_construction_report(self):
        print("PYTHON NETWORK CONSTRUCTION REPORT")
        print("="*80)
        
        if not self.load_preprocessed_data():
            return None
        
        self.calculate_advanced_node_attributes()
        self.calculate_edge_weights()
        self.construct_network()
        self.calculate_network_statistics()
        self.save_network()
        
        print("\nNetwork construction completed successfully!")
        print("Network ready for analysis")
        
        return {
            'network_stats': self.network_stats,
            'nodes': self.network.number_of_nodes(),
            'edges': self.network.number_of_edges()
        }

def main():
    constructor = MusicNetworkConstructor()
    results = constructor.generate_construction_report()
    print("\nNetwork construction completed successfully!")

if __name__ == "__main__":
    main()