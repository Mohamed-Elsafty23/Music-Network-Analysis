import networkx as nx
import pickle
import json
import pandas as pd

class CentralityAnalyzer:
    
    def __init__(self, 
                 network_file="network_output/music_cover_network.pickle",
                 artist_file="cleaned_data/artist_lookup_cleaned.csv"):
        self.network_file = network_file
        self.artist_file = artist_file
        self.G = None
        self.artist_map = {}
    
    def load_data(self):
        """Load network and artist data"""
        # Load network
        with open(self.network_file, "rb") as f:
            self.G = pickle.load(f)

        # Load artist data
        artist_df = pd.read_csv(self.artist_file)
        self.artist_map = pd.Series(artist_df.common_name.values, index=artist_df.artist_id).to_dict()
    
    def compute_centralities(self):
        """Compute all centrality measures"""
        print("Computing centrality measures...")
        
        # Compute centralities
        degree_centrality = nx.degree_centrality(self.G)
        in_degree_centrality = nx.in_degree_centrality(self.G)
        out_degree_centrality = nx.out_degree_centrality(self.G)
        
        try:
            eigenvector_centrality = nx.eigenvector_centrality(self.G, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            print("Warning: Eigenvector centrality failed to converge, using zero values")
            eigenvector_centrality = {node: 0 for node in self.G.nodes()}
        
        pagerank_centrality = nx.pagerank(self.G, alpha=0.85)

        return {
            'degree': degree_centrality,
            'in_degree': in_degree_centrality,
            'out_degree': out_degree_centrality,
            'eigenvector': eigenvector_centrality,
            'pagerank': pagerank_centrality
        }
    
    def prepare_centrality_data(self, centralities):
        """Prepare centrality data with names and round to 5 decimals"""
        data = []
        for node in self.G.nodes():
            data.append({
                "ArtistID": node,
                "ArtistName": self.artist_map.get(node, str(node)),
                "DegreeCentrality": round(centralities['degree'].get(node, 0), 5),
                "InDegreeCentrality": round(centralities['in_degree'].get(node, 0), 5),
                "OutDegreeCentrality": round(centralities['out_degree'].get(node, 0), 5),
                "EigenvectorCentrality": round(centralities['eigenvector'].get(node, 0), 5),
                "PageRank": round(centralities['pagerank'].get(node, 0), 5)
            })
        return data
    
    def get_top_10_rankings(self, data):
        """Get top 10 rankings for each centrality measure"""
        top10 = {
            "DegreeCentrality": sorted(data, key=lambda x: x["DegreeCentrality"], reverse=True)[:10],
            "InDegreeCentrality": sorted(data, key=lambda x: x["InDegreeCentrality"], reverse=True)[:10],
            "OutDegreeCentrality": sorted(data, key=lambda x: x["OutDegreeCentrality"], reverse=True)[:10],
            "EigenvectorCentrality": sorted(data, key=lambda x: x["EigenvectorCentrality"], reverse=True)[:10],
            "PageRank": sorted(data, key=lambda x: x["PageRank"], reverse=True)[:10]
        }
        return top10
    
    def save_results(self, data, top10):
        """Save centrality results to files"""
        # Save complete results
        print("Saving centrality analysis results...")
        
        # Save top 10 per centrality measure in JSON
        with open("centrality_top10_with_names.json", "w") as f:
            json.dump(top10, f, indent=2)
        
        # Print summary to console
        print("\n" + "="*80)
        print("CENTRALITY ANALYSIS SUMMARY")
        print("="*80)
        print(f"Total artists analyzed: {len(data)}")
        print(f"Top 10 rankings saved to: centrality_top10_with_names.json")
    
    def generate_centrality_analysis(self):
        """Main method to generate centrality analysis"""
        print("Starting centrality analysis...")
        
        # Load data
        self.load_data()
        
        # Compute centralities
        centralities = self.compute_centralities()
        
        # Prepare data
        data = self.prepare_centrality_data(centralities)
        
        # Get top 10 rankings
        top10 = self.get_top_10_rankings(data)
        
        # Save results
        self.save_results(data, top10)
        
        results = {
            "status": "completed",
            "files_generated": ["centrality_top10_with_names.json"],
            "total_artists_analyzed": len(data),
            "centrality_measures_computed": list(centralities.keys())
        }
        
        print("Centrality analysis completed successfully.")
        return results

if __name__ == "__main__":
    analyzer = CentralityAnalyzer()
    analyzer.generate_centrality_analysis()
if __name__ == "__main__":
    analyzer = CentralityAnalyzer()
    analyzer.generate_centrality_analysis()