import json
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class StatisticalVisualizationGenerator:
    
    def __init__(self, 
                 network_file="network_output/music_cover_network.pickle",
                 topology_file="topology_analysis.json",
                 basic_file="basic_network_analysis.json"):
        self.network_file = network_file
        self.topology_file = topology_file
        self.basic_file = basic_file
        self.G = None
        self.pagerank = {}
        self.node_info = {}
    
    def load_data(self):
        """Load network and analysis data"""
        # Load network
        with open(self.network_file, "rb") as f:
            self.G = pickle.load(f)

        # Load PageRank scores
        with open(self.topology_file, "r") as f:
            topology_data = json.load(f)
        pagerank_raw = topology_data['topology_statistics']['centralities']['pagerank']
        self.pagerank = {int(k): v for k, v in pagerank_raw.items()}

        # Load artist names and degree hubs
        with open(self.basic_file, "r") as f:
            basic_data = json.load(f)
        self.node_info = {entry['node']: entry['name'] for entry in basic_data['network_hubs']['total_degree_hubs']}
    
    def create_pagerank_bar_chart(self):
        """Create bar chart of top PageRank scores"""
        # Build DataFrame for top 10 PageRank artists
        top_10 = sorted(self.pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
        df_top10 = pd.DataFrame({
            "Node": [node for node, _ in top_10],
            "Name": [self.node_info.get(node, str(node)) for node, _ in top_10],
            "PageRank": [score for _, score in top_10],
            "InDegree": [self.G.in_degree(node) for node, _ in top_10],
            "OutDegree": [self.G.out_degree(node) for node, _ in top_10]
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
        
        return df_top10
    
    def create_centrality_heatmap(self, df_top10):
        """Create heatmap of centrality metrics"""
        # --- 2. Heatmap of Centrality Metrics ---
        heatmap_data = df_top10.set_index("Name")[["PageRank", "InDegree", "OutDegree"]]
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, cmap="magma", fmt=".2f")
        plt.title("Centrality Metrics for Top 10 Artists")
        plt.tight_layout()
        plt.savefig("top_10_metrics_heatmap.png", dpi=300)
        plt.close()
    
    def create_pagerank_vs_indegree_scatter(self, df_top10):
        """Create scatter plot of PageRank vs InDegree"""
        # --- 3. Scatter Plot: PageRank vs. In-Degree ---
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='InDegree', y='PageRank', data=df_top10, hue='Name', size='OutDegree', sizes=(50, 500))
        plt.title('PageRank vs. In-Degree Centrality')
        plt.xlabel('In-Degree Centrality')
        plt.ylabel('PageRank Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("pagerank_vs_indegree_scatter.png", dpi=300)
        plt.close()
    
    def create_full_network_scatter(self):
        """Create scatter plot for entire network"""
        # --- 4. Full Network Scatter Plot ---
        all_nodes = list(self.G.nodes())
        df_all = pd.DataFrame({
            "Node": all_nodes,
            "Name": [self.node_info.get(node, str(node)) for node in all_nodes],
            "PageRank": [self.pagerank.get(node, 0) for node in all_nodes],
            "InDegree": [self.G.in_degree(node) for node in all_nodes],
            "OutDegree": [self.G.out_degree(node) for node in all_nodes]
        })

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='InDegree', y='PageRank', data=df_all, alpha=0.6, color='blue')
        plt.title('PageRank vs. In-Degree (All Artists)')
        plt.xlabel('In-Degree')
        plt.ylabel('PageRank Score')
        plt.tight_layout()
        plt.savefig("pagerank_vs_indegree_scatter_02.png", dpi=300)
        plt.close()
        
        return df_all
    
    def generate_statistical_visualizations(self):
        """Main method to generate all statistical visualizations"""
        print("Generating statistical visualizations...")
        
        # Load data
        self.load_data()
        
        # Create visualizations
        df_top10 = self.create_pagerank_bar_chart()
        self.create_centrality_heatmap(df_top10)
        self.create_pagerank_vs_indegree_scatter(df_top10)
        df_all = self.create_full_network_scatter()
        
        results = {
            "status": "completed",
            "files_generated": [
                "top_10_pagerank_bar_chart.png",
                "top_10_metrics_heatmap.png",
                "pagerank_vs_indegree_scatter.png",
                "pagerank_vs_indegree_scatter_02.png"
            ],
            "top_artists_analyzed": len(df_top10),
            "total_artists_analyzed": len(df_all)
        }
        
        print("Statistical visualizations completed successfully.")
        return results

if __name__ == "__main__":
    generator = StatisticalVisualizationGenerator()
    generator.generate_statistical_visualizations()