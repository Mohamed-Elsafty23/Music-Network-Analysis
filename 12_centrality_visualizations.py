import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class CentralityVisualizationGenerator:
    
    def __init__(self, 
                 centrality_file='centrality_top10_with_names.json',
                 output_dir='network_output'):
        self.centrality_file = centrality_file
        self.output_dir = output_dir
        self.data = None
    
    def load_data(self):
        """Load centrality data"""
        # Set up output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Load JSON file
        with open(self.centrality_file) as f:
            self.data = json.load(f)
    
    def create_pagerank_bar_chart(self):
        """Create bar chart of top PageRank scores"""
        # Extract DegreeCentrality data (adjust if there are other keys)
        degree_data = self.data.get('DegreeCentrality', [])
        df = pd.DataFrame(degree_data)

        # Debug: Check df
        print("DataFrame preview:")
        print(df.head())

        # Print summary
        print("\nTop 10 Artists by PageRank:")
        for _, row in df.iterrows():
            print(f"{row['ArtistName']}: {row['PageRank']:.6f}")

        # 1. Bar Chart of Top PageRank Scores
        plt.figure(figsize=(10, 6))
        sns.barplot(x='ArtistName', y='PageRank', data=df, palette='viridis')
        plt.title('Top 10 Artists by PageRank Score')
        plt.xlabel('Artist')
        plt.ylabel('PageRank Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'top_pagerank_bar.png'))
        plt.close()
        
        return df
    
    def create_centrality_heatmap(self, df):
        """Create heatmap of centrality metrics"""
        # 2. Heatmap of Centrality Metrics
        centrality_df = df[['ArtistName', 'PageRank', 'InDegreeCentrality', 'OutDegreeCentrality', 'EigenvectorCentrality']].set_index('ArtistName')
        plt.figure(figsize=(8, 6))
        sns.heatmap(centrality_df, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title('Centrality Metrics for Top 10 Artists')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'centrality_heatmap.png'))
        plt.close()
    
    def create_pagerank_vs_indegree_scatter(self, df):
        """Create scatter plot of PageRank vs InDegree"""
        # 3. Scatter Plot: PageRank vs. In-Degree
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='InDegreeCentrality', y='PageRank', data=df, hue='ArtistName', size='OutDegreeCentrality', sizes=(50, 500))
        plt.title('PageRank vs. In-Degree Centrality')
        plt.xlabel('In-Degree Centrality')
        plt.ylabel('PageRank Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pagerank_vs_indegree_scatter_02.png'))
        plt.close()
    
    def generate_centrality_visualizations(self):
        """Main method to generate all centrality visualizations"""
        print("Generating centrality visualizations...")
        
        # Load data
        self.load_data()
        
        # Create visualizations
        df = self.create_pagerank_bar_chart()
        self.create_centrality_heatmap(df)
        self.create_pagerank_vs_indegree_scatter(df)
        
        results = {
            "status": "completed",
            "files_generated": [
                os.path.join(self.output_dir, 'top_pagerank_bar.png'),
                os.path.join(self.output_dir, 'centrality_heatmap.png'),
                os.path.join(self.output_dir, 'pagerank_vs_indegree_scatter_02.png')
            ],
            "artists_visualized": len(df)
        }
        
        print("Centrality visualizations completed successfully.")
        return results

if __name__ == "__main__":
    generator = CentralityVisualizationGenerator()
    generator.generate_centrality_visualizations()