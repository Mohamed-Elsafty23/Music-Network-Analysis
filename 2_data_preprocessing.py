import pandas as pd
import numpy as np
import ast
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MusicDataPreprocessor:
    
    def __init__(self, data_path="MusicData/MusicData"):
        self.data_path = Path(data_path)
        self.raw_data = {}
        self.cleaned_data = {}
        
    def load_raw_datasets(self):
        print("Loading Raw Music Datasets...")
        
        artists_path = self.data_path / "artists.csv" / "neo4j_artists.csv"
        self.raw_data['artists'] = pd.read_csv(artists_path)
        print(f"Raw artists dataset: {len(self.raw_data['artists'])} records")
        
        originals_path = self.data_path / "originals.csv" / "originals.csv"
        self.raw_data['originals'] = pd.read_csv(originals_path)
        print(f"Raw originals dataset: {len(self.raw_data['originals'])} records")
        
        covers_path = self.data_path / "covers.csv" / "covers.csv" 
        self.raw_data['covers'] = pd.read_csv(covers_path)
        print(f"Raw covers dataset: {len(self.raw_data['covers'])} records")
        
        releases_path = self.data_path / "releases.csv" / "releases.csv"
        self.raw_data['releases'] = pd.read_csv(releases_path)
        print(f"Raw releases dataset: {len(self.raw_data['releases'])} records")
        
    def assess_data_quality(self):
        print("\nDATA QUALITY ASSESSMENT")
        print("="*60)
        
        quality_report = {}
        
        for dataset_name, df in self.raw_data.items():
            print(f"\n{dataset_name.upper()} Dataset Quality:")
            
            total_records = len(df)
            total_columns = len(df.columns)
            
            missing_counts = df.isnull().sum()
            missing_percentages = (missing_counts / total_records) * 100
            
            duplicate_count = df.duplicated().sum()
            dtype_info = df.dtypes.value_counts()
            
            quality_metrics = {
                'total_records': total_records,
                'total_columns': total_columns,
                'missing_values': missing_counts.sum(),
                'missing_percentage': (missing_counts.sum() / (total_records * total_columns)) * 100,
                'duplicate_records': duplicate_count,
                'data_types': dict(dtype_info)
            }
            
            quality_report[dataset_name] = quality_metrics
            
            print(f"  Records: {total_records:,}")
            print(f"  Columns: {total_columns}")
            print(f"  Missing values: {missing_counts.sum():,} ({quality_metrics['missing_percentage']:.2f}%)")
            print(f"  Duplicates: {duplicate_count:,}")
            
            high_missing = missing_percentages[missing_percentages > 20]
            if len(high_missing) > 0:
                print(f"  Columns with >20% missing: {list(high_missing.index)}")
        
        return quality_report
    
    def clean_artists_dataset(self):
        print("\nCleaning Artists Dataset...")
        
        df = self.raw_data['artists'].copy()
        initial_count = len(df)
        
        df = df.dropna(subset=['artist_id', 'common_name'])
        print(f"Removed {initial_count - len(df)} records missing critical info")
        
        df['common_name'] = df['common_name'].str.strip()
        df['common_name'] = df['common_name'].str.replace(r'\s+', ' ', regex=True)
        
        df['home_country'] = df['home_country'].fillna('Unknown')
        df['artist_type'] = df['artist_type'].fillna('Unknown')
        
        df['birth_year'] = pd.to_numeric(df['birth_year'], errors='coerce')
        df['birth_year'] = df['birth_year'].fillna(0)
        
        df = df.drop_duplicates(subset=['artist_id'])
        print(f"Final artists dataset: {len(df)} records")
        
        self.cleaned_data['artists'] = df
        return df
    
    def clean_originals_dataset(self):
        print("\nCleaning Originals Dataset...")
        
        df = self.raw_data['originals'].copy()
        initial_count = len(df)
        
        df = df.dropna(subset=['artist_id', 'song_title'])
        print(f"Removed {initial_count - len(df)} records missing critical info")
        
        df['song_title'] = df['song_title'].str.strip()
        df['song_title'] = df['song_title'].str.replace(r'\s+', ' ', regex=True)
        
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['year'] = df['year'].fillna(0)
        
        df = df.drop_duplicates(subset=['artist_id', 'song_title'])
        print(f"Final originals dataset: {len(df)} records")
        
        self.cleaned_data['originals'] = df
        return df
    
    def clean_covers_dataset(self):
        print("\nCleaning Covers Dataset...")
        
        df = self.raw_data['covers'].copy()
        initial_count = len(df)
        
        df = df.dropna(subset=['cover_artist_id', 'original_artist_id', 'song_title'])
        print(f"Removed {initial_count - len(df)} records missing critical info")
        
        df['song_title'] = df['song_title'].str.strip()
        df['song_title'] = df['song_title'].str.replace(r'\s+', ' ', regex=True)
        
        df['cover_year'] = pd.to_numeric(df['cover_year'], errors='coerce')
        df['original_year'] = pd.to_numeric(df['original_year'], errors='coerce')
        
        df['cover_year'] = df['cover_year'].fillna(0)
        df['original_year'] = df['original_year'].fillna(0)
        
        df = df.drop_duplicates(subset=['cover_artist_id', 'original_artist_id', 'song_title'])
        print(f"Final covers dataset: {len(df)} records")
        
        self.cleaned_data['covers'] = df
        return df
    
    def clean_releases_dataset(self):
        print("\nCleaning Releases Dataset...")
        
        df = self.raw_data['releases'].copy()
        initial_count = len(df)
        
        df = df.dropna(subset=['artist_id', 'release_title'])
        print(f"Removed {initial_count - len(df)} records missing critical info")
        
        df['release_title'] = df['release_title'].str.strip()
        df['release_title'] = df['release_title'].str.replace(r'\s+', ' ', regex=True)
        
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['year'] = df['year'].fillna(0)
        
        df['release_type'] = df['release_type'].fillna('Unknown')
        
        df = df.drop_duplicates(subset=['artist_id', 'release_title'])
        print(f"Final releases dataset: {len(df)} records")
        
        self.cleaned_data['releases'] = df
        return df
    
    def create_artist_lookup(self):
        print("\nCreating Artist Lookup Table...")
        
        artists_df = self.cleaned_data['artists']
        
        artist_lookup = artists_df[['artist_id', 'common_name', 'artist_type', 'home_country', 'birth_year']].copy()
        artist_lookup = artist_lookup.rename(columns={'common_name': 'name'})
        
        artist_lookup['in_degree'] = 0
        artist_lookup['out_degree'] = 0
        artist_lookup['total_degree'] = 0
        artist_lookup['influence_score'] = 0.0
        artist_lookup['diversity_score'] = 0.0
        artist_lookup['node_type'] = 'isolated'
        artist_lookup['career_stage'] = 'unknown'
        artist_lookup['activity_span'] = 0.0
        artist_lookup['age'] = 0
        
        self.cleaned_data['artist_lookup'] = artist_lookup
        return artist_lookup
    
    def create_network_edges(self):
        print("\nCreating Network Edges...")
        
        covers_df = self.cleaned_data['covers']
        artists_df = self.cleaned_data['artists']
        
        edges = []
        
        for _, cover in covers_df.iterrows():
            source_artist = cover['cover_artist_id']
            target_artist = cover['original_artist_id']
            song_title = cover['song_title']
            cover_year = cover['cover_year']
            original_year = cover['original_year']
            
            if source_artist in artists_df['artist_id'].values and target_artist in artists_df['artist_id'].values:
                time_gap = cover_year - original_year if cover_year > 0 and original_year > 0 else 0
                
                edge = {
                    'source_artist': source_artist,
                    'target_artist': target_artist,
                    'song_title': song_title,
                    'original_year': original_year,
                    'cover_year': cover_year,
                    'time_gap': time_gap,
                    'edge_weight': 1.0,
                    'edge_weight_normalized': 1.0,
                    'edge_category': 'standard',
                    'cross_decade': 1 if abs(time_gap) >= 10 else 0
                }
                edges.append(edge)
        
        edges_df = pd.DataFrame(edges)
        print(f"Created {len(edges_df)} network edges")
        
        self.cleaned_data['network_edges'] = edges_df
        return edges_df
    
    def save_cleaned_data(self):
        print("\nSaving Cleaned Data...")
        
        output_dir = Path("cleaned_data")
        output_dir.mkdir(exist_ok=True)
        
        for name, df in self.cleaned_data.items():
            output_path = output_dir / f"{name}_cleaned.csv"
            df.to_csv(output_path, index=False)
            print(f"Saved {name}: {len(df)} records to {output_path}")
    
    def generate_preprocessing_report(self):
        print("DATA PREPROCESSING AND CLEANING REPORT")
        print("="*80)
        
        self.load_raw_datasets()
        quality_report = self.assess_data_quality()
        
        self.clean_artists_dataset()
        self.clean_originals_dataset()
        self.clean_covers_dataset()
        self.clean_releases_dataset()
        
        self.create_artist_lookup()
        self.create_network_edges()
        
        self.save_cleaned_data()
        
        print("\nPreprocessing completed successfully!")
        print("Clean datasets ready for network construction")
        
        return {
            'quality_report': quality_report,
            'cleaned_datasets': list(self.cleaned_data.keys())
        }

def main():
    preprocessor = MusicDataPreprocessor()
    results = preprocessor.generate_preprocessing_report()
    print("\nData preprocessing completed successfully!")

if __name__ == "__main__":
    main()