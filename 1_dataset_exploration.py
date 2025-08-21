import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MusicDataExplorer:
    
    def __init__(self, data_path="MusicData/MusicData"):
        self.data_path = Path(data_path)
        self.artists_df = None
        self.originals_df = None
        self.covers_df = None
        self.releases_df = None
        
    def load_datasets(self):
        print("Loading Music Datasets...")
        
        artists_path = self.data_path / "artists.csv" / "neo4j_artists.csv"
        self.artists_df = pd.read_csv(artists_path)
        print(f"Artists dataset: {len(self.artists_df)} records")
        
        originals_path = self.data_path / "originals.csv" / "originals.csv"
        self.originals_df = pd.read_csv(originals_path)
        print(f"Originals dataset: {len(self.originals_df)} records")
        
        covers_path = self.data_path / "covers.csv" / "covers.csv" 
        self.covers_df = pd.read_csv(covers_path)
        print(f"Covers dataset: {len(self.covers_df)} records")
        
        releases_path = self.data_path / "releases.csv" / "releases.csv"
        self.releases_df = pd.read_csv(releases_path)
        print(f"Releases dataset: {len(self.releases_df)} records")
        
    def explore_artists_dataset(self):
        print("\nARTISTS DATASET CHARACTERISTICS")
        print("="*60)
        
        if self.artists_df is None:
            print("Error: Artists dataset not loaded")
            return None
            
        print(f"Dataset Shape: {self.artists_df.shape}")
        print(f"Columns: {list(self.artists_df.columns)}")
        
        missing_stats = self.artists_df.isnull().sum()
        missing_pct = (missing_stats / len(self.artists_df)) * 100
        print("\nMissing Values Analysis:")
        for col, pct in missing_pct.items():
            if pct > 0:
                print(f"  {col}: {pct:.1f}%")
        
        print("\nArtist Types Distribution:")
        print(self.artists_df['artist_type'].value_counts())
        
        print("\nTop 10 Countries:")
        print(self.artists_df['home_country'].value_counts().head(10))
        
        birth_years = self.artists_df['birth_year']
        valid_birth_years = birth_years[birth_years > 0]
        if len(valid_birth_years) > 0:
            print(f"\nBirth Year Statistics:")
            print(f"  Range: {valid_birth_years.min()} - {valid_birth_years.max()}")
            print(f"  Mean: {valid_birth_years.mean():.1f}")
            print(f"  Median: {valid_birth_years.median()}")
        
        rs_ranked = len(self.artists_df[self.artists_df['RS_Ranking'] > 0])
        allmusic_ranked = len(self.artists_df[self.artists_df['allmusic_Ranking'] > 0])
        print(f"\nRankings Analysis:")
        print(f"  Rolling Stone Rankings: {rs_ranked} artists")
        print(f"  AllMusic Rankings: {allmusic_ranked} artists")
        
        return self.artists_df.describe()
    
    def explore_originals_dataset(self):
        print("\nORIGINALS DATASET CHARACTERISTICS") 
        print("="*60)
        
        if self.originals_df is None:
            print("Error: Originals dataset not loaded")
            return None
            
        print(f"Dataset Shape: {self.originals_df.shape}")
        print(f"Columns: {list(self.originals_df.columns)}")
        
        missing_stats = self.originals_df.isnull().sum()
        missing_pct = (missing_stats / len(self.originals_df)) * 100
        print("\nMissing Values Analysis:")
        for col, pct in missing_pct.items():
            if pct > 0:
                print(f"  {col}: {pct:.1f}%")
        
        print("\nTop 10 Artists by Number of Originals:")
        artist_counts = self.originals_df['artist_id'].value_counts().head(10)
        print(artist_counts)
        
        print("\nYear Distribution:")
        year_counts = self.originals_df['year'].value_counts().sort_index()
        print(f"  Range: {year_counts.index.min()} - {year_counts.index.max()}")
        print(f"  Most productive year: {year_counts.idxmax()} ({year_counts.max()} songs)")
        
        return self.originals_df.describe()
    
    def explore_covers_dataset(self):
        print("\nCOVERS DATASET CHARACTERISTICS")
        print("="*60)
        
        if self.covers_df is None:
            print("Error: Covers dataset not loaded")
            return None
            
        print(f"Dataset Shape: {self.covers_df.shape}")
        print(f"Columns: {list(self.covers_df.columns)}")
        
        missing_stats = self.covers_df.isnull().sum()
        missing_pct = (missing_stats / len(self.covers_df)) * 100
        print("\nMissing Values Analysis:")
        for col, pct in missing_pct.items():
            if pct > 0:
                print(f"  {col}: {pct:.1f}%")
        
        print("\nTop 10 Cover Artists:")
        cover_artist_counts = self.covers_df['cover_artist_id'].value_counts().head(10)
        print(cover_artist_counts)
        
        print("\nTop 10 Original Artists (Most Covered):")
        original_artist_counts = self.covers_df['original_artist_id'].value_counts().head(10)
        print(original_artist_counts)
        
        print("\nYear Distribution:")
        year_counts = self.covers_df['cover_year'].value_counts().sort_index()
        print(f"  Range: {year_counts.index.min()} - {year_counts.index.max()}")
        print(f"  Most active cover year: {year_counts.idxmax()} ({year_counts.max()} covers)")
        
        time_gaps = self.covers_df['cover_year'] - self.covers_df['original_year']
        print(f"\nTime Gap Analysis:")
        print(f"  Average gap: {time_gaps.mean():.1f} years")
        print(f"  Median gap: {time_gaps.median():.1f} years")
        print(f"  Max gap: {time_gaps.max()} years")
        
        return self.covers_df.describe()
    
    def explore_releases_dataset(self):
        print("\nRELEASES DATASET CHARACTERISTICS")
        print("="*60)
        
        if self.releases_df is None:
            print("Error: Releases dataset not loaded")
            return None
            
        print(f"Dataset Shape: {self.releases_df.shape}")
        print(f"Columns: {list(self.releases_df.columns)}")
        
        missing_stats = self.releases_df.isnull().sum()
        missing_pct = (missing_stats / len(self.releases_df)) * 100
        print("\nMissing Values Analysis:")
        for col, pct in missing_pct.items():
            if pct > 0:
                print(f"  {col}: {pct:.1f}%")
        
        print("\nRelease Types Distribution:")
        print(self.releases_df['release_type'].value_counts())
        
        print("\nTop 10 Artists by Number of Releases:")
        artist_counts = self.releases_df['artist_id'].value_counts().head(10)
        print(artist_counts)
        
        print("\nYear Distribution:")
        year_counts = self.releases_df['year'].value_counts().sort_index()
        print(f"  Range: {year_counts.index.min()} - {year_counts.index.max()}")
        print(f"  Most productive year: {year_counts.idxmax()} ({year_counts.max()} releases)")
        
        return self.releases_df.describe()
    
    def create_dataset_overview_visualization(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Music Datasets Overview', fontsize=16, fontweight='bold')
        
        if self.artists_df is not None:
            axes[0, 0].hist(self.artists_df['birth_year'].dropna(), bins=30, alpha=0.7, color='skyblue')
            axes[0, 0].set_title('Artist Birth Years Distribution')
            axes[0, 0].set_xlabel('Birth Year')
            axes[0, 0].set_ylabel('Count')
        
        if self.originals_df is not None:
            year_counts = self.originals_df['year'].value_counts().sort_index()
            axes[0, 1].plot(year_counts.index, year_counts.values, color='green', alpha=0.7)
            axes[0, 1].set_title('Originals by Year')
            axes[0, 1].set_xlabel('Year')
            axes[0, 1].set_ylabel('Number of Songs')
        
        if self.covers_df is not None:
            year_counts = self.covers_df['cover_year'].value_counts().sort_index()
            axes[1, 0].plot(year_counts.index, year_counts.values, color='orange', alpha=0.7)
            axes[1, 0].set_title('Covers by Year')
            axes[1, 0].set_xlabel('Year')
            axes[1, 0].set_ylabel('Number of Covers')
        
        if self.releases_df is not None:
            year_counts = self.releases_df['year'].value_counts().sort_index()
            axes[1, 1].plot(year_counts.index, year_counts.values, color='red', alpha=0.7)
            axes[1, 1].set_title('Releases by Year')
            axes[1, 1].set_xlabel('Year')
            axes[1, 1].set_ylabel('Number of Releases')
        
        plt.tight_layout()
        plt.savefig('music_datasets_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_comprehensive_report(self):
        print("MUSIC DATASETS COMPREHENSIVE EXPLORATION REPORT")
        print("="*80)
        
        self.load_datasets()
        
        artists_stats = self.explore_artists_dataset()
        originals_stats = self.explore_originals_dataset()
        covers_stats = self.explore_covers_dataset()
        releases_stats = self.explore_releases_dataset()
        
        self.create_dataset_overview_visualization()
        
        return {
            'artists_stats': artists_stats,
            'originals_stats': originals_stats,
            'covers_stats': covers_stats,
            'releases_stats': releases_stats
        }

def main():
    explorer = MusicDataExplorer()
    results = explorer.generate_comprehensive_report()
    print("\nDataset exploration completed successfully!")

if __name__ == "__main__":
    main()