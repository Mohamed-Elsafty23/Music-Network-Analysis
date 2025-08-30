# Music Network Analysis

A comprehensive network analysis project exploring musical influence patterns through cover song relationships. This project analyzes how musical artists influence each other through cover performances, creating a directed network where edges represent artistic influence transmission.

## 📊 Project Overview

This project implements a complete pipeline for analyzing music cover networks, from data exploration to advanced network analysis and visualization. The analysis focuses on understanding cultural diffusion patterns and artistic influence in the music industry through the lens of network science.

### Key Features

- **Comprehensive Data Pipeline**: Automated workflow from raw data to network insights
- **Network Model Design**: Theoretically grounded approach based on cultural diffusion theory
- **Advanced Analytics**: Multiple centrality measures, community detection, and topology analysis
- **Interactive Visualizations**: Dynamic network visualizations and statistical plots
- **Scalable Architecture**: Modular design supporting various network analysis workflows

## 🎵 Dataset

The project analyzes music cover relationships using four main datasets:

- **Artists**: Musician profiles with metadata (country, birth year, career span)
- **Originals**: Original song compositions and their creators
- **Covers**: Cover song performances and relationships
- **Releases**: Album and release information

## 🏗️ Project Structure

```
Music-Network-Analysis/
├── 0_main_workflow.py              # Main execution pipeline
├── 1_dataset_exploration.py        # Data exploration and profiling
├── 2_data_preprocessing.py         # Data cleaning and preparation
├── 3_network_model_design.py       # Network model specification
├── 4_network_construction_python.py # Network building algorithms
├── 6_basic_network_analysis.py     # Fundamental network metrics
├── 7_connectivity_analysis.py      # Network connectivity analysis
├── 8_network_topology_analysis.py  # Advanced topology metrics
├── 9_visualize_graph_excerpt.py    # Network visualization
├── 10_generate_visualizations.py   # Statistical visualizations
├── 11_centralities.py             # Centrality measure calculations
├── 12_centrality_visualizations.py # Centrality analysis plots
├── requirements.txt                # Python dependencies
├── network_model_specification.txt # Detailed network model documentation
├── MusicData/                     # Raw dataset files
├── cleaned_data/                  # Processed data outputs
├── network_output/               # Network files and exports
└── lib/                         # Web visualization libraries
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required packages (see installation below)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Mohamed-Elsafty23/Music-Network-Analysis.git
cd Music-Network-Analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Analysis

Execute the complete analysis pipeline:

```bash
python 0_main_workflow.py
```

Or run individual analysis components:

```bash
# Data exploration
python 1_dataset_exploration.py

# Network construction
python 4_network_construction_python.py

# Basic network analysis
python 6_basic_network_analysis.py

# Visualization
python 10_generate_visualizations.py
```

## 📈 Analysis Components

### 1. Data Exploration (`1_dataset_exploration.py`)
- Dataset profiling and statistics
- Data quality assessment
- Initial pattern discovery

### 2. Data Preprocessing (`2_data_preprocessing.py`)
- Data cleaning and standardization
- Missing value handling
- Feature engineering

### 3. Network Model Design (`3_network_model_design.py`)
- Theoretical network model specification
- Node and edge definition
- Attribute design

### 4. Network Construction (`4_network_construction_python.py`)
- Graph building from cover relationships
- Network export in multiple formats (GraphML, GML, Pickle)
- Node and edge attribute assignment

### 5. Network Analysis

#### Basic Analysis (`6_basic_network_analysis.py`)
- Fundamental network metrics
- Degree distributions
- Basic structural properties

#### Connectivity Analysis (`7_connectivity_analysis.py`)
- Component analysis
- Connectivity patterns
- Network robustness

#### Topology Analysis (`8_network_topology_analysis.py`)
- Advanced structural metrics
- Clustering coefficients
- Path length distributions

#### Centrality Analysis (`11_centralities.py`)
- Multiple centrality measures
- Influence ranking
- Centrality correlations

### 6. Visualization

#### Network Visualization (`9_visualize_graph_excerpt.py`)
- Interactive network plots
- Subgraph extraction
- Layout optimization

#### Statistical Visualization (`10_generate_visualizations.py`)
- Distribution plots
- Correlation analysis
- Comparative visualizations

#### Centrality Visualization (`12_centrality_visualizations.py`)
- Centrality heatmaps
- Ranking visualizations
- Scatter plot analysis

## 🔬 Network Model

The project implements a **Music Cover Influence Network** with the following specifications:

### Nodes (Artists)
- **Entity Type**: Musical Artists/Musicians
- **Identifier**: Unique artist ID
- **Attributes**: Name, type, country, birth year, career span, genres

### Edges (Cover Relationships)
- **Type**: Directed edges (Original Artist → Covering Artist)
- **Interpretation**: Musical influence and cultural transmission
- **Attributes**: Song information, temporal data, performance metrics

### Theoretical Foundation
Based on **Cultural Diffusion and Artistic Influence Theory**, following network science principles from Brandes et al.

## 📊 Key Metrics and Analysis

### Network-Level Metrics
- **Size**: Number of nodes and edges
- **Density**: Network connectivity ratio
- **Components**: Connected component analysis
- **Clustering**: Local and global clustering coefficients

### Node-Level Metrics
- **Degree Centrality**: Direct influence connections
- **Betweenness Centrality**: Bridge roles in cultural transmission
- **Closeness Centrality**: Accessibility in influence network
- **PageRank**: Overall influence ranking
- **Eigenvector Centrality**: Influence from influential neighbors

### Specialized Analysis
- **Community Detection**: Musical genre and style communities
- **Temporal Analysis**: Evolution of influence patterns
- **Geographic Analysis**: Cultural diffusion across countries

## 📁 Output Files

The analysis generates various outputs:

### Network Files
- `music_cover_network.graphml` - Standard graph format
- `music_cover_network.gml` - Graph Modeling Language
- `music_cover_network.pickle` - Python serialized network
- `music_cover_network_nodes.csv` - Node data
- `music_cover_network_edges.csv` - Edge data

### Analysis Results
- `basic_network_analysis.json` - Fundamental metrics
- `connectivity_analysis.json` - Connectivity statistics
- `topology_analysis.json` - Topology measures
- `network_statistics.json` - Comprehensive statistics

### Visualizations
- Network plots (PNG format)
- Statistical charts and distributions
- Centrality analysis plots
- Interactive HTML visualizations

## 🛠️ Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **networkx**: Network analysis and algorithms
- **matplotlib**: Static plotting
- **seaborn**: Statistical visualization

### Specialized Libraries
- **pyvis**: Interactive network visualization
- **scipy**: Scientific computing
- **python-louvain**: Community detection
- **scikit-learn**: Machine learning utilities
- **tqdm**: Progress bars

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request

## 📚 Academic Context

This project was developed as part of network analysis coursework at Leuphana University, focusing on practical applications of network science theory to cultural and artistic domains.

### References
- Brandes, U., et al. "What is network science?" Network Science principles
- Cultural diffusion theory in network analysis
- Music information retrieval and network analysis

## 📄 License

This project is available for academic and research purposes. Please cite appropriately if used in academic work.

## 📧 Contact

**Author**: Mohamed Elsafty  
**Institution**: Leuphana University  
**Course**: Analyzing Networks (Semester 2)

For questions or collaboration opportunities, please open an issue on GitHub.

---

*This project demonstrates the application of network science principles to understand cultural influence patterns in the music industry through systematic analysis of cover song relationships.*
