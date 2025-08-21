import argparse
import sys
import time
import math
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


try:
    import pandas as pd
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Error: Required packages not installed: {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)

try:
    from pyvis.network import Network
    _PYVIS_AVAILABLE = True
except Exception:
    _PYVIS_AVAILABLE = False

try:
    from pathlib import Path
    import sys
    
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    sys.path.insert(0, '.')
    from importlib import import_module
    
    dataset_exploration = import_module('1_dataset_exploration')
    MusicDataExplorer = dataset_exploration.MusicDataExplorer
    
    data_preprocessing = import_module('2_data_preprocessing')
    MusicDataPreprocessor = data_preprocessing.MusicDataPreprocessor
    
    network_model_design = import_module('3_network_model_design')
    NetworkModelDesigner = network_model_design.NetworkModelDesigner
    
    network_construction_python = import_module('4_network_construction_python')
    MusicNetworkConstructor = network_construction_python.MusicNetworkConstructor
    
    basic_network_analysis = import_module('6_basic_network_analysis')
    BasicNetworkAnalyzer = basic_network_analysis.BasicNetworkAnalyzer
    
    connectivity_analysis = import_module('7_connectivity_analysis')
    ConnectivityAnalyzer = connectivity_analysis.ConnectivityAnalyzer
    
    network_topology_analysis = import_module('8_network_topology_analysis')
    NetworkTopologyAnalyzer = network_topology_analysis.NetworkTopologyAnalyzer
    
except ImportError:
    try:
        import importlib.util
        
        def import_from_file(module_name, file_path):
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        
        module1 = import_from_file("dataset_exploration", "1_dataset_exploration.py")
        MusicDataExplorer = module1.MusicDataExplorer
        
        module2 = import_from_file("data_preprocessing", "2_data_preprocessing.py") 
        MusicDataPreprocessor = module2.MusicDataPreprocessor
        
        module3 = import_from_file("network_model_design", "3_network_model_design.py")
        NetworkModelDesigner = module3.NetworkModelDesigner
        
        module4 = import_from_file("network_construction_python", "4_network_construction_python.py")
        MusicNetworkConstructor = module4.MusicNetworkConstructor
        
        module6 = import_from_file("basic_network_analysis", "6_basic_network_analysis.py")
        BasicNetworkAnalyzer = module6.BasicNetworkAnalyzer
        
        module7 = import_from_file("connectivity_analysis", "7_connectivity_analysis.py")
        ConnectivityAnalyzer = module7.ConnectivityAnalyzer
        
        module8 = import_from_file("network_topology_analysis", "8_network_topology_analysis.py")
        NetworkTopologyAnalyzer = module8.NetworkTopologyAnalyzer

    except Exception as e:
        print(f"Error importing modules: {e}")
        print("Make sure all workflow modules are in the same directory")
        sys.exit(1)

def log(message: str) -> None:
    print(f"[export] {message}")

def ensure_exports_dir(output_dir: Path) -> Path:
    exports_dir = output_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    return exports_dir

def load_graph_from_graphml(path: Path) -> nx.Graph | nx.DiGraph:
    log(f"Loading GraphML: {path.name}")
    G = nx.read_graphml(path)
    if not isinstance(G, (nx.DiGraph, nx.Graph)):
        G = nx.DiGraph(G)
    return G

def load_graph_from_gml(path: Path) -> nx.Graph | nx.DiGraph:
    log(f"Loading GML: {path.name}")
    try:
        G = nx.read_gml(path)
    except Exception:
        G = nx.read_gml(path, label=None)
    if not isinstance(G, (nx.DiGraph, nx.Graph)):
        G = nx.DiGraph(G)
    return G

def load_graph_from_pickle(path: Path) -> nx.Graph | nx.DiGraph:
    log(f"Loading Pickle: {path.name}")
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G

def load_graph_from_csv(nodes_csv: Path, edges_csv: Path) -> nx.DiGraph:
    log(f"Loading CSV nodes: {nodes_csv.name}")
    log(f"Loading CSV edges: {edges_csv.name}")
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    G = nx.DiGraph()

    node_id_col = "node_id" if "node_id" in nodes_df.columns else nodes_df.columns[0]
    for _, node_row in nodes_df.iterrows():
        node_id = node_row[node_id_col]
        attrs = node_row.to_dict()
        attrs.pop(node_id_col, None)
        G.add_node(node_id, **attrs)

    required_edge_cols = {"source", "target"}
    missing = required_edge_cols - set(edges_df.columns)
    if missing:
        raise ValueError(f"Edges CSV missing required columns: {missing}")

    for _, edge_row in edges_df.iterrows():
        src = edge_row["source"]
        tgt = edge_row["target"]
        attrs = edge_row.to_dict()
        attrs.pop("source", None)
        attrs.pop("target", None)
        G.add_edge(src, tgt, **attrs)

    return G

def detect_directed(G: nx.Graph | nx.DiGraph) -> bool:
    return isinstance(G, (nx.DiGraph, nx.MultiDiGraph))

def pick_largest_component_nodes(G: nx.Graph | nx.DiGraph) -> set:
    if detect_directed(G):
        components = nx.weakly_connected_components(G)
    else:
        components = nx.connected_components(G)
    largest = max(components, key=len)
    return set(largest)

def kcore_or_topdegree_induced(G: nx.Graph | nx.DiGraph, max_nodes: int) -> nx.Graph | nx.DiGraph:
    if G.number_of_nodes() <= max_nodes:
        return G.copy()

    undirected_view = G.to_undirected()

    degrees = dict(undirected_view.degree())
    max_degree = max(degrees.values()) if degrees else 0
    low, high = 1, max_degree
    best_core_nodes = None

    while low <= high:
        mid = (low + high) // 2
        core_subgraph = nx.k_core(undirected_view, k=mid)
        n = core_subgraph.number_of_nodes()
        if n == 0:
            high = mid - 1
            continue
        if n <= max_nodes:
            best_core_nodes = set(core_subgraph.nodes())
            low = mid + 1
        else:
            high = mid - 1

    if best_core_nodes and len(best_core_nodes) > 0:
        return G.subgraph(best_core_nodes).copy()

    degree_pairs = sorted(undirected_view.degree(), key=lambda kv: kv[1], reverse=True)
    selected_nodes = []
    for node, _deg in degree_pairs:
        selected_nodes.append(node)
        if len(selected_nodes) >= max_nodes:
            break
    return G.subgraph(selected_nodes).copy()

def reduce_for_preview(G: nx.Graph | nx.DiGraph, max_nodes: int) -> nx.Graph | nx.DiGraph:
    if G.number_of_nodes() <= max_nodes:
        return G.copy()

    largest_nodes = pick_largest_component_nodes(G)
    G_largest = G.subgraph(largest_nodes).copy()

    return kcore_or_topdegree_induced(G_largest, max_nodes)

def _export_html_visjs_fallback(G: nx.Graph | nx.DiGraph, out_path: Path, title: str) -> None:
    log(f"Exporting HTML (fallback) → {out_path.name}")
    directed = detect_directed(G)
    nodes_list = []
    for node, attrs in G.nodes(data=True):
        label = str(attrs.get("name", node))
        nodes_list.append({"id": str(node), "label": label})
    edges_list = []
    for u, v in G.edges():
        edge = {"from": str(u), "to": str(v)}
        if directed:
            edge["arrows"] = "to"
        edges_list.append(edge)

    nodes_json = json.dumps(nodes_list)
    edges_json = json.dumps(edges_list)

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>{title}</title>
  <script type=\"text/javascript\" src=\"https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js\"></script>
  <style>html, body, #mynetwork {{ height: 100%; margin: 0; padding: 0; }}</style>
  <style>#mynetwork {{ width: 100%; height: 95vh; border: 1px solid #ddd; }}</style>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  </head>
<body>
  <div id=\"mynetwork\"></div>
  <script type=\"text/javascript\">
    const nodes = new vis.DataSet({nodes_json});
    const edges = new vis.DataSet({edges_json});
    const container = document.getElementById('mynetwork');
    const data = {{ nodes: nodes, edges: edges }};
    const options = {{
      interaction: {{ hover: true }},
      physics: {{ stabilization: true }},
      nodes: {{ shape: 'dot', size: 6 }},
      edges: {{ color: {{ color: '#999' }}, width: 1 }}
    }};
    const network = new vis.Network(container, data, options);
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")

def export_html_pyvis(G: nx.Graph | nx.DiGraph, out_path: Path, title: str = "Network Preview") -> None:
    if not _PYVIS_AVAILABLE:
        _export_html_visjs_fallback(G, out_path, title)
        return
    log(f"Exporting HTML → {out_path.name}")
    directed = detect_directed(G)
    net = Network(height="900px", width="100%", directed=directed, notebook=False)
    net.barnes_hut(gravity=-30000, central_gravity=0.3, spring_length=110, spring_strength=0.01, damping=0.9)
    net.force_atlas_2based(gravity=-50)
    net.show_buttons(filter_=['physics'])
    net.title = title

    net.from_nx(G)
    try:
        net.write_html(str(out_path), notebook=False)
    except Exception as exc:
        log(f"pyvis write_html failed ({exc}); using fallback HTML renderer.")
        _export_html_visjs_fallback(G, out_path, title)

def export_png_matplotlib(G: nx.Graph | nx.DiGraph, out_path: Path, seed: int = 42) -> None:
    log(f"Exporting PNG → {out_path.name}")
    n = G.number_of_nodes()
    if n == 0:
        log("Graph is empty; skipping PNG export.")
        return

    k = 1.0 / math.sqrt(n)
    pos = nx.spring_layout(G.to_undirected(), k=k, seed=seed)

    degrees = dict(G.degree())
    min_size = 10
    max_size = 80
    node_sizes = [min(max(min_size, deg * 2), max_size) for deg in degrees.values()]

    fig, ax = plt.subplots(figsize=(12, 9), dpi=200)
    ax.axis('off')
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#4C78A8", alpha=0.8, linewidths=0)
    nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.4, edge_color="#999999")

    try:
        top_labels = sorted(degrees.items(), key=lambda kv: kv[1], reverse=True)[:50]
        label_nodes = [n for n, _ in top_labels]
        labels = {n: str(n) for n in label_nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=4, font_color="#222222")
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

def process_single_graph(G: nx.Graph | nx.DiGraph, exports_dir: Path, base_name: str, max_nodes_html: int, max_nodes_png: int) -> None:
    G_html = reduce_for_preview(G, max_nodes=max_nodes_html)
    export_html_pyvis(G_html, exports_dir / f"{base_name}.preview.html", title=f"{base_name} (n={G_html.number_of_nodes()}, m={G_html.number_of_edges()})")

    if G.number_of_nodes() > max_nodes_png:
        G_png = reduce_for_preview(G, max_nodes=max_nodes_png)
    else:
        G_png = G
    export_png_matplotlib(G_png, exports_dir / f"{base_name}.preview.png")

def discover_and_export(input_dir: Path, max_nodes_html: int, max_nodes_png: int) -> None:
    exports_dir = ensure_exports_dir(input_dir)

    pickle_path = input_dir / "music_cover_network.pickle"
    graphml = input_dir / "music_cover_network.graphml"
    gml = input_dir / "music_cover_network.gml"
    nodes_csv = input_dir / "music_cover_network_nodes.csv"
    edges_csv = input_dir / "music_cover_network_edges.csv"

    any_exported = False

    if pickle_path.exists():
        G = load_graph_from_pickle(pickle_path)
        base = pickle_path.name
        process_single_graph(G, exports_dir, base_name=base, max_nodes_html=max_nodes_html, max_nodes_png=max_nodes_png)
        any_exported = True

    if graphml.exists():
        G = load_graph_from_graphml(graphml)
        base = graphml.name
        process_single_graph(G, exports_dir, base_name=base, max_nodes_html=max_nodes_html, max_nodes_png=max_nodes_png)
        any_exported = True

    if gml.exists():
        G = load_graph_from_gml(gml)
        base = gml.name
        process_single_graph(G, exports_dir, base_name=base, max_nodes_html=max_nodes_html, max_nodes_png=max_nodes_png)
        any_exported = True

    if nodes_csv.exists() and edges_csv.exists():
        G = load_graph_from_csv(nodes_csv, edges_csv)
        base = "music_cover_network_csv"
        process_single_graph(G, exports_dir, base_name=base, max_nodes_html=max_nodes_html, max_nodes_png=max_nodes_png)
        any_exported = True

    if not any_exported:
        raise FileNotFoundError("No recognized network files found in the input directory.")

def export_visualizations(input_dir: str | Path = "network_output", max_nodes_html: int = 2000, max_nodes_png: int = 1000) -> None:
    dir_path = Path(input_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Input directory not found: {dir_path}")
    log(f"(API) Input dir: {dir_path}")
    log(f"(API) Max nodes HTML: {max_nodes_html}")
    log(f"(API) Max nodes PNG: {max_nodes_png}")
    discover_and_export(input_dir=dir_path, max_nodes_html=max_nodes_html, max_nodes_png=max_nodes_png)
    log("(API) Done.")

class MusicNetworkWorkflowOrchestrator:
    
    def __init__(self, data_path="MusicData/MusicData", skip_existing: bool = False):
        self.data_path = data_path
        self.skip_existing = skip_existing
        self.workflow_steps = {
            1: ("Network Model Design", self.run_network_model_design),
            2: ("Dataset Description and Exploration", self.run_dataset_exploration),
            3: ("Data Preprocessing and Cleaning", self.run_data_preprocessing),
            4: ("Network Data Selection and Filtering", self.run_network_data_filtering),
            5: ("Python Implementation", self.run_network_construction_python),
            6: ("Basic Network Properties", self.run_basic_network_analysis),
            7: ("Connectivity Analysis", self.run_connectivity_analysis),
            8: ("Network Topology Characteristics", self.run_network_topology_analysis),
            9: ("Export Visualizations", self.run_export_visualizations)
        }
        
        self.results = {}
        
    def print_banner(self):
        banner = """
        ================================================================================
                              MUSIC NETWORK ANALYSIS PROJECT                        
        ================================================================================
        
            Comprehensive Network Science Analysis of Music Cover Relationships       
        
            Workflow Steps:                                                            
            1. Network Model Design                                         
            2. Dataset Description and Exploration                     
            3. Data Preprocessing and Cleaning                                  
            4. Network Data Selection and Filtering                             
            5. Python Implementation                                          
            6. Basic Network Properties                                
            7. Connectivity Analysis                                   
            8. Network Topology Characteristics                                 
            9. Export Visualizations
        ================================================================================
        """
        print(banner)
    
    def run_network_model_design(self):
        print("\nSTEP 1: NETWORK MODEL DESIGN")
        print("="*80)
        
        try:
            if self.skip_existing and Path("network_model_design.png").exists():
                print("Skipped: network model design already generated")
                return True
            designer = NetworkModelDesigner()
            results = designer.generate_model_design_report()
            self.results['step_1'] = results
            print("Step 1 completed successfully")
            return True
        except Exception as e:
            print(f"Step 1 failed: {e}")
            return False

    def run_dataset_exploration(self):
        print("\nSTEP 2: DATASET DESCRIPTION AND EXPLORATION")
        print("="*80)
        
        try:
            if self.skip_existing and Path("music_datasets_overview.png").exists():
                print("Skipped: dataset exploration already generated")
                return True
            explorer = MusicDataExplorer(self.data_path)
            results = explorer.generate_comprehensive_report()
            self.results['step_2'] = results
            print("Step 2 completed successfully")
            return True
        except Exception as e:
            print(f"Step 2 failed: {e}")
            return False
    
    def run_data_preprocessing(self):
        print("\nSTEP 3: DATA PREPROCESSING AND CLEANING")
        print("="*80)
        
        try:
            if self.skip_existing and Path("cleaned_data").exists():
                print("Skipped: cleaned_data already present")
                return True
            preprocessor = MusicDataPreprocessor(self.data_path)
            results = preprocessor.generate_preprocessing_report()
            self.results['step_3'] = results
            print("Step 3 completed successfully")
            return True
        except Exception as e:
            print(f"Step 3 failed: {e}")
            return False
    
    def run_network_data_filtering(self):
        print("\nSTEP 4: NETWORK DATA SELECTION AND FILTERING")
        print("="*80)
        
        try:
            if self.skip_existing and Path("cleaned_data").exists():
                print("Skipped: network data filtering completed with preprocessing")
                return True
            print("Network data selection and filtering completed as part of preprocessing")
            self.results['step_4'] = {"status": "completed", "integrated_with": "preprocessing"}
            print("Step 4 completed successfully")
            return True
        except Exception as e:
            print(f"Step 4 failed: {e}")
            return False
    
    def run_network_construction_python(self):
        print("\nSTEP 5: PYTHON IMPLEMENTATION")
        print("="*80)
        
        try:
            expected = [
                Path("network_output/music_cover_network.pickle"),
                Path("network_output/music_cover_network.graphml"),
                Path("network_output/music_cover_network.gml"),
                Path("network_output/music_cover_network_nodes.csv"),
                Path("network_output/music_cover_network_edges.csv"),
            ]
            if self.skip_existing and any(p.exists() for p in expected):
                print("Skipped: network files already exist in network_output/")
                return True
            constructor = MusicNetworkConstructor()
            results = constructor.generate_construction_report()
            self.results['step_5'] = results
            print("Step 5 completed successfully")
            return True
        except Exception as e:
            print(f"Step 5 failed: {e}")
            return False
    
    def run_basic_network_analysis(self):
        print("\nSTEP 6: BASIC NETWORK PROPERTIES")
        print("="*80)
        
        try:
            expected = [Path("basic_network_analysis.png"), Path("basic_network_analysis.json")]
            if self.skip_existing and all(p.exists() for p in expected):
                print("Skipped: basic network analysis artifacts already present")
                return True
            analyzer = BasicNetworkAnalyzer()
            results = analyzer.generate_basic_analysis_report()
            self.results['step_6'] = results
            print("Step 6 completed successfully")
            return True
        except Exception as e:
            print(f"Step 6 failed: {e}")
            return False
    
    def run_connectivity_analysis(self):
        print("\nSTEP 7: CONNECTIVITY ANALYSIS")
        print("="*80)
        
        try:
            expected = [Path("connectivity_analysis.png"), Path("connectivity_analysis.json")]
            if self.skip_existing and all(p.exists() for p in expected):
                print("Skipped: connectivity analysis artifacts already present")
                return True
            analyzer = ConnectivityAnalyzer()
            results = analyzer.generate_connectivity_analysis_report()
            self.results['step_7'] = results
            print("Step 7 completed successfully")
            return True
        except Exception as e:
            print(f"Step 7 failed: {e}")
            return False
    
    def run_network_topology_analysis(self):
        print("\nSTEP 8: NETWORK TOPOLOGY CHARACTERISTICS")
        print("="*80)
        
        try:
            expected = [Path("network_topology_analysis.png"), Path("topology_analysis.json")]
            if self.skip_existing and all(p.exists() for p in expected):
                print("Skipped: topology analysis artifacts already present")
                return True
            analyzer = NetworkTopologyAnalyzer()
            results = analyzer.generate_topology_analysis_report()
            self.results['step_8'] = results
            print("Step 8 completed successfully")
            return True
        except Exception as e:
            print(f"Step 8 failed: {e}")
            return False

    def run_export_visualizations(self):
        print("\nSTEP 9: EXPORT VISUALIZATIONS")
        print("="*80)
        
        try:
            export_visualizations("network_output", max_nodes_html=2000, max_nodes_png=1000)
            self.results['step_9'] = {"status": "completed"}
            print("Step 9 completed successfully")
            return True
        except Exception as e:
            print(f"Step 9 failed: {e}")
            return False

    def run_all_steps(self):
        self.print_banner()
        
        start_time = time.time()
        successful_steps = 0
        
        print(f"\nStarting complete music network analysis workflow")
        print(f"Data path: {self.data_path}")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        for step_num, (step_name, step_function) in self.workflow_steps.items():
            print(f"\nExecuting Step {step_num}: {step_name}")
            step_start = time.time()
            
            success = step_function()
            step_duration = time.time() - step_start
            
            if success:
                successful_steps += 1
                print(f"Step {step_num} completed in {step_duration:.1f} seconds")
            else:
                print(f"Step {step_num} failed after {step_duration:.1f} seconds")
                print("Workflow stopped due to failure")
                break
        
        total_duration = time.time() - start_time
        
        print("\nWORKFLOW COMPLETION SUMMARY")
        print("="*80)
        print(f"Successful steps: {successful_steps}/{len(self.workflow_steps)}")
        print(f"Total duration: {total_duration/60:.1f} minutes")
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if successful_steps == len(self.workflow_steps):
            print(f"\nCOMPLETE SUCCESS: All workflow steps completed successfully")
            print("Your music network analysis is ready for interpretation")
            self.print_output_summary()
        else:
            print(f"\nPartial completion: {successful_steps}/{len(self.workflow_steps)} steps successful")
        
        return successful_steps == len(self.workflow_steps)
    
    def run_selected_steps(self, step_numbers):
        self.print_banner()
        
        print(f"\nRunning selected steps: {step_numbers}")
        
        for step_num in step_numbers:
            if step_num in self.workflow_steps:
                step_name, step_function = self.workflow_steps[step_num]
                print(f"\nExecuting Step {step_num}: {step_name}")
                
                success = step_function()
                if not success:
                    print(f"Step {step_num} failed - stopping execution")
                    return False
            else:
                print(f"Warning: Step {step_num} not found - skipping")
        
        print(f"\nSelected steps completed successfully")
        return True
    
    def run_interactive_mode(self):
        self.print_banner()
        
        print(f"\nInteractive Mode Activated")
        print("Select which steps to run:\n")
        
        for step_num, (step_name, _) in self.workflow_steps.items():
            print(f"  {step_num}. {step_name}")
        
        while True:
            try:
                user_input = input("\nEnter step numbers (comma-separated) or 'all' for complete workflow: ").strip()
                
                if user_input.lower() == 'all':
                    return self.run_all_steps()
                elif user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye")
                    return False
                else:
                    step_numbers = [int(x.strip()) for x in user_input.split(',')]
                    return self.run_selected_steps(step_numbers)
                    
            except ValueError:
                print("Error: Invalid input. Please enter numbers separated by commas.")
            except KeyboardInterrupt:
                print("\nGoodbye")
                return False
    
    def print_output_summary(self):
        print(f"\nGENERATED OUTPUT FILES:")
        print("-" * 40)
        
        expected_outputs = [
            ("network_model_design.png", "Network Model Design visualizations"),
            ("network_model_specification.txt", "Network Model specification"),
            ("music_datasets_overview.png", "Dataset exploration visualizations"),
            ("cleaned_data/", "Data preprocessing and cleaning"),
            ("network_output/", "Python network implementation"),
            ("basic_network_analysis.png", "Basic network properties"),
            ("basic_network_analysis.json", "Network statistics"),
            ("connectivity_analysis.png", "Connectivity analysis"),
            ("connectivity_analysis.json", "Connectivity statistics"),
            ("network_topology_analysis.png", "Network topology characteristics"),
            ("topology_analysis.json", "Topology analysis results"),
            ("network_output/exports/", "Interactive visualizations")
        ]
        
        for filename, description in expected_outputs:
            if Path(filename).exists():
                print(f"  {filename} - {description}")
            else:
                print(f"  {filename} - {description} (not found)")

def main():
    parser = argparse.ArgumentParser(description="Music Network Analysis Workflow Orchestrator")
    parser.add_argument("--run-all", action="store_true", help="Run complete workflow")
    parser.add_argument("--steps", type=str, help="Run specific steps (comma-separated, e.g., '1,2,3')")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--data-path", type=str, default="MusicData/MusicData", 
                       help="Path to music data directory")
    parser.add_argument("--skip-existing", action="store_true", help="Skip steps whose outputs already exist")
    
    args = parser.parse_args()

    no_flags = not args.run_all and not args.steps and not args.interactive
    effective_skip = True if no_flags else args.skip_existing
    orchestrator = MusicNetworkWorkflowOrchestrator(args.data_path, skip_existing=effective_skip)
    
    if args.run_all:
        success = orchestrator.run_all_steps()
        sys.exit(0 if success else 1)
    
    elif args.steps:
        try:
            step_numbers = [int(x.strip()) for x in args.steps.split(',')]
            success = orchestrator.run_selected_steps(step_numbers)
            sys.exit(0 if success else 1)
        except ValueError:
            print("Error: Invalid step numbers. Please provide comma-separated integers.")
            sys.exit(1)
    
    elif args.interactive:
        success = orchestrator.run_interactive_mode()
        sys.exit(0 if success else 1)
    
    else:
        print("Music Network Analysis Project")
        print("Running complete workflow automatically (skip-existing by default)...")
        print("(Use --help to see command line options)")
        print()
        
        success = orchestrator.run_all_steps()
        
        if success:
            print("\nWorkflow completed successfully!")
            sys.exit(0)
        else:
            print("\nWorkflow failed!")
            sys.exit(1)

if __name__ == "__main__":
    main()