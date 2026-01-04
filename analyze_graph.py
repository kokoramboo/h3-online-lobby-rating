import csv
from pathlib import Path
import networkx as nx

# Configuration
MATCHES_FILE = Path("data/matches_jc_filtered.csv")
K_CORE = 5

def main():
    if not MATCHES_FILE.exists():
        print(f"[!] {MATCHES_FILE} not found.")
        return

    print(f"[*] Loading matches from {MATCHES_FILE}...")
    G = nx.Graph()
    with open(MATCHES_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            p1 = row['p1_id']
            p2 = row['p2_id']
            G.add_edge(p1, p2)

    print(f"[*] Initial Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Extract K-Core
    print(f"[*] Extracting {K_CORE}-core...")
    G.remove_edges_from(nx.selfloop_edges(G))
    core = nx.k_core(G, k=K_CORE)
    print(f"[*] {K_CORE}-Core Graph: {core.number_of_nodes()} nodes, {core.number_of_edges()} edges")

    # Connected Components
    components = list(nx.connected_components(core))
    print(f"[*] Number of connected components: {len(components)}")
    if components:
        main_component_size = len(max(components, key=len))
        print(f"[*] Main component size: {main_component_size} nodes")

    # Find Articulation Points (Conjugation Points)
    print(f"[*] Finding articulation points (conjugation points)...")
    aps = list(nx.articulation_points(core))
    print(f"[*] Found {len(aps)} articulation points.")

    if not aps:
        print("[*] No single conjugation points found. The 5-core graph is highly redundant.")
    else:
        # Load player names for reporting
        player_names = {}
        with open(MATCHES_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                player_names[row['p1_id']] = row['p1_name']
                player_names[row['p2_id']] = row['p2_name']

        print("\n=== Top Articulation Points (by degree) ===")
        # Sort by degree in the core graph to find the most "important" bridges
        aps_with_degree = [(ap, core.degree(ap)) for ap in aps]
        aps_with_degree.sort(key=lambda x: x[1], reverse=True)

        for i, (ap, deg) in enumerate(aps_with_degree[:20]):
            name = player_names.get(ap, "Unknown")
            print(f"{i+1:2}. {name:20} (ID: {ap:8}) | Degree: {deg}")

if __name__ == "__main__":
    main()
