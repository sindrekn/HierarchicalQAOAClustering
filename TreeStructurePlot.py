import h5py
import json
import argparse
import networkx as nx
import matplotlib.pyplot as plt

def load_hierarchical_result(filename: str) -> dict:
    """
    Loads the hierarchical clustering result from an .h5 file and returns a dictionary containing
    the given labels. 

    LLM assisted
    ------------
    Tool: Claude (2026)
    Created by Claude, tested by me. 
    """
    with h5py.File(filename, 'r') as f:
        best_labels     = f['best_labels'][:]
        best_modularity = float(f['best_modularity'][()])
        n_clusters      = int(f['n_clusters'][()])
        tree            = json.loads(f['tree'][()])
    return {
        'best_labels':     best_labels,
        'best_modularity': best_modularity,
        'n_clusters':      n_clusters,
        'tree':            tree,
    }

def parse_tree_from_list(node_list):
    """
    Takes a list of node names and builds a directed graph.
    Parents are automatically deduced by stripping the last '_CX' component.

    LLM assisted
    ------------
    Tool: Claude (2026)
    Created by Claude, tested by me. 
    """
    G = nx.DiGraph()
    
    for node in node_list:
        G.add_node(node)
        if node == 'root':
            continue
            
        parts = node.split('_')
        parent = "_".join(parts[:-1])
        
        if parent in G:
            G.add_edge(parent, node)
            
    return G

def calculate_tree_positions(G, node='root', pos=None, x=0, y=0, layer_width=1.0):
    """
    Recursively calculates (x, y) coordinates for a perfect tree layout.
    
    LLM assisted
    ------------
    Tool: Claude (2026)
    Created by Claude, tested by me. 
    """
    if pos is None:
        pos = {}
        
    pos[node] = (x, y)
    neighbors = list(G.neighbors(node))
    
    if len(neighbors) == 0:
        return pos

    neighbors.sort() 

    if len(neighbors) == 2:
        pos = calculate_tree_positions(G, neighbors[0], pos, x - layer_width, y - 1, layer_width * 0.5)
        pos = calculate_tree_positions(G, neighbors[1], pos, x + layer_width, y - 1, layer_width * 0.5)
    elif len(neighbors) == 1:
        direction = -0.5 if '_C0' in neighbors[0].split('_')[-1] else 0.5
        pos = calculate_tree_positions(G, neighbors[0], pos, x + direction, y - 1, layer_width * 0.5)
        
    return pos

def create_label(root_names: list, tree_res: dict) -> dict:
    custom_labels = {}
    for name in root_names: 
        if 'qaoa_results' in tree_res[name] and 'proposed_modularity' in tree_res[name]:
            node = tree_res[name]
            success_prob = 2 * max(node['qaoa_results']['probs'])
            string = f'Q={node["proposed_modularity"]:.3f}\nα={node["alpha"]:.3f}\nSub N={node["sub_n"]:.0f}\nP={success_prob:.3f}\n({"Accepted" if node["split_accepted"] else "Rejected"})'
            custom_labels[name] = string
        elif 'qaoa_results' in tree_res[name]:
            node = tree_res[name]
            success_prob = 2 * max(node['qaoa_results']['probs'])
            string = f'α={node["alpha"]:.3f}\nSub N={node["sub_n"]:.0f}\nP={success_prob:.3f}\n(Rejected:\n0 nodes in a cluster)'
            custom_labels[name] = string
        else:
            node = tree_res[name]
            string = f'α={node["alpha"]:.3f}\nSub N={node["sub_n"]:.0f}\n(Rejected)'
            custom_labels[name] = string
    return custom_labels

def parse_args():
    """
    Parses command-line arguments for the tree structure plotting script.
    """
    parser = argparse.ArgumentParser(description="Plot tree structure from hierarchical clustering results.")
    parser.add_argument("--result_path", type=str, required=True, help="Path to the hierarchical result .h5 file")
    parser.add_argument("--save_path", type=str, required=True, help="Path to the graph file")
    return parser.parse_args()

def main(): 
    """Main function to load hierarchical results, construct the tree graph, and plot it."""
    args = parse_args()

    result_hierarchical_path = args.result_path

    result = load_hierarchical_result(result_hierarchical_path)
    tree_res = result['tree']
    root_names_list = sorted(tree_res.keys(), key=lambda k: (len(k), k))

    G = parse_tree_from_list(root_names_list)
    positions = calculate_tree_positions(G)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    node_colors = ['gold' if n == 'root' else 'lightblue' for n in G.nodes()]

    custom_labels = create_label(root_names_list, tree_res)
    labels_to_show = {node: custom_labels.get(node, node) for node in root_names_list}

    nx.draw(
        G, pos=positions, 
        with_labels=False,  
        node_color=node_colors, node_size=4500, 
        arrowsize=30, edge_color='gray', ax=ax
    )

    nx.draw_networkx_labels(
        G, pos=positions, 
        labels=labels_to_show,  
        font_size=9, font_weight='bold', ax=ax
    )

    # Clean up Matplotlib borders
    ax.set_title("Bisection Structure", fontsize=12, fontweight='bold', y=1.05)
    ax.set_xmargin(0.2)
    ax.set_ymargin(0.2)
    plt.axis('off')
    plt.savefig(args.save_path, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()