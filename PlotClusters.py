import h5py
import json
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =============================================================================
# Create the graph from the CSV file
# ==============================================================================
def add_edges(graph: nx.Graph, edges: list, weights: list):
    for edge, weight in zip(edges, weights):
        graph.add_edge(edge[0], edge[1], weight=weight)

def construct_graph_from_csv(graph_path: str) -> np.ndarray:
    network_graph = nx.Graph()
    graph = pd.read_csv(graph_path)
    edges   = [(row['Node1'], row['Node2']) for _, row in graph.iterrows()]
    weights = [row['Weight'] for _, row in graph.iterrows()]

    add_edges(network_graph, edges, weights)

    nodelist = sorted(network_graph.nodes())
    A = np.array(nx.adjacency_matrix(network_graph, nodelist=nodelist, weight='weight').todense())
    return A, network_graph

# =============================================================================
# Loading Results
# =============================================================================
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

def load_analyse_results(filename: str) -> tuple:
    """
    Loads the QAOA analysis results from an .h5 file and returns two dictionaries: 
    one for the gamma-beta sweep results and one for the p-sweep results.

    LLM assisted
    ------------
    Tool: Claude (2026)
    Created by Claude, tested by me. 
    """
    gamm_beta_result = {'gammas': [], 'betas': [], 'expected_values': [], 'state_probs': []}
    p_result = {'p': [], 'gamma_opt': [], 'beta_opt': [], 'state probabilities': [], 'expected values': []}

    with h5py.File(filename, 'r') as f:

        # --- gamma_beta: flat arrays ---
        gb = f['gamma_beta']
        gamm_beta_result = {
            'gamma':           gb['gamma'][:],
            'beta':            gb['beta'][:],
            'expected_values': gb['expected_values'][:],
            'state_probs':     gb['state_probs'][:],
        }

        # --- p_sweep ---
        pr = f['p_sweep']
        p_result['p']               = pr['p'][:].tolist()
        p_result['expected values'] = pr['expected_values'][:].tolist()
        for key in sorted(pr.keys()):
            if key.startswith('p_'):
                layer = pr[key]
                p_result['gamma_opt'].append(layer['gamma_opt'][:])
                p_result['beta_opt'].append(layer['beta_opt'][:])
                p_result['state probabilities'].append(layer['state_probs'][:])

    return gamm_beta_result, p_result

def plot_optimal_configurations(result_path: str, graph_name: str, A: np.ndarray, network_graph: nx.Graph):    
    """
    Plot the optimal clustering configuration from the hierarchical clustering result.

    LLM assisted
    ------------
    Tool: Claude (2026)
    Created by Claude, tested by me.
    """
    res = load_hierarchical_result(result_path)
    
    # Plot the resulting partition
    G = nx.from_numpy_array(A)
    print(f"Graph has {network_graph.number_of_nodes()} nodes and {network_graph.number_of_edges()} edges.")

    labels = res["best_labels"]
    unique_labels = np.unique(labels)
    color_map = plt.get_cmap('tab10')
    colors = [color_map(label) for label in labels]

    plt.figure(figsize=(8, 6))
    nx.draw_networkx(G, with_labels=True, node_color=colors, node_size=500)
    # plt.title(f"Hierarchical QAOA Clustering (Q={res['best_modularity']:.4f}, k={res['n_clusters']})")
    plt.axis('off')
    plt.savefig(f"/home/sindrekampennesheim/Documents/PhD/FYS5419/Project_ClusteringQAOA/Plots/OptimalConfigurations/{graph_name}_OptimalConfig.pdf", bbox_inches='tight')
    plt.show()

def plot_probability_distribution_at_diff_p(
    p_result: dict,
    top_n: int = 10,
):
    """
    Plot the probability distribution of the top_n most probable basis states
    for each QAOA circuit depth p in p_result.

    The Z_2 symmetry means the optimal partition always appears as two
    degenerate bitstrings (complements of each other) — their combined
    probability is shown in the title.

    LLM assisted
    ------------
    Tool: Claude (2026)
    Created by Claude, tested by me.
    """
    p_vals   = p_result['p']
    p_num    = len(p_vals)
    n_qubits = int(np.log2(len(p_result['state probabilities'][0])))

    # Build subplot grid
    if p_num <= 2:
        fig, axes = plt.subplots(1, p_num, figsize=(5 * p_num, 5))
        axes = np.atleast_1d(axes)  # ensure iterable when p_num == 1
    else:
        ncols = 2
        nrows = (p_num + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows))
        axes = axes.flatten()

    for i, p_val in enumerate(p_vals):
        # p_result['state probabilities'][i] is the state prob vector for depth p_val
        # (already the best over restarts from qaoa_analyse_p)
        state_probs_p = p_result['state probabilities'][i]
        expected_val  = p_result['expected values'][i]

        # Top_n most probable states, sorted ascending then reverse for bar plot
        top_indices    = np.argsort(state_probs_p)[-top_n:][::-1]
        top_probs      = state_probs_p[top_indices]
        top_bitstrings = [format(idx, f'0{n_qubits}b') for idx in top_indices]

        # Combined probability of the two degenerate Z_2-symmetric ground states
        total_top_prob = np.sum(top_probs[:2])

        ax = axes[i]
        ax.bar(range(top_n), top_probs)
        ax.set_xticks(range(top_n))
        ax.set_xticklabels(top_bitstrings, rotation=30, ha='right', fontsize=11)
        
        if i % 2 == 0:
            ax.set_ylabel('Probability', fontsize=14)

        legend_label = f'p={p_val}\n$\langle H_c \\rangle$={expected_val:.4f}\nP={total_top_prob:.3f}'        
        blank_handle = Line2D([0], [0], linestyle='none', marker='none', label=legend_label)
        ax.legend(
            handles=[blank_handle], 
            loc='upper right', 
            handlelength=0, 
            handletextpad=0,
            frameon=True,
            fontsize=14
        )


    # Hide any unused subplots
    for j in range(p_num, len(axes)):
        axes[j].set_visible(False)

    # Scale all the plots to have the same y-axis limit for better comparison
    max_prob = max(np.max(p_result['state probabilities'][i]) for i in range(p_num))
    for ax in axes[:p_num]:
        ax.set_ylim(0, max_prob * 1.1)

    # fig.tight_layout()
    fig.subplots_adjust(
    left=0.08,    # Space on the left edge of the figure
    right=0.99,   # Space on the right edge of the figure
    top=0.97,     # Space on the top edge of the figure
    bottom=0.08,  # Space on the bottom edge of the figure
    wspace=0.3,  # Width spacing BETWEEN the subplot boxes
    hspace=0.3   # Height spacing BETWEEN the subplot boxes (great for rotated labels)
)
    plt.show()

def plot_gamma_beta_heatmaps(gamm_beta_result: dict):
    """
    Plot heatmaps of expected values over the gamma-beta grid.
    
    LLM assisted
    ------------
    Tool: Claude (2026)
    Created by Claude, tested by me.
    """
    gammas = np.array(gamm_beta_result['gamma'])
    betas  = np.array(gamm_beta_result['beta'])

    n_gamma = len(np.unique(gammas))
    n_beta  = len(np.unique(betas))

    # Shape: (n_gamma, n_beta) — then transpose so rows=beta, cols=gamma
    EV = np.array(gamm_beta_result['expected_values']).reshape(n_gamma, n_beta).T

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(
        EV,
        origin='lower',
        aspect='auto',
        extent=[gammas.min()/(np.pi), gammas.max()/(np.pi), betas.min()/(np.pi), betas.max()/(np.pi)],
    )
    ax.set_xlabel(r'$\gamma / \pi$')
    ax.set_ylabel(r'$\beta / \pi$')
    # ax.set_title(r'Expected value $\langle H_C \rangle$ over $(\gamma, \beta)$ grid')
    fig.colorbar(im, ax=ax, label=r'$\langle H_C \rangle$')
    plt.tight_layout()
    plt.savefig(f"/home/sindrekampennesheim/Documents/PhD/FYS5419/Project_ClusteringQAOA/Plots/GammaBetaHeatPlots/Gamma_beta_{graph_name}_bigRange.pdf", bbox_inches='tight')
    plt.show()


def parse_args(): 
    """Parse command-line arguments for plotting."""
    parser = argparse.ArgumentParser(description="Plot QAOA clustering results.")

    # --- Mode selection ---
    parser.add_argument('--heatmap', action='store_true', help='Plot gamma-beta heatmaps.')
    parser.add_argument('--optimal-config', action='store_true', help='Plot optimal clustering configuration.')
    parser.add_argument('--prob-dist', action='store_true', help='Plot probability distribution at different p values.')

    # --- Common arguments ---
    parser.add_argument('--save_path', type=str, help='Path to save the generated plot.')

    # --- Specific arguments ---
    parser.add_argument('--analyse_path', type=str, help='Path to the analyse results HDF5 file (required for heatmap and probability distribution).')
    parser.add_argument('--hierarchical_path', type=str, help='Path to the hierarchical results HDF5 file (required for optimal configuration plot).')
    parser.add_argument('--graph_path', type=str, help='Path to the graph CSV file (required for optimal configuration plot).')
    return parser.parse_args()

def main():
    """Main function to execute the desired plotting based on command-line arguments."""
    args = parse_args()

    if args.heatmap:
        if not args.analyse_path:
            print("Error: --analyse_path is required for heatmap plotting.")
            return
        gamm_beta_result, _ = load_analyse_results(args.analyse_path)
        plot_gamma_beta_heatmaps(gamm_beta_result)

    if args.optimal_config:
        if not args.hierarchical_path:
            print("Error: --hierarchical_path is required for optimal configuration plotting.")
            return
        A, network_graph = construct_graph_from_csv(args.graph_path)
        plot_optimal_configurations(args.hierarchical_path, graph_name, A, network_graph)

    if args.prob_dist:
        if not args.analyse_path:
            print("Error: --analyse_path is required for probability distribution plotting.")
            return
        _, p_result = load_analyse_results(args.analyse_path)
        plot_probability_distribution_at_diff_p(p_result, top_n=10)

if __name__ == "__main__":
    main()

