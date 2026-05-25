import h5py
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# =============================================================================
# Create the graph from the CSV file
# ==============================================================================
def add_edges(graph: nx.Graph, edges: list, weights: list):
    for edge, weight in zip(edges, weights):
        graph.add_edge(edge[0], edge[1], weight=weight)

network_graph = nx.Graph()
graph = pd.read_csv("graphs/graph.csv")
edges   = [(row['Node1'], row['Node2']) for _, row in graph.iterrows()]
weights = [row['Weight'] for _, row in graph.iterrows()]

add_edges(network_graph, edges, weights)

nodelist = sorted(network_graph.nodes())
A = np.array(nx.adjacency_matrix(network_graph, nodelist=nodelist, weight='weight').todense())

# =============================================================================
# Loading Results
# =============================================================================
def load_hierarchical_result(filename: str) -> dict:
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

def load_test_results(filename: str) -> tuple:
    gamm_beta_result = {'gammas': [], 'betas': [], 'expected_values': [], 'state_probs': []}
    p_result = {'p': [], 'gamma_opt': [], 'beta_opt': [], 'state probabilities': [], 'expected values': []}

    with h5py.File(filename, 'r') as f:

        gb = f['gamma_beta']
        for key in sorted(gb.keys()):
            restart = gb[key]
            gamm_beta_result['gammas'].append(restart['gammas'][:])
            gamm_beta_result['betas'].append(restart['betas'][:])
            gamm_beta_result['expected_values'].append(restart['expected_values'][:])
            gamm_beta_result['state_probs'].append(restart['state_probs'][:])

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

def plot_optimal_configurations():    
    """Plot the optimal clustering configuration from the hierarchical clustering result."""
    res = load_hierarchical_result("results/hierarchical_result.h5")
    
    # Plot the resulting partition
    G = nx.from_numpy_array(A)
    print(f"Graph has {network_graph.number_of_nodes()} nodes and {network_graph.number_of_edges()} edges.")

    labels = res["best_labels"]
    unique_labels = np.unique(labels)
    color_map = plt.get_cmap('tab10')
    colors = [color_map(label) for label in labels]

    plt.figure(figsize=(8, 6))
    nx.draw_networkx(G, with_labels=True, node_color=colors, node_size=500)
    plt.title(f"Hierarchical QAOA Clustering (Q={res['best_modularity']:.4f}, k={res['n_clusters']})")
    plt.axis('off')
    plt.show()

def probability_distribution_at_diff_p(
    p_result: dict,
    top_n: int = 10,
):
    """
    Plot the probability distribution of the top_n most probable basis states
    for each QAOA circuit depth p in p_result.

    The Z_2 symmetry means the optimal partition always appears as two
    degenerate bitstrings (complements of each other) — their combined
    probability is shown in the title.
    """
    p_vals   = p_result['p']
    p_num    = len(p_vals)
    n_qubits = int(np.log2(len(p_result['state probabilities'][0])))

    # Build subplot grid
    if p_num <= 3:
        fig, axes = plt.subplots(1, p_num, figsize=(5 * p_num, 5))
        axes = np.atleast_1d(axes)  # ensure iterable when p_num == 1
    else:
        ncols = 3
        nrows = (p_num + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
        axes = axes.flatten()

    for i, p_val in enumerate(p_vals):
        # p_result['state probabilities'][i] is the state prob vector for depth p_val
        # (already the best over restarts from qaoa_test_p)
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
        ax.set_xticklabels(top_bitstrings, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Basis state')
        ax.set_ylabel('Probability')
        ax.set_title(
            f'p={p_val} | E={expected_val:.4f} | '
            f'Z$_2$ prob={total_top_prob:.3f}',
            fontsize=10,
        )

    # Hide any unused subplots
    for j in range(p_num, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f'QAOA probability distributions — top {top_n} states per depth',
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    plt.show()

probability_distribution_at_diff_p(load_test_results("results/test_results.h5")[1], top_n=20)