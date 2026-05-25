import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import minimize
import h5py
import json
from itertools import product
import argparse

# =============================================================================
# Graph Construction
# =============================================================================
def add_edges(graph: nx.Graph, edges: list, weights: list):
    for edge, weight in zip(edges, weights):
        graph.add_edge(edge[0], edge[1], weight=weight)

network_graph = nx.Graph()
graph = pd.read_csv("result/graph.csv")
edges   = [(row['Node1'], row['Node2']) for _, row in graph.iterrows()]
weights = [row['Weight'] for _, row in graph.iterrows()]

add_edges(network_graph, edges, weights)

print(f"Graph has {network_graph.number_of_nodes()} nodes and {network_graph.number_of_edges()} edges.")

nodelist = sorted(network_graph.nodes())
A = np.array(nx.adjacency_matrix(network_graph, nodelist=nodelist, weight='weight').todense())
N_QUBITS = len(network_graph.nodes)
DIM = 2 ** N_QUBITS

# =============================================================================
# Pauli Matrices
# ============================================================================= 
I2 = np.eye(2, dtype=complex)
Zp = np.array([[1, 0], [0, -1]], dtype=complex)

# =============================================================================
# Modularity and Ising Hamiltonian
# =============================================================================

def modularity_calc(A: np.ndarray, alpha: float, x: np.ndarray) -> float:
    """
    Calculate the modularity Q of a partition x given adjacency matrix A.

    The resolution parameter alpha scales the null model term k_i k_j / 2m.
    alpha=1.0 recovers standard Newman-Girvan modularity. Values alpha > 1.0
    penalise large clusters more heavily, encouraging finer partitions —
    useful when recursively bisecting subgraphs.
    """
    num_nodes = len(A)
    m = np.sum(A) / 2
    k = A.sum(axis=1)  # degree vector

    modularity = 0.0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if x[i] == x[j]:
                modularity += A[i, j] - alpha * (k[i] * k[j]) / (2 * m)

    return modularity / (2 * m)


def ising_hamiltonian_k2_modularity(A: np.ndarray, alpha: float) -> tuple:
    """
    Derive the coupling matrix J and constant offset for the k=2 modularity
    Ising Hamiltonian H = sum_{i<j} J_ij s_i s_j + const.

    The modularity matrix B_ij = (A_ij - alpha * k_i k_j / 2m) / 2m is
    split into an interaction part (off-diagonal, i < j) and a constant
    contribution from the diagonal.
    """
    num_nodes = len(A)
    m = np.sum(A) / 2
    k = A.sum(axis=1)
    B = (A - alpha * np.outer(k, k) / (2 * m)) / (2 * m)

    J = np.zeros((num_nodes, num_nodes))
    const = 0.0

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            J[i, j] += B[i, j]
        const += B[i, i]

    return J, const / 2

# =============================================================================
# Pauli-Z Hamiltonian Construction
# =============================================================================

def kron_op(op: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """
    Embed a single-qubit operator on qubit i in the full n-qubit Hilbert space
    via tensor product: I x ... x op x ... x I.
    Qubit 0 is the leftmost (MSB) factor.
    """
    ops = [I2] * n_qubits
    ops[qubit] = op
    result = ops[0]
    for o in ops[1:]:
        result = np.kron(result, o)
    return result


def kron_two(op_i: np.ndarray, qi: int, op_j: np.ndarray, qj: int, n_qubits: int) -> np.ndarray:
    """
    Embed two single-qubit operators on qubits qi and qj simultaneously.
    Used to construct ZZ interaction terms.
    """
    ops = [I2] * n_qubits
    ops[qi] = op_i
    ops[qj] = op_j
    result = ops[0]
    for o in ops[1:]:
        result = np.kron(result, o)
    return result


def pauli_z_hamiltonian_k2_modularity(A: np.ndarray, alpha: float) -> np.ndarray:
    """
    Construct the full 2^n x 2^n cost Hamiltonian H_C in the computational basis.

    Classical spin variables s_i in {-1, +1} are promoted to Pauli-Z operators,
    and two-spin interactions J_ij s_i s_j become ZZ tensor products.
    H_C is diagonal in the computational basis — its eigenvalues are the
    modularity values of all 2^n spin configurations.
    """
    num_nodes = len(A)
    DIM = 2 ** num_nodes
    H_C = np.zeros((DIM, DIM), dtype=complex)
    J, const = ising_hamiltonian_k2_modularity(A, alpha)

    # Sum ZZ interaction terms weighted by J_ij
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            ZZ = kron_two(Zp, i, Zp, j, num_nodes)
            H_C += J[i, j] * ZZ

    # Add the constant offset as a scaled identity
    H_C += const * np.eye(DIM, dtype=complex)

    return H_C

# =============================================================================
# QAOA State Evolution
# =============================================================================

class QAOAClustering:
    """
    Statevector simulation of the QAOA circuit for k=2 graph clustering.

    The circuit alternates between:
      - Cost unitary:  exp(-i gamma H_C), applied as elementwise phase on
                       the diagonal of H_C (cheap since H_C is diagonal)
      - Mixer unitary: exp(-i beta H_M), applied as single-qubit Rx rotations
                       via tensordot contraction qubit by qubit
    """

    def __init__(self, A: np.ndarray, alpha: float):
        self.A = A
        self.alpha = alpha
        self.N_QUBITS = len(A)
        self.dim = 2 ** self.N_QUBITS
        self.qubit_shape = [2] * self.N_QUBITS
        # Only the diagonal is needed since H_C is diagonal in the comp. basis
        self.cost_diag = np.diag(pauli_z_hamiltonian_k2_modularity(A, alpha=alpha)).real
        self.state = None

    def apply_cost_unitary(self, gamma: float):
        # Elementwise multiplication — exact because H_C is diagonal
        self.state = np.exp(-1j * gamma * self.cost_diag) * self.state

    def apply_mixer(self, beta: float):
        # Rx(2beta) rotation applied independently to each qubit
        # exp(-i beta H_M) = tensor product of Rx(2beta) over all qubits
        c, s = np.cos(beta / 2), np.sin(beta / 2)
        Rx = np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

        # Reshape statevector into tensor form for qubit-wise contraction
        self.state = self.state.reshape(self.qubit_shape)
        for q in range(self.N_QUBITS):
            self.state = np.tensordot(Rx, self.state, axes=[[1], [q]])
            self.state = np.moveaxis(self.state, 0, q)
        self.state = self.state.reshape(self.dim)

    def qaoa_run(self, p: int, params: np.ndarray):
        """Prepare the QAOA state |psi_p(gamma, beta)> from |+>^n."""
        gammas, betas = params[:p], params[p:]
        # Initialise to uniform superposition |+>^n
        self.state = np.ones(self.dim, dtype=complex) / np.sqrt(self.dim)
        for k in range(p):
            self.apply_cost_unitary(gammas[k])
            self.apply_mixer(betas[k])

    def expectation_value(self, p: int, params: np.ndarray) -> float:
        """Return <psi_p | H_C | psi_p> = sum_x cost_diag[x] * |<x|psi>|^2."""
        self.qaoa_run(p, params)
        return float(np.dot(self.cost_diag, np.abs(self.state) ** 2))

    def get_most_probable_bitstring(self) -> str:
        """Return the computational basis state with highest measurement probability."""
        if self.state is None:
            raise ValueError("State not initialised — call qaoa_run() first.")
        probs = np.abs(self.state) ** 2
        most_probable_idx = np.argmax(probs)
        return format(most_probable_idx, f'0{self.N_QUBITS}b')

# =============================================================================
# QAOA Optimizer — 2-cluster
# =============================================================================

def qaoa_k2_cluster(
    A: np.ndarray,
    alpha: float,
    p: int,
    n_restarts: int,
) -> dict:
    """
    Optimize QAOA variational parameters for 2-cluster modularity maximization.

    Uses COBYLA (gradient-free) via scipy.minimize with multiple random
    restarts to mitigate local optima. The objective is negated since
    scipy minimizes and we want to maximize modularity.

    Returns a dict with the best parameters, measurement probabilities,
    and the most probable bitstring partition.
    """
    k2_cluster = QAOAClustering(A, alpha=alpha)

    best_val = np.inf
    best_res = None

    for _ in range(n_restarts):
        # Random initialization of gamma in [0, pi] and beta in [0, pi/2]
        g0 = np.random.uniform(0, np.pi,     p)
        b0 = np.random.uniform(0, np.pi / 2, p)
        x0 = np.concatenate([g0, b0])

        res = minimize(
            lambda params: -k2_cluster.expectation_value(p, params),
            x0,
            method='COBYLA',
            options={'maxiter': 200, 'rhobeg': 0.5},
        )

        if res.fun < best_val:
            best_val = res.fun
            best_res = res

    # Re-run with best found parameters to get the final state
    k2_cluster.qaoa_run(p, best_res.x)
    probs_opt = np.abs(k2_cluster.state) ** 2

    results = {
        'params':          best_res.x,
        'probs':           probs_opt,
        'best_partition':  k2_cluster.get_most_probable_bitstring(),
    }
    return results

def qaoa_test_gamma_beta(
    A: np.ndarray,
    alpha: float,
    n_restarts: int,
) -> dict:
    """
    Test function to evaluate the distribution of optimized gamma and beta
    parameters across multiple random restarts of the 2-cluster QAOA optimization.
    Logs every function evaluation made by the optimizer, not just the final result.
    """
    k2_cluster = QAOAClustering(A, alpha=alpha)
    results = {
        'gammas':           [],  
        'betas':            [],  
        'state_probs':      [],  
        'expected_values':  [],   
    }

    for _ in range(n_restarts):
        # Log every step the optimizer takes
        trajectory = {
            'gammas':          [],
            'betas':           [],
            'expected_values': [],
            'state_probs':     [],
        }

        def objective(params):
            ev = k2_cluster.expectation_value(1, params)
            # Log current evaluation
            trajectory['gammas'].append(params[0])
            trajectory['betas'].append(params[1])
            trajectory['expected_values'].append(ev)
            trajectory['state_probs'].append(np.abs(k2_cluster.state) ** 2)
            return -ev

        g0 = np.random.uniform(0, np.pi)
        b0 = np.random.uniform(0, np.pi / 2)
        x0 = np.array([g0, b0])

        res = minimize(
            objective,
            x0,
            method='COBYLA',
            options={'maxiter': 5, 'rhobeg': 0.5},
        )

        results['gammas'].append(np.array(trajectory['gammas']))
        results['betas'].append(np.array(trajectory['betas']))
        results['state_probs'].append(np.array(trajectory['state_probs']))
        results['expected_values'].append(np.array(trajectory['expected_values']))

    return results

def qaoa_test_p(
    A: np.ndarray,
    alpha: float,
    n_restarts: int,
) -> dict:
    """
    Test function to evaluate the effect of varying the QAOA circuit depth p on the 
    optimized parameters and resulting state probabilities for the 2-cluster 
    modularity maximization problem.
    """
    k2_cluster = QAOAClustering(A, alpha=alpha)

    results = {'p': [],'gamma_opt': [], 'beta_opt': [], 'state probabilities': [], 'expected values': []}

    for p in range(1, 7):  # Test p from 1 to 6
        best_val = np.inf
        best_res = None
        for _ in range(n_restarts):
            # Random initialization of gamma in [0, pi] and beta in [0, pi/2]
            g0 = np.random.uniform(0, np.pi,     p)
            b0 = np.random.uniform(0, np.pi / 2, p)
            x0 = np.concatenate([g0, b0])

            res = minimize(
                lambda params: -k2_cluster.expectation_value(p, params),
                x0,
                method='COBYLA',
                options={'maxiter': 200, 'rhobeg': 0.5},
            )

            if res.fun < best_val:
                best_val = res.fun
                best_res = res

        # Re-run with best found parameters to get the final state
        k2_cluster.qaoa_run(p, best_res.x)
        probs_opt = np.abs(k2_cluster.state) ** 2

        results['p'].append(p)
        results['gamma_opt'].append(best_res.x[:p])
        results['beta_opt'].append(best_res.x[p:])
        results['state probabilities'].append(probs_opt)
        results['expected values'].append(-best_val)

    return results

# =============================================================================
# Hierarchical k-cluster QAOA
# =============================================================================

def k_cluster_qaoa(
    A: np.ndarray,
    p: int = 1,
    n_restarts: int = 3,
    alpha_scale: float = 1.5,
    min_cluster_size: int = 2,
    max_level: int = 7,
) -> dict:
    """
    Hierarchical bisection clustering using QAOA.

    Repeatedly applies 2-cluster QAOA to subgraphs, building a binary
    bisection tree. A split is accepted only if it strictly improves the
    global modularity of the original graph. alpha is scaled up at each
    level to incentivise partitioning of increasingly cohesive subgraphs.

    Parameters
    ----------
    A                : Adjacency matrix of the original graph (never modified).
    p                : QAOA circuit depth.
    n_restarts       : Random restarts per QAOA optimization.
    alpha_scale      : Multiplicative alpha increase per recursion depth.
    min_cluster_size : Do not attempt to split clusters smaller than this.
    max_level        : Maximum bisection level.
    """
    n = len(A)

    # Shared mutable state across all recursive calls
    state = {
        "best_labels":      np.zeros(n, dtype=int),
        "best_modularity":  modularity_calc(A, alpha=1.0, x=np.zeros(n, dtype=int)),
        "n_clusters":       1,
        "tree":             {},
    }

    def _bisect(
        subA: np.ndarray,
        global_indices: np.ndarray,
        current_labels: np.ndarray,
        depth: int,
        alpha: float,
        node_key: str,
    ) -> None:
        """
        Recursively bisect a subgraph and update shared state if accepted.

        global_indices maps local subgraph node indices back to the original
        graph, allowing local QAOA results to update the global label array.
        """
        sub_n = len(subA)

        # Record this node in the bisection tree regardless of outcome
        tree_node = {
            "depth":          depth,
            "alpha":          alpha,
            "global_indices": global_indices.tolist(),
            "sub_n":          sub_n,
            "split_accepted": False,
            "children":       {},
        }
        state["tree"][node_key] = tree_node

        # Base cases: subgraph too small or maximum level reached
        if sub_n < min_cluster_size or depth >= max_level:
            return

        # Run 2-cluster QAOA on this subgraph
        results = qaoa_k2_cluster(subA, alpha=alpha, p=p, n_restarts=n_restarts)
        local_config = np.array(list(results["best_partition"]), dtype=int)
        tree_node["qaoa_results"] = {
            "params":       results["params"],
            "local_config": local_config.tolist(),
            "probs":        results["probs"].tolist(),
        }

        mask0 = local_config == 0
        mask1 = local_config == 1

        # Reject trivial splits where one side is empty
        if mask0.sum() == 0 or mask1.sum() == 0:
            return

        # Assign a new global cluster id to nodes in the local cluster-1 group
        new_cluster_id = state["n_clusters"]
        proposed_labels = current_labels.copy()
        proposed_labels[global_indices[mask1]] = new_cluster_id

        # Evaluate the split against the *global* graph modularity
        global_modularity = modularity_calc(A, alpha=1.0, x=proposed_labels)
        tree_node["proposed_modularity"] = float(global_modularity)
        tree_node["previous_modularity"] = float(state["best_modularity"])

        if global_modularity > state["best_modularity"]:
            # Accept: update global best and recurse into both children
            state["best_modularity"] = global_modularity
            state["best_labels"]     = proposed_labels.copy()
            state["n_clusters"]     += 1
            tree_node["split_accepted"] = True

            print(
                f"[depth={depth}] Split accepted | "
                f"nodes {global_indices.tolist()} → "
                f"C{new_cluster_id - 1}:{global_indices[mask0].tolist()} "
                f"C{new_cluster_id}:{global_indices[mask1].tolist()} | "
                f"Q={global_modularity:.4f}"
            )

            child_alpha = alpha * alpha_scale
            _bisect(
                subA=subA[np.ix_(mask0, mask0)],
                global_indices=global_indices[mask0],
                current_labels=proposed_labels,
                depth=depth + 1,
                alpha=child_alpha,
                node_key=f"{node_key}_C0",
            )
            _bisect(
                subA=subA[np.ix_(mask1, mask1)],
                global_indices=global_indices[mask1],
                current_labels=proposed_labels,
                depth=depth + 1,
                alpha=child_alpha,
                node_key=f"{node_key}_C1",
            )
        else:
            print(
                f"[depth={depth}] Split rejected | "
                f"nodes {global_indices.tolist()} | "
                f"Q={global_modularity:.4f} ≤ best={state['best_modularity']:.4f}"
            )

    # Start recursion from the full graph with standard alpha=1.0
    _bisect(
        subA=A,
        global_indices=np.arange(n),
        current_labels=state["best_labels"].copy(),
        depth=0,
        alpha=1.0,
        node_key="root",
    )

    print(f"\nFinal partition into {state['n_clusters']} clusters:")
    print(f"  Labels     : {state['best_labels'].tolist()}")
    print(f"  Modularity : {state['best_modularity']:.4f}")

    return {
        "best_labels":      state["best_labels"],
        "best_modularity":  state["best_modularity"],
        "n_clusters":       state["n_clusters"],
        "tree":             state["tree"],
    }

# =============================================================================
# Storage
# =============================================================================

def save_test_results(filename: str, gamm_beta_result: dict, p_result: dict):
    with h5py.File(filename, 'w') as f:

        # --- qaoa_test_gamma_beta results ---
        gb = f.create_group('gamma_beta')
        for i in range(len(gamm_beta_result['gammas'])):
            restart = gb.create_group(f'restart_{i}')
            restart.create_dataset('gammas',          data=gamm_beta_result['gammas'][i])
            restart.create_dataset('betas',           data=gamm_beta_result['betas'][i])
            restart.create_dataset('expected_values', data=gamm_beta_result['expected_values'][i])
            restart.create_dataset('state_probs',     data=gamm_beta_result['state_probs'][i])

        # --- qaoa_test_p results ---
        pr = f.create_group('p_sweep')
        pr.create_dataset('p',               data=np.array(p_result['p']))
        pr.create_dataset('expected_values', data=np.array(p_result['expected values']))
        for i, p in enumerate(p_result['p']):
            layer = pr.create_group(f'p_{p}')
            layer.create_dataset('gamma_opt',        data=p_result['gamma_opt'][i])
            layer.create_dataset('beta_opt',         data=p_result['beta_opt'][i])
            layer.create_dataset('state_probs',      data=p_result['state probabilities'][i])

class NumpyEncoder(json.JSONEncoder):
    """Convert numpy types to native Python types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)

def save_hierarchical_result(filename: str, result: dict):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('best_labels',     data=result['best_labels'])
        f.create_dataset('best_modularity', data=result['best_modularity'])
        f.create_dataset('n_clusters',      data=result['n_clusters'])
        # Store tree as JSON string with numpy-safe encoder
        tree_json = json.dumps(result['tree'], cls=NumpyEncoder)
        f.create_dataset('tree', data=tree_json)

# Find the optimal k-cluster partition using brute-force search for comparison (only feasible for small graphs due to exponential scaling)
def brute_force_k_cluster(A: np.ndarray, k: int) -> dict:
    n = len(A)
    best_modularity = -np.inf
    best_partition = None

    for partition in product(range(k), repeat=n):
        modularity = modularity_calc(A, alpha=1.0, x=np.array(partition))
        if modularity > best_modularity:
            best_modularity = modularity
            best_partition = partition

    return {
        "best_partition": best_partition,
        "best_modularity": best_modularity,
    }

# =============================================================================
# Main Execution
# =============================================================================
# =============================================================================
# Main Execution
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="QAOA-based graph clustering from adjacency CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Mode flags ---
    parser.add_argument('--hierarchical', action='store_true',
                        help="Run hierarchical k-cluster QAOA bisection.")
    parser.add_argument('--test', action='store_true',
                        help="Run gamma/beta and p-sweep diagnostic tests.")

    # --- Shared parameters ---
    parser.add_argument('--n_restarts', type=int, default=5,
                        help="Number of random restarts per QAOA optimization.")

    # --- Hierarchical-only parameters ---
    parser.add_argument('--hierarchical_result', type=str, default="hierarchical_result.h5",
                        help="Filename to save hierarchical clustering result.")
    parser.add_argument('--p', type=int, default=1,
                        help="QAOA circuit depth.")
    parser.add_argument('--alpha_scale', type=float, default=1.5,
                        help="[hierarchical] Multiplicative alpha increase per bisection level.")
    parser.add_argument('--min_cluster_size', type=int, default=2,
                        help="[hierarchical] Minimum subgraph size to attempt a split.")
    parser.add_argument('--max_level', type=int, default=7,
                        help="[hierarchical] Maximum bisection level.")

    # --- Test-only parameters ---
    parser.add_argument('--test_results', type=str, default="test_results.h5",
                        help="Filename to save diagnostic test results.")
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="[test] Resolution parameter for diagnostic tests.")

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.hierarchical and not args.test:
        print("Nothing to run — pass --hierarchical and/or --test.")
        return

    # Validate test-specific args
    if args.test and args.alpha is None:
        raise ValueError("--alpha is required when running --test.")

    # ------------------------------------------------------------------
    # Hierarchical clustering
    # ------------------------------------------------------------------
    if args.hierarchical:
        print("\n" + "="*60)
        print("Running hierarchical k-cluster QAOA bisection")
        print("="*60)
        res = k_cluster_qaoa(
            A,
            p=args.p,
            n_restarts=args.n_restarts,
            alpha_scale=args.alpha_scale,
            min_cluster_size=args.min_cluster_size,
            max_level=args.max_level,
        )
        save_hierarchical_result(f'results/{args.hierarchical_result}', res)

        print(f"\n  Final number of clusters : {res['n_clusters']}")
        print(f"  Best modularity Q        : {res['best_modularity']:.4f}")
        print(f"  Node labels              : {res['best_labels'].tolist()}")
        for c in range(res['n_clusters']):
            nodes = np.where(res['best_labels'] == c)[0].tolist()
            print(f"    Cluster {c} ({len(nodes)} nodes): {nodes}")

        if N_QUBITS < 15:
            print("\n  Brute-force comparison:")
            for k in range(2, res['n_clusters'] + 1):
                bf = brute_force_k_cluster(A, k)
                print(f"    k={k} | Q={bf['best_modularity']:.4f} | "
                      f"Partition={list(bf['best_partition'])}")

    # ------------------------------------------------------------------
    # Diagnostic tests
    # ------------------------------------------------------------------
    if args.test:
        print("\n" + "="*60)
        print(f"Running diagnostic tests (alpha={args.alpha})")
        print("="*60)

        gamm_beta_result = qaoa_test_gamma_beta(A, alpha=args.alpha, n_restarts=args.n_restarts)
        p_result         = qaoa_test_p(A, alpha=args.alpha, n_restarts=args.n_restarts)
        save_test_results(f'results/{args.test_results}', gamm_beta_result, p_result)

        print("\n  Gamma/Beta sweep summary:")
        for i in range(args.n_restarts):
            final_ev = gamm_beta_result['expected_values'][i][-1]
            final_g  = gamm_beta_result['gammas'][i][-1]
            final_b  = gamm_beta_result['betas'][i][-1]
            n_iters  = len(gamm_beta_result['expected_values'][i])
            print(f"    Restart {i}: {n_iters} iters | "
                  f"gamma={final_g:.3f}  beta={final_b:.3f}  E={final_ev:.4f}")

        print("\n  P-sweep summary:")
        for i, p_val in enumerate(p_result['p']):
            ev = p_result['expected values'][i]
            print(f"    p={p_val} | E={ev:.4f} | "
                  f"gamma={np.round(p_result['gamma_opt'][i], 3).tolist()} | "
                  f"beta={np.round(p_result['beta_opt'][i], 3).tolist()}")


if __name__ == "__main__":
    main()
