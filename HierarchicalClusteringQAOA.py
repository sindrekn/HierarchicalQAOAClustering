import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import minimize
import h5py
import json
from itertools import product
import argparse
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler

# =============================================================================
# Graph Construction
# =============================================================================
def add_edges(graph: nx.Graph, edges: list, weights: list):
    for edge, weight in zip(edges, weights):
        graph.add_edge(edge[0], edge[1], weight=weight)

def construct_graph_from_csv(filename: str) -> nx.Graph:
    network_graph = nx.Graph()
    graph = pd.read_csv(filename)
    edges   = [(row['Node1'], row['Node2']) for _, row in graph.iterrows()]
    weights = [row['Weight'] for _, row in graph.iterrows()]

    add_edges(network_graph, edges, weights)

    print(f"Graph has {network_graph.number_of_nodes()} nodes and {network_graph.number_of_edges()} edges.")

    nodelist = sorted(network_graph.nodes())
    return np.array(nx.adjacency_matrix(network_graph, nodelist=nodelist, weight='weight').todense())

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
# QAOA on IBM quantum device
# =============================================================================

_PARITY = np.array(
    [-1 if bin(i).count("1") % 2 else 1 for i in range(256)],
    dtype=np.complex128,
)

def evaluate_sparse_pauli(state: int, observable: SparsePauliOp) -> complex:
    """Expectation value of a SparsePauliOp on a single computational-basis state.

    For a Z-only observable (which QAOA cost Hamiltonians are, after the
    QUBO-to-Hamiltonian mapping), the eigenvalue of each Pauli term on a
    computational-basis state is simply (-1)**popcount(z_mask AND state),
    i.e., the parity of the bitwise-AND of the term's Z-support and the
    measured bitstring.

    This routine packs the Z-support of every Pauli term into bytes, ANDs
    them against the measured state in a single vectorized op, and looks up
    the parity in _PARITY. For a 100-qubit / ~hundreds-of-terms Hamiltonian
    over 10_000 samples, this is dramatically faster than calling
    SparsePauliOp.expectation_value per sample.
    """
    packed_uint8 = np.packbits(observable.paulis.z, axis=1, bitorder="little")
    state_bytes = np.frombuffer(
        state.to_bytes(packed_uint8.shape[1], "little"), dtype=np.uint8
    )
    reduced = np.bitwise_xor.reduce(packed_uint8 & state_bytes, axis=1)
    return np.sum(observable.coeffs * _PARITY[reduced])

def best_solution(samples, hamiltonian):
    """Return the sampled bitstring (as int) with the lowest Hamiltonian cost."""
    min_cost = float("inf")
    min_sol = None
    for bit_str in samples.keys():
        candidate_sol = int(bit_str)
        fval = evaluate_sparse_pauli(candidate_sol, hamiltonian).real
        if fval <= min_cost:
            min_cost = fval
            min_sol = candidate_sol
    return min_sol

def to_array(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return np.array([int(digit) for digit in result])

class QAOAClusteringIBM:
    """
    QAOA clustering using IBM quantum hardware via Qiskit Runtime.
    """
    def __init__(
        self,
        A: np.ndarray,
        alpha: float,
        p: int,
        shots: int,
    ):
        self.A           = A
        self.alpha       = alpha
        self.p           = p
        self.shots       = shots
        self.n_qubits    = len(A)

        # Build cost Hamiltonian as SparsePauliOp (excludes constant)
        J, self.const = ising_hamiltonian_k2_modularity(A, alpha)
        pauli_list = []
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                if J[i, j] != 0.0:
                    pauli_list.append(("ZZ", [i, j], J[i, j]))
        self.cost_hamiltonian = SparsePauliOp.from_sparse_list(
            pauli_list, num_qubits=self.n_qubits
        )

        # Build and transpile the QAOA ansatz
        circuit = QAOAAnsatz(
            cost_operator=self.cost_hamiltonian,
            reps=self.p,
            name='qaoa_clustering',
        )
        circuit.measure_all()  # For final Sampler measurement

        # Select backend and transpile with preset pass manager for optimal performance
        service = QiskitRuntimeService()
        self.backend = service.least_busy(
            operational=True, simulator=False, min_num_qubits=self.n_qubits
        )
        print(f"Using backend: {self.backend.name}")

        # Transpile with a preset pass manager for the target backend
        pm = generate_preset_pass_manager(
            optimization_level=3,
            backend=self.backend,
        )
        self.candidate_circuit = pm.run(circuit)

    def run(self) -> dict:
        """
        Run the full QAOA optimization.
        """
        trajectory = {'gammas': [], 'betas': [], 'energies': []}

        with Session(backend=self.backend) as session:
            estimator = Estimator(mode=session)
            estimator.options.default_shots = 1000

            # Set simple error suppression/mitigation options
            estimator.options.dynamical_decoupling.enable = True
            estimator.options.dynamical_decoupling.sequence_type = "XY4"
            estimator.options.twirling.enable_gates = True
            estimator.options.twirling.num_randomizations = "auto"
            estimator.options.environment.job_tags = ["CLUDTER_QAOA"]

            print("Running QAOA optimization...")

            def objective(params: np.ndarray) -> float:
                pub       = (self.candidate_circuit, self.cost_hamiltonian, params)
                job       = estimator.run([pub])

                ev        = float(job.result()[0].data.evs) + self.const
                trajectory['gammas'].append(params[:self.p].tolist())
                trajectory['betas'].append(params[self.p:].tolist())
                trajectory['energies'].append(ev)
                return -ev   # negate: scipy minimizes (we want to maximize modularity)

            g0 = np.random.uniform(0, np.pi, self.p)
            b0 = np.random.uniform(0, np.pi/2, self.p)
            x0 = np.concatenate([g0, b0])

            res = minimize(
                objective,
                x0,
                method='COBYLA',
                options={'maxiter': 200},
            )

            print(f"  Best energy: {-res.fun:.4f}")
            best_val    = res.fun
            best_params = res.x.copy()

        # ------------------------------------------------------------------
        # Final measurement with Sampler at optimal parameters
        # ------------------------------------------------------------------
        print("\n  Running final Sampler measurement at optimal parameters...")

        optimized_circuit = self.candidate_circuit.assign_parameters(best_params)

        sampler = Sampler(mode=self.backend)
        sampler.options.default_shots = self.shots

        # Set simple error suppression/mitigation options
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.num_randomizations = "auto"

        # Add a unique tag to the job execution
        sampler.options.environment.job_tags = ["CLUSTER_QAOA"]

        pub = (optimized_circuit,)
        job = sampler.run([pub], shots=self.shots)

        counts_int = job.result()[0].data.meas.get_int_counts()
        shots = sum(counts_int.values())
        final_dist = {
            key:val / shots for key, val in counts_int.items()
        }

        best_sol = best_solution(final_dist, self.cost_hamiltonian)
        best_sol_array = to_array(int(best_sol), self.n_qubits)
        best_sol_array.reverse()

        print(f"\n  Optimal partition : {best_sol_array} with modularity Q = {modularity_calc(self.A, alpha=1.0, x=best_sol_array):.4f}")
        print(f"  Best energy <H_C> : {-best_val:.4f}")
        print(f"  Optimal params    : gamma={np.round(best_params[:self.p], 3).tolist()}"
              f"  beta={np.round(best_params[self.p:], 3).tolist()}")

        # Transform trajectory dict into a 2d numpy array
        trajectory_array = np.array([trajectory['gammas'], trajectory['betas'], trajectory['energies']])

        return {
            'best_params':    best_params,
            'best_energy':    -best_val,
            'best_partition': best_sol_array,
            'trajectories':   trajectory_array
        }

# =============================================================================
# Hierarchical QAOA
# =============================================================================

def HierarchicalBisection(
    A: np.ndarray,
    p: int = 1,
    alpha_scale: float = 1.5,
    min_cluster_size: int = 2,
    max_level: int = 7,
    shots: int = 1000,
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
    shots            : Number of shots for each QAOA optimization.
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
        shots: int = 1000,
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
        k2_cluster = QAOAClusteringIBM(A, alpha=alpha, p=p, shots=shots)
        result = k2_cluster.run()
        local_config = np.array(list(result["best_partition"]), dtype=int)
        tree_node["qaoa_results"] = {
            "params":       result["best_params"],
            "local_config": local_config.tolist(),
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

        # Evaluate the split against the global graph modularity
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
                shots=shots
            )
            _bisect(
                subA=subA[np.ix_(mask1, mask1)],
                global_indices=global_indices[mask1],
                current_labels=proposed_labels,
                depth=depth + 1,
                alpha=child_alpha,
                node_key=f"{node_key}_C1",
                shots=shots
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
        shots=shots
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
# Test the depth p
# =============================================================================
def test_p(A):
    """Run a sweep over p to observe convergence of energy and state probabilities."""
    results = {
        'p': [],
        'best_energy': [],
        'best_partition': [],
        'trajectories': [],
    }
    for p in range(1, 7): 
        print(f"Testing p={p}...")
        result = QAOAClusteringIBM(A, alpha=1.0, p=p, shots=1000).run()
        results['p'].append(p)
        results['best_energy'].append(result['best_energy'])
        results['best_partition'].append(result['best_partition'])
        results['trajectories'].append(result['trajectories'])

    return results

# =============================================================================
# Storage
# =============================================================================

def save_test_results(filename: str, gamm_beta_result: dict, p_result: dict):
    with h5py.File(filename, 'w') as f:

        # --- qaoa_test_gamma_beta results (flat grid, no restart structure) ---
        gb = f.create_group('gamma_beta')
        gb.create_dataset('gamma',          data=np.array(gamm_beta_result['gamma']))
        gb.create_dataset('beta',           data=np.array(gamm_beta_result['beta']))
        gb.create_dataset('expected_values', data=np.array(gamm_beta_result['expected_values']))
        gb.create_dataset('state_probs',    data=np.array(gamm_beta_result['state_probs']))

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
    parser.add_argument('--IBM quantum', action='store_true',
                        help="Run 2-cluster QAOA on IBM quantum hardware (requires Qiskit Runtime setup).")

    # --- Shared parameters ---
    parser.add_argument('--n_restarts', type=int, default=5,
                        help="Number of random restarts per QAOA optimization.")
    parser.add_argument('--gamma_max', type=float, default=np.pi,
                        help="Maximum gamma value for random initialization.")
    parser.add_argument('--beta_max', type=float, default=np.pi / 2,
                        help="Maximum beta value for random initialization.")
    parser.add_argument('--graphfile', type=str, default="graphs/graph.csv",
                        help="CSV file containing graph edges and weights.")

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

    # Construct graph from CSV
    A = construct_graph_from_csv(args.graphfile)

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
            gamma_max=args.gamma_max,
            beta_max=args.beta_max
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

        gamma_beta_result = qaoa_test_gamma_beta(A, alpha=args.alpha, gamma_max=args.gamma_max, beta_max=args.beta_max)
        p_result         = qaoa_test_p(A, alpha=args.alpha, n_restarts=args.n_restarts, gamma_max=args.gamma_max, beta_max=args.beta_max)
        save_test_results(f'results/{args.test_results}', gamma_beta_result, p_result)

        print("\n  Gamma/Beta best result:")

        final_ev = np.array(gamma_beta_result['expected_values'])
        final_g  = np.array(gamma_beta_result['gamma'])
        final_b  = np.array(gamma_beta_result['beta'])
        max_ev = np.max(final_ev)
        max_idx = np.argmax(final_ev)
        gamma = final_g[max_idx]
        beta = final_b[max_idx]
        print(f"    E={max_ev:.4f} | gamma={gamma:.3f} | beta={beta:.3f}")

        print("\n  P-sweep summary:")
        for i, p_val in enumerate(p_result['p']):
            ev = p_result['expected values'][i]
            print(f"    p={p_val} | E={ev:.4f} | "
                  f"gamma={np.round(p_result['gamma_opt'][i], 3).tolist()} | "
                  f"beta={np.round(p_result['beta_opt'][i], 3).tolist()}")


if __name__ == "__main__":
    # Check the eigenvalue gaps
    k2_cluster = QAOAClustering(A, alpha=1.0)
    eigs = np.sort(np.unique(np.round(k2_cluster.cost_diag, 8)))
    gaps = np.diff(eigs)
    min_gap = gaps[gaps > 0].min()
    nyquist_gamma = np.pi / min_gap  # grid step must be smaller than this
    print(f"Min eigenvalue gap : {min_gap:.6f}")
    print(f"Max safe gamma step: {nyquist_gamma:.4f}")
    print(f"Your current step  : {50/100:.4f}")
    

    eigs = np.sort(np.unique(np.round(k2_cluster.cost_diag, 6)))
    print(f"Number of distinct eigenvalues: {len(eigs)}")
    print(f"Eigenvalues: {eigs}")