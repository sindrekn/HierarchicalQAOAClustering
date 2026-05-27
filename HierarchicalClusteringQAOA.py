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
# QAOA on IBM quantum device
# =============================================================================
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler

class QAOAClusteringIBM:
    """
    QAOA clustering using IBM quantum hardware via Qiskit Runtime.

    Uses EstimatorV2 inside a Session for the classical optimization loop
    (evaluating <H_C> at each step), then SamplerV2 for a final high-shot
    measurement to extract the partition bitstring.

    The optimizer trajectory — (gamma, beta, <H_C>) at every function
    evaluation — is logged via a closure around the objective function,
    mirroring the statevector implementation.
    """

    def __init__(
        self,
        A: np.ndarray,
        alpha: float,
        p: int,
        n_restarts: int = 3,
        gamma_max: float = np.pi,
        beta_max: float  = np.pi / 2,
        shots: int       = 4096,
        optimization_level: int = 3,
    ):
        self.A           = A
        self.alpha       = alpha
        self.p           = p
        self.n_restarts  = n_restarts
        self.gamma_max   = gamma_max
        self.beta_max    = beta_max
        self.shots       = shots
        self.n_qubits    = len(A)

        # Build cost Hamiltonian as SparsePauliOp (excludes constant — Estimator
        # measures operator expectation values, not absolute energies)
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
        # QAOAAnsatz parameters are ordered [gamma_0,...,gamma_p, beta_0,...,beta_p]
        ansatz_raw = QAOAAnsatz(
            cost_operator=self.cost_hamiltonian,
            reps=self.p,
            name='qaoa_clustering',
        )
        # Store parameter order for binding later
        self.param_names = [p.name for p in ansatz_raw.parameters]

        service = QiskitRuntimeService()
        self.backend = service.least_busy(
            operational=True, simulator=False, min_num_qubits=self.n_qubits
        )
        print(f"Using backend: {self.backend.name}")

        pm = generate_preset_pass_manager(
            optimization_level=optimization_level,
            backend=self.backend,
        )
        self.isa_ansatz = pm.run(ansatz_raw)

        # ISA observable — layout may reorder qubits after transpilation
        self.isa_hamiltonian = self.cost_hamiltonian.apply_layout(
            self.isa_ansatz.layout
        )

        # Sampler circuit needs measurements
        self.isa_ansatz_meas = self.isa_ansatz.copy()
        self.isa_ansatz_meas.measure_all()

    def _bind_params(self, params: np.ndarray) -> np.ndarray:
        """
        QAOAAnsatz uses the parameter order [beta_0,...,beta_p, gamma_0,...gamma_p]
        internally (mixer before cost). Our convention is [gammas | betas].
        Remap accordingly.
        """
        gammas = params[:self.p]
        betas  = params[self.p:]
        # QAOAAnsatz parameter name pattern: 'β[k]' and 'γ[k]'
        bound  = np.zeros(len(self.param_names))
        for idx, name in enumerate(self.param_names):
            if name.startswith('β') or name.startswith('beta'):
                layer = int(''.join(filter(str.isdigit, name)))
                bound[idx] = betas[layer]
            elif name.startswith('γ') or name.startswith('gamma'):
                layer = int(''.join(filter(str.isdigit, name)))
                bound[idx] = gammas[layer]
        return bound

    def run(self) -> dict:
        """
        Run the full QAOA optimization over multiple restarts.

        Returns
        -------
        dict with keys:
            best_params      : optimized (gamma, beta) array
            best_energy      : best <H_C> found (including constant offset)
            best_partition   : most probable bitstring at optimal params
            trajectories     : list of per-restart dicts with full optimizer paths
            counts           : raw measurement counts from final Sampler run
        """
        best_val        = np.inf
        best_params     = None
        trajectories    = []

        with Session(backend=self.backend) as session:
            estimator = Estimator(mode=session)
            estimator.options.default_shots = 1024

            for restart in range(self.n_restarts):
                print(f"\n  Restart {restart + 1}/{self.n_restarts}")
                trajectory = {'gammas': [], 'betas': [], 'energies': []}

                def objective(params: np.ndarray) -> float:
                    bound     = self._bind_params(params)
                    pub       = (self.isa_ansatz, self.isa_hamiltonian, [bound])
                    job       = estimator.run([pub])
                    ev        = float(job.result()[0].data.evs[0]) + self.const
                    # Log trajectory
                    trajectory['gammas'].append(params[:self.p].tolist())
                    trajectory['betas'].append(params[self.p:].tolist())
                    trajectory['energies'].append(ev)
                    return -ev   # negate: scipy minimizes, we maximize

                g0 = np.random.uniform(0, self.gamma_max, self.p)
                b0 = np.random.uniform(0, self.beta_max,  self.p)
                x0 = np.concatenate([g0, b0])

                res = minimize(
                    objective,
                    x0,
                    method='COBYLA',
                    options={'maxiter': 200, 'rhobeg': 0.5},
                )

                trajectories.append({
                    'gammas':   np.array(trajectory['gammas']),
                    'betas':    np.array(trajectory['betas']),
                    'energies': np.array(trajectory['energies']),
                })

                print(f"    Best energy this restart: {-res.fun:.4f}")
                if res.fun < best_val:
                    best_val    = res.fun
                    best_params = res.x.copy()

            # ------------------------------------------------------------------
            # Final measurement with Sampler at optimal parameters
            # ------------------------------------------------------------------
            print("\n  Running final Sampler measurement at optimal parameters...")
            sampler = Sampler(mode=session)
            sampler.options.default_shots = self.shots

            bound_final = self._bind_params(best_params)
            pub         = (self.isa_ansatz_meas, [bound_final])
            job         = sampler.run([pub])
            counts      = job.result()[0].data.meas.get_counts()

        # Most probable bitstring → partition
        best_bitstring = max(counts, key=counts.get)
        # Reverse bit order: Qiskit is LSB-first, our convention is MSB-first
        best_bitstring_msb = best_bitstring[::-1]

        print(f"\n  Optimal partition : {best_bitstring_msb}")
        print(f"  Best energy <H_C> : {-best_val:.4f}")
        print(f"  Optimal params    : gamma={np.round(best_params[:self.p], 3).tolist()}"
              f"  beta={np.round(best_params[self.p:], 3).tolist()}")

        return {
            'best_params':    best_params,
            'best_energy':    -best_val,
            'best_partition': best_bitstring_msb,
            'trajectories':   trajectories,
            'counts':         counts,
        }


# =============================================================================
# QAOA Optimizer — 2-cluster
# =============================================================================

def qaoa_k2_cluster(
    A: np.ndarray,
    alpha: float,
    p: int,
    n_restarts: int,
    gamma_max: float,
    beta_max: float
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
        g0 = np.random.uniform(0, gamma_max,     p)
        b0 = np.random.uniform(0, beta_max, p)
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
    gamma_max: float,
    beta_max: float
) -> dict:
    """
    Test function to evaluate the distribution of optimized gamma and beta
    parameters across multiple random restarts of the 2-cluster QAOA optimization.
    Logs every function evaluation made by the optimizer, not just the final result.
    """
    k2_cluster = QAOAClustering(A, alpha=alpha)
    results = {
        'gamma':           [],  
        'beta':            [],  
        'state_probs':      [],  
        'expected_values':  [],   
    }   

    gamma_list = np.linspace(25, 65, 100)
    beta_list = np.linspace(0, beta_max, 100)  

    for gamma in gamma_list:
        for beta in beta_list:
            ev = k2_cluster.expectation_value(1, np.array([gamma, beta]))
            results['gamma'].append(gamma)
            results['beta'].append(beta)
            results['expected_values'].append(ev)
            results['state_probs'].append(np.abs(k2_cluster.state) ** 2)

    return results

def qaoa_test_p(
    A: np.ndarray,
    alpha: float,
    n_restarts: int,
    gamma_max: float,
    beta_max: float
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
            g0 = np.random.uniform(0, gamma_max,     p)
            b0 = np.random.uniform(0, beta_max, p)
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
    gamma_max: float = np.pi,
    beta_max: float = np.pi / 2
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
        gamma_max: float, 
        beta_max: float
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
        results = qaoa_k2_cluster(subA, alpha=alpha, p=p, n_restarts=n_restarts, gamma_max=gamma_max, beta_max=beta_max)
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
                gamma_max=gamma_max,
                beta_max=beta_max
            )
            _bisect(
                subA=subA[np.ix_(mask1, mask1)],
                global_indices=global_indices[mask1],
                current_labels=proposed_labels,
                depth=depth + 1,
                alpha=child_alpha,
                node_key=f"{node_key}_C1",
                gamma_max=gamma_max,
                beta_max=beta_max
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
        gamma_max=gamma_max,
        beta_max=beta_max
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