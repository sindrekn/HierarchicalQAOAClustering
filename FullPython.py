import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import networkx as nx
from numba import njit

# ------------------------------------------------------------------------------
# Create a graph
# ------------------------------------------------------------------------------
def add_edges(graph: nx.Graph, edges: list):
    for edge in edges:
        graph.add_edge(edge[0], edge[1])

network_graph = nx.Graph()

# edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), 
#          (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6), 
#          (6, 7), (6, 8), (7, 8), (7, 9), (8, 9)]
edges = [(0, 1), (0, 6), (0, 2), (1, 2), (1, 6), (2, 3), (3, 4), (3, 5), (4, 5)]

add_edges(network_graph, edges)
A = np.array(nx.adjacency_matrix(network_graph).todense())

N_QUBITS = len(network_graph.nodes)
DIM = 2 ** N_QUBITS

# Pauli matrices
I2 = np.eye(2, dtype=complex)
Zp = np.array([[1, 0], [0, -1]], dtype=complex)

def ising_hamiltonian_k2_modularity(A: np.array, alpha: float) -> np.array:
    """
    Get the Ising Hamiltonian matrix J and constant shift for the k=2 
    modularity problem (special "easy" case).
    """
    num_nodes = len(A)
    m = np.sum(A) / 2
    k = A.sum(axis=1)                          
    B = (A - alpha * np.outer(k, k) / (2 * m)) / (2 * m)
    J = np.zeros((num_nodes, num_nodes))
    const = 0.0

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            J[i, j] += B[i, j]
        const += B[i, i]

    return J, const / 2

def kron_op(op: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """
    Embed single-qubit op on 'qubit' in n-qubit space.
    """
    ops = [I2] * n_qubits
    ops[n_qubits - 1 - qubit] = op   # qubit 0 is the rightmost tensor factor
    result = ops[0]
    for o in ops[1:]:
        result = np.kron(result, o)
    return result

def kron_two(op_i: np.ndarray, qi: int, op_j: np.ndarray, qj: int, n_qubits: int) -> np.ndarray:
    """
    Embed two single-qubit operators on different qubits.
    """
    ops = [I2] * n_qubits
    ops[n_qubits - 1 - qi] = op_i
    ops[n_qubits - 1 - qj] = op_j
    result = ops[0]
    for o in ops[1:]:
        result = np.kron(result, o)
    return result

def pauli_z_hamiltonian_k2_modularity(A: np.array, alpha: float) -> np.array:
    """
    The full 2^n x 2^n Hamiltonian matrix for the k=2 modularity problem, expressed in the computational basis.
    This is a sum of ZZ terms with coefficients from the Ising Hamiltonian, plus a constant shift.
    """
    num_nodes = len(A)
    DIM = 2 ** num_nodes
    H_C = np.zeros((DIM, DIM), dtype=complex)
    J, const = ising_hamiltonian_k2_modularity(A, alpha)

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            ZZ = kron_two(Zp, i, Zp, j, num_nodes)
            H_C += J[i, j] * ZZ

    H_C += const * np.eye(DIM, dtype=complex)
    return H_C

def bitstring_energy(H_C: np.ndarray, x: np.array) -> float:
    """
    Get energy of a computational basis state from the Pauli-Z Hamiltonian.
    x: binary array of length n, e.g. [0, 1, 0, 1]
    """
    n = len(x)
    # Encode bitstring to matrix index: qubit 0 = most significant bit
    idx = sum(int(x[i]) * (2 ** (n - 1 - i)) for i in range(n))
    return H_C[idx, idx].real


H_C = pauli_z_hamiltonian_k2_modularity(A, alpha=1.0)

cost_diag = np.diag(H_C).real

# print("Cost Hamiltonian diagonal (cost for each bitstring):")
# for i, val in enumerate(cost_diag):
#     print(f"  Bitstring {i:0{N_QUBITS}b}: {val:.4f}")

def apply_cost_unitary(state: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-1j * gamma * cost_diag) * state  # O(n) elementwise

# ------------------------------------------------------------------------------
# Apply the mixer unitary (Rx rotations) to the state
# ------------------------------------------------------------------------------
def apply_mixer(state: np.ndarray, beta: float, N_QUBITS: int) -> np.ndarray:
    c, s = np.cos(beta/2), np.sin(beta/2)
    mat = np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)
    state = state.reshape([2]*N_QUBITS)
    for q in range(N_QUBITS):
        state = np.tensordot(mat, state, axes=[[1], [q]])
        state = np.moveaxis(state, 0, q)
    return state.reshape(-1)

def qaoa_run(p, gammas, betas, N_QUBITS):
    dim = 2**N_QUBITS
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)
    for k in range(p):
        state = apply_cost_unitary(state, gammas[k])   
        state = apply_mixer(state, betas[k], N_QUBITS) 
    return state

def expectation_value(p, params, N_QUBITS):
    gammas, betas = params[:p], params[p:]
    state = qaoa_run(p, gammas, betas, N_QUBITS)
    return float(np.real(np.dot(cost_diag, np.abs(state)**2))) 


# Optimise for different p's using COBYLA with random restarts
results = {}
print('Optimising QAOA parameters:\n')

for p in [1, 3, 5]:
    best_val = np.inf
    best_res = None
    n_restarts = 50 if p == 1 else 100

    for _ in range(n_restarts):
        g0 = np.random.uniform(0, np.pi,    p)
        b0 = np.random.uniform(0, np.pi/2,  p)
        x0 = np.concatenate([g0, b0])
        res = minimize(lambda params: -expectation_value(p, params, N_QUBITS), x0,
                       method='COBYLA', options={'maxiter': 1000, 'rhobeg': 0.5})
        if res.fun < best_val:
            best_val = res.fun
            best_res = res

    energy     = -best_val
    psi_opt    = qaoa_run(p, best_res.x[:p], best_res.x[p:], N_QUBITS)
    probs_opt  = np.abs(psi_opt) ** 2

    results[p] = {'params': best_res.x, 'energy': energy,
                  'probs': probs_opt, 'state': psi_opt}

    print(f'  p={p}: E={energy:.4f}  params={best_res.x}')
    print(f'  Probabilities of top 5 bitstrings:')
    top_indices = np.argsort(probs_opt)[-5:][::-1]
    for idx in top_indices:
        print(f'    {idx:0{N_QUBITS}b}  Prob={probs_opt[idx]:.4f}')

    # Find the most probable bitstring and print it
    most_probable = np.argmax(probs_opt)
    print(f'  Most probable bitstring: {most_probable:0{N_QUBITS}b}  Prob={probs_opt[most_probable]:.4f}\n')
          

# Should see variation across gamma with beta fixed
gammas = np.linspace(0, np.pi, 100)
energies = [expectation_value(1, np.array([g, np.pi/4]), N_QUBITS) for g in gammas]
plt.plot(gammas, energies)
plt.xlabel('Gamma')
plt.ylabel('Expected Energy')
plt.title('QAOA Performance')
plt.show()

# ------------------------------------------------------------------------------
# Brute-force optimal k-cluster modularity Q calculation
# ------------------------------------------------------------------------------
from itertools import product
@njit
def modularity_calc(A: np.array, alpha: float, z: np.array) -> float: 
    num_nodes = len(A)
    m = np.sum(A) / 2  # Since A is symmetric, we divide by 2 to get the actual sum of edges

    k = np.zeros(num_nodes)
    for i in range(num_nodes):
        k[i] = np.sum(A[i])

    modularity = 0.0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if z[i] == z[j]:  # Only consider pairs in the same cluster
                modularity += A[i, j] - alpha*(k[i] * k[j]) / (2 * m)
            
    return modularity / (2 * m)

print("Brute-force optimal k-cluster modularity Q calculation: \n")
print("-----------------------------\n")

for n_clusters in [2, 3]:
    best_val = -np.inf
    best_config = None

    # Enumerate all n_clusters^N_QUBITS labellings
    # To avoid counting permutations of labels as different solutions,
    # we fix node 0 to cluster 0 (breaks label symmetry)
    for z in product(range(n_clusters), repeat=N_QUBITS - 1):
        z_full = np.array([0] + list(z), dtype=int)

        # Skip configurations that don't use all n_clusters labels
        # (those are really k<n_clusters partitions)
        if len(np.unique(z_full)) < n_clusters:
            continue

        obj_val = modularity_calc(A, alpha=1.0, z=z_full)
        if obj_val > best_val:
            best_val = obj_val
            best_config = z_full.copy()

    # Summary
    print(f"\nOptimal {n_clusters}-cluster partition (Q = {best_val:.4f})")
    for c in range(n_clusters):
        nodes = np.where(best_config == c)[0].tolist()
        print(f"  Cluster {c} ({len(nodes)} nodes): {nodes}")
    print(f"  Labelling: {best_config.tolist()}")

    if n_clusters == 2:
        best_3_config = best_config.copy()
        best_3_val = best_val

# Plot
cmap = ['red', 'blue', 'green', 'orange', 'purple']
color_map = [cmap[best_3_config[i]] for i in range(N_QUBITS)]
plt.figure()
nx.draw(network_graph, with_labels=True, node_color=color_map)
plt.title(
    f"Optimal 3-cluster partition (Q = {best_3_val:.4f})"
)
plt.show()


print("Modualrity for 0101: ", modularity_calc(A, alpha=1.0, z=np.array([0, 1, 0, 1], dtype=int)))
