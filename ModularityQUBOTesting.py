import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd

def add_edges(graph: nx.Graph, edges: list, weights: list):
    """Add edges with weights to the graph."""
    for edge, weight in zip(edges, weights):
        graph.add_edge(edge[0], edge[1], weight=weight)

def construct_graph_from_csv(filename: str) -> nx.Graph:
    """
    Construct a NetworkX graph from a CSV file with columns Node1,Node2,Weight.
    Node1 and Node2 has to be ints, while Weight can be any numeric type.

    LLM-assisted
    ------------
    Tool: Github Copilot (2026)
    Created print statement
    """
    network_graph = nx.Graph()
    graph = pd.read_csv(filename)
    edges   = [(row['Node1'], row['Node2']) for _, row in graph.iterrows()]
    weights = [row['Weight'] for _, row in graph.iterrows()]

    add_edges(network_graph, edges, weights)

    print(f"Graph has {network_graph.number_of_nodes()} nodes and {network_graph.number_of_edges()} edges.")

    nodelist = sorted(network_graph.nodes())
    return np.array(nx.adjacency_matrix(network_graph, nodelist=nodelist, weight='weight').todense())

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


# Pauli matrices
I2 = np.eye(2, dtype=complex)
Zp = np.array([[1, 0], [0, -1]], dtype=complex)

def kron_two(op_i: np.ndarray, qi: int, op_j: np.ndarray, qj: int, n_qubits: int) -> np.ndarray:
    """Embed two single-qubit operators on different qubits."""
    ops = [I2] * n_qubits
    ops[n_qubits - 1 - qi] = op_i
    ops[n_qubits - 1 - qj] = op_j
    result = ops[0]
    for o in ops[1:]:
        result = np.kron(result, o)
    return result

def pauli_z_hamiltonian(A: np.ndarray, alpha: float) -> np.ndarray:
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

def bitstring_energy(H_C: np.ndarray, x: np.array, n_qubits: int) -> float:
    """
    Get energy of a computational basis state from the Pauli-Z Hamiltonian.
    x: binary array of length n, e.g. [0, 1, 0, 1]
    """
    x = np.flip(x)
    # Map x to a bitstring: 
    x = ''.join(x.astype(int).astype(str))
    idx = 0
    while True: 
        format(idx, f'0{n_qubits}b')
        if format(idx, f'0{n_qubits}b') == x:
            break
        idx += 1

    return H_C[idx, idx].real

def qubo_k2_modularity(A: np.array, alpha: float = 1.0) -> tuple[np.array, float]:
    num_nodes = len(A)
    m = np.sum(A) / 2
    k = A.sum(axis=1)                          

    B = (A - alpha * np.outer(k, k) / (2 * m)) / (2 * m)

    qubo = np.zeros((num_nodes, num_nodes))
    constant = B.sum()
    
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            qubo[i, j] += 4 * B[i, j]
        qubo[i, i] += 2 * B[i, i]
        qubo[i, i] -= 2*B.sum(axis=1)[i] 

    return qubo, constant

def modularity_calc(A: np.array, alpha: float, x: np.array) -> float: 
    num_nodes = len(A)
    m = np.sum(A) / 2  # Since A is symmetric, we divide by 2 to get the actual sum of edges

    k = np.zeros(num_nodes)
    for i in range(num_nodes):
        k[i] = np.sum(A[i])

    modularity = 0.0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if x[i] == x[j]:  # Only consider pairs in the same cluster
                modularity += A[i, j] - alpha*(k[i] * k[j]) / (2 * m)
            
    return modularity / (2 * m)

def modularity_calc(A: np.ndarray, alpha: float, x: np.ndarray) -> float:
    k = A.sum(axis=1)
    m = k.sum() / 2
    # Outer product mask for same-community pairs
    same_comm = (x[:, None] == x[None, :])  # (n, n) bool
    null_model = np.outer(k, k) / (2 * m)
    B = A - alpha * null_model
    return B[same_comm].sum() / (2 * m)

def modularity_calc(A: np.ndarray, alpha: float, x: np.ndarray) -> float:
    k = A.sum(axis=1)
    deg_sum = k.sum()
    m = deg_sum / 2
    norm = 1 / deg_sum**2

    Q = 0.0
    for comm_id in np.unique(x):
        members = np.where(x == comm_id)[0]
        # L_c: intra-community edge weight (each edge once, like nx.G.edges())
        subA = A[np.ix_(members, members)]
        L_c = subA.sum() / 2  # divide by 2 because A is symmetric
        k_c = k[members].sum()
        Q += L_c / m - alpha * k_c**2 * norm

    return Q

# ------------------------------------------------------------------------------
def build_modularity_hamiltonian_org(A: np.array, alpha: float = 1.0) -> tuple[np.array, float]:
    num_nodes = len(A)
    m = np.sum(A) / 2
    k = A.sum(axis=1)            

    B = (A - alpha * np.outer(k, k) / (2 * m)) / (4 * m)

    H = np.zeros((num_nodes, num_nodes))

    constant = np.diag(np.diag(B)).sum()

    H += B
    H = 2 * np.triu(H)                       # keep upper triangle (standard QUBO form)
    H -= np.diag(np.diag(H))                   # ensure diagonal is zero

    return H, constant

def build_modularity_hamiltonian(A: np.array, alpha: float = 1.0) -> tuple[np.array, float]:
    num_nodes = len(A)
    m = np.sum(A) / 2
    k = A.sum(axis=1)            

    B = (A - alpha * np.outer(k, k) / (2 * m)) / (4 * m)

    H = np.zeros((num_nodes, num_nodes))

    constant = np.diag(B).sum()

    H += B
    H = 2 * np.triu(H)                       # keep upper triangle (standard QUBO form)
    H -= np.diag(np.diag(H))                   # ensure diagonal is zero

    return H, constant

def modularity_spin_calc(A: np.array, alpha: float, s: np.array) -> float: 
    num_nodes = len(A)
    m = np.sum(A) / 2  # Since A is symmetric, we divide by 2 to get the actual sum of edges
    k = A.sum(axis=1)

    B = (A - alpha * np.outer(k, k) / (2 * m)) / (4 * m)

    modularity = B.sum()
    for i in range(num_nodes):
        for j in range(num_nodes):
            modularity += B[i, j]*(s[i] * s[j])
            
    return modularity

def QUBO_k2_modularity(A: np.array, alpha: float, x: np.array) -> float:
    num_nodes = len(A)
    m = np.sum(A) / 2  
    k = A.sum(axis=1)                        
    
    B = (A - alpha * np.outer(k, k) / (2 * m)) / (2 * m)

    modularity = B.sum()
    for i in range(num_nodes):
        for j in range(num_nodes):
            modularity += B[i, j] * (2*x[i] * x[j])
        modularity -= 2*B.sum(axis=1)[i] * x[i]

    return modularity
# ------------------------------------------------------------------------------

graph_path = "graphs/XXSgraph.csv"
A = construct_graph_from_csv(graph_path)
network_graph = nx.from_numpy_array(A)
N_QUBITS = A.shape[0]

H, const = ising_hamiltonian_k2_modularity(A, alpha=1.0)
Q, const_Q = qubo_k2_modularity(A, alpha=1.0)
H_C = pauli_z_hamiltonian(A, alpha=1.0)

np.set_printoptions(precision=4, suppress=True)
cost_diag = np.real(np.diag(H_C)) 
print(cost_diag)

print("Brute-force optimal k-cluster modularity Q calculation: \n")
print("-----------------------------\n")

best_val = -np.inf
best_config = None

prods = np.array(list(product(range(2), repeat=N_QUBITS)))

sols = []

for i, x in enumerate(prods):
    z_full = np.array([2*x_i - 1 for x_i in x])  # Map {0,1} to {-1,1}

    obj_val = modularity_calc(A, 1.0, x=x)
    networkx_modularity = nx.algorithms.community.quality.modularity(network_graph, [np.where(x == 0)[0].tolist(), np.where(x == 1)[0].tolist()])
    H_modularity = z_full @ H @ z_full + const   # Calculate modularity using the Hamiltonian matrix
    Q_mod = x @ Q @ x + const_Q           # Calculate modularity using the QUBO matrix with x
    
    # Pauli Z Hamiltonian modularity
    H_C_modularity = bitstring_energy(H_C, x, N_QUBITS)

    # print(f"State: {x}  Mod: {obj_val:.4f}     Netx: {networkx_modularity:.4f}    H: {H_modularity:.4f}   Q: {Q_mod:.4f}     H_C: {H_C_modularity:.4f}")

    # Check if all the modularity calculations match
    sols.append(obj_val)
    assert np.isclose(obj_val, networkx_modularity), f"Modularity mismatch for state {x}"
    assert np.isclose(obj_val, H_modularity), f"Hamiltonian modularity mismatch for state {x}"
    assert np.isclose(obj_val, Q_mod), f"QUBO modularity mismatch for state {x}"
    assert np.isclose(obj_val, H_C_modularity), f"Pauli Z Hamiltonian modularity mismatch for state {x}"

# compare cost_diag with sols
for value in cost_diag:
    count_sol = len(np.where(np.isclose(sols, value, atol=1e-4))[0])
    count_dia = len(np.where(np.isclose(cost_diag, value, atol=1e-4))[0])
    assert np.isclose(count_sol, count_dia), f"Mismatch in counts for value {value:.4f}"

# Test modularity for different alpha values: 
good_sol = np.array([0, 0, 1, 1, 1])
bad_sol = np.array([0, 1, 0, 0, 1])
for alpha in [1, 1.5, 2, 2.5, 3]: 
    mod_good = modularity_calc(A, alpha, x=good_sol)
    mod_bad = modularity_calc(A, alpha, x=bad_sol)
    print(f"Alpha: {alpha:.1f}  Good Modularity: {mod_good:.4f}   Bad Modularity: {mod_bad:.4f}    Diff: {mod_good - mod_bad:.4f}")