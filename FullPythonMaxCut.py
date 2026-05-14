import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import networkx as nx
from numba import njit

# ------------------------------------------------------------------------------
# Create a graph
# ------------------------------------------------------------------------------
def add_edges(graph, edges):
    for edge in edges:
        graph.add_edge(edge[0], edge[1])

network_graph = nx.Graph()

edges = [(0, 1), (1, 2), (2, 3), (3, 0)]   # 4-cycle C4
add_edges(network_graph, edges)
A = np.array(nx.adjacency_matrix(network_graph).todense())

N_QUBITS = len(network_graph.nodes)
DIM = 2 ** N_QUBITS

# Pauli matrices
I2 = np.eye(2, dtype=complex)
Zp = np.array([[1, 0], [0, -1]], dtype=complex)

def kron_op(op, qubit, n_qubits):
    """Embed single-qubit op on 'qubit' in n-qubit space (LSB = qubit 0)."""
    ops = [I2] * n_qubits
    ops[n_qubits - 1 - qubit] = op   # qubit 0 is the rightmost tensor factor
    result = ops[0]
    for o in ops[1:]:
        result = np.kron(result, o)
    return result


def kron_two(op_i, qi, op_j, qj, n_qubits):
    """Embed two single-qubit operators on different qubits."""
    ops = [I2] * n_qubits
    ops[n_qubits - 1 - qi] = op_i
    ops[n_qubits - 1 - qj] = op_j
    result = ops[0]
    for o in ops[1:]:
        result = np.kron(result, o)
    return result


# ── Cost Hamiltonian H_C = sum_{(i,j)} (I - Z_i Z_j) / 2 ─────────────────
H_C = np.zeros((DIM, DIM), dtype=complex)
Q = np.zeros((N_QUBITS, N_QUBITS))
H = np.zeros((N_QUBITS, N_QUBITS))

for i, j in edges:
    ZZ   = kron_two(Zp, i, Zp, j, N_QUBITS)
    H_C += (np.eye(DIM, dtype=complex) - ZZ) / 2
    Q[i, i] += 1  
    Q[j, j] += 1
    Q[i, j] -= 2 
    H[i, j] -= 0.5

cost_diag = np.diag(H_C).real      # H_C is diagonal

print("QUBO Matrix Q:")
print(Q)
print(H)

print("Eigenvalues of H:")
eigenvals = np.linalg.eigvals(H)
for val in eigenvals:
    print(f"  {val:.4f}")

print("Cost Hamiltonian diagonal (cost for each bitstring):")
for i, val in enumerate(cost_diag):
    print(f"  Bitstring {i:0{N_QUBITS}b}: {val:.4f}")

def apply_cost_unitary(state: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-1j * gamma * cost_diag) * state  # O(n) elementwise
aa
# ------------------------------------------------------------------------------
# Apply the mixer unitary (Rx rotations) to the state
# ------------------------------------------------------------------------------
def apply_mixer(state: np.ndarray, beta: float, N_QUBITS: int) -> np.ndarray:
    c, s = np.cos(beta/2), np.sin(beta/2)
    mat = np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)  # Rx(2β) up to convention
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

for p in [1, 2, 3]:
    best_val = np.inf
    best_res = None
    n_restarts = 100 if p == 1 else 300

    for _ in range(n_restarts):
        g0 = np.random.uniform(0, np.pi,    p)
        b0 = np.random.uniform(0, np.pi/2,  p)
        x0 = np.concatenate([g0, b0])
        res = minimize(lambda params: -expectation_value(p, params, N_QUBITS), x0,
                       method='COBYLA', options={'maxiter': 2000, 'rhobeg': 0.5})
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
          



