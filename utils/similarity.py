import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity_from_logits_cos(probs):
    return cosine_similarity(probs)

def compute_normalized_aggregate(probs, edge_index, num_nodes):
    from scipy.sparse import csr_matrix, eye
    row, col = edge_index
    data = np.ones(len(row))
    A = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    A = A + eye(num_nodes, format='csr')
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv = np.zeros_like(deg)
    deg_inv[deg > 0] = 1.0 / deg[deg > 0]
    from scipy.sparse import csr_matrix as sp_csr
    D_inv = sp_csr(np.diag(deg_inv))
    D_inv_A = D_inv.dot(A)
    N = D_inv_A.dot(probs)
    sim_mat = cosine_similarity(N)
    return sim_mat, N
