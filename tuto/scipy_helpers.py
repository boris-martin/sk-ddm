import numpy as np
import scipy.sparse as sp


def restriction_matrix(n, idx):
    """
    Build a restriction matrix R such that R @ u == u[idx].

    n   : length of the full vector
    idx : ordered, unique list/array of indices to keep
    """

    idx = np.asarray(idx)

    # Safety checks avoid silent, hard-to-debug errors
    if idx.ndim != 1:
        raise ValueError("idx must be 1D.")
    if np.any(idx < 0) or np.any(idx >= n):
        raise IndexError("Index out of bounds.")
    if len(np.unique(idx)) != len(idx):
        raise ValueError("Expected unique indices.")
    
    m = len(idx)                      # size of reduced vector
    data = np.ones(m)
    rows = np.arange(m)
    cols = idx                        # selects u[idx[i]]

    return sp.csr_matrix((data, (rows, cols)), shape=(m, n))

def bmat(list_of_list):
    return sp.bmat(list_of_list, format='csr')


def build_swap(g, g_ij_offsets, ndom, total_g_size):
    total_size = total_g_size
    row_indices = []
    col_indices = []
    data = []
    print(g_ij_offsets)

    for idom in range(1, ndom + 1):
        for j in g[idom - 1].keys():
            offset_ij = g_ij_offsets[(idom, j)]
            offset_ji = g_ij_offsets[(j, idom)]
            size_ij = g[idom - 1][j][1].shape[0]
            for k in range(size_ij):
                row_indices.append(offset_ij + k)
                col_indices.append(offset_ji + k)
                data.append(1)
    return sp.csr_matrix((data, (row_indices, col_indices)), shape=(total_size, total_size))