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
