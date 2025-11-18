import numpy as np
import scipy.sparse as sp


def restriction_matrix(n, idx, SKToGmsh, gmshToSK):
    gmsh_nodes = sorted([SKToGmsh[i] for i in idx])
    sk_nodes = [gmshToSK[gmsh_node] for gmsh_node in gmsh_nodes]

    sk_to_g_idx = dict()

    m = len(sk_nodes)
    # Column is a sk node index, row is a location in the gmsh nodes
    rows = [i for i in range(m)]
    cols = []
    data = [1.0 for _ in range(m)]
    for i in range(m):
        cols.append(sk_nodes[i])
        sk_to_g_idx[sk_nodes[i]] = i

    #print("Shape of restrict", m, n)
    return sp.csr_matrix((data, (rows, cols)), shape=(m, n)), sk_to_g_idx

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