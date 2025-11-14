# ddm_utils.py
import numpy as np

def build_offsets_and_total_size(g, ndom: int):
    """
    Build:
      - offsets[(idom, j)] : starting index of interface (idom, j) in global g
      - istart[idom]       : starting index for subdomain idom in global g-block
      - total_g_size       : total length of the global g vector

    Assumes each g[idom-1] is a dict j -> (fs, proj),
    and uses sorted(j) to define a consistent ordering.
    """
    offsets = {}
    istart = []
    counter = 0
    for idom in range(1, ndom + 1):
        istart.append(counter)
        gi = g[idom - 1]
        for j in sorted(gi):
            _, proj = gi[j]
            offsets[(idom, j)] = counter
            counter += proj.shape[0]
    istart.append(counter)
    total_g_size = counter
    return offsets, istart, total_g_size


def build_full_rhs(phys_b_list):
    """
    Concatenate per-subdomain physical RHS contributions into a single global RHS.
    """
    return np.concatenate(phys_b_list)
