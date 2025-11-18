import numpy as np
import gmsh
from collections import defaultdict

def circular_neighbors_triplets(values: np.ndarray, x: int):
    """
    Given a sorted NumPy array of unique integers and a value x contained in it,
    return the previous and next elements in circular order.

    Raises:
        ValueError: if x is not contained inside values.
    """
    # Find all indices where values == x
    matches = np.where(values == x)[0]

    if matches.size == 0:
        raise ValueError(f"{x} not found in array")

    idx = matches[0]
    size = values.size

    prev_idx = (idx - 1) % size
    next_idx = (idx + 1) % size

    return values[prev_idx], values[next_idx]


def build_cycle_2d(partition_vertex_tag: int):
    partitions_using_vertex = set()
    partitions = gmsh.model.getPartitions(0, partition_vertex_tag)
    for p in partitions:
        partitions_using_vertex.add(p)
    partition_to_neighbor_set = defaultdict(set)
    entities_1d = gmsh.model.getEntities(1)
    for dim, tag in entities_1d:
        parentDim, _ = gmsh.model.getParent(dim, tag)
        if parentDim != 2:
            continue # Check it's an interface
        partitions = gmsh.model.getPartitions(dim, tag)
        if len(partitions) == 1:
            raise ValueError("Interface with only 1 partition ?")
        for p in partitions:
            for pp in partitions:
                if p != pp:
                    if pp in partitions_using_vertex and p in partitions_using_vertex:
                        partition_to_neighbor_set[p].add(pp)
                        partition_to_neighbor_set[pp].add(p)

    # Now, partition_to_neighbor_set contains for each partition the set of its neighbors
    for p, neighbors in partition_to_neighbor_set.items():
        if len(neighbors) != 2:
            raise ValueError(f"Partition {p} does not have exactly 2 neighbors, has {len(neighbors)}")
    
    # pick an arbitrary start
    start = next(iter(partition_to_neighbor_set))
    order = [start]

    # pick arbitrary next
    n1, n2 = tuple(partition_to_neighbor_set[start])
    order.append(n1)

    prev = start
    curr = n1

    for _ in range(2, len(partition_to_neighbor_set)):
        a, b = tuple(partition_to_neighbor_set[curr])
        nxt = b if a == prev else a
        order.append(nxt)
        prev, curr = curr, nxt

    # closure check
    last_neighbors = partition_to_neighbor_set[order[-1]]
    if start not in last_neighbors:
        raise RuntimeError("Failed to close cycle properly; graph not a single cycle?")

    # uniqueness check
    if len(set(order)) != len(order):
        raise RuntimeError("Cycle reconstruction contains duplicates — malformed structure")

    print("Final order:", order)
    return order


def cycle_find_prev_and_next(order: list, partition: int):
    idx = order.index(partition)
    size = len(order)
    prev_idx = (idx - 1) % size
    next_idx = (idx + 1) % size
    return order[prev_idx], order[next_idx]