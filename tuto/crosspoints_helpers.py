import numpy as np

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
