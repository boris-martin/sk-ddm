#!/usr/bin/env python3
import numpy as np
from skfem import MeshTri
from skfem.visuals.matplotlib import plot
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1) Hard-coded coordinates and connectivity
# ----------------------------------------------------------
# Node coordinates: shape (2, n_vertices)
p = np.array([
    [0.0, 1.0, 0.0],   # x-coordinates
    [0.0, 0.0, 1.0],   # y-coordinates
])

# Triangle connectivity: shape (3, n_elements)
# Each column lists vertex indices (0-based)
t = np.array([
    [0],  # vertex 0
    [1],  # vertex 1
    [2],  # vertex 2
])

# ----------------------------------------------------------
# 2) Build the MeshTri
# ----------------------------------------------------------
mesh = MeshTri(p, t)

# ----------------------------------------------------------
# 3) Inspect and plot
# ----------------------------------------------------------
print(mesh)  # prints MeshTri(1 elements, 3 vertices)

plot(mesh, np.array([1., 2, 3]), shading='gouraud', colorbar=True)
plt.axis("equal")
plt.title("Single Triangle Mesh")
plt.show()
