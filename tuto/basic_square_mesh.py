#!/usr/bin/env python3
import numpy as np
import skfem
from skfem import MeshTri, FacetBasis, ElementTriP1, LinearForm, BilinearForm
from skfem.visuals.matplotlib import plot
import matplotlib.pyplot as plt
import gmsh



gmsh.initialize()
gmsh.model.add("unit-square")
lc = 0.05
p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
p2 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, lc)
p3 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, lc)
p4 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0, lc)
l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)
cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
s1 = gmsh.model.geo.addPlaneSurface([cl])
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write("square.msh")


ntags, ncoords, _ = gmsh.model.mesh.get_nodes(2, s1, includeBoundary=True)
print(f"Number of nodes: {len(ntags)} and coordinates shape: {ncoords.shape}")

gmshNodeTagToCoords = dict()
gmshNodeTagToSKFemIndex = dict()

counter = 0
# Iterate over zip of tags and 3 coords
for nodeTag, coords in zip(ntags, ncoords.reshape(-1, 3)):
    gmshNodeTagToCoords[nodeTag] = coords[:2]
    gmshNodeTagToSKFemIndex[nodeTag] = counter
    counter += 1
assert(len(gmshNodeTagToSKFemIndex) == len(ntags))
assert(len(gmshNodeTagToCoords) == len(ntags))
# Build SKFEM vertex array
skfem_points = np.zeros((2, len(ntags)))
for gmshTag, skfemIndex in gmshNodeTagToSKFemIndex.items():
    skfem_points[:, skfemIndex] = gmshNodeTagToCoords[gmshTag]

types, elem, nodes = gmsh.model.mesh.get_elements(2, s1)
assert types[0] == 2  # triangle
assert len(types) == 1  # only triangles
num_elements = len(nodes[0]) // 3
skfem_elements = np.zeros((3, num_elements), dtype=int)
for eidx in range(num_elements):
    for lidx in range(3):
        gmshNodeTag = nodes[0][3 * eidx + lidx]
        skfemIndex = gmshNodeTagToSKFemIndex[gmshNodeTag]
        skfem_elements[lidx, eidx] = skfemIndex

mesh = MeshTri(skfem_points, skfem_elements)
gmsh.finalize()

# Element wise plot
elem_data = np.linspace(1, skfem_elements.shape[1], skfem_elements.shape[1])
print(elem_data)
plot(mesh, elem_data, colorbar=True)
plt.show()

node_data = np.linspace(1, skfem_points.shape[1], skfem_points.shape[1])
print(node_data)
plot(mesh, node_data, shading='gouraud', colorbar=True)
plt.show()


# Assemble a mass matrix and linear source
basis = skfem.Basis(mesh, ElementTriP1())
@BilinearForm
def mass(u, v, _):
    return u * v
@LinearForm
def source(v, w):
    return v * w['x'][0]

M = skfem.asm(mass, basis)
b = skfem.asm(source, basis)
# Solve M x = b
from scipy.sparse.linalg import spsolve
x = spsolve(M.tocsr(), b)
# Plot solution
plot(basis, x, shading='gouraud', colorbar=True)
plt.show()