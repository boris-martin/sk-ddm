#!/usr/bin/env python3
import numpy as np
import skfem
from skfem import MeshTri, FacetBasis, ElementTriP1, LinearForm, BilinearForm
from skfem.helpers import grad, dot
from skfem.visuals.matplotlib import plot
import matplotlib.pyplot as plt
import gmsh



gmsh.initialize()
gmsh.model.add("unit-square")
lc = 0.02
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
gmsh.model.mesh.partition(2)
gmsh.write("square.msh")

@BilinearForm
def helmholtz(u, v, w):
    k = w['k']
    return dot(grad(u), grad(v)) - k**2 * u * v
@BilinearForm(facet=True, dtype=np.complex128)
def absorbing(u, v, w):
    k = w['k']
    return np.complex128(1j * k * u * v)
@LinearForm
def source(v, w):
    return v * w['x'][0]


target_part = 1
omega_tag = -1
gamma_tags = []
sigma_tags = dict() # j to tag

for dim, tag in  gmsh.model.get_entities(-1):
    print(f"Dimension: {dim}, Tag: {tag}")
    pdim, ptag = gmsh.model.getParent(dim, tag)
    partitions = gmsh.model.getPartitions(dim, tag)
    if target_part not in partitions:
        continue
    if dim == 2 and ptag == 1:
        omega_tag = tag
    elif dim == 1:
        if pdim == 2:
            assert(len(partitions) == 2)
            other_part = partitions[0] if partitions[1] == target_part else partitions[1]
            sigma_tags[other_part] = tag
        elif pdim == 1:
            gamma_tags.append(tag)

print(f"Omega tag: {omega_tag}")
print(f"Gamma tags: {gamma_tags}")
print(f"Sigma tags: {sigma_tags}")


def buildNodes(surface_tag):
    gmshNodeTagToCoords = dict()
    gmshNodeTagToSKFemIndex = dict()
    ntags, ncoords, _ = gmsh.model.mesh.get_nodes(2, surface_tag, includeBoundary=True)
    print(f"Number of nodes: {len(ntags)} and coordinates shape: {ncoords.shape}")
    counter = 0
    # Iterate over zip of tags and 3 coords
    for nodeTag, coords in zip(ntags, ncoords.reshape(-1, 3)):
        gmshNodeTagToCoords[nodeTag] = coords[:2]
        gmshNodeTagToSKFemIndex[nodeTag] = counter
        counter += 1
    assert(len(gmshNodeTagToSKFemIndex) == len(ntags))
    assert(len(gmshNodeTagToCoords) == len(ntags))
    # SKFEM node list
    skfem_points = np.zeros((2, len(ntags)))
    for gmshTag, skfemIndex in gmshNodeTagToSKFemIndex.items():
        skfem_points[:, skfemIndex] = gmshNodeTagToCoords[gmshTag]

    return skfem_points, gmshNodeTagToSKFemIndex

def buildTriangularMesh(surface_tag, gmshToSK):
    etypes, _, nodes = gmsh.model.mesh.get_elements(2, surface_tag)
    # We expect only triangles
    assert(len(etypes) == 1)
    assert(etypes[0] == 2)  # triangle
    triangles = nodes[0].reshape(-1, 3)
    skfem_elements = np.zeros(triangles.shape, dtype=int)
    for i in range(triangles.shape[0]):
        for j in range(3):
            gmshNodeTag = triangles[i, j]
            skfemIndex = gmshToSK[gmshNodeTag]
            skfem_elements[i, j] = skfemIndex
    return skfem_elements.T

skfem_points, gmshToSK = buildNodes(omega_tag)
skfem_elements = buildTriangularMesh(omega_tag, gmshToSK)
center_of_mass = np.mean(skfem_points, axis=1)
# Index to closest point to center of mass
distances = np.linalg.norm(skfem_points - center_of_mass[:, np.newaxis], axis=0)
center_idx = np.argmin(distances)
print(f"Center of mass: {center_of_mass}, closest node index: {center_idx}, coords: {skfem_points[:, center_idx]}")

mesh = MeshTri(skfem_points, skfem_elements)
facets = mesh.facets
facets_dict = dict()
for j in range(facets.shape[1]):
    assert(facets[0, j] < facets[1, j])
    facets_dict[(facets[0, j], facets[1, j])] = j

def findFacetsGamma(gammaTags):
    gamma_facets = []
    for tag in gammaTags:
        etypes, _, nodes = gmsh.model.mesh.get_elements(1, tag)
        assert(len(etypes) == 1)
        assert(etypes[0] == 1)  # line elements
        lines = nodes[0].reshape(-1, 2)
        for line in lines:
            n1, n2 = line
            sk_n1 = gmshToSK[n1]
            sk_n2 = gmshToSK[n2]
            if sk_n1 > sk_n2:
                sk_n1, sk_n2 = sk_n2, sk_n1
            facet_idx = facets_dict[(sk_n1, sk_n2)]
            gamma_facets.append(facet_idx)
    return np.array(gamma_facets)

def findFullSigma(sigma_tags):
    facets = []
    for _, tag in sigma_tags.items():
        etypes, _, nodes = gmsh.model.mesh.get_elements(1, tag)
        assert(len(etypes) == 1)
        assert(etypes[0] == 1)
        lines = nodes[0].reshape(-1, 2)
        for line in lines:
            n1, n2 = line
            sk_n1 = gmshToSK[n1]
            sk_n2 = gmshToSK[n2]
            if sk_n1 > sk_n2:
                sk_n1, sk_n2 = sk_n2, sk_n1
            facet_idx = facets_dict[(sk_n1, sk_n2)]
            facets.append(facet_idx)
    return np.array(facets)



gamma_facets = findFacetsGamma(gamma_tags)
sigma_facets = findFullSigma(sigma_tags)
# Concat gamma and sigma
all_facets = np.concatenate((gamma_facets, sigma_facets))
print("Number of facets for absorbing BC:", len(all_facets))
basis = skfem.Basis(mesh, ElementTriP1())
print(all_facets)
facet_basis = skfem.FacetBasis(mesh, ElementTriP1(), facets=all_facets)
print(facet_basis)
wavelength = 0.3
k = 2.0 * np.pi / wavelength
A_bnd = skfem.asm(absorbing, facet_basis, k=k).astype(np.complex128)
A_vol = skfem.asm(helmholtz, basis, k=k).astype(np.complex128)
print(A_bnd)
A_vol += A_bnd


#plot(mesh, np.arange(mesh.nelements), colorbar=True)
#plt.title("Element indices")
#plt.show()

import scipy.sparse.linalg
b = np.zeros(mesh.nvertices, dtype=np.complex128)
b[center_idx] = 1.0 + 0.0j
plot(mesh, np.real(scipy.sparse.linalg.spsolve(A_vol, b)), shading='gouraud', colorbar=True)
plt.title("Solution")
plt.show()