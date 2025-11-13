#!/usr/bin/env python3
import numpy as np
from numpy import conjugate
import skfem
from skfem import MeshTri, FacetBasis, ElementTriP1, LinearForm, BilinearForm
from skfem.helpers import grad, dot
from skfem.visuals.matplotlib import plot
import matplotlib.pyplot as plt
import gmsh
import scipy.sparse.linalg

from mesh_helpers import create_square, find_entities_on_domain
import mesh_helpers
import plane_wave
import scipy_helpers

ndom = 6
g = [] # List (i-dom) of (j, (g_ij, vertexSet)} with g_ij a function space and the set of DOFs)
local_mats = [] # List (i-dom) of local matrices (u + output g as in gmshDDM)
phys_b = []
theta = np.pi / 4

wavelength = 0.4
k = 2 * np.pi / wavelength

create_square(.1, ndom)

@BilinearForm
def helmholtz(u, v, w):
    k = w['k']
    return dot(grad(u), conjugate(grad(v))) - k**2 * u * conjugate(v)
@BilinearForm(facet=True, dtype=np.complex128)
def mass_bnd(u, v, w):
    return np.complex128(u * conjugate(v))
@BilinearForm(facet=True, dtype=np.complex128)
def absorbing(u, v, w):
    k = w['k']
    return np.complex128(-1j * k * u * conjugate(v))
@BilinearForm(facet=True, dtype=np.complex128)
def transmission(u, v, w):
    k = w['k']
    return np.complex128(-2j * k * u * conjugate(v))
@LinearForm
def source(v, w):
    return conjugate(v) * w['x'][0]

for idom in range(1, ndom+1):
    omega_tag, gamma_tags, sigma_tags = find_entities_on_domain(idom)
    g.append(dict())
    gi = g[-1]
    print(f"Domain {idom}:")
    print(f"-- Omega tag: {omega_tag}")
    print(f"-- Gamma tags: {gamma_tags}")
    print(f"-- Sigma tags: {sigma_tags}")

    # Nodes have different SK-fem indices in each subdomain, mapping needed!
    skfem_points, gmshToSK = mesh_helpers.buildNodes(omega_tag)
    skfem_elements = mesh_helpers.buildTriangleSet(omega_tag, gmshToSK)
    mesh = MeshTri(skfem_points, skfem_elements)
    facets_dict = mesh_helpers.buildFacetDict(mesh)
    all_sigma_facets = mesh_helpers.findFullSigma(sigma_tags, gmshToSK, facets_dict)

    for j, sigma_tag in sigma_tags.items():
        nodeSet = set()
        if j not in all_sigma_facets:
            raise ValueError(f"Domain {idom}: Sigma tag {sigma_tag} not found in facets.")
        for edge in all_sigma_facets[j]:
            for node in mesh.facets[:, edge]:
                nodeSet.update({node})
        function_space = skfem.FacetBasis(mesh, ElementTriP1(), facets=all_sigma_facets[j])
        projector = scipy_helpers.restriction_matrix(mesh.nvertices, list(nodeSet))
        gi[j] = (function_space, projector)

    sizes = [mesh.nvertices] + [proj.shape[0] for (j, (fs, proj)) in gi.items()]
    print("Local sizes: ", sizes)
    num_interface_fields = len(gi)
    mats = [[None for _ in range(num_interface_fields + 1)] for _ in range(num_interface_fields + 1)]
    mats[0][0] = skfem.asm(helmholtz, skfem.Basis(mesh, ElementTriP1()), k=k) + \
                    skfem.asm(absorbing, FacetBasis(mesh, ElementTriP1()), k=k) # Should be ALL bnd facets
    for idx_j, (j, (fs_j, pj)) in enumerate(gi.items()):
        mass = pj @ skfem.asm(mass_bnd, fs_j) @ pj.T
        mats[idx_j + 1][idx_j + 1] = mass
        mat_s = pj @ skfem.asm(transmission, fs_j, k=k)# @ pj.T # Map from u to g
        mats[idx_j + 1][0] = mat_s
    print(scipy_helpers.bmat(mats).shape)

    gamma_facets = mesh_helpers.findFacetsGamma(gamma_tags, gmshToSK, facets_dict)
    
    if len(gamma_facets) == 0:
        local_source = np.zeros(mesh.nvertices, dtype=np.complex128)
    else:
        gamma_basis = skfem.FacetBasis(mesh, ElementTriP1(), facets=gamma_facets)
        local_source = skfem.asm(plane_wave.plane_wave, gamma_basis, k=k, theta=theta).astype(np.complex128)

    full_rhs = np.zeros(sum(sizes), dtype=np.complex128)
    full_rhs[0:mesh.nvertices] = local_source
    b_substructured = scipy.sparse.linalg.spsolve(scipy_helpers.bmat(mats), full_rhs)[mesh.nvertices:]
    phys_b.append(b_substructured)
    print("b_substructured size: ", b_substructured.shape)
    print(b_substructured)



offsets = dict()
counter = 0
for idom in range(1, ndom+1):
    gi = g[idom - 1]
    for j in sorted(gi):
        fs, proj = gi[j]
        offsets[(idom, j)] = counter
        counter += proj.shape[0]
    
total_g_size = sum([sum([proj.shape[0] for (j, (fs, proj)) in gi.items()]) for gi in g])
print("Total g size: ", total_g_size)
print(offsets)

rhs = np.concat(phys_b)
print("Global rhs size: ", rhs.shape)
print(rhs)