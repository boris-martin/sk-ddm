#!/usr/bin/env python3
import numpy as np
from numpy import conjugate
import skfem
from skfem import MeshTri, FacetBasis, ElementTriP1, LinearForm, BilinearForm
from skfem.helpers import grad, dot
from skfem.visuals.matplotlib import plot
import matplotlib.pyplot as plt
import gmsh


from mesh_helpers import create_square, find_entities_on_domain
import mesh_helpers
import plane_wave
import scipy_helpers

ndom = 2
g = [] # List (i-dom) of (j, (g_ij, vertexSet)} with g_ij a function space and the set of DOFs)

create_square(.2, ndom)

@BilinearForm
def helmholtz(u, v, w):
    k = w['k']
    return dot(grad(u), grad(conjugate(v))) - k**2 * u * conjugate(v)
@BilinearForm(facet=True, dtype=np.complex128)
def mass_bnd(u, v, w):
    return np.complex128(u * conjugate(v))
@BilinearForm(facet=True, dtype=np.complex128)
def absorbing(u, v, w):
    k = w['k']
    return np.complex128(-1j * k * u * conjugate(v))
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
        print(projector)
        M = skfem.asm(mass_bnd, function_space)
        print((projector@M@projector.T).toarray())
        gi[j] = (function_space, nodeSet)
        
    
print(g)