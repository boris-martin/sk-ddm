#!/usr/bin/env python3
import numpy as np
import skfem
from skfem import MeshTri, FacetBasis, ElementTriP1, LinearForm, BilinearForm
from skfem.helpers import grad, dot
from skfem.visuals.matplotlib import plot
import matplotlib.pyplot as plt
import gmsh


from mesh_helpers import create_square, find_entities_on_domain
import mesh_helpers

create_square(.02, 2)

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


target_part = 2
omega_tag, gamma_tags, sigma_tags = find_entities_on_domain(target_part)

print(f"Omega tag: {omega_tag}")
print(f"Gamma tags: {gamma_tags}")
print(f"Sigma tags: {sigma_tags}")


skfem_points, gmshToSK = mesh_helpers.buildNodes(omega_tag)
skfem_elements = mesh_helpers.buildTriangleSet(omega_tag, gmshToSK)

center_of_mass = np.mean(skfem_points, axis=1)
# Index to closest point to center of mass
distances = np.linalg.norm(skfem_points - center_of_mass[:, np.newaxis], axis=0)
center_idx = np.argmin(distances)
print(f"Center of mass: {center_of_mass}, closest node index: {center_idx}, coords: {skfem_points[:, center_idx]}")

mesh = MeshTri(skfem_points, skfem_elements)
facets_dict = mesh_helpers.buildFacetDict(mesh)



def findFullSigma(sigma_tags):
    facets = dict()
    for j, tag in sigma_tags.items():
        facets_local = []
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
            facets_local.append(facet_idx)
        facets.update({j: facets_local})
    return facets



gamma_facets = mesh_helpers.findFacetsGamma(gamma_tags, gmshToSK, facets_dict)
sigma_facets = findFullSigma(sigma_tags)
# Concat gamma and sigma
all_facets = np.concatenate([gamma_facets] + [v for k, v in sigma_facets.items()])
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