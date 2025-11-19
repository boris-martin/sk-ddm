#!/usr/bin/env python3
import matplotlib.pyplot as plt
import mesh_helpers
import numpy as np
import plane_wave
import scipy.sparse.linalg
import skfem
from mesh_helpers import create_square, find_entities_on_domain
from skfem import BilinearForm, ElementTriP1, LinearForm, MeshTri
from skfem.helpers import dot, grad
from skfem.visuals.matplotlib import plot

create_square(0.01, 17)


@BilinearForm
def helmholtz(u, v, w):
    k = w["k"]
    return dot(grad(u), grad(v)) - k**2 * u * v


@BilinearForm(facet=True, dtype=np.complex128)
def absorbing(u, v, w):
    k = w["k"]
    return np.complex128(-1j * k * u * v)


@LinearForm
def source(v, w):
    return v * w["x"][0]


target_part = 4
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

mesh = MeshTri(skfem_points, skfem_elements)
facets_dict = mesh_helpers.buildFacetDict(mesh)


gamma_facets = mesh_helpers.findFacetsGamma(gamma_tags, gmshToSK, facets_dict)
sigma_facets = mesh_helpers.findFullSigma(sigma_tags, gmshToSK, facets_dict)
# Concat gamma and sigma
all_facets = np.concatenate([gamma_facets] + [v for k, v in sigma_facets.items()]).astype(np.int64)
print("Number of facets for absorbing BC:", len(all_facets))
basis = skfem.Basis(mesh, ElementTriP1())
print(all_facets)
facet_basis = skfem.FacetBasis(mesh, ElementTriP1(), facets=all_facets)
print(facet_basis)
wavelength = 0.10
k = 2.0 * np.pi / wavelength
A_bnd = skfem.asm(absorbing, facet_basis, k=k).astype(np.complex128)
A_vol = skfem.asm(helmholtz, basis, k=k).astype(np.complex128)
print(A_bnd)
A_vol += A_bnd


# plot(mesh, np.arange(mesh.nelements), colorbar=True)
# plt.title("Element indices")
# plt.show()


b = np.zeros(mesh.nvertices, dtype=np.complex128)
b[center_idx] = 1.0 + 0.0j
theta = 0.0
b = skfem.asm(plane_wave.plane_wave, facet_basis, k=k, theta=theta).astype(np.complex128)
x = scipy.sparse.linalg.spsolve(A_vol, b)
expected_x = plane_wave.plane_wave_value(skfem_points, k, theta)
error = np.linalg.norm(x - expected_x) / np.linalg.norm(expected_x)
print("Relative error compared to analytical solution:", error * 100, "%")
plot(mesh, np.real(x), shading="gouraud", colorbar=True)
plt.title("Solution")
plt.show()
