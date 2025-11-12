#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poisson with complex arithmetic & inhomogeneous Robin BC using scikit-fem.
Manufactured solution: u(x, y) = x + i*y on Ω = (0,1)^2.

PDE:
    -Δu = f in Ω,  with  α u + β ∂u/∂n = g on ∂Ω,
where for the manufactured solution:
    f = -Δu = 0,
    ∂u/∂n = ∇u·n = 1 * n_x + (i) * n_y,
    g = α u + β ∂u/∂n.

We pick α=1, β=1 for simplicity (any β≠0 works).
"""
import numpy as np
import gmsh
import meshio
from pathlib import Path

from skfem import (
    MeshTri, Basis, FacetBasis, ElementTriP1, asm
)
from skfem.assembly import BilinearForm, LinearForm
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# -------------------------------
# 1) Mesh the unit square in Gmsh
# -------------------------------
gmsh.initialize()
gmsh.model.add("unit-square")

# Characteristic length (target edge size). Keep it modest for speed.
lc = 0.15

# Geometry: points
p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
p2 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, lc)
p3 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, lc)
p4 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0, lc)

# Lines & loop
l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)
cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
s1 = gmsh.model.geo.addPlaneSurface([cl])

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)

# Write to .msh (v4), then read with meshio -> scikit-fem
msh_path = Path("square.msh")
gmsh.write(str(msh_path))
gmsh.finalize()

meshio_mesh = meshio.read(msh_path)
# Make sure only triangles are passed to scikit-fem as cells
# (Gmsh may also include line elements for the boundary.)
triangle_cells = [c for c in meshio_mesh.cells if c.type in ("triangle", "tri")]
assert len(triangle_cells) >= 1, "No triangular cells found; check Gmsh setup."

meshio_tri = meshio.Mesh(
    points=meshio_mesh.points,
    cells=triangle_cells,
)

# Import into scikit-fem
from skfem.io.meshio import from_meshio
from skfem.helpers import dot, grad
mesh: MeshTri = from_meshio(meshio_tri)


# -------------------------------
# 2) FE space and PDE definition
# -------------------------------
elem = ElementTriP1()
basis = Basis(mesh, elem)
fbasis = FacetBasis(mesh, elem)  # all exterior facets

# Parameters for Robin BC
alpha = 1.0
beta = 1.0  # must be nonzero

# Manufactured solution and helpers
def u_exact_xy(x):
    """
    x: (2, npts) array of coordinates.
    returns u = x + i*y at those points.
    """
    return x[0] + 1j * x[1]

def dudn_w(w):
    """
    On facets: ∂u/∂n = ∇u · n = (1, i) · (n_x, n_y) = n_x + i*n_y.
    w.n has shape (2, nqp).
    """
    n = w.n
    return n[0] + 1j * n[1]

# Bilinear forms
@BilinearForm
def a(u, v, w):
    # ∫Ω ∇u · ∇v
    return dot( grad(u), grad(v) )

@BilinearForm
def a_robin(u, v, w):
    # ∫∂Ω (α/β) u v
    return (alpha / beta) * u * v

# RHS: f=0 in the volume for this manufactured solution
@LinearForm
def L_domain(v, w):
    return 0.0 * v  # explicit zero; keeps dtype consistent

@LinearForm
def L_robin(v, w):
    # ∫∂Ω (g/β) v, where g = α u + β ∂u/∂n
    x = w.x
    ue = u_exact_xy(x)
    g = alpha * ue + beta * dudn_w(w)
    return (g / beta) * v


# -------------------------------
# 3) Assemble & solve (complex)
# -------------------------------
A = asm(a, basis) + asm(a_robin, fbasis)
b = asm(L_domain, basis).astype(np.complex128) + asm(L_robin, fbasis).astype(np.complex128)

# Solve the complex linear system
uh = spsolve(A.tocsr().astype(np.complex128), b)

# -------------------------------
# 4) Error check (complex L2 norm)
# -------------------------------
# Use a mass matrix that correctly handles complex conjugation on the test function.
@BilinearForm
def mass(u, v, w):
    return u * np.conj(v)

M = asm(mass, basis).astype(np.complex128)

# Exact values at DOF locations
x_dofs = basis.doflocs  # shape (2, ndof)
u_ex_dofs = u_exact_xy(x_dofs)

err_vec = uh - u_ex_dofs
l2_err = np.sqrt(np.real(err_vec.conj() @ (M @ err_vec)))

# Report
ndof = uh.size
print(f"ndof = {ndof}")
print(f"L2 error ≈ {l2_err:.3e}")

# Optional: sanity checks at corners to catch sign/normal issues quickly
corners = np.array([[0., 1., 1., 0.],
                    [0., 0., 1., 1.]])  # (x;y) for (0,0),(1,0),(1,1),(0,1)
# Interpolate FE solution at these points (P1 -> nodal projection is enough for a quick check)
# For a robust check, build an Interpolator; here we just print the exact values.
print("Exact u at corners:", u_exact_xy(corners))


# Plot
from skfem.visuals.matplotlib import plot, plot3, plot_basis
plot(basis, np.real(uh), shading='gouraud', colorbar=True)
plt.show()
