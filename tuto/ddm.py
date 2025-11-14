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

from collections import defaultdict

ndom = 4
g = [] # List (i-dom) of (j, (g_ij, vertexSet)} with g_ij a function space and the set of DOFs)
local_mats = [] # List (i-dom) of local matrices (u + output g as in gmshDDM)
local_rhs_mats = [] # List (i-dom) of map from local gijs to RHS of the local problem
local_physical_sources = []
local_solves = []
all_g_masses = []
phys_b = []
meshes = []
theta = np.pi / 4

cross_points_gmsh_tags = set()

wavelength = 0.3
k = 2 * np.pi / wavelength

create_square(.05, ndom)

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
    return np.complex128(-(-0.0 + 1j) * k * u * conjugate(v))

# GPT fix
def make_local_solve(gi, sizes, local_rhs_mat, local_mat, nvertices):
    # Precompute for asserts and speed
    actual_length = sum(proj.shape[0] for (j, (_, proj)) in gi.items())
    total_size = sum(sizes)

    def local_solve(gloc):
        # Basic sanity checks; useful while debugging
        assert gloc.shape[0] == actual_length, (
            f"Wrong length in local solve: expected {actual_length}, got {gloc.shape[0]}"
        )
        rhs = local_rhs_mat @ gloc
        assert rhs.shape[0] == total_size, (
            f"RHS size mismatch: expected {total_size}, got {rhs.shape[0]}"
        )
        u_and_g = scipy.sparse.linalg.spsolve(local_mat, rhs)
        # Return only the interface part
        return u_and_g[nvertices:]

    return local_solve


if ndom == 1:
    # Mono domain solve
    omega_tag, gamma_tags, sigma_tags = find_entities_on_domain(1)
    skfem_points, gmshToSK = mesh_helpers.buildNodes(omega_tag)
    skfem_elements = mesh_helpers.buildTriangleSet(omega_tag, gmshToSK)
    mesh = MeshTri(skfem_points, skfem_elements)
    basis = skfem.Basis(mesh, ElementTriP1())
    fbasis = skfem.FacetBasis(mesh, ElementTriP1())  # all exterior facets
    A = skfem.asm(helmholtz, basis, k=k) + skfem.asm(absorbing, fbasis, k=k)
    gamma_facets = mesh_helpers.findFacetsGamma(gamma_tags, gmshToSK, mesh_helpers.buildFacetDict(mesh))
    if len(gamma_facets) == 0:
        local_source = np.zeros(mesh.nvertices, dtype=np.complex128)
    else:
        gamma_basis = skfem.FacetBasis(mesh, ElementTriP1(), facets=gamma_facets)
        local_source = skfem.asm(plane_wave.plane_wave, gamma_basis, k=k, theta=theta).astype(np.complex128)
    u = scipy.sparse.linalg.spsolve(A, local_source)
    print("Mono domain solution norm: ", np.linalg.norm(u))
    plot(mesh, np.real(u), shading='gouraud')
    plt.show()
    exit()



for idom in range(1, ndom+1):
    omega_tag, gamma_tags, sigma_tags = find_entities_on_domain(idom)
    g.append(dict())
    gi = g[-1]
    print(f"Domain {idom}:")
    #print(f"-- Omega tag: {omega_tag}")
    #print(f"-- Gamma tags: {gamma_tags}")
    #print(f"-- Sigma tags: {sigma_tags}")

    # Nodes have different SK-fem indices in each subdomain, mapping needed!
    skfem_points, gmshToSK = mesh_helpers.buildNodes(omega_tag)
    SKToGmsh = mesh_helpers.reverseNodeDict(gmshToSK)
    skfem_elements = mesh_helpers.buildTriangleSet(omega_tag, gmshToSK)
    mesh = MeshTri(skfem_points, skfem_elements)
    meshes.append(mesh)
    facets_dict = mesh_helpers.buildFacetDict(mesh)
    all_sigma_facets = mesh_helpers.findFullSigma(sigma_tags, gmshToSK, facets_dict)

    nodesToJset = defaultdict(set)


    for j, sigma_tag in sigma_tags.items():
        nodeSet = set()
        if j not in all_sigma_facets:
            raise ValueError(f"Domain {idom}: Sigma tag {sigma_tag} not found in facets.")
        for edge in all_sigma_facets[j]:
            for node in mesh.facets[:, edge]:
                nodeSet.update({node})
        function_space = skfem.FacetBasis(mesh, ElementTriP1(), facets=all_sigma_facets[j])
        for node in nodeSet:
            nodesToJset[SKToGmsh[node]].add(j)
        projector = scipy_helpers.restriction_matrix(mesh.nvertices, list(nodeSet), SKToGmsh, gmshToSK)
        gi[j] = (function_space, projector)

    sizes = [int(mesh.nvertices)] + [proj.shape[0] for (j, (fs, proj)) in gi.items()]
    print("Local sizes (u and each g): ", sizes)
    num_interface_fields = len(gi)
    gamma_facets = mesh_helpers.findFacetsGamma(gamma_tags, gmshToSK, facets_dict)

    mats = [[None for _ in range(num_interface_fields + 1)] for _ in range(num_interface_fields + 1)]
    mats[0][0] = skfem.asm(helmholtz, skfem.Basis(mesh, ElementTriP1()), k=k) + \
                    skfem.asm(absorbing, FacetBasis(mesh, ElementTriP1()), k=k) # Should be ALL bnd facets
    for idx_j, j in enumerate(sorted(gi.keys())):
        fs_j, pj = gi[j]
        print("IDX J, J :", idx_j, j)
        mass = pj @ skfem.asm(mass_bnd, fs_j) @ pj.T
        all_g_masses.append(mass)
        mats[idx_j + 1][idx_j + 1] = mass
        mat_s = -2 * pj @ skfem.asm(transmission, fs_j, k=k)# @ pj.T # Map from u to g
        mats[idx_j + 1][0] = mat_s

    
    if len(gamma_facets) == 0:
        local_source = np.zeros(mesh.nvertices, dtype=np.complex128)
    else:
        gamma_basis = skfem.FacetBasis(mesh, ElementTriP1(), facets=gamma_facets)
        local_source = skfem.asm(plane_wave.plane_wave, gamma_basis, k=k, theta=theta).astype(np.complex128)

    full_rhs = np.zeros(sum(sizes), dtype=np.complex128)
    full_rhs[0:mesh.nvertices] = local_source
    local_physical_sources.append(np.array(full_rhs))
    local_mats.append(scipy_helpers.bmat(mats))
    b_substructured = scipy.sparse.linalg.spsolve(local_mats[-1], full_rhs)[mesh.nvertices:]
    phys_b.append(b_substructured)
    # Now, mats goes from the gs to the local solution including u
    # So, structure is num_interface_fields to num_interface_fields + 1
    mats = [[None for _ in range(num_interface_fields)] for _ in range(num_interface_fields + 1)]
    for idx_j, j in enumerate(sorted(gi.keys())):
        P = gi[j][1]
        volumetric_mass_j = skfem.asm(mass_bnd, gi[j][0], k=k) @ P.T
        print("Idx j", idx_j)
        print("Vol mass shape", volumetric_mass_j.shape)
        mats[0][idx_j] = volumetric_mass_j
        print("P times vol mass shape", (P @ volumetric_mass_j).shape)
        mats[idx_j + 1][idx_j] = -P @ volumetric_mass_j
    local_rhs_mats.append(scipy_helpers.bmat(mats))

    this_rhs_mat = local_rhs_mats[-1]
    print("RHS generator shape", local_rhs_mats[-1].shape)

    
    local_rhs = local_rhs_mats[-1]
    local_mat = local_mats[-1]
    nvertices = mesh.nvertices

    num_rows = sum(sizes) - nvertices
    num_cols = sum(proj.shape[0] for (j, (_, proj)) in gi.items())

    local_solve = make_local_solve(
        gi=dict(gi),                  # shallow copy for safety
        sizes=list(sizes),
        local_rhs_mat=local_rhs,
        local_mat=local_mat,
        nvertices=nvertices,
    )

    linearOp = scipy.sparse.linalg.LinearOperator(
        (num_rows, num_cols),
        matvec=local_solve,
        dtype=np.complex128,
    )
    local_solves.append(linearOp)
    """
    def local_solve(gloc):
        actual_length = sum([proj.shape[0] for (j, (fs, proj)) in gi.items()])
        assert(len(gloc) == actual_length), "Wrong length in local solve: expected {}, got {}".format(actual_length, len(gloc))
        rhs = local_rhs_mats[idom-1] @ gloc
        assert(rhs.shape[0] == sum(sizes))
        u_and_g = scipy.sparse.linalg.spsolve(local_mats[idom-1], rhs)
        return u_and_g[mesh.nvertices:]

    num_rows = sum(sizes) - mesh.nvertices
    num_cols = sum([proj.shape[0] for (j, (fs, proj)) in gi.items()])
    print("Shape: ", (num_rows, num_cols))
    linearOp = scipy.sparse.linalg.LinearOperator((num_rows, num_cols), matvec=local_solve)
    local_solves.append(linearOp)
    """


full_mass = scipy.sparse.block_diag(all_g_masses)
print("Full mass shape: ", full_mass.shape)




offsets = dict()
istart = []
counter = 0
for idom in range(1, ndom+1):
    istart.append(counter)
    gi = g[idom - 1]
    for j in sorted(gi):
        fs, proj = gi[j]
        offsets[(idom, j)] = counter
        counter += proj.shape[0]
istart.append(counter)    

total_g_size = sum([sum([proj.shape[0] for (j, (fs, proj)) in gi.items()]) for gi in g])
print("Total g size: ", total_g_size)
print(offsets)

rhs = np.concatenate(phys_b)
print("Global rhs size: ", rhs.shape)
#print(rhs)

swap  = scipy_helpers.build_swap(g, offsets, ndom, total_g_size)

def apply_local(g):
    assert(g.shape[0] == total_g_size)
    g_solved = np.zeros_like(g, dtype=np.complex128)
    for idom in range(1, ndom+1):
        working_range_start = istart[idom - 1]
        working_range_end = istart[idom]
        gloc = g[working_range_start:working_range_end]
        gloc_solved = local_solves[idom - 1].matvec(gloc)
        g_solved[working_range_start:working_range_end] = gloc_solved
    return g_solved

def ddm_operator(g):
    g_swap = swap @ g
    g_solved = apply_local(g_swap)
    return g_solved

ddm_linop = scipy.sparse.linalg.LinearOperator((total_g_size, total_g_size), matvec=ddm_operator)
id_minus_ddm = scipy.sparse.linalg.LinearOperator((total_g_size, total_g_size), matvec=lambda x: x - ddm_operator(x))
x, info = scipy.sparse.linalg.gmres(id_minus_ddm, rhs, rtol=1e-6, callback=lambda r: print("GMRES residual: ", r))
print(x, info)
ddm_dense = ddm_linop @ np.eye(total_g_size, dtype=np.complex128)
x_dense = np.linalg.solve(np.eye(total_g_size) - ddm_dense, rhs)
print("Norm of RHS: ", np.linalg.norm(rhs))
print("Residual: ", np.linalg.norm(id_minus_ddm @ x_dense - rhs)/np.linalg.norm(rhs))

# Build full solution
if False:
    x = swap @ x
    for idom in range(1, ndom+1):
        working_range_start = istart[idom - 1]
        working_range_end = istart[idom]
        gloc = x[working_range_start:working_range_end]
        artifical_source = local_rhs_mats[idom-1] @ gloc
        physical_source = local_physical_sources[idom-1]
        mesh = meshes[idom-1]
        u_local = scipy.sparse.linalg.spsolve(local_mats[idom-1], artifical_source+physical_source)[0:mesh.nvertices]
        print(f"Domain {idom} local solution norm: ", np.linalg.norm(u_local))
        print("Shape and nvertices : ", u_local.shape, meshes[idom-1].nvertices)
        plot(meshes[idom-1], np.real(u_local[:mesh.nvertices]), shading='gouraud')
        plt.show()

from scipy.sparse.linalg import svds

#u, s, vt = svds(ddm_linop, k=6, which='SM')
#print("Singular vlaues: ", s)

from numpy.linalg import svd

u, s, vt = svd(np.eye(total_g_size)- ddm_dense)
print("Singular values: ", s[s < 1e-8])
ker = vt.T[:, s < 1e-8]
print("Approximate kernel dimension: ", ker.shape[1])
for i in range(ker.shape[1]):
    print("Kernel vector ", i, " norm: ", np.linalg.norm(ker[:, i]))
    #Filter near-zero entries for display
    vec = full_mass @ ker[:, i]
    # Make it have unit L-inf norm
    vec = vec / np.max(np.abs(vec))
    vec[np.abs(vec) < 1e-3] = 0.0
    # Filter both real and imag parts separately
    vec[np.abs(np.real(vec)) < 1e-3] = np.imag(vec[np.abs(np.real(vec)) < 1e-3]) * 1j
    vec[np.abs(np.imag(vec)) < 1e-3] = np.real(vec[np.abs(np.imag(vec)) < 1e-3])
    # Print real part if imaginary part is small
    norm_imag = np.linalg.norm(np.imag(vec))
    norm_real = np.linalg.norm(np.real(vec))
    if norm_imag < 1e-6:
        vec = np.real(vec)
    elif norm_real < 1e-6:
        vec = 1j * np.imag(vec)
    print(vec)



# Compute spectrum of ddm_operator
eigs = scipy.sparse.linalg.eigs(ddm_linop, k=total_g_size-2, which='LM')
eigs_from_dense = np.linalg.eigvals(np.eye(total_g_size) - ddm_dense)
# Plot eigenvalues
if False:
    plt.figure()
    plt.scatter(eigs_from_dense.real, eigs_from_dense.imag)
    plt.title("Eigenvalues of DDM operator")
    plt.xlabel("Real part")
    plt.ylabel("Imaginary part")
    # Plot unit circle for reference
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(theta)+1, np.sin(theta), 'r--', label='Unit Circle')
    plt.grid()
    plt.show()



for node, doms in nodesToJset.items():
    if len(doms) >= 2:
        cross_points_gmsh_tags.add(node)

print("Cross points Gmsh tags: ", cross_points_gmsh_tags)


gmsh_vertices = gmsh.model.get_entities(0)
for dim, tag in gmsh_vertices:
    parDim, parTag = gmsh.model.getParent(dim, tag)
    if parDim != 2:
        continue
    partitions = gmsh.model.getPartitions(dim, tag)
    print("Vertex tag ", tag, " partitions: ", partitions)
    if len(partitions) >= 2:
        print(f"Vertex tag {tag} is shared by partitions {partitions}")



print("Ker shape: ", ker.shape)
