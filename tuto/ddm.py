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
import scipy.sparse
from scipy.sparse import csr_matrix

from mesh_helpers import create_square, find_entities_on_domain
import mesh_helpers
import plane_wave
import scipy_helpers
from ddm_utils import helmholtz, absorbing, mass_bnd, transmission, Subdomain

from collections import defaultdict
from typing import Dict, Tuple, Callable, List
from crosspoints_helpers import circular_neighbors_triplets, build_cycle_2d, cycle_find_prev_and_next

ndom = 20
local_physical_sources = []
local_solves = []
all_g_masses = []
all_s_masses = []
phys_b = []
theta = np.pi / 4
subdomains = []



cross_points_gmsh_tags: set[int] = set()

wavelength = 0.2
k = 2 * np.pi / wavelength

create_square(.02, ndom)
crosspoints_gmsh_node_tags = set()
crosspoints_gmsh_to_graph = {}

gmsh_vertices = gmsh.model.get_entities(0)
for dim, tag in gmsh_vertices:
    parDim, parTag = gmsh.model.getParent(dim, tag)
    if parDim != 2:
        continue
    partitions = gmsh.model.getPartitions(dim, tag)
    nodes = gmsh.model.mesh.getNodes(dim, tag)[0]

    cycle = build_cycle_2d(tag)
    assert len(nodes) == 1, "Vertex should have exactly one node"
    print("Vertex tag ", tag, " partitions: ", partitions)
    if len(partitions) >= 2:
        crosspoints_gmsh_node_tags.add(nodes[0])
        crosspoints_gmsh_to_graph.update({nodes[0]: cycle})

crosspoints_gmsh_to_kernel_column = {tag: idx for idx, tag in enumerate(crosspoints_gmsh_node_tags)}


# GPT fix
def make_local_solve(
        gi: Dict[int, Tuple],
        sizes: List[int],
        local_rhs_mat: scipy.sparse.spmatrix,
        local_mat: scipy.sparse.spmatrix,
        nvertices: int,
    ) -> Callable[[ np.ndarray ], np.ndarray]:
        """
        Build and return a closure that solves the local substructured DDM system.
        Uses explicit exceptions instead of assertions to ensure shape validation
        is preserved even under Python -O.
        """

        actual_length = sum(proj.shape[0] for (_, (_, proj, _)) in gi.items())
        total_size = sum(sizes)

        def local_solve(gloc: np.ndarray) -> np.ndarray:
            if gloc.shape[0] != actual_length:
                raise ValueError(
                    f"Wrong length in local solve: expected {actual_length}, got {gloc.shape[0]}"
                )

            rhs = local_rhs_mat @ gloc
            if rhs.shape[0] != total_size:
                raise ValueError(
                    f"RHS size mismatch: expected {total_size}, got {rhs.shape[0]}"
                )

            u_and_g = scipy.sparse.linalg.spsolve(local_mat, rhs)
            return u_and_g[nvertices:]  # only return interface part

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
    print(f"Domain {idom}:")

    subdomain = Subdomain(idom, omega_tag, gamma_tags, sigma_tags)
    subdomains.append(subdomain)
    gi = subdomain.gi
    mesh = subdomain.mesh
    nodesToJset = defaultdict(set)

    for j, sigma_tag in sigma_tags.items():
        nodeSet = set()
        if j not in subdomain.all_sigma_facets:
            raise ValueError(f"Domain {idom}: Sigma tag {sigma_tag} not found in facets.")
        for edge in subdomain.all_sigma_facets[j]:
            for node in mesh.facets[:, edge]:
                nodeSet.update({node})
        function_space = skfem.FacetBasis(mesh, ElementTriP1(), facets=subdomain.all_sigma_facets[j])
        for node in nodeSet:
            nodesToJset[subdomain.SKToGmsh[node]].add(j)
        projector, sk_to_g = scipy_helpers.restriction_matrix(mesh.nvertices, list(nodeSet), subdomain.SKToGmsh, subdomain.gmshToSK)
        gi[j] = (function_space, projector, sk_to_g)


    for node_sk, node_gmsh in subdomain.SKToGmsh.items():
        if node_gmsh in crosspoints_gmsh_node_tags:
            column = crosspoints_gmsh_to_kernel_column[node_gmsh]
            partitions = crosspoints_gmsh_to_graph[node_gmsh]
            prev, next = cycle_find_prev_and_next(partitions, idom)
            crosspoint_tag = crosspoints_gmsh_to_kernel_column[node_gmsh]
            print(f"Found kernel mode {crosspoint_tag} on domain {idom} at node {node_gmsh} shared with partitions {partitions}. Prev is {prev}, next is {next}")
            print("Local node value is at skfem node ", node_sk)
            subdomain.add_kernel_mode(crosspoint_tag, node_sk, next, prev)

    sizes = [int(mesh.nvertices)] + [proj.shape[0] for (j, (fs, proj, _)) in gi.items()]
    print("Local sizes (u and each g): ", sizes)
    num_interface_fields = len(gi)

    gamma_facets = mesh_helpers.findFacetsGamma(gamma_tags, subdomain.gmshToSK, subdomain.facets_dict)
    print("gamma_facets has size ", len(gamma_facets))

    mats: list[list[csr_matrix | None]] = [[None for _ in range(num_interface_fields + 1)] for _ in range(num_interface_fields + 1)]
    mats[0][0] = skfem.asm(helmholtz, skfem.Basis(mesh, ElementTriP1()), k=k)
    if len(gamma_facets) > 0:
        mats[0][0] += skfem.asm(absorbing, FacetBasis(mesh, ElementTriP1(), facets=gamma_facets), k=k)

    subdomain.set_neuman_mat(scipy.sparse.csr_matrix(mats[0][0]) if mats[0][0] is not None else None) # Deep copy
    all_sigma = np.concatenate(list(subdomain.all_sigma_facets.values()))
    global_sigma_basis = skfem.FacetBasis(mesh, ElementTriP1(), facets=all_sigma)
    subdomain.set_sigma_mass_mat(skfem.asm(mass_bnd, global_sigma_basis, k=k))
    
    rank_of_sigma = np.linalg.svd(subdomain.get_sigma_mass_mat().toarray(), compute_uv=False)
    print("Sigma rank: ", np.sum(rank_of_sigma > 1e-10), " out of ", rank_of_sigma.shape[0])

    sigma_mass = subdomain.get_neuman_mat()
    bnd_dofs = subdomain.get_bnd_dofs()
    sigma_mass_restricted = sigma_mass[bnd_dofs, :][:, bnd_dofs]


    # Generalized EVP: small real parts of neuman * X = lambda * sigma_mass * X
    from scipy.sparse.linalg import eigs
    nev = 4
    neuman_mat = subdomain.get_neuman_mat()
    sigma_mass_mat = subdomain.get_sigma_mass_mat()
    eigvals, eigvecs = eigs(neuman_mat, M=sigma_mass_mat, k=nev, which='SR', sigma=0.1)
    A_s = skfem.asm(transmission, global_sigma_basis, k=k)

    print(f"Domain {idom} generalized EVP eigenvalues (neuman * X = lambda * sigma_mass * X):")
    for l in range(nev):
        print(f"  Eigenvalue {l}: {eigvals[l]}")
        ev = eigvals[l]
        x = eigvecs[:, l]
        g_continuous = ev * x
        # g = lambda x + M^-1 A_S x where A_s is the absorption mat
        g_continuous[bnd_dofs] += scipy.sparse.linalg.spsolve(sigma_mass_restricted, (A_s @ x)[bnd_dofs])
        subdomain.add_continuous_g_coarse(g_continuous[bnd_dofs])

    # local coarse space
    z = np.column_stack(subdomain.continuous_g_coarse)
    print("Local coarse space shape: ", z.shape)
    # rank
    rank_of_z = np.linalg.matrix_rank(z)
    print("Local coarse space rank: ", rank_of_z)
    

    for idx_j, j in enumerate(sorted(gi.keys())):
        transmission_contribution = skfem.asm(transmission, gi[j][0], k=k)
        mats[0][0] += transmission_contribution

    for idx_j, j in enumerate(sorted(gi.keys())):
        fs_j, pj, _ = gi[j]
        mass = pj @ skfem.asm(mass_bnd, fs_j) @ pj.T
        all_g_masses.append(mass)
        all_s_masses.append(pj @ skfem.asm(transmission, fs_j, k=k) @ pj.T * 1.0j)

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

    subdomain.set_problem_mat(scipy_helpers.bmat(mats))

    b_substructured = scipy.sparse.linalg.spsolve(subdomain.get_problem_mat(), full_rhs)[mesh.nvertices:]
    phys_b.append(b_substructured)
    # Now, mats goes from the gs to the local solution including u
    # So, structure is num_interface_fields to num_interface_fields + 1
    mats = [[None for _ in range(num_interface_fields)] for _ in range(num_interface_fields + 1)]
    for idx_j, j in enumerate(sorted(gi.keys())):
        P = gi[j][1]
        volumetric_mass_j = skfem.asm(mass_bnd, gi[j][0], k=k) @ P.T
        mats[0][idx_j] = volumetric_mass_j
        mats[idx_j + 1][idx_j] = -P @ volumetric_mass_j

    rhs_mat  = scipy_helpers.bmat(mats)
    subdomain.set_rhs_mat(rhs_mat)


    
    nvertices = mesh.nvertices

    num_rows = sum(sizes) - nvertices
    num_cols = sum(proj.shape[0] for (j, (_, proj, _)) in gi.items())

    local_solve = make_local_solve(
        gi=dict(gi),                  # shallow copy for safety
        sizes=list(sizes),
        local_rhs_mat=subdomain.get_rhs_mat(),
        local_mat=subdomain.get_problem_mat(),
        nvertices=nvertices,
    )

    linearOp = scipy.sparse.linalg.LinearOperator(
        (num_rows, num_cols),
        matvec=local_solve,
        dtype=np.complex128,
    )
    local_solves.append(linearOp)

full_mass = scipy.sparse.block_diag(all_g_masses, format='csr')
full_s_mass = scipy.sparse.block_diag(all_s_masses, format='csr')


from ddm_utils import build_offsets_and_total_size, build_full_rhs, LocalDDMSolver

offsets, istart, total_g_size = build_offsets_and_total_size(subdomains)
print("Total g size: ", total_g_size)
delta_kernel = np.zeros((total_g_size, len(crosspoints_gmsh_node_tags)), dtype=np.complex128)
for idom in range(1, ndom+1):
    subdomain = subdomains[idom - 1]
    for mode in subdomain.ker:
        kernel_column = mode['kernel_column']
        node_sk = mode['node_sk']
        jplus = int(mode['jplus'])
        jminus = mode['jminus']
        offset_jplus = subdomain.gi[jplus][2][node_sk]
        global_dof_index_plus =  offset_jplus + offsets[(idom, jplus)]
        offset_jminus = subdomain.gi[jminus][2][node_sk]
        global_dof_index_minus =  offset_jminus + offsets[(idom, jminus)]
        
        delta_kernel[global_dof_index_plus, kernel_column] = 1.0
        delta_kernel[global_dof_index_minus, kernel_column] = -1.0
rhs = build_full_rhs(phys_b)


rhs = np.concatenate(phys_b)

swap  = scipy_helpers.build_swap(subdomains, offsets, ndom, total_g_size)

ddm_op = LocalDDMSolver(local_solves, istart, ndom, total_g_size)
ddm_op.set_swap(swap)



def ddm_operator(g):
    g_swap = swap @ g
    g_solved = ddm_op.apply(g_swap)
    return g_solved


all_coarse_spaces = [subdomains[i].gen_global_coarse_space_contrib(total_g_size, istart[i]) for i in range(ndom)]
print("All coarse spaces shapes: ", [cs.shape for cs in all_coarse_spaces])
total_cs_size = sum(cs.shape[1] for cs in all_coarse_spaces if cs is not None)
print("Total coarse space size: ", total_cs_size)
Z = np.zeros((total_g_size, total_cs_size), dtype=np.complex128)
for i, cs in enumerate(all_coarse_spaces):
    if cs is not None:
        start_col = sum(all_coarse_spaces[j].shape[1] for j in range(i) if all_coarse_spaces[j] is not None)
        end_col = start_col + cs.shape[1]
        start_row = istart[i]
        end_row = istart[i+1]
        Z[start_row:end_row, start_col:end_col] = cs


#plt.spy(Z, markersize=1)
print("Shape of Z: ", Z.shape, " rank of Z and AZ", np.linalg.matrix_rank(Z, tol=1e-8), np.linalg.matrix_rank(ddm_op.A @ Z, tol=1e-8))
ZAZ = np.conjugate(Z.T) @ (ddm_op.A @ Z)
rank_zaz = np.linalg.matrix_rank(ZAZ, tol=1e-8)
print("Rank of Z^* A Z: ", rank_zaz, " out of ", ZAZ.shape[0])

#plt.show()

ddm_dense = ddm_op.A @ np.eye(total_g_size, dtype=np.complex128)

"""
x, info = scipy.sparse.linalg.gmres(ddm_op.A, rhs, rtol=1e-6, callback=lambda r: print("GMRES residual: ", r))
#print(x, info)
ddm_dense = ddm_op.A @ np.eye(total_g_size, dtype=np.complex128)
x_dense = np.linalg.solve(np.eye(total_g_size) - ddm_dense, rhs)
print("Norm of RHS: ", np.linalg.norm(rhs))
print("Residual: ", np.linalg.norm(ddm_op.A @ x_dense - rhs)/np.linalg.norm(rhs))

# Build full solution
if False:
    x = swap @ x
    for idom in range(1, ndom+1):
        working_range_start = istart[idom - 1]
        working_range_end = istart[idom]
        gloc = x[working_range_start:working_range_end]
        artifical_source = local_rhs_mats[idom-1] @ gloc
        physical_source = local_physical_sources[idom-1]
        mesh = subdomains[idom-1].mesh
        u_local = scipy.sparse.linalg.spsolve(local_mats[idom-1], artifical_source+physical_source)[0:mesh.nvertices]
        print(f"Domain {idom} local solution norm: ", np.linalg.norm(u_local))
        print("Shape and nvertices : ", u_local.shape, subdomains[idom-1].mesh.nvertices)
        plot(subdomains[idom-1].mesh, np.real(u_local[:mesh.nvertices]), shading='gouraud')
        plt.show()
"""
from scipy.sparse.linalg import svds

#u, s, vt = svds(ddm_op.T, k=6, which='SM')
#print("Singular vlaues: ", s)

from numpy.linalg import svd
import scipy.sparse.linalg as spla


m_inv_delta = scipy.sparse.linalg.spsolve(full_mass, delta_kernel.reshape(total_g_size, -1)).reshape(total_g_size, -1)
print("Is it a kernel ? Ax has norm ", np.linalg.norm(ddm_op.A @ m_inv_delta), " and x has norm ", np.linalg.norm(m_inv_delta))
m_s_inv_delta = full_mass @ scipy.sparse.linalg.spsolve(full_s_mass, delta_kernel.reshape(total_g_size, -1)).reshape(total_g_size, -1)
print("Shape of msinv delta: ", m_s_inv_delta.shape)
print("Is it a kernel for the adjoint? A^*x has norm ", np.linalg.norm(np.conj(ddm_dense.T) @ m_s_inv_delta), " and x has norm ", np.linalg.norm(m_s_inv_delta))


print("Dense Svd... size is ", ddm_dense.shape)
u, s, vt = svd(ddm_op.A @ np.eye(total_g_size, dtype=np.complex128), full_matrices=False)
print("Number of singular values near zero: ", np.sum(s < 1e-8))

print("Solving randomized RHS orthogonalized to ker A* of shape", m_s_inv_delta.shape)
rhs = np.random.normal(size=(total_g_size,)) + 1j * np.random.normal(size=(total_g_size,))
norm_2 = np.linalg.norm(m_s_inv_delta, ord=2) ** 2
# Orthogonalize rhs to m_s_inv_delta if that matrix is not orthogonal
Q, R = np.linalg.qr(m_s_inv_delta)
rhs = rhs - Q @ (np.conjugate(Q.T) @ rhs)
#x_rand, info_rand = scipy.sparse.linalg.gmres(ddm_op.A, rhs, rtol=1e-6, callback=lambda r: print("GMRES residual (rand RHS): ", r))


# Build deflated operator from Z, ZAZ
def prec(g):
    print("Applygin on g of shape ", g.shape)
    print("Z has shape ", Z.shape)
    print("ZAZ has shape ", ZAZ.shape)
    zt = np.conjugate(Z.T) @ g
    print("Ztg has shape ", zt.shape)
    Qg = Z @ (np.linalg.solve(ZAZ, zt))
    AQg = ddm_op.A @ Qg
    return Qg - AQg + g

def prec_mat(G: np.ndarray) -> np.ndarray:
    # G is shape (N, m)
    # compute Z^H G : (k, N) @ (N, m) = (k, m)
    ZHG = np.conjugate(Z.T) @ G
    X = np.linalg.solve(ZAZ, ZHG)  # solves multiple RHS
    QG = Z @ X
    return G + QG - ddm_op.A @ QG
    
precond = scipy.sparse.linalg.LinearOperator(
    (total_g_size, total_g_size),
    matvec=prec,
    matmat=prec_mat,
)

precond_ide = precond @ np.eye(total_g_size, dtype=np.complex128)




# Hermitian part of the preconditioned operator
# compar GMRES and GMRES precond
residual = []
scipy.sparse.linalg.gmres(ddm_op.A, rhs, rtol=1e-6, callback=lambda r: residual.append(r), maxiter=1000)
counter = 0
residual_prec = []
scipy.sparse.linalg.gmres(ddm_op.A@precond_ide, rhs, rtol=1e-6, callback=lambda r: residual_prec.append(r), maxiter=1000)
plt.plot(residual, label='No precond')
plt.plot(residual_prec, label='With deflation precond')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('GMRES residual')
plt.legend()
plt.title('GMRES convergence with and without deflation preconditioner')
plt.grid()
plt.show()


eigs_precond = np.linalg.eigvals(ddm_dense @ precond_ide)
eigs_from_dense = np.linalg.eigvals(ddm_dense @ np.eye(total_g_size, dtype=np.complex128))


def fov_sample(B, nsamples=2000):
    N = B.shape[0]
    vals = []
    for _ in range(nsamples):
        x = np.random.randn(N) + 1j*np.random.randn(N)
        x /= np.linalg.norm(x)
        vals.append(np.conjugate(x.T) @ (B @ x))
    return np.array(vals)

B = ddm_dense @ precond_ide
fov_vals = fov_sample(B, nsamples=2000)

plt.figure()
plt.scatter(fov_vals.real, fov_vals.imag, s=2, alpha=0.5, label="Approx FOV")
plt.scatter(eigs_from_dense.real, eigs_from_dense.imag, s=8, label="Eigenvalues")
plt.xlabel("Real")
plt.ylabel("Imag")
plt.legend()
plt.grid(True)
plt.title("Field of Values vs Spectrum")
plt.show()


# Plot eigenvalues
if True:
    plt.figure()
    plt.scatter(eigs_precond.real, eigs_precond.imag, label='Deflated operator eigenvalues')
    plt.scatter(eigs_from_dense.real, eigs_from_dense.imag, label='Original operator eigenvalues')
    plt.title("Eigenvalues of DDM operator")
    plt.xlabel("Real part")
    plt.ylabel("Imaginary part")
    # Plot unit circle for reference
    thetas = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(thetas)+1, np.sin(thetas), 'r--', label='Unit Circle')
    plt.grid()
    plt.show()


exit()

u, s, vt = svd(ddm_op.A @ spla.spsolve(full_mass, np.eye(total_g_size, dtype=np.complex128)), full_matrices=False)
#u, s, vt = svd(ddm_op.A @ np.eye(total_g_size, dtype=np.complex128), full_matrices=False)
print("Singular values: ", s[s < 1e-8])
ker = np.conjugate(vt.T[:, s < 1e-8])
ker_of_adj = u[:, s < 1e-8]
concatenated_kers = np.concatenate((ker, ker_of_adj), axis=1)

# Is RHS of the problem orthogonal to the kernel of the operator?
rhs_dot_ker = np.conjugate(ker.T) @ rhs
print("RHS dot kernel vectors: ", abs(rhs_dot_ker))
# Is it M-orthogonal ?
rhs_dot_ker_M = np.conjugate(ker.T @ full_mass) @ (rhs)
# Compute contributions of the dot per subdomain, before the summation
for l in range(ker.shape[1]):
    ker_vec = ker[:, l]
    piecewise_product = np.conjugate(ker_vec) * (full_s_mass @ rhs)
    for idom in range(1, ndom+1):
        start = istart[idom - 1]
        end = istart[idom]
        contribution = np.sum(piecewise_product[start:end])
        #print(f"Contribution to RHS M-orthogonality from domain {idom}, kernel vector {l}: ", contribution)

print("RHS M-orthogonal to kernel vectors: ", abs(rhs_dot_ker_M))
rhs_dot_ker_M = np.conjugate(ker.T) @ (full_s_mass @ rhs)
print("RHS real(S)-orthogonal to kernel vectors: ", abs(rhs_dot_ker_M))

sim = scipy.sparse.linalg.spsolve(full_s_mass, full_mass @ rhs)
sinv_mass_dot_ker = np.conjugate(ker.T) @ sim
print("RHS S^-1 M-orthogonal to kernel vectors: ", abs(sinv_mass_dot_ker))
mis = scipy.sparse.linalg.spsolve(full_mass, full_s_mass @ rhs)
massinv_dot_ker = np.conjugate(ker.T) @ mis
print("RHS M^-1 S-orthogonal to kernel vectors: ", abs(massinv_dot_ker))
mism = scipy.sparse.linalg.spsolve(full_mass, full_s_mass @ scipy.sparse.linalg.spsolve(full_mass, full_mass @ rhs))
this_dot = np.conjugate(ker.T) @ mism
print("RHS M^-1 S M-orthogonal to kernel vectors: ", abs(this_dot))
# Is it M^-1-orthogonal ?
rhs_dot_ker_Minv = np.conjugate(ker.T) @ scipy.sparse.linalg.spsolve(full_mass, rhs)
print("RHS M^-1-orthogonal to kernel vectors: ", abs(rhs_dot_ker_Minv))

print("Solving randomized RHS")
rhs = np.random.normal(size=(total_g_size,)) + 1j * np.random.normal(size=(total_g_size,))
x_rand, info_rand = scipy.sparse.linalg.gmres(ddm_op.A, rhs, rtol=1e-6, callback=lambda r: print("GMRES residual (rand RHS): ", r))
print("M-orthogonalizing the RHS against the kernel")
y = scipy.sparse.linalg.spsolve(np.conjugate(ker.T) @ full_mass @ ker, np.conjugate(ker.T) @ (full_mass @ rhs))
rhs = rhs - ker @ y
print("Checking orthogonality after M-orthogonalization: ", np.conjugate(ker.T @ full_mass) @ rhs)
x_rand_ortho, info_rand_ortho = scipy.sparse.linalg.gmres(ddm_op.A, rhs, rtol=1e-6, callback=lambda r: print("GMRES residual (rand ortho RHS): ", r))

#raise NotImplementedError("Think again about stuff with adjoint kernel?")

_, ss, _ = svd(concatenated_kers)
print("Singular values of concatenated ker and ker of adj: ", ss)
number_of_nonzeros_in_concatenated = np.sum(ss > 1e-8)
print("Cumulated rank is : ", number_of_nonzeros_in_concatenated)


print("Approximate kernel dimension: ", ker.shape, ker_of_adj.shape)
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
    #print(vec)



# Compute spectrum of ddm_operator
eigs = scipy.sparse.linalg.eigs(ddm_op.T, k=total_g_size-2, which='LM')
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






print("Ker shape: ", ker.shape)
