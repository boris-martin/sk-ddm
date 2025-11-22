# We first import petsc4py and sys to initialize PETSc
import sys, petsc4py
petsc4py.init(sys.argv)
from typing import Dict, List, Tuple
from typing import cast
# Import the PETSc module
from petsc4py import PETSc
from line_profiler import profile
from src.lib_subdomain.distributed_domain_set import SubdomainsOnMyRank
from src.lib_subdomain.load_local_mesh import LocalGeometry
from src.tuto.mesh_helpers import create_square
import math

from skfem import BilinearForm, LinearForm
from skfem.helpers import dot, grad
import skfem
import numpy as np
from numpy import conjugate
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, spsolve, splu


import matplotlib.pyplot as plt
from mpi4py import MPI

@BilinearForm
def helmholtz(u, v, w):
    k = w["k"]
    x = w["x"]
    x_left = x[0] < 0.5
    #k_eff = k * (1.0 + np.sin(x[0] * 4 * 3.1415) ** 2) * x_left + k * (
    #    1.0 + np.cos(x[0] * 4 * 3.1415) ** 2
    #) * (~x_left)
    k_eff = k
    return dot(grad(u), conjugate(grad(v))) - k_eff**2 * u * conjugate(v)


@BilinearForm(facet=True, dtype=np.complex128)
def mass_bnd(u, v, w):
    return np.complex128(u * conjugate(v))


@BilinearForm(facet=True, dtype=np.complex128)
def absorbing(u, v, w):
    k = w["k"]
    return np.complex128(-1j * k * u * conjugate(v))


@BilinearForm(facet=True, dtype=np.complex128)
def transmission(u, v, w):
    k = w["k"]
    x = w["x"]
    # Mask for knowing if x[0] < 0.5
    x_left = x[0] < 0.5  # Unused in current implementation

    # k eff = k * (1 + x²) - currently unused
    k_eff = k * (1.0 + np.sin(x[0] * 4 * math.pi) ** 2) * x_left + k * (
        1.0 + np.cos(x[0] * 4 * 3.1415) ** 2
    ) * (~x_left)
    k_eff = k  # ignore heterogeneity for transmission
    return np.complex128(-(1j-0.1)* k_eff * u * conjugate(v))


# Time convention: S = iku, we remove S_gamma on the boundary (-dnu v term)

class Formulation:
    def __init__(self, domains: SubdomainsOnMyRank, k: float):
        self.domains: SubdomainsOnMyRank = domains
        self.k = k
        self.volume_mats: List[csr_matrix]
        self.local_masses: dict[tuple[int, int], csr_matrix] = {}
        self.local_transmission: dict[tuple[int, int], csr_matrix] = {}
        self.swap: PETSc.Mat = domains.build_swap_operator()

        self.volume_fact: List = []
        self.mass_fact: dict[tuple[int, int], object] = {}
        self.offset_i_cache: Dict[int, int] | None = None

    def assemble_volume(self):
        self.volume_mats = []
        self.volume_fact = []
        for dom in self.domains.subdomains:
            A = skfem.asm(helmholtz, dom.volume_basis , k=self.k)
            if dom.has_gamma():
                A += skfem.asm(absorbing, dom.gamma_basis, k=self.k)
            for _, basis in dom.sigma_basis.items():
                A += skfem.asm(transmission, basis, k=self.k)
            A = A.tocsr()
            self.volume_mats.append(A)
            self.volume_fact.append(splu(A))
        assert len(self.volume_mats) == len(self.domains.subdomains)

    def assemble_interface_mats(self):
        for ki, i in enumerate(self.domains.partitions):
            for j in self.domains.subdomains[ki].all_neighboring_partitions():
                if (i, j) in self.local_masses:
                    raise ValueError("Interface matrices already assembled")
                M = skfem.asm(
                    mass_bnd,
                    self.domains.subdomains[ki].sigma_basis[j],
                    k=self.k,
                )
                S = skfem.asm(
                    transmission,
                    self.domains.subdomains[ki].sigma_basis[j],
                    k=self.k,
                )
                gdofs = self.domains.subdomains[ki].dofs_on_interface(j)
                M_reduced = M[gdofs, :][:, gdofs]
                S_reduced = S[gdofs, :][:, gdofs]
                self.local_masses[(i, j)] = M_reduced.tocsr()
                self.mass_fact[(i, j)] = splu(M_reduced.tocsr())
                self.local_transmission[(i, j)] = S_reduced.tocsr()

        # setup cached offsets
        self.offset_ij_cache: Dict[Tuple[int, int], int] = {}
        offsets = self.domains.local_offset_list()
        for iidx, i in enumerate(self.domains.partitions):
            for j in self.domains.subdomains[iidx].all_neighboring_partitions():
                # find idx in the offsets list
                for idx in range(len(offsets[0])):
                    if offsets[0][idx] == i and offsets[1][idx] == j:
                        local_offset = offsets[2][idx]
                        break
                self.offset_ij_cache[(i, j)] = local_offset
    @profile
    def apply_scatter(self, x: PETSc.Vec, y: PETSc.Vec):
        """
        Scatter from global g-vector x to local subdomain vectors y.
        """
        self.swap.mult(x, y)
        y_numpy = y.getArray()
        output = -y_numpy.copy()

        offsets = self.domains.local_offset_list()
        if self.offset_i_cache is None:
            self.offset_i_cache = {i: self.domains.offset_of_domain(i) for i in self.domains.partitions}

        for iidx, i in enumerate(self.domains.partitions):
            dom = self.domains.subdomains[iidx]
            offset = self.offset_i_cache[i]
            g_size = self.domains.g_vector_size_for_domain(i)
            # print("For domain ", i, " g-size=", g_size)

            rhs = np.zeros(dom.volume_size(), dtype=np.complex128)
            for j in dom.all_neighboring_partitions():
                local_offset = self.offset_ij_cache[(i, j)]
                dofs = cast(List[int], dom.dofs_on_interface(j))
                local_mass = self.local_masses[(i, j)]
                mass_g = local_mass @ y_numpy[local_offset:local_offset+len(dofs)]
                rhs[dofs] += mass_g

            #u_i = spsolve(self.volume_mats[iidx], rhs)
            u_i = self.volume_fact[iidx].solve(rhs)
            for j in dom.all_neighboring_partitions():
                
                local_offset = self.offset_ij_cache[(i, j)]
                dofs = dom.dofs_on_interface(j)
                su = self.local_transmission[(i, j)] @ u_i[dofs]
                #m_inv_su = spsolve(self.local_masses[(i, j)], su)
                m_inv_su = self.mass_fact[(i, j)].solve(su)
                output[local_offset:local_offset+len(dofs)] += 2 * m_inv_su



        y.setArray(output)

    def compute_substructured_rhs(self, f: dict[int, np.ndarray]) -> PETSc.Vec:
        """
        Compute the substructured RHS vector from local volume RHS f.
        """
        for i, src in f.items():
            assert src.shape == (self.domains.find_domain(i).volume_size(),)
            assert i in self.domains.partitions
        
        x = self.domains.create_petsc_g_vector()
        x_numpy = x.getArray()
        offsets = self.domains.local_offset_list()
    
        for iidx, i in enumerate(self.domains.partitions):
            dom = self.domains.subdomains[iidx]
            offset = self.domains.offset_of_domain(i)
            g_size = self.domains.g_vector_size_for_domain(i)
            print("For domain ", i, " g-size=", g_size)

            rhs = f[i]
            #u_i = spsolve(self.volume_mats[iidx], rhs)
            u_i = self.volume_fact[iidx].solve(rhs)
            for j in dom.all_neighboring_partitions():
                # find idx in the offsets list
                for idx in range(len(offsets[0])):
                    if offsets[0][idx] == i and offsets[1][idx] == j:
                        local_offset = offsets[2][idx]
                        break
                dofs = dom.dofs_on_interface(j)
                #m_inv_su = spsolve(self.local_masses[(i, j)], self.local_transmission[(i, j)] @ u_i[dofs])
                m_inv_su = self.mass_fact[(i, j)].solve( self.local_transmission[(i, j)] @ u_i[dofs])
                x_numpy[local_offset:local_offset+len(dofs)] = m_inv_su

        x.setArray(x_numpy)
        return x
    
    def mult(self, mat, x, y):
        """
        PETSc MatMult operation: y = (I - T) * x
        """
        # y = 0
        y.zeroEntries()
        # y = x
        y.axpy(1.0, x)

        # temp = T(x)
        temp = y.duplicate()
        self.apply_scatter(x, temp)

        # y = x - temp
        y.axpy(-1.0, temp)
        temp.destroy()

    def build_dtn_coarse(self, nev: int):
        # Build local DTN bases (Z columns)
        for dom in self.domains.subdomains:
            local_dtn = LocalDTN(dom, self.k)
            local_dtn.build_basis(nev)
            dom.local_dtn = local_dtn

        g_size_on_rank = self.domains.g_vector_local_size()
        
        # Local size of the coarse space on this rank (Coarse Columns)
        local_coarse_size = nev * len(self.domains.subdomains)
        self.Z = np.zeros((g_size_on_rank, local_coarse_size), dtype=np.complex128)
        
        # --- NEW: Calculate and Store Global Coarse Metadata ---
        comm = MPI.COMM_WORLD
        
        # 1. Store local size for indexing Z matrix rows
        self.coarse_size = local_coarse_size 
        
        # 2. Gather sizes (counts) from all ranks
        self.coarse_counts = np.array(comm.allgather(local_coarse_size), dtype=np.int32)
        
        # 3. Calculate global offsets (prefix sums)
        self.coarse_offsets = np.cumsum(np.concatenate(([0], self.coarse_counts)))
        print("Coarse offsets:", self.coarse_offsets)
        self.total_coarse_size = self.coarse_offsets[-1]

        # 4. Fill Z (Prolongation matrix)
        for iidx, dom in enumerate(self.domains.subdomains):
            for j in dom.all_neighboring_partitions():
                dofs = dom.dofs_on_interface(j)
                offset = self.offset_ij_cache[(dom.partition, j)]
                nev_dom = dom.local_dtn.get_eigvecs().shape[1]
                for ev in range(nev_dom):
                    self.Z[offset:offset+len(dofs), iidx * nev + ev] = dom.local_dtn.get_eigvecs()[dofs, ev]
                
        print("Coarse basis Z shape:", self.Z.shape)
    
    def build_Z_operator(self):
        """
        Builds the PETSc operator Z mapping from Coarse Space -> Fine Space.
        Also supports Z* (Hermitian Transpose) mapping Fine Space -> Coarse Space.
        """
        assert hasattr(self, 'Z'), "Coarse basis Z not built yet."

        class ZContext:
            def __init__(ctx, Z_numpy):
                # Z_numpy shape: (Local_Fine_Size, Local_Coarse_Size)
                ctx.Z = Z_numpy

            def mult(ctx, mat, x, y):
                """ y = Z * x """
                # 1. Get Read-Only access to input vector x
                # This bypasses the PETSc lock check but returns a standard numpy array
                x_arr = x.getArray(readonly=True)
                
                # 2. Get Write access to output vector y
                y_arr = y.getArray()
                
                # 3. Perform computation
                # (Fine, Coarse) @ (Coarse) -> (Fine)
                np.dot(ctx.Z, x_arr, out=y_arr)
                
                # Note: No explicit 'release' needed for standard getArray in Python, 
                # but cleaning up references is good practice.

            def multHermitianTranspose(ctx, mat, x, y):
                """ y = Z^H * x (Adjoint) """
                # 1. Get Read-Only access to input vector x
                x_arr = x.getArray(readonly=True)
                
                # 2. Get Write access to output vector y
                y_arr = y.getArray()
                
                # 3. Perform computation
                # (Coarse, Fine) @ (Fine) -> (Coarse)
                # Z.conj().T @ x
                # We use slice assignment [:] to ensure we write into the PETSc buffer
                y_arr[:] = ctx.Z.conj().T @ x_arr

            # --- BATCHED / MATRIX OPERATIONS (BLAS Level 3) ---
            def multMat(ctx, _, X, Y):
                """ Y = Z * X (Matrix-Matrix multiplication) """
                # X is input Dense Matrix (Coarse Size x N_vecs)
                # Y is output Dense Matrix (Fine Size x N_vecs)
                
                # getDenseArray returns the local chunk of the dense matrix as a 2D NumPy array
                X_arr = X.getDenseArray(readonly=True)
                Y_arr = Y.getDenseArray()
                
                # Perform Dense Matrix-Matrix Multiplication (GEMM)
                # (Fine, Coarse) @ (Coarse, N_vecs) -> (Fine, N_vecs)
                np.matmul(ctx.Z, X_arr, out=Y_arr)

            def multHermitianTransposeMat(ctx, mat, X, Y):
                """ Y = Z^H * X (Matrix-Matrix multiplication) """
                # X is input Dense Matrix (Fine Size x N_vecs)
                # Y is output Dense Matrix (Coarse Size x N_vecs)
                
                X_arr = X.getDenseArray(readonly=True)
                Y_arr = Y.getDenseArray()
                
                # (Coarse, Fine) @ (Fine, N_vecs) -> (Coarse, N_vecs)
                # Use slice assignment to write into PETSc buffer
                Y_arr[:] = ctx.Z.conj().T @ X_arr

            # Optional: If you strictly need Z^T (non-conjugate)
            def multTranspose(ctx, mat, x, y):
                x_arr = x.getArray(readonly=True)
                y_arr = y.getArray()
                y_arr[:] = ctx.Z.T @ x_arr

        # 1. Dimensions
        local_rows = self.Z.shape[0] 
        local_cols = self.Z.shape[1]

        # 2. Create the PETSc Matrix
        Z_mat = PETSc.Mat().createPython(
            [(local_rows, PETSc.DECIDE), (local_cols, PETSc.DECIDE)], 
            context=ZContext(self.Z),
            comm=PETSc.COMM_WORLD
        )
        Z_mat.setUp()
        
        self.Z_op = Z_mat
        print(f"Built Z Operator. Local dims: {local_rows}x{local_cols}")
        return Z_mat


if __name__ == "__main__":

    from src.lib_subdomain.local_dtn import LocalDTN
    ndom = 8
    subdomains_on_rank = SubdomainsOnMyRank(ndom)
    create_square(0.025, ndom)
    for dom in subdomains_on_rank.subdomains:
        dom.init_all()

    print("Partitions on this rank:", subdomains_on_rank.partitions)
    for i in subdomains_on_rank.partitions:
        offset = subdomains_on_rank.offset_of_domain(i)
        print(f"Domain {i} offset in g-vector:", offset)
    for dom in subdomains_on_rank.subdomains:
        print(f"Domain {dom.partition} neighbors:", dom.all_neighboring_partitions())

    formulation = Formulation(subdomains_on_rank, k=10.0)
    formulation.assemble_volume()
    formulation.assemble_interface_mats()

    f: Dict[int, np.ndarray] = {}
    for i in subdomains_on_rank.partitions:
        dom = subdomains_on_rank.find_domain(i)
        f[i] = np.ones(dom.volume_size(), dtype=np.complex128) if i == 2 else np.zeros(dom.volume_size(), dtype=np.complex128)
    rhs = formulation.compute_substructured_rhs(f)

    formulation.build_dtn_coarse(nev=10)
    Z = formulation.build_Z_operator()

    # 1. Create Vectors
    # Create a Coarse Vector (Input for Z)
    x_coarse = Z.createVecRight() 
    # Create a Fine Vector (Input for Z*)
    x_fine = Z.createVecLeft()

    # 2. Test Z * x_coarse -> y_fine
    x_coarse.set(1.0 + 0j) # Fill with ones
    y_fine = Z.createVecLeft()
    Z.mult(x_coarse, y_fine)

    # 3. Test Z* * x_fine -> y_coarse
    x_fine.set(1.0 + 0j)
    y_coarse = Z.createVecRight()
    Z.multHermitian(x_fine, y_coarse)

    print(f"Z mult output norm: {y_fine.norm()}")
    print(f"Z* mult output norm: {y_coarse.norm()}")

    # Create a 5 columns dense matrix for testing batched operations. Output has G-shape
    local_size = subdomains_on_rank.g_vector_local_size()
    x_coarse_mat = PETSc.Mat().createDense(((formulation.coarse_size, PETSc.DETERMINE), (PETSc.DETERMINE, 5)), comm=PETSc.COMM_WORLD)
    #x_coarse_mat.set(1.0 + 0j)
    x_numpy = x_coarse_mat.getDenseArray()
    x_numpy.fill(1.0 + 0j)

    y_fine_mat = PETSc.Mat().createDense(((local_size, PETSc.DETERMINE), (PETSc.DETERMINE, 5)), comm=PETSc.COMM_WORLD)
    Z.matMult(x_coarse_mat, y_fine_mat)
    print("Z multMat output norm:", y_fine_mat.norm())
    y_fine_col_1 = y_fine_mat.getDenseArray()[:, 0]
    print("First column norm (local):", np.linalg.norm(y_fine_col_1))
    norm_2 = np.linalg.norm(y_fine_col_1)**2
    norm_2 = MPI.COMM_WORLD.allreduce(norm_2, op=MPI.SUM)
    print("First column norm (global):", math.sqrt(norm_2))

    # Adjoint batched
    #Z.matMultHermitian(y_fine_mat, x_coarse_mat)
    Z.getPythonContext().multHermitianTransposeMat(Z, y_fine_mat, x_coarse_mat)
    print("Z* multMat output norm:", x_coarse_mat.norm())
    # Norm of col 1
    x_coarse_col_1 = x_coarse_mat.getDenseArray()[:, 0]
    print("First column norm (local):", np.linalg.norm(x_coarse_col_1))
    norm_2 = np.linalg.norm(x_coarse_col_1)**2
    norm_2 = MPI.COMM_WORLD.allreduce(norm_2, op=MPI.SUM)
    print("First column norm (global):", math.sqrt(norm_2))


    #rhs.view()

    x = rhs.duplicate()
    # create a mat shell
    A = PETSc.Mat().createPython([(rhs.getLocalSize(), rhs.getSize()), (rhs.getLocalSize(), rhs.getSize())], context=formulation)
    A.setType(PETSc.Mat.Type.PYTHON)
    A.setPythonContext(formulation)
    A.setUp()

    A.mult(rhs, x)
    #x.view()

    ksp = PETSc.KSP().create()  # noqa: F841
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.solve(rhs, x)


   