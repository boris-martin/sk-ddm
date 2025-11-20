# We first import petsc4py and sys to initialize PETSc
import sys, petsc4py
petsc4py.init(sys.argv)
from typing import Dict, List, Tuple
from typing import cast
# Import the PETSc module
from petsc4py import PETSc

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
from scipy.sparse.linalg import LinearOperator, spsolve


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

    def assemble_volume(self):
        self.volume_mats = []
        for dom in self.domains.subdomains:
            A = skfem.asm(helmholtz, dom.volume_basis , k=self.k)
            if dom.has_gamma():
                A += skfem.asm(absorbing, dom.gamma_basis, k=self.k)
            for _, basis in dom.sigma_basis.items():
                A += skfem.asm(transmission, basis, k=self.k)
            self.volume_mats.append(A.tocsr())
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
                self.local_transmission[(i, j)] = S_reduced.tocsr()

    def apply_scatter(self, x: PETSc.Vec, y: PETSc.Vec):
        """
        Scatter from global g-vector x to local subdomain vectors y.
        """
        self.swap.mult(x, y)
        y_numpy = y.getArray()
        output = -y_numpy.copy()

        offsets = self.domains.local_offset_list()

        for iidx, i in enumerate(self.domains.partitions):
            dom = self.domains.subdomains[iidx]
            offset = self.domains.offset_of_domain(i)
            g_size = self.domains.g_vector_size_for_domain(i)
            # print("For domain ", i, " g-size=", g_size)

            rhs = np.zeros(dom.volume_size(), dtype=np.complex128)
            for j in dom.all_neighboring_partitions():
                # find idx in the offsets list
                for idx in range(len(offsets[0])):
                    if offsets[0][idx] == i and offsets[1][idx] == j:
                        local_offset = offsets[2][idx]
                        break
                dofs = cast(List[int], dom.dofs_on_interface(j))
                mass_g = self.local_masses[(i, j)] @ y_numpy[local_offset:local_offset+len(dofs)]
                rhs[dofs] += mass_g

            u_i = spsolve(self.volume_mats[iidx], rhs)

            for j in dom.all_neighboring_partitions():
                # find idx in the offsets list
                for idx in range(len(offsets[0])):
                    if offsets[0][idx] == i and offsets[1][idx] == j:
                        local_offset = offsets[2][idx]
                        break
                dofs = dom.dofs_on_interface(j)
                su = self.local_transmission[(i, j)] @ u_i[dofs]
                m_inv_su = spsolve(self.local_masses[(i, j)], su)
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
            u_i = spsolve(self.volume_mats[iidx], rhs)
            for j in dom.all_neighboring_partitions():
                # find idx in the offsets list
                for idx in range(len(offsets[0])):
                    if offsets[0][idx] == i and offsets[1][idx] == j:
                        local_offset = offsets[2][idx]
                        break
                dofs = dom.dofs_on_interface(j)
                m_inv_su = spsolve(self.local_masses[(i, j)], self.local_transmission[(i, j)] @ u_i[dofs])
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



if __name__ == "__main__":


    ndom = 8
    subdomains_on_rank = SubdomainsOnMyRank(ndom)
    create_square(0.04, ndom)
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


   