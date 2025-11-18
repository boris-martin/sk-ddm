# ddm_utils.py
import numpy as np
from skfem import MeshTri, FacetBasis, ElementTriP1, LinearForm, BilinearForm
from skfem.helpers import grad, dot
from numpy import conjugate
import scipy.sparse.linalg as spla
import scipy.sparse

from mesh_helpers import create_square, find_entities_on_domain
import mesh_helpers

def build_offsets_and_total_size(subdomains: list):
    """
    Build:
      - offsets[(idom, j)] : starting index of interface (idom, j) in global g
      - istart[idom]       : starting index for subdomain idom in global g-block
      - total_g_size       : total length of the global g vector

    Assumes each g[idom-1] is a dict j -> (fs, proj),
    and uses sorted(j) to define a consistent ordering.
    """
    g = [subdomain.gi for subdomain in subdomains]
    ndom = len(subdomains)
    offsets = {}
    istart = []
    counter = 0
    for idom in range(1, ndom + 1):
        istart.append(counter)
        gi = g[idom - 1]
        for j in sorted(gi):
            _, proj, _ = gi[j]
            offsets[(idom, j)] = counter
            counter += proj.shape[0]
    istart.append(counter)
    total_g_size = counter
    return offsets, istart, total_g_size


def build_full_rhs(phys_b_list):
    """
    Concatenate per-subdomain physical RHS contributions into a single global RHS.
    """
    return np.concatenate(phys_b_list)


class LocalDDMSolver:
    """
    Lightweight wrapper around a set of per-domain DDM local solves that operate
    on disjoint ranges of the global interface vector g.

    Attributes
    ----------
    local_solves : list[LinearOperator]
        One linear operator per subdomain.
    ranges : list[tuple[int, int]]
        List of (start,end) ranges that reference blocks inside global g.
    total_g_size : int
        Length of the global interface vector.
    """

    def __init__(self, local_solves, istart, ndom, total_g_size):
        self.local_solves = local_solves
        self.ranges = [(istart[i-1], istart[i]) for i in range(1, ndom+1)]
        self.total_g_size = total_g_size
        self.swap = None

        # Defensive runtime check
        for (s, e), op in zip(self.ranges, self.local_solves):
            assert op.shape[0] == (e - s), (
                f"Local operator shape {op.shape[0]} does not match block size {e-s}"
            )

    def apply(self, g: np.ndarray) -> np.ndarray:
        """Apply and return local block-solve results on each domain."""
        assert g.shape[0] == self.total_g_size, (
            f"Expected g of size {self.total_g_size}, got {g.shape[0]}"
        )

        g_solved = np.zeros_like(g, dtype=np.complex128)

        for (s, e), op in zip(self.ranges, self.local_solves):
            g_solved[s:e] = op.matvec(g[s:e])

        return g_solved
    
    def set_swap(self, swap_op: scipy.sparse.csr_matrix):
        """Set a swap operator to be applied after local solves."""
        self.swap = swap_op
        self.T = spla.LinearOperator(
            swap_op.shape,
            matvec=lambda x: self.apply(self.swap @(x))
        )
        # A = Id - T
        self.A = spla.LinearOperator(
            swap_op.shape,
            matvec=lambda x: x - self.T.matvec(x)
        )


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
    x = w['x']
    # k eff = k * (1 + x²)
    k_eff = k * (1.0 + x[0]**2 + x[1]**2)
    #k_eff = k
    return np.complex128(-(-0.0 + 1j) * k_eff * u * conjugate(v))



class Subdomain:
    def __init__(self, idom, omega_tag: int, gamma_tags: list, sigma_tags: dict):
        self.idom = idom
        self.omega_tag = omega_tag
        self.gamma_tags = gamma_tags
        self.sigma_tags = sigma_tags

        self.skfem_points, self.gmshToSK = mesh_helpers.buildNodes(omega_tag)
        self.SKToGmsh = mesh_helpers.reverseNodeDict(self.gmshToSK)
        self.elements = mesh_helpers.buildTriangleSet(omega_tag, self.gmshToSK)
        self.mesh = MeshTri(self.skfem_points, self.elements)
        self.facets_dict = mesh_helpers.buildFacetDict(self.mesh) # Pair of nodes to face ID
        self.all_sigma_facets = mesh_helpers.findFullSigma(sigma_tags, self.gmshToSK, self.facets_dict)

        self.ker = []
        self.gi = dict()

        self.mats = dict()

    def add_kernel_mode(self, kernel_column: int, node_sk: int, jplus: int, jminus: int):
        self.ker.append({'kernel_column': kernel_column, 'node_sk': node_sk, 'jplus': jplus, 'jminus': jminus})

    def set_problem_mat(self, mat):
        self.mats['problem'] = mat

    def get_problem_mat(self):
        return self.mats['problem']
    
    def set_rhs_mat(self, mat):
        self.mats['rhs'] = mat

    def get_rhs_mat(self):
        return self.mats['rhs']