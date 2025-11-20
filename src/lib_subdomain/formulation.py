from src.lib_subdomain.distributed_domain_set import SubdomainsOnMyRank
from src.lib_subdomain.load_local_mesh import LocalGeometry
from src.tuto.mesh_helpers import create_square

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
    x_left = x[0] < 0.5

    # k eff = k * (1 + x²)
    k_eff = k * (1.0 + np.sin(x[0] * 4 * 3.1415) ** 2) * x_left + k * (
        1.0 + np.cos(x[0] * 4 * 3.1415) ** 2
    ) * (~x_left)
    k_eff = k # ignore heterogeneity for transmission
    return np.complex128(-1j * k_eff * u * conjugate(v))


# Time convention: S = iku, we remove S_gamma on the boundary (-dnu v term)

class Formulation:
    def __init__(self, domains: SubdomainsOnMyRank, k: float):
        self.domains: SubdomainsOnMyRank = domains
        self.k = k
        self.volume_mats: list[csr_matrix]

    def assemble_volume(self):
        self.volume_mats = []
        for dom in self.domains.subdomains:
            A = skfem.asm(helmholtz, dom.volume_basis , k=self.k)
            if dom.has_gamma():
                A -= skfem.asm(absorbing, dom.gamma_basis, k=self.k)
            for _, basis in dom.sigma_basis.items():
                A -= skfem.asm(transmission, basis, k=self.k)
            self.volume_mats.append(A.tocsr())


if __name__ == "__main__":
    ndom = 4
    subdomains_on_rank = SubdomainsOnMyRank(ndom)
    create_square(0.04, ndom)
    for dom in subdomains_on_rank.subdomains:
        dom.init_all()

    print("Partitions on this rank:", subdomains_on_rank.partitions)
    for dom in subdomains_on_rank.subdomains:
        print(f"Domain {dom.partition} neighbors:", dom.all_neighboring_partitions())

    formulation = Formulation(subdomains_on_rank, k=10.0)
    formulation.assemble_volume()