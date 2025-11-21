import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from skfem import asm

from src.lib_subdomain.formulation import (absorbing, helmholtz, mass_bnd,
                                           transmission)
from src.lib_subdomain.load_local_mesh import LocalGeometry


class LocalDTN:
    domain: LocalGeometry

    def __init__(self, domain: LocalGeometry, k:float) -> None:
        self.domain = domain
        self.M_sigma: sp.csr_matrix = self.compute_mass_sigma() # In the full volume indexing
        self.A_neuman: sp.csr_matrix = self.compute_A_neuman()
        self.k = k

    def compute_mass_sigma(self) -> sp.csr_matrix:
        ndof = self.domain.volume_basis.N
        result = sp.csr_matrix((ndof, ndof), dtype=np.complex128)
        for _, basis in enumerate(self.domain.sigma_basis):
            result += asm(mass_bnd, self.domain.mesh, basis, k=self.k)

        return result
    
    def compute_A_neuman(self) -> sp.csr_matrix:
        A = asm(helmholtz, self.domain.mesh, self.domain.volume_basis, k=self.k)
        if self.domain.has_gamma():
            A += asm(absorbing, self.domain.mesh, self.domain.volume_basis, k=self.k)
        return A