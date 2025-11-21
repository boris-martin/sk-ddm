import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from skfem import asm

from src.lib_subdomain.formulation import (absorbing, helmholtz, mass_bnd,
                                           transmission)
from src.lib_subdomain.load_local_mesh import LocalGeometry


class LocalDTN:
    domain: LocalGeometry
    M_sigma: sp.csr_matrix
    A_neuman: sp.csr_matrix
    k: float

    def __init__(self, domain: LocalGeometry, k:float) -> None:
        self.domain = domain
        self.k = k
        self.M_sigma: sp.csr_matrix = self.compute_mass_sigma() # In the full volume indexing
        self.A_neuman: sp.csr_matrix = self.compute_A_neuman()

    def compute_mass_sigma(self) -> sp.csr_matrix:
        return asm(mass_bnd, self.domain.all_sigma_basis)
    
    def compute_A_neuman(self) -> sp.csr_matrix:
        A = asm(helmholtz,  self.domain.volume_basis, k=self.k)
        if True and self.domain.has_gamma():
            A += asm(absorbing, self.domain.gamma_basis, k=self.k)
        return A
    
    def build_basis(self, nev):
        from scipy.sparse.linalg import eigs
        
        # Add these checks
        print(f"A_neuman shape: {self.A_neuman.shape}")
        print(f"M_sigma shape: {self.M_sigma.shape}")
        # Note: toarray() can be slow/memory intensive for very large matrices
        M_rank = np.linalg.matrix_rank(self.M_sigma.toarray(), tol=1e-6)
        print(f"M_sigma Rank: {M_rank} out of {self.M_sigma.shape[0]}")
        # ---
        
        vals, vecs = eigs(self.A_neuman, M=self.M_sigma, k=nev, which='LM', sigma=0)
        if max(np.abs(vals)) > 1e10:
            print("Warning: Large eigenvalues detected, possible singularity in M_sigma.")

        self.vals = vals
        self.vecs = vecs

    def get_eigvals(self):
        return self.vals
    def get_eigvecs(self):
        return self.vecs