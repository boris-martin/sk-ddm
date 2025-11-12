from skfem import LinearForm
from skfem.helpers import dot
import numpy as np

@LinearForm(facet=True, dtype=np.complex128)
def plane_wave(v, w):
    k = w['k']
    theta = w['theta']
    n = w['n']
    d = np.array([np.cos(theta), np.sin(theta)])                 # incidence direction
    phase = np.exp(1j * k * (d[0]*w.x[0] + d[1]*w.x[1]))         # e^{ik d·x}
    dn = d[0]*w.n[0] + d[1]*w.n[1]                               # d·n
    return (1j * k) * (dn - 1.0) * phase * v  
