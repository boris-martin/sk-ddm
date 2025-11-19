import mpi4py.MPI as MPI
import petsc4py.PETSc as PETSc

from src.lib_subdomain.load_local_mesh import LocalGeometry
from src.tuto.mesh_helpers import create_square

def evenly_distribute_domains(num_domains: int) -> list[int]:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert num_domains >= size, "Number of domains must be at least number of processes"

    return [i + 1 for i in range(num_domains) if i % size == rank]

def rank_of_domain(domain_id: int, num_domains: int) -> int:
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    assert 1 <= domain_id <= num_domains, "Domain ID out of range"

    return (domain_id - 1) % size

class SubdomainsOnMyRank:
    def __init__(self, ndomains: int):
        # So far, we assume all partitions are in a unique file
        self.partitions = evenly_distribute_domains(ndomains)
        self.subdomains = [LocalGeometry(partition=p) for p in self.partitions]

    def g_vector_local_size(self) -> int:
        return sum(len(dom.dofs_on_interface(j)) for dom in self.subdomains for j in dom.all_neighboring_partitions())
    
    def create_petsc_g_vector(self) -> PETSc.Vec:
        local_size = self.g_vector_local_size()
        g_vector = PETSc.Vec().createMPI((local_size, PETSc.DECIDE), comm=MPI.COMM_WORLD)
        g_vector.set(0.0)
        return g_vector

if __name__ == "__main__":
    print("Distributed domain assignment:")
    num_total_domains = 8  # Example total number of domains
    assigned_domains = evenly_distribute_domains(num_total_domains)
    print(f"Process {MPI.COMM_WORLD.Get_rank()} assigned domains: {assigned_domains}")
    print("Rank of all domains: ", [rank_of_domain(i + 1, num_total_domains) for i in range(num_total_domains)])
    create_square(0.03, num_total_domains)
    subdomains_on_rank = SubdomainsOnMyRank(num_total_domains)
    for subdomain in subdomains_on_rank.subdomains:
        subdomain.init_all()
        print(f"Process {MPI.COMM_WORLD.Get_rank()} initialized subdomain {subdomain.partition} with omega_tag {subdomain.omega_tag}")
    print(f"Process {MPI.COMM_WORLD.Get_rank()} total local g-vector size: {subdomains_on_rank.g_vector_local_size()}")

    x = subdomains_on_rank.create_petsc_g_vector()