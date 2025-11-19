import mpi4py.MPI as MPI

from src.lib_subdomain.load_local_mesh import LocalGeometry

def evenly_distribute_domains(num_domains: int) -> list[int]:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert num_domains >= size, "Number of domains must be at least number of processes"

    return [i + 1 for i in range(num_domains) if i % size == rank]

if __name__ == "__main__":
    print("Distributed domain assignment:")
    num_total_domains = 8  # Example total number of domains
    assigned_domains = evenly_distribute_domains(num_total_domains)
    print(f"Process {MPI.COMM_WORLD.Get_rank()} assigned domains: {assigned_domains}")