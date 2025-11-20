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
        self.ndomains = ndomains
        self.partitions = evenly_distribute_domains(ndomains)
        self.subdomains: list[LocalGeometry] = [LocalGeometry(partition=p) for p in self.partitions]

    def g_vector_local_size(self) -> int:
        return sum(len(dom.dofs_on_interface(j)) for dom in self.subdomains for j in dom.all_neighboring_partitions())
    
    def g_vector_size_for_domain(self, i: int) -> int:
        dom = self.subdomains[self.partitions.index(i)]
        return sum(len(dom.dofs_on_interface(j)) for j in dom.all_neighboring_partitions())

    def create_petsc_g_vector(self) -> PETSc.Vec:
        local_size = self.g_vector_local_size()
        g_vector = PETSc.Vec().createMPI((local_size, PETSc.DECIDE), comm=MPI.COMM_WORLD)
        g_vector.set(0.0)
        return g_vector
    
    def local_offset_list(self) -> tuple[list[int], list[int], list[int]]:
        """
        Returns an ordered lit of [i, j, offset] where i is the local subdomain index, j is the neighboring partition,
        and offset is the starting index in the rank-local numbering.
        i and j are 1-based.
        """
        iset, jset, offsets = [], [], []
        current_offset = 0
        for i in self.partitions:
            dom = self.subdomains[self.partitions.index(i)]
            for j in dom.all_neighboring_partitions():
                dof_count = len(dom.dofs_on_interface(j))
                iset.append(i)
                jset.append(j)
                offsets.append(current_offset)
                current_offset += dof_count
        return iset, jset, offsets
    
    def offset_of_domain(self, i: int) -> int:
        """
        Returns the starting offset of subdomain i in the rank-local g-vector numbering.
        i is 1-based.
        """
        iset, jset, offsets = self.local_offset_list()
        for idx, subdomain_id in enumerate(iset):
            if subdomain_id == i:
                return offsets[idx]
        raise ValueError(f"Subdomain {i} not found on this rank.")

    def build_swap_operator(self) -> PETSc.Mat:
        """
        Build a PETSc permutation matrix S such that
            (S g)_(j,i,·) = g_(i,j,·)
        i.e. it swaps g_ij <-> g_ji while preserving the ordering along the interface.

        The matrix layout matches the MPI distribution used for the g-vector.
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # ----- 1. Local and global sizes consistent with g-vector -----
        local_size = self.g_vector_local_size()
        global_size = comm.allreduce(local_size, op=MPI.SUM)

        # PETSc will use contiguous ranges; we reproduce them explicitly
        # so we can compute global indices from local offsets.
        scan_total = comm.scan(local_size)          # prefix sum up to *this* rank
        local_start = scan_total - local_size       # first global index on this rank
        local_end   = local_start + local_size      # one past last

        # ----- 2. Build the matrix with the same row distribution -----
        swap_mat = PETSc.Mat().createAIJ(
            size=((local_size, PETSc.DETERMINE), (local_size, PETSc.DETERMINE)),
            comm=comm,
        )
        swap_mat.setUp()

        # Check what rows this rank actually owns (should match [local_start, local_end))
        rstart, rend = swap_mat.getOwnershipRange()

        # ----- 3. Collect interface segments (i, j) -> [global_start, length] -----

        # Map partition id -> LocalGeometry instance (for fast lookup)
        part2dom = {p: dom for p, dom in zip(self.partitions, self.subdomains)}

        iset, jset, offsets = self.local_offset_list()
        assert len(iset) == len(jset) == len(offsets)

        # Local list of interface segments on this rank:
        # (i, j, global_start, length)
        local_segments: list[tuple[int, int, int, int]] = []

        for i, j, offset in zip(iset, jset, offsets):
            dom = part2dom[i]
            dofs_ij = dom.dofs_on_interface(j)
            n_ij = len(dofs_ij)
            assert n_ij >= 0

            gstart_ij = local_start + offset
            assert 0 <= gstart_ij <= global_size
            assert 0 <= gstart_ij + n_ij <= global_size

            local_segments.append((i, j, gstart_ij, n_ij))

        # Gather all interface segments on all ranks
        all_segments = comm.allgather(local_segments)

        # Flatten and build a global dictionary: (i, j) -> (global_start, length)
        seg_dict: dict[tuple[int, int], tuple[int, int]] = {}
        for rank_segs in all_segments:
            for i, j, gstart, n in rank_segs:
                key = (i, j)
                # You can tighten this to assert if you expect uniqueness.
                if key in seg_dict:
                    # Sanity: same segment should have same metadata
                    old_gstart, old_n = seg_dict[key]
                    assert old_gstart == gstart
                    assert old_n == n
                else:
                    seg_dict[key] = (gstart, n)

        # ----- 4. Insert permutation entries: g_ij <-> g_ji -----
        #
        # For each unordered pair {i, j}, with i < j, we create:
        #   for k in [0, n):
        #       row = global index of (j,i,k)
        #       col = global index of (i,j,k)
        #       S[row, col] = 1
        #       S[col, row] = 1
        #
        # This guarantees S^2 = I on the subspace spanned by interface DOFs.

        for (i, j), (gstart_ij, n_ij) in seg_dict.items():
            if i == j:
                # Self-interfaces (if any) are ignored or could be set to identity.
                continue

            # Only handle each unordered pair once
            if i > j:
                continue

            partner_key = (j, i)
            assert partner_key in seg_dict, f"Missing partner interface ({j},{i}) for ({i},{j})"

            gstart_ji, n_ji = seg_dict[partner_key]
            assert n_ij == n_ji, f"Interface size mismatch between ({i},{j}) and ({j},{i})"

            for k in range(n_ij):
                # Map g_ij[k] -> g_ji[k]
                row_ji = gstart_ji + k  # output index
                col_ij = gstart_ij + k  # input index

                if rstart <= row_ji < rend:
                    swap_mat.setValue(row_ji, col_ij, 1.0)

                # Map g_ji[k] -> g_ij[k] (the inverse direction)
                row_ij = gstart_ij + k
                col_ji = gstart_ji + k

                if rstart <= row_ij < rend:
                    swap_mat.setValue(row_ij, col_ji, 1.0)

        swap_mat.assemblyBegin()
        swap_mat.assemblyEnd()

        return swap_mat
    

if __name__ == "__main__":
    print("Distributed domain assignment:")
    num_total_domains = 3  # Example total number of domains
    assigned_domains = evenly_distribute_domains(num_total_domains)
    print(f"Process {MPI.COMM_WORLD.Get_rank()} assigned domains: {assigned_domains}")
    print("Rank of all domains: ", [rank_of_domain(i + 1, num_total_domains) for i in range(num_total_domains)])
    
    create_square(0.05, num_total_domains)
    MPI.COMM_WORLD.Barrier()

    subdomains_on_rank = SubdomainsOnMyRank(num_total_domains)
    for subdomain in subdomains_on_rank.subdomains:
        subdomain.init_all()
        print(f"Process {MPI.COMM_WORLD.Get_rank()} initialized subdomain {subdomain.partition} with omega_tag {subdomain.omega_tag}")
    print(f"Process {MPI.COMM_WORLD.Get_rank()} total local g-vector size: {subdomains_on_rank.g_vector_local_size()}")

    x = subdomains_on_rank.create_petsc_g_vector()
    x.set(MPI.COMM_WORLD.Get_rank() + 1.0)
    global_g_lenth = x.getSize()
    print(f"Process {MPI.COMM_WORLD.Get_rank()} created PETSc g-vector of local size {x.getLocalSize()} (global size {global_g_lenth})")

    print(f"Process {MPI.COMM_WORLD.Get_rank()} has offsets: {subdomains_on_rank.local_offset_list()}")
    iset, jset, offsets = subdomains_on_rank.local_offset_list()
    gathered_offsets = MPI.COMM_WORLD.gather((iset, jset, offsets), root=0)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Gathered offsets from all ranks:")
        for rank, (iset, jset, offsets) in enumerate(gathered_offsets):
            print(f"Rank {rank}:")
            for i, j, offset in zip(iset, jset, offsets):
                print(f"  Subdomain {i}, Neighbor {j}, Offset {offset}")


    swap = subdomains_on_rank.build_swap_operator()
    swap.view()
    swap.view(PETSc.Viewer.DRAW(comm=MPI.COMM_WORLD))
    PETSc.Sys.sleep(5)  # Ensure output order

    nnz_global = swap.getInfo()[ 'nz_allocated' ]
    print("Global nnz:", nnz_global)
    y = x.duplicate()
    y = swap @ x
    print("Result of swap operator on g-vector:")
    y.view()

    print("Global and local sizes of swap result:", swap.getSize(), swap.getLocalSize())
    print("Global and local sizes of original g-vector:", x.getSize(), x.getLocalSize())
