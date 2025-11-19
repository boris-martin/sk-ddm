import gmsh
import numpy as np

import skfem
import src.tuto.mesh_helpers as mesh_helpers

class LocalGeometry:
    def __init__(self, partition: int) -> None:
        self.partition = partition

        self.omega_tag: int | None = None
        self.gamma_tags: list[int] = []
        self.sigma_tags: dict[int, list[int]] = {}

        self.mesh: skfem.Mesh | None = None
        self.gmshToSK: dict[int, int] = {}
        self.SKToGmsh: dict[int, int] = {}

        self.gamma_facets: list[int] = []
        self.sigma_facets: dict[int, list[int]] = {}

        self.volume_basis: skfem.Basis | None = None
        self.gamma_basis: skfem.FacetBasis | None = None
        self.sigma_basis: dict[int, skfem.FacetBasis] = {}

    def discover_entities(self) -> None:
        entities_2d = gmsh.model.getEntities(2)
        for dim, tag in entities_2d:
            pdim, ptag = gmsh.model.getParent(dim, tag)
            partitions = gmsh.model.getPartitions(dim, tag)
            if self.partition in partitions and pdim == 2 and ptag == 1:
                self.omega_tag = tag
                break

        entities_1d = gmsh.model.getEntities(1)
        for dim, tag in entities_1d:
            pdim, ptag = gmsh.model.getParent(dim, tag)
            partitions = gmsh.model.getPartitions(dim, tag)
            if self.partition not in partitions:
                continue

            if pdim == 2:
                assert len(partitions) == 2, "expected interface"
                other = partitions[0] if partitions[1] == self.partition else partitions[1]
                self.sigma_tags.setdefault(other, []).append(tag)
            elif pdim == 1:
                self.gamma_tags.append(tag)

    def build_skfem_mesh(self):
        assert self.omega_tag is not None, "omega_tag not set"
        skfem_points, self.gmshToSK = mesh_helpers.buildNodes(self.omega_tag)
        self.SKToGmsh = mesh_helpers.reverseNodeDict(self.gmshToSK)
        skfem_elements = mesh_helpers.buildTriangleSet(self.omega_tag, self.gmshToSK)
        self.mesh = skfem.MeshTri(skfem_points, skfem_elements)

    def find_facets(self):
        assert self.mesh is not None, "mesh not built"
        self.facet_dict = mesh_helpers.buildFacetDict(self.mesh)
        self.gamma_facets = mesh_helpers.findFacetsGamma(self.gamma_tags, self.gmshToSK, self.facet_dict)
        self.sigma_facets = self._find_sigma()
        self.all_sigma_facets = [f for facets in self.sigma_facets.values() for f in facets]

    def build_bases(self, order: int = 1):
        assert self.mesh is not None, "mesh not built"
        assert order == 1, "Only order 1 is supported currently"
        self.volume_basis = skfem.Basis(self.mesh, skfem.ElementTriP1())
        self.gamma_basis = skfem.FacetBasis(self.mesh, skfem.ElementTriP1(), facets=self.gamma_facets)
        for j, facets in self.sigma_facets.items():
            self.sigma_basis[j] = skfem.FacetBasis(self.mesh, skfem.ElementTriP1(), facets=list(facets))

    def dofs_on_interface(self, j: int) -> np.ndarray:
        assert j in self.sigma_basis, f"No sigma basis for partition {j}"
        basis = self.sigma_basis[j]
        dofs = sorted(basis.get_dofs(facets=list(self.sigma_facets[j])).flatten())
        return dofs
    
    def all_neighboring_partitions(self) -> list[int]:
        return sorted(self.sigma_tags.keys())
    
    def local_g_size(self):
        return sum(len(dom.dofs_on_interface(j)) for j in self.all_neighboring_partitions())
    
    def volume_size(self):
        assert self.volume_basis is not None, "volume basis not built"
        return self.volume_basis.N

    def init_all(self):
        self.discover_entities()
        self.build_skfem_mesh()
        self.find_facets()
        self.build_bases(order=1)

    def is_initialized(self) -> bool:
        return (
            self.omega_tag is not None and
            self.mesh is not None and
            self.sigma_facets != {} and
            self.volume_basis is not None and
            self.sigma_basis != {}
        )


    def _find_sigma(self):
        facets: dict[int, set[int]] = {}

        for j, tags in self.sigma_tags.items():
            # Normalize tags to a list
            if not isinstance(tags, (list, tuple, set)):
                tags = [tags]

            facets_local: set[int] = set()

            for tag in tags:
                etypes, _, nodes = gmsh.model.mesh.get_elements(1, tag)
                assert len(etypes) == 1, f"Unexpected number of element types for tag {tag}"
                assert etypes[0] == 1,   f"Unexpected element type {etypes[0]} (expected 1=edges)"

                lines = nodes[0].reshape(-1, 2)

                for n1, n2 in lines:
                    sk_n1 = self.gmshToSK[n1]
                    sk_n2 = self.gmshToSK[n2]
                    if sk_n1 > sk_n2:
                        sk_n1, sk_n2 = sk_n2, sk_n1

                    facet_idx = self.facet_dict[(sk_n1, sk_n2)]
                    facets_local.add(facet_idx)

            facets[j] = facets_local

        return facets



if __name__ == "__main__":
    mesh_helpers.create_square(0.03, 4)
    dom = LocalGeometry(partition=1)
    dom.discover_entities()
    print(dom.omega_tag, dom.gamma_tags, dom.sigma_tags)
    print("Building skfem mesh...")
    dom.build_skfem_mesh()
    print("Done.")
    print("Number of nodes:", dom.mesh.nvertices)
    print("Number of elements:", dom.mesh.nelements)
    dom.find_facets()
    print("Gamma facets:", dom.gamma_facets)
    print("Sigma facets:", dom.sigma_facets)
    print("All sigma facets:", dom.all_sigma_facets)
    dom.build_bases(order=1)
    print("Volume basis size:", dom.volume_basis.N)
    print("Sigma bases sizes:", {j: b.N for j, b in dom.sigma_basis.items()})
    print("Gamma basis size:", len(dom.gamma_basis.get_dofs(facets=dom.gamma_facets).flatten()))
    for j in dom.all_neighboring_partitions():
        dofs = dom.dofs_on_interface(j)
        print(f"DOFs on interface with partition {j}:", len(dofs))
    print("Local g size:", dom.local_g_size())
    print("Volume size:", dom.volume_size())