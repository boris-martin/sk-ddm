import gmsh
import numpy as np

def create_square(lc = 0.02, ndom = 2, name="square"):
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.model.add("unit-square")
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0, lc)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s1 = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.partition(ndom)
    gmsh.write(f"{name}.msh")


def find_entities_on_domain(target_partition):
    omega_tag = -1
    gamma_tags = []
    sigma_tags = dict() # j to tag

    for dim, tag in  gmsh.model.get_entities(-1):
        print(f"Dimension: {dim}, Tag: {tag}")
        pdim, ptag = gmsh.model.getParent(dim, tag)
        partitions = gmsh.model.getPartitions(dim, tag)
        if target_partition not in partitions:
            continue
        if dim == 2 and ptag == 1:
            omega_tag = tag
        elif dim == 1:
            if pdim == 2:
                assert(len(partitions) == 2)
                other_part = partitions[0] if partitions[1] == target_partition else partitions[1]
                sigma_tags[other_part] = tag
            elif pdim == 1:
                gamma_tags.append(tag)

    return omega_tag, gamma_tags, sigma_tags


def buildNodes(surface_tag):
    gmshNodeTagToCoords = dict()
    gmshNodeTagToSKFemIndex = dict()
    ntags, ncoords, _ = gmsh.model.mesh.get_nodes(2, surface_tag, includeBoundary=True)
    print(f"Number of nodes: {len(ntags)} and coordinates shape: {ncoords.shape}")
    counter = 0
    # Iterate over zip of tags and 3 coords
    for nodeTag, coords in zip(ntags, ncoords.reshape(-1, 3)):
        gmshNodeTagToCoords[nodeTag] = coords[:2]
        gmshNodeTagToSKFemIndex[nodeTag] = counter
        counter += 1
    assert(len(gmshNodeTagToSKFemIndex) == len(ntags))
    assert(len(gmshNodeTagToCoords) == len(ntags))
    # SKFEM node list
    skfem_points = np.zeros((2, len(ntags)))
    for gmshTag, skfemIndex in gmshNodeTagToSKFemIndex.items():
        skfem_points[:, skfemIndex] = gmshNodeTagToCoords[gmshTag]

    return skfem_points, gmshNodeTagToSKFemIndex

def buildTriangleSet(surface_tag, gmshToSK):
    etypes, _, nodes = gmsh.model.mesh.get_elements(2, surface_tag)
    # We expect only triangles
    assert(len(etypes) == 1)
    assert(etypes[0] == 2)  # triangle
    triangles = nodes[0].reshape(-1, 3)
    skfem_elements = np.zeros(triangles.shape, dtype=int)
    for i in range(triangles.shape[0]):
        for j in range(3):
            gmshNodeTag = triangles[i, j]
            skfemIndex = gmshToSK[gmshNodeTag]
            skfem_elements[i, j] = skfemIndex
    return skfem_elements.T

def buildFacetDict(mesh):
    facets = mesh.facets
    facets_dict = dict()
    for j in range(facets.shape[1]):
        assert(facets[0, j] < facets[1, j])
        facets_dict[(facets[0, j], facets[1, j])] = j
    return facets_dict