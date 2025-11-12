import gmsh

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