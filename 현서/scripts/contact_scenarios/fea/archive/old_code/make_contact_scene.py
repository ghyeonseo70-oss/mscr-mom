"""
튜브 + 강체 구(indenter) 두 물체를 한 메쉬에 만들어서 CalculiX 접촉해석용으로 내보냄.
구는 튜브 옆면(접촉위치 지정 가능)에 살짝 닿아있는 위치에서 시작해서, 해석 스텝에서 안쪽으로
더 밀어넣으며(변위제어) 실제 접촉면적·압력을 계산하게 한다.

sweep_contact.py에서 여러 시나리오(접촉위치, 구 크기)를 반복 생성할 수 있도록 함수화함.
"""
import os
import gmsh

D_OUT = 2.0     # mm, 튜브 바깥지름
D_IN = 1.0      # mm, 튜브 안지름
L_TEST = 100.0  # mm, 실제 로봇 전체 길이(논문/force_model.py 기준)와 동일하게 맞춤
MESH_SIZE_TUBE = 0.3   # mm

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def build_mesh(contact_z=L_TEST / 2, ball_r=0.4, gap0=0.02, mesh_size_ball=0.15,
               inp_name="contact_mesh.inp", sets_name="contact_node_sets.inp", verbose=True):
    """튜브+구 메쉬를 만들어 inp_name/sets_name으로 저장. 구 중심좌표(mm)를 반환."""
    gmsh.initialize()
    gmsh.model.add("tube_contact")

    outer = gmsh.model.occ.addDisk(0, 0, 0, D_OUT / 2, D_OUT / 2)
    inner = gmsh.model.occ.addDisk(0, 0, 0, D_IN / 2, D_IN / 2)
    annulus = gmsh.model.occ.cut([(2, outer)], [(2, inner)])
    gmsh.model.occ.synchronize()
    ext = gmsh.model.occ.extrude(annulus[0], 0, 0, L_TEST)
    gmsh.model.occ.synchronize()
    tube_vol = [e[1] for e in ext if e[0] == 3][0]

    ball_x = D_OUT / 2 + gap0 + ball_r
    ball_center = (ball_x, 0, contact_z)
    ball_tag = gmsh.model.occ.addSphere(*ball_center, ball_r)
    gmsh.model.occ.synchronize()

    surfaces_tube = gmsh.model.getBoundary([(3, tube_vol)], oriented=False)
    fixed_faces, outer_faces = [], []
    for dim, tag in surfaces_tube:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(dim, tag)
        if abs(zmax - zmin) < 1e-6 and abs(zmin) < 1e-3:
            fixed_faces.append(tag)
        elif abs(zmax - zmin) > L_TEST - 1e-3:
            r_bbox = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
            if r_bbox > D_OUT / 4:
                outer_faces.append(tag)

    surfaces_ball = gmsh.model.getBoundary([(3, ball_tag)], oriented=False)
    ball_faces = [tag for dim, tag in surfaces_ball]

    gmsh.model.addPhysicalGroup(3, [tube_vol], name="TUBE")
    gmsh.model.addPhysicalGroup(3, [ball_tag], name="BALL")
    gmsh.model.addPhysicalGroup(2, fixed_faces, name="FIXED")
    gmsh.model.addPhysicalGroup(2, outer_faces, name="TUBE_OUTER")
    gmsh.model.addPhysicalGroup(2, ball_faces, name="BALL_SURF")

    gmsh.model.mesh.setSize(gmsh.model.getBoundary([(3, ball_tag)], recursive=True), mesh_size_ball)
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_ball)
    gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE_TUBE)

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(2)  # C3D10 (shear locking 회피)

    inp_path = os.path.join(OUT_DIR, inp_name)
    gmsh.write(inp_path)

    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    if verbose:
        print(f"메쉬 생성 완료: 절점 {n_nodes}개")

    # CalculiX는 CPS(2D 평면)요소가 파일에 정의만 돼 있어도 gen3delem에서 z=0 평면 체크로
    # 에러를 내므로, 접촉에 실제 쓰는 BALL_SURF(Surface5)만 남기고 나머지는 제거.
    # BALL_SURF 자체도 CPS6->S6(셸, 순수 표면참조용)로 바꿔 그 체크를 피함.
    with open(inp_path, encoding="latin-1") as f:
        lines = f.readlines()

    remove_elem_sets = {"Surface1", "Surface2", "Surface3"}
    remove_elset_names = {"FIXED", "TUBE_OUTER"}
    filtered = []
    skip = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("*ELEMENT") and any(f"ELSET={n}" in stripped for n in remove_elem_sets):
            skip = True
            continue
        if stripped.startswith("*ELSET") and any(f"ELSET={n}" in stripped for n in remove_elset_names):
            skip = True
            continue
        if skip and stripped.startswith("*"):
            skip = False
        if skip:
            continue
        filtered.append(line)
    filtered = [line.replace("type=CPS6", "type=S6") for line in filtered]

    with open(inp_path, "w", encoding="latin-1") as f:
        f.writelines(filtered)

    def nodes_of(dim, name):
        for d, t in gmsh.model.getPhysicalGroups(dim):
            if gmsh.model.getPhysicalName(d, t) == name:
                return gmsh.model.mesh.getNodesForPhysicalGroup(d, t)[0]
        raise RuntimeError(f"physical group {name} not found")

    fixed_nodes = nodes_of(2, "FIXED")
    outer_nodes = nodes_of(2, "TUBE_OUTER")
    ball_nodes = nodes_of(2, "BALL_SURF")
    ball_vol_nodes = nodes_of(3, "BALL")

    sets_path = os.path.join(OUT_DIR, sets_name)
    with open(sets_path, "w") as f:
        for name, nodes in [("N_FIXED", fixed_nodes), ("N_TUBE_OUTER", outer_nodes),
                             ("N_BALL_SURF", ball_nodes), ("N_BALL_ALL", ball_vol_nodes)]:
            f.write(f"*NSET, NSET={name}\n")
            for i in range(0, len(nodes), 10):
                f.write(",".join(str(int(n)) for n in nodes[i:i + 10]) + ",\n")

    if verbose:
        print(f"저장: {inp_path}, {sets_path}")
        print(f"  N_FIXED={len(fixed_nodes)}, N_TUBE_OUTER={len(outer_nodes)}, "
              f"N_BALL_SURF={len(ball_nodes)}, N_BALL_ALL={len(ball_vol_nodes)}")
        print(f"접촉위치: z={contact_z}mm, 구중심=({ball_center[0]:.3f},{ball_center[1]:.3f},{ball_center[2]:.3f}), "
              f"반지름={ball_r}mm, 초기간격={gap0}mm")

    gmsh.finalize()
    return ball_center


if __name__ == "__main__":
    build_mesh()
