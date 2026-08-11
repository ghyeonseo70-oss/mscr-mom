"""find_safe_force_range.py가 테스트한 것과 완전히 같은 방식(임의 방향 점하중, 접촉공이 아님)으로
FEA를 검증하기 위한 메쉬 생성. make_bent_tube_mesh.py(공 없는 굽은 튜브)를 기반으로, 여기에
접촉위치 s에 가장 가까운 바깥표면 절점들을 "N_LOAD" 세트로 추가 추출한다(그 위치에 *CLOAD로
직접 임의방향 힘을 걸고 변위를 읽기 위함 - 공으로 누르는 접촉해석이 아니라 해석모델의
solve_shape_robust(loads=[{"type":"point",...}])와 동일한 하중 조건).
"""
import json
import math
import os
import sys

import gmsh
import numpy as np

D_OUT = 2.0
D_IN = 1.0
MESH_SIZE_TUBE = 0.3

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "force_model"))


def _point_at_s(points, s_target):
    s_arr = np.array([p["s"] for p in points])
    x_arr = np.array([p["x"] for p in points])
    y_arr = np.array([p["y"] for p in points])
    z_arr = np.array([p["z"] for p in points])
    th_arr = np.array([p["theta_deg"] for p in points])
    return (np.interp(s_target, s_arr, x_arr), np.interp(s_target, s_arr, y_arr),
            np.interp(s_target, s_arr, z_arr), np.interp(s_target, s_arr, th_arr))


def build_mesh(centerline_path, contact_s, mesh_size=MESH_SIZE_TUBE,
               inp_name="point_load_mesh.inp", sets_name="point_load_node_sets.inp",
               verbose=True, load_radius=1.0):
    with open(centerline_path) as f:
        cl = json.load(f)
    pts = cl["points"]
    target_x, target_y, target_z, _ = _point_at_s(pts, contact_s)

    gmsh.initialize()
    gmsh.model.add("bent_tube_point_load")

    point_tags = [gmsh.model.occ.addPoint(p["x"], p["y"], p["z"]) for p in pts]
    spline_tag = gmsh.model.occ.addSpline(point_tags)
    wire_tag = gmsh.model.occ.addWire([spline_tag])
    gmsh.model.occ.synchronize()

    outer = gmsh.model.occ.addDisk(pts[0]["x"], pts[0]["y"], pts[0]["z"], D_OUT / 2, D_OUT / 2)
    inner = gmsh.model.occ.addDisk(pts[0]["x"], pts[0]["y"], pts[0]["z"], D_IN / 2, D_IN / 2)
    annulus = gmsh.model.occ.cut([(2, outer)], [(2, inner)])
    gmsh.model.occ.rotate(annulus[0], pts[0]["x"], pts[0]["y"], pts[0]["z"], 1, 0, 0, -math.pi / 2)
    gmsh.model.occ.synchronize()

    piped = gmsh.model.occ.addPipe(annulus[0], wire_tag)
    gmsh.model.occ.synchronize()
    tube_vol = [t for d, t in piped if d == 3][0]

    all_surfaces = gmsh.model.getBoundary([(3, tube_vol)], oriented=False)
    base_y = pts[0]["y"]
    path_y_span = max(p["y"] for p in pts) - min(p["y"] for p in pts)
    fixed_faces, tip_faces, wall_candidates = [], [], []
    for dim, tag in all_surfaces:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(dim, tag)
        if abs(ymin - base_y) < 1e-6 and abs(ymax - base_y) < 1e-6:
            fixed_faces.append(tag)
        elif (ymax - ymin) < 0.3 * path_y_span:
            tip_faces.append(tag)
        else:
            wall_candidates.append((tag, zmax - zmin))
    wall_candidates.sort(key=lambda t: t[1], reverse=True)
    outer_faces = [wall_candidates[0][0]] if wall_candidates else []

    gmsh.model.addPhysicalGroup(3, [tube_vol], name="TUBE")
    gmsh.model.addPhysicalGroup(2, fixed_faces, name="FIXED")
    gmsh.model.addPhysicalGroup(2, outer_faces, name="TUBE_OUTER")
    gmsh.model.addPhysicalGroup(2, tip_faces, name="TIP")

    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(2)

    inp_path = os.path.join(HERE, inp_name)
    gmsh.write(inp_path)

    # gmsh가 표면(2D 물리그룹) 태깅 때문에 CPS3/CPS6 면요소도 같이 써놓는데, 이걸 그대로
    # 두면 CalculiX가 "plane stress/strain 요소는 z=0 평면에 있어야 한다"는 에러를 냄
    # (실제로 겪음 - make_bent_contact_scene.py는 볼 껍질(S6)이 필요해서 남기고 지웠지만,
    # 여기는 절점하중만 걸 거라 표면요소 자체가 필요 없어서 전부 제거).
    with open(inp_path, encoding="latin-1") as f:
        lines = f.readlines()
    filtered, skip = [], False
    for line in lines:
        stripped = line.strip().upper()
        if stripped.startswith("*ELEMENT"):
            skip = "CPS3" in stripped or "CPS6" in stripped
            if skip:
                continue
        elif stripped.startswith("*"):
            skip = False
        if not skip:
            filtered.append(line)
    with open(inp_path, "w", encoding="latin-1") as f:
        f.writelines(filtered)

    def nodes_of(dim, name):
        for d, t in gmsh.model.getPhysicalGroups(dim):
            if gmsh.model.getPhysicalName(d, t) == name:
                return gmsh.model.mesh.getNodesForPhysicalGroup(d, t)
        raise RuntimeError(f"physical group {name} not found")

    fixed_nodes, _ = nodes_of(2, "FIXED")
    outer_nodes, outer_coords = nodes_of(2, "TUBE_OUTER")
    outer_coords = np.array(outer_coords).reshape(-1, 3)

    target = np.array([target_x, target_y, target_z])
    dists = np.linalg.norm(outer_coords - target, axis=1)
    load_mask = dists < load_radius
    load_nodes = outer_nodes[load_mask]
    if len(load_nodes) == 0:
        nearest = np.argsort(dists)[:20]
        load_nodes = outer_nodes[nearest]

    sets_path = os.path.join(HERE, sets_name)
    with open(sets_path, "w") as f:
        for name, nodes in [("N_FIXED", fixed_nodes), ("N_LOAD", load_nodes)]:
            f.write(f"*NSET, NSET={name}\n")
            for i in range(0, len(nodes), 10):
                f.write(",".join(str(int(n)) for n in nodes[i:i + 10]) + ",\n")

    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    if verbose:
        print(f"메쉬 생성 완료: 절점 {n_nodes}개, N_FIXED={len(fixed_nodes)}, "
              f"N_LOAD={len(load_nodes)}개 (목표점 {target}, 반경 {load_radius}mm)")

    gmsh.finalize()
    return {"target_point": target.tolist(), "n_load_nodes": int(len(load_nodes))}


if __name__ == "__main__":
    build_mesh(centerline_path=os.path.join(HERE, "bent_centerline.json"), contact_s=50.0)
