"""
force_model.py가 계산한 굽은 중심선(bent_centerline.json)을 따라 실리콘관 단면(환형)을
스윕(pipe)해서 실제로 굽은 형태의 3D 튜브 메쉬를 만든다. 접촉(충돌) 없이 형상만 먼저 검증.

핵심 아이디어: 직선 튜브는 그냥 z축으로 밀어냈지만(extrude), 굽은 튜브는 force_model이 계산한
곡선 경로(spline)를 따라 단면을 밀어내야(pipe) 하고, 그 단면은 경로 시작점의 접선 방향에
수직이어야 함(base에서 접선은 로컬 +x 방향이므로, 단면은 YZ평면에 놓이도록 회전).
"""
import json
import math
import os
import gmsh

D_OUT = 2.0   # mm
D_IN = 1.0    # mm
MESH_SIZE_TUBE = 0.3  # mm

HERE = os.path.dirname(os.path.abspath(__file__))


def build_bent_mesh(centerline_path=None, mesh_size=MESH_SIZE_TUBE,
                     inp_name="bent_tube_mesh.inp", verbose=True):
    if centerline_path is None:
        centerline_path = os.path.join(HERE, "bent_centerline.json")
    with open(centerline_path) as f:
        cl = json.load(f)
    pts = cl["points"]

    gmsh.initialize()
    gmsh.model.add("bent_tube")

    # ── 경로(스플라인) 생성 ──────────────────────────────
    point_tags = [gmsh.model.occ.addPoint(p["x"], p["y"], p["z"]) for p in pts]
    spline_tag = gmsh.model.occ.addSpline(point_tags)
    wire_tag = gmsh.model.occ.addWire([spline_tag])
    gmsh.model.occ.synchronize()

    # ── 단면(환형) 프로파일: 경로 시작점(홀센서 보드좌표 (90,0,3))에서 접선(보드 +y, "자라나는
    #    방향")에 수직이 되도록 XY평면(법선 Z) 기본 원판을 X축 기준 -90도 회전
    #    -> 법선이 Y를 향하게(XZ평면에 놓임) ──
    outer = gmsh.model.occ.addDisk(pts[0]["x"], pts[0]["y"], pts[0]["z"], D_OUT / 2, D_OUT / 2)
    inner = gmsh.model.occ.addDisk(pts[0]["x"], pts[0]["y"], pts[0]["z"], D_IN / 2, D_IN / 2)
    annulus = gmsh.model.occ.cut([(2, outer)], [(2, inner)])
    gmsh.model.occ.rotate(annulus[0], pts[0]["x"], pts[0]["y"], pts[0]["z"], 1, 0, 0, -math.pi / 2)
    gmsh.model.occ.synchronize()

    # ── 스윕(pipe): 단면을 경로를 따라 밀어냄 ──────────────────────────────
    piped = gmsh.model.occ.addPipe(annulus[0], wire_tag)
    gmsh.model.occ.synchronize()
    tube_vol = [t for d, t in piped if d == 3][0]

    # ── 물리그룹: 시작면(고정단), 전체 부피 ──────────────────────────────
    all_surfaces = gmsh.model.getBoundary([(3, tube_vol)], oriented=False)
    fixed_faces = []
    base_y = pts[0]["y"]
    for dim, tag in all_surfaces:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(dim, tag)
        # 시작점에서 y=base_y로 완전히 평평한 면 = 고정단(시작 캡). (보드좌표계에서는
        # y_board가 "자라나는 방향"이라 시작점에서 y가 거의 안 변하는 면이 캡이 됨.
        # 옆면/팁캡은 y가 넓게 퍼져 있어 이 조건 하나로 충분히 구분됨)
        if abs(ymin - base_y) < 1e-6 and abs(ymax - base_y) < 1e-6:
            fixed_faces.append(tag)

    gmsh.model.addPhysicalGroup(3, [tube_vol], name="TUBE")
    if fixed_faces:
        gmsh.model.addPhysicalGroup(2, fixed_faces, name="FIXED")

    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(2)

    inp_path = os.path.join(HERE, inp_name)
    gmsh.write(inp_path)
    msh_path = inp_path.replace(".inp", ".msh")
    gmsh.write(msh_path)

    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    if verbose:
        print(f"메쉬 생성 완료: 절점 {n_nodes}개, FIXED면 후보 {len(fixed_faces)}개")
        print(f"저장: {inp_path}, {msh_path}")

    # 팁 근처 절점 좌표 확인 (force_model 예측 x_L,y_L과 비교용)
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes(3, tube_vol, includeBoundary=True)
    import numpy as np
    coords = np.array(node_coords).reshape(-1, 3)
    tip_target = np.array([cl["x_L"], cl["y_L"], cl.get("board_z", 3.0)])
    dists = np.linalg.norm(coords - tip_target, axis=1)
    nearest_idx = np.argmin(dists)
    if verbose:
        print(f"force_model 예측 팁: ({cl['x_L']:.2f},{cl['y_L']:.2f})")
        print(f"메쉬에서 가장 가까운 절점 거리: {dists[nearest_idx]:.3f}mm "
              f"(작을수록 형상이 잘 맞는것)")

    gmsh.finalize()
    return cl


if __name__ == "__main__":
    build_bent_mesh()
