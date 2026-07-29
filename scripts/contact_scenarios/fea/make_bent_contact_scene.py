"""
굽은 튜브(force_model.py 예측 중심선을 스윕한 형상) + 강체 구(인덴터)를 한 메쉬로 만들어서
CalculiX 접촉해석용으로 내보냄. 접촉위치는 호길이(s, mm, 베이스=0부터)로 지정하면
그 지점의 접선각도(theta)를 이용해 법선 방향으로 구를 배치한다.
"""
import json
import math
import os
import gmsh
import numpy as np

D_OUT = 2.0    # mm
D_IN = 1.0     # mm
MESH_SIZE_TUBE = 0.3
MESH_SIZE_BALL = 0.15

HERE = os.path.dirname(os.path.abspath(__file__))


def _point_and_normal_at_s(points, s_target):
    """중심선 점 목록에서 s_target 위치의 (x,y,z)와 법선 벡터(접선에 수직, 보드 xy평면 내)를 보간.
    theta_deg는 force_model의 로컬좌표계 각도(get_bent_centerline.py 참고)라서, 보드좌표계
    (x_board=90+y_local, y_board=x_local)로 변환된 접선/법선 공식을 써야 함:
    로컬 접선(cos,sin,0) -> 보드 접선(sin,cos,0)이 되므로, 그에 수직인 법선은
    (-cos(theta), sin(theta), 0)이 됨(기존 로컬용 공식 (-sin,cos,0)에서 치환)."""
    s_arr = np.array([p["s"] for p in points])
    x_arr = np.array([p["x"] for p in points])
    y_arr = np.array([p["y"] for p in points])
    z_arr = np.array([p["z"] for p in points])
    th_arr = np.array([p["theta_deg"] for p in points])
    x = np.interp(s_target, s_arr, x_arr)
    y = np.interp(s_target, s_arr, y_arr)
    z = np.interp(s_target, s_arr, z_arr)
    theta = np.radians(np.interp(s_target, s_arr, th_arr))
    normal = np.array([-math.cos(theta), math.sin(theta), 0.0])
    return np.array([x, y, z]), normal


def build_mesh(contact_s, ball_r=0.4, gap0=0.02, mesh_size_ball=MESH_SIZE_BALL,
               centerline_path=None, inp_name="bent_contact_mesh.inp",
               sets_name="bent_contact_node_sets.inp", verbose=True):
    if centerline_path is None:
        centerline_path = os.path.join(HERE, "bent_centerline.json")
    with open(centerline_path) as f:
        cl = json.load(f)
    pts = cl["points"]

    center_pt, normal = _point_and_normal_at_s(pts, contact_s)
    ball_center = center_pt + normal * (D_OUT / 2 + gap0 + ball_r)

    gmsh.initialize()
    gmsh.model.add("bent_tube_contact")

    # ── 굽은 튜브: 중심선 스플라인 경로를 따라 환형 단면을 스윕 ──────────────
    point_tags = [gmsh.model.occ.addPoint(p["x"], p["y"], p["z"]) for p in pts]
    spline_tag = gmsh.model.occ.addSpline(point_tags)
    wire_tag = gmsh.model.occ.addWire([spline_tag])
    gmsh.model.occ.synchronize()

    # 단면(환형) 프로파일: 경로 시작점(보드좌표, 보통 (90,0,3))에서 접선(보드 +y, "자라나는
    # 방향")에 수직이 되도록 XY평면(법선 Z) 기본 원판을 X축 기준 -90도 회전 -> 법선이 Y를 향함
    base_pt = (pts[0]["x"], pts[0]["y"], pts[0]["z"])
    outer = gmsh.model.occ.addDisk(*base_pt, D_OUT / 2, D_OUT / 2)
    inner = gmsh.model.occ.addDisk(*base_pt, D_IN / 2, D_IN / 2)
    annulus = gmsh.model.occ.cut([(2, outer)], [(2, inner)])
    gmsh.model.occ.rotate(annulus[0], *base_pt, 1, 0, 0, -math.pi / 2)
    gmsh.model.occ.synchronize()

    piped = gmsh.model.occ.addPipe(annulus[0], wire_tag)
    gmsh.model.occ.synchronize()
    tube_vol = [t for d, t in piped if d == 3][0]

    # ── 강체 구(인덴터) ──────────────────────────────
    ball_tag = gmsh.model.occ.addSphere(*ball_center, ball_r)
    gmsh.model.occ.synchronize()

    # ── 면 태깅: FIXED(시작캡), TIP캡, 옆면(바깥/안쪽), BALL_SURF ──────
    # 굽은 파이프의 경계면은 항상 4개: 시작캡 + 팁캡 + 바깥옆면 + 안쪽옆면.
    # 처음엔 "접촉점에 가장 가까운 면"으로 옆면을 골랐는데, 안쪽옆면이 바깥옆면보다 더
    # 가까울 수 있어서(둘 다 접촉점 근처를 지나가므로) 안쪽을 잘못 고르는 버그가 있었음
    # (그러면 접촉이 물리적으로 불가능해서 힘이 0으로 나옴). 그래서(보드좌표계 기준,
    # y가 "자라나는 방향"이라 x/y 역할이 로컬좌표계 때와 반대임):
    #  1) y범위가 base_y로 고정된 면 = FIXED(시작캡)
    #  2) 나머지 중 y범위가 짧은(전체 경로 길이 대비) 면 = TIP캡
    #  3) 나머지 2개(둘 다 y범위가 김, 옆면) 중 z범위(zmax-zmin)가 더 큰 쪽 = 바깥옆면(반경 큼),
    #     작은 쪽 = 안쪽옆면 (D_OUT=2mm가 D_IN=1mm의 2배라 z폭도 대략 2배 차이남)
    tube_surfaces = gmsh.model.getBoundary([(3, tube_vol)], oriented=False)
    base_y = pts[0]["y"]
    path_y_span = max(p["y"] for p in pts) - min(p["y"] for p in pts)
    fixed_faces, tip_faces, wall_candidates = [], [], []
    for dim, tag in tube_surfaces:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(dim, tag)
        if abs(ymin - base_y) < 1e-6 and abs(ymax - base_y) < 1e-6:
            fixed_faces.append(tag)
        elif (ymax - ymin) < 0.3 * path_y_span:
            tip_faces.append(tag)
        else:
            wall_candidates.append((tag, zmax - zmin))

    wall_candidates.sort(key=lambda t: t[1], reverse=True)  # z범위 큰 순
    outer_faces = [wall_candidates[0][0]] if wall_candidates else []
    if verbose:
        print(f"  면 분류: FIXED={fixed_faces}, TIP={tip_faces}, "
              f"OUTER={outer_faces}(z범위 {wall_candidates[0][1]:.2f}), "
              f"INNER={[t for t,_ in wall_candidates[1:]]}")

    surfaces_ball = gmsh.model.getBoundary([(3, ball_tag)], oriented=False)
    ball_faces = [tag for dim, tag in surfaces_ball]

    gmsh.model.addPhysicalGroup(3, [tube_vol], name="TUBE")
    gmsh.model.addPhysicalGroup(3, [ball_tag], name="BALL")
    gmsh.model.addPhysicalGroup(2, fixed_faces, name="FIXED")
    gmsh.model.addPhysicalGroup(2, outer_faces, name="TUBE_OUTER")
    gmsh.model.addPhysicalGroup(2, ball_faces, name="BALL_SURF")
    # 팁(자유단) 캡 면 - 접촉으로 인한 팁 변위를 직접 뽑아서(홀센서가 보는 자석 위치 변화와
    # 동급인 값) 빔이론 단순모델이 예측한 것과 실제(FEA) 변위가 얼마나 다른지 비교하기 위함.
    gmsh.model.addPhysicalGroup(2, tip_faces, name="TIP")

    gmsh.model.mesh.setSize(gmsh.model.getBoundary([(3, ball_tag)], recursive=True), mesh_size_ball)
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_ball)
    gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE_TUBE)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(2)

    inp_path = os.path.join(HERE, inp_name)
    gmsh.write(inp_path)

    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    if verbose:
        print(f"메쉬 생성 완료: 절점 {n_nodes}개")
        print(f"접촉위치 s={contact_s}mm, 중심선점={center_pt}, 법선={normal}, 구중심={ball_center}")

    # ── CPS(2D 평면)요소 필터링 (BALL_SURF만 남기고 S6로 재지정) ──────────────
    with open(inp_path, encoding="latin-1") as f:
        lines = f.readlines()

    # *NODE 블록에서 절점좌표 사전을 만들어, 각 CPS 요소블록이 "구 중심 근처"인지
    # 실제 좌표로 판별(요소수/이름 같은 간접적 방식은 오판 위험이 있어서 직접 좌표로 확인).
    node_coord = {}
    in_node_block = False
    for line in lines:
        s2 = line.strip()
        if s2.startswith("*NODE"):
            in_node_block = True
            continue
        if in_node_block:
            if s2.startswith("*"):
                in_node_block = False
                continue
            parts = [p.strip() for p in s2.split(",")]
            if len(parts) >= 4:
                try:
                    nid = int(parts[0])
                    node_coord[nid] = (float(parts[1]), float(parts[2]), float(parts[3]))
                except ValueError:
                    pass

    def parse_elem_blocks(lines):
        blocks = []
        cur = None
        for i, line in enumerate(lines):
            s2 = line.strip()
            if s2.startswith("*ELEMENT") and ("CPS3" in s2 or "CPS6" in s2):
                if cur:
                    blocks.append(cur)
                cur = {"start": i, "lines": [line]}
            elif cur is not None and s2.startswith("*") and not s2.startswith("*ELEMENT"):
                blocks.append(cur)
                cur = None
            elif cur is not None:
                cur["lines"].append(line)
        if cur:
            blocks.append(cur)
        return blocks

    blocks = parse_elem_blocks(lines)
    ball_center_arr = np.array(ball_center)
    remove_line_idx = set()
    for b in blocks:
        block_idx = set(range(b["start"], b["start"] + len(b["lines"])))
        # 이 블록 첫 요소줄의 첫 절점ID로 좌표 조회
        first_elem_line = b["lines"][1] if len(b["lines"]) > 1 else ""
        parts = [p.strip() for p in first_elem_line.strip().split(",") if p.strip()]
        is_ball = False
        if len(parts) >= 2:
            try:
                first_node_id = int(parts[1])
                if first_node_id in node_coord:
                    p_xyz = np.array(node_coord[first_node_id])
                    is_ball = np.linalg.norm(p_xyz - ball_center_arr) < ball_r * 3
            except ValueError:
                pass
        if not is_ball:
            remove_line_idx |= block_idx

    filtered = [line for i, line in enumerate(lines) if i not in remove_line_idx]
    filtered = [line.replace("type=CPS6", "type=S6").replace("type=CPS3", "type=S3") for line in filtered]

    with open(inp_path, "w", encoding="latin-1") as f:
        f.writelines(filtered)

    def nodes_of(dim, name):
        for d, t in gmsh.model.getPhysicalGroups(dim):
            if gmsh.model.getPhysicalName(d, t) == name:
                return gmsh.model.mesh.getNodesForPhysicalGroup(d, t)[0]
        raise RuntimeError(f"physical group {name} not found")

    fixed_nodes = nodes_of(2, "FIXED")
    outer_nodes = nodes_of(2, "TUBE_OUTER")
    ball_surf_nodes = nodes_of(2, "BALL_SURF")
    ball_vol_nodes = nodes_of(3, "BALL")
    tip_nodes = nodes_of(2, "TIP")

    sets_path = os.path.join(HERE, sets_name)
    with open(sets_path, "w") as f:
        for name, nodes in [("N_FIXED", fixed_nodes), ("N_TUBE_OUTER", outer_nodes),
                             ("N_BALL_SURF", ball_surf_nodes), ("N_BALL_ALL", ball_vol_nodes),
                             ("N_TIP", tip_nodes)]:
            f.write(f"*NSET, NSET={name}\n")
            for i in range(0, len(nodes), 10):
                f.write(",".join(str(int(n)) for n in nodes[i:i + 10]) + ",\n")

    if verbose:
        print(f"저장: {inp_path}, {sets_path}")
        print(f"  N_FIXED={len(fixed_nodes)}, N_TUBE_OUTER={len(outer_nodes)}, "
              f"N_BALL_SURF={len(ball_surf_nodes)}, N_BALL_ALL={len(ball_vol_nodes)}, N_TIP={len(tip_nodes)}")

    gmsh.finalize()
    return {"ball_center": ball_center.tolist(), "normal": normal.tolist(), "contact_point": center_pt.tolist()}


if __name__ == "__main__":
    build_mesh(contact_s=50.0)
