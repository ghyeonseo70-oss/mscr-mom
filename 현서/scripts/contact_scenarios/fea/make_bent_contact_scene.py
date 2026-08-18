"""
굽은 튜브(force_model.py 예측 중심선을 스윕한 형상) + 강체 구(인덴터)를 한 메쉬로 만들어서
CalculiX 접촉해석용으로 내보냄. 접촉위치는 호길이(s, mm, 베이스=0부터)로 지정하면
그 지점의 접선각도(theta)를 이용해 법선 방향으로 구를 배치한다.
"""
import json
import math
import os
import sys
import gmsh
import numpy as np

D_OUT = 2.0    # mm
D_IN = 1.0     # mm
MESH_SIZE_TUBE = 0.3
MESH_SIZE_BALL = 0.15

# 2026-08-18 추가: 니티놀 와이어(사용자 실측값 - 지름 100um, E=28GPa, K1 구간에만).
# 3D 고체(구멍 뚫은 파이프)로 직접 메싱하는 방식은 극단적 크기비율(와이어 0.1mm vs 튜브
# 100mm) 때문에 자코비안 음수/세그폴트로 실패해서, 1D 빔요소 + *EMBEDDED ELEMENT 방식으로
# 전환함(콘크리트 속 철근과 동일한 표준 기법) - WIRE_R은 *BEAM SECTION의 단면 반지름으로만
# 쓰임(run_contact.py). 재질(E=28GPa)도 run_contact.py에서 지정.
# 위치: 원래 "중심축(내강 정중앙, 반지름 0)"이었는데, 거긴 완전히 빈 공간이라 CalculiX
# *EMBEDDED ELEMENT가 물릴 고체 요소가 없음(사용자와 논의: 내강을 채우는 별도 충전재를 새로
# 만드는 대신, 이미 고체인 실리콘 벽 안쪽으로 옮기기로 함). WIRE_WALL_OFFSET_R은 벽 두께
# (D_IN/2=0.5mm ~ D_OUT/2=1.0mm) 중간인 0.75mm - 벽 안에 항상 파묻히도록.
WIRE_D = 0.1          # mm, 지름 100um
WIRE_R = WIRE_D / 2
WIRE_WALL_OFFSET_R = (D_IN / 2 + D_OUT / 2) / 2  # mm, 벽 두께 중간(0.75mm)에 임베드

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "force_model"))
import force_model as fm


def _point_and_normal_at_s(points, s_target, beta_deg=0.0):
    """중심선 점 목록에서 s_target 위치의 (x,y,z)와 접촉방향 벡터를 보간.
    theta_deg는 force_model의 로컬좌표계 각도(get_bent_centerline.py 참고)라서, 보드좌표계
    (x_board=90+y_local, y_board=x_local)로 변환된 접선/법선 공식을 써야 함:
    로컬 접선(cos,sin,0) -> 보드 접선(sin,cos,0)이 되므로, 그에 수직인 법선은
    (-cos(theta), sin(theta), 0)이 됨(기존 로컬용 공식 (-sin,cos,0)에서 치환).

    beta_deg: 접선을 축으로 한 원주방향 회전각(0=굽힘평면 바깥쪽, 90=평면 밖 위쪽, 180=안쪽,
    270=평면 밖 아래쪽). 굽힘이 항상 z=board_z 평면 안에서만 일어난다고 가정하므로(force_model.py/
    get_bent_centerline.py 전체가 이 가정), 바이노멀(B)은 s와 무관하게 항상 (0,0,1)로 고정 —
    그래서 국소 프레네 프레임 전체를 다시 계산할 필요 없이 push_dir = cos(beta)*N(s) + sin(beta)*B
    로 원주 어느 각도든 표현 가능함. 관 바깥면(TUBE_OUTER) 자체가 이미 원주 전체를 포함하므로
    메쉬는 그대로 두고 공(indenter) 위치/방향만 이 식으로 바꾸면 됨."""
    s_arr = np.array([p["s"] for p in points])
    x_arr = np.array([p["x"] for p in points])
    y_arr = np.array([p["y"] for p in points])
    z_arr = np.array([p["z"] for p in points])
    th_arr = np.array([p["theta_deg"] for p in points])
    x = np.interp(s_target, s_arr, x_arr)
    y = np.interp(s_target, s_arr, y_arr)
    z = np.interp(s_target, s_arr, z_arr)
    theta = np.radians(np.interp(s_target, s_arr, th_arr))
    in_plane_normal = np.array([-math.cos(theta), math.sin(theta), 0.0])
    binormal = np.array([0.0, 0.0, 1.0])
    beta = math.radians(beta_deg)
    direction = math.cos(beta) * in_plane_normal + math.sin(beta) * binormal
    return np.array([x, y, z]), direction


def _points_up_to_s(points, s_max):
    """센터라인 점 목록에서 s<=s_max인 점들 + s=s_max 지점 보간점(정확한 절단면 확보용,
    theta_deg도 함께 보간 - 와이어를 벽 안쪽으로 오프셋할 때 국소 법선 방향 계산에 필요)."""
    s_arr = np.array([p["s"] for p in points])
    x_arr = np.array([p["x"] for p in points])
    y_arr = np.array([p["y"] for p in points])
    z_arr = np.array([p["z"] for p in points])
    theta_arr = np.array([p["theta_deg"] for p in points])
    sub = [p for p in points if p["s"] <= s_max + 1e-9]
    if not sub or sub[-1]["s"] < s_max - 1e-6:
        sub.append({
            "s": s_max,
            "x": float(np.interp(s_max, s_arr, x_arr)),
            "y": float(np.interp(s_max, s_arr, y_arr)),
            "z": float(np.interp(s_max, s_arr, z_arr)),
            "theta_deg": float(np.interp(s_max, s_arr, theta_arr)),
        })
    return sub


def _offset_into_wall(p, r_offset):
    """센터라인 점 p를 국소 법선 방향(N = (-cos theta, sin theta, 0), _point_and_normal_at_s와
    동일 관례)으로 r_offset만큼 밀어서 실리콘 벽 두께 안쪽 좌표를 얻음."""
    theta = math.radians(p["theta_deg"])
    nx, ny = -math.cos(theta), math.sin(theta)
    return {"x": p["x"] + r_offset * nx, "y": p["y"] + r_offset * ny, "z": p["z"]}


def build_mesh(contact_s, ball_r=0.4, gap0=0.02, mesh_size_ball=MESH_SIZE_BALL,
               centerline_path=None, inp_name="bent_contact_mesh.inp",
               sets_name="bent_contact_node_sets.inp", verbose=True, beta_deg=0.0,
               include_wire=False):
    """include_wire: 2026-08-18 추가 실험 기능(니티놀 와이어, K1 구간). 기본값 False로 명시적
    opt-in 필요 - 이전에 이 플래그 없이 함수 자체를 바로 고쳐서, 이미 돌고 있던(와이어와 무관한)
    matv2 재료값 검증 스윕이 새로 시작하는 조합부터 의도치 않게 와이어 코드를 타는 사고가
    있었음(phi=-90 같은 급한 굽힘에서 메쉬 왜곡으로 실패 유발, 실제 저장된 데이터 오염은 없었음
    - 전부 실패 처리라 결과 JSON엔 안 들어감). 앞으로 이 함수 자체를 바꾸더라도 기본 동작은
    항상 와이어 없음을 유지할 것 - 와이어 스윕은 별도로 include_wire=True를 명시해서 돌릴 것."""
    if centerline_path is None:
        centerline_path = os.path.join(HERE, "bent_centerline.json")
    with open(centerline_path) as f:
        cl = json.load(f)
    pts = cl["points"]

    center_pt, normal = _point_and_normal_at_s(pts, contact_s, beta_deg=beta_deg)
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

    # ── 니티놀 와이어(K1 구간 s=0~a1에만, 실리콘 벽 두께 안쪽에 임베드) ──────────
    # 2026-08-18: 처음엔 와이어를 실제 지름(0.1mm)의 3D 고체(구멍 뚫은 파이프)로 직접 메싱해서
    # 내강(원래 비어있는 중심축)에 넣으려 했으나, 지름 0.1mm 와이어가 지름 2mm 튜브/100mm 전체
    # 형상 안에 들어가면서 생기는 극단적 크기비율(20~40배) 때문에 2차(곡면) 요소가 자코비안
    # 음수가 되는 문제를 못 피함(메쉬를 세밀화해도 요소수만 폭증하고 CalculiX가 세그폴트로 죽음).
    # 이런 "굵은 재질 속 아주 가는 보강재"는 3D 고체로 직접 메싱하지 않고 1D 빔요소로 표현해
    # CalculiX *EMBEDDED ELEMENT로 주변 솔리드에 묻는 게 표준적인 방법(콘크리트 속 철근 모델링과
    # 동일한 기법) - 와이어가 이 정도로 가늘고 긴(길이/지름비 200:1 이상) 형상에서는 빔이론이
    # 사실상 정확해서, 국소 응력 디테일만 포기하고 접촉힘/굽힘강성 기여 같은 집합값의 정확도는
    # 거의 그대로 유지됨(사용자와 논의 후 결정).
    # 임베드 위치: 빔 노드가 "속이 빈 공간"(내강 중심축)에 있으면 embed할 호스트 요소가 없어서,
    # 처음엔 내강을 K1 구간만 실리콘 충전재로 채웠는데(gmsh fragment), 굽은 형상(phi!=0)에서
    # 충전재의 독립 스플라인이 튜브 스플라인과 미세하게 어긋나 자코비안 음수 요소가 생기는
    # 문제가 있었음. 사용자와 논의 후, 별도 충전재 없이 **와이어를 이미 고체인 실리콘 벽
    # 두께(D_IN/2~D_OUT/2) 안쪽으로 옮겨서** 기존 튜브 재질에 바로 embed하는 방식으로 변경 -
    # 기하가 훨씬 단순해지고 fragment 관련 문제가 원천적으로 사라짐(실제 카테터도 이런 보강재를
    # 속이 빈 내강 정중앙보다 벽 속에 매립하는 경우가 흔해서 물리적으로도 무리 없는 가정).
    a1_mm = float(np.clip(cl["L_M"] - fm.H_M / 2, 0.0, fm.L))
    wire_beam_curve = None
    if include_wire and a1_mm > 0.5:  # K1 구간이 있을 때만 (L_M=0이면 a1=0, 와이어 없음)
        k1_pts = _points_up_to_s(pts, a1_mm)
        wall_pts = [_offset_into_wall(p, WIRE_WALL_OFFSET_R) for p in k1_pts]
        beam_pt_tags = [gmsh.model.occ.addPoint(p["x"], p["y"], p["z"]) for p in wall_pts]
        wire_beam_curve = gmsh.model.occ.addSpline(beam_pt_tags)
        gmsh.model.occ.synchronize()
        if verbose:
            print(f"  와이어(빔) 추가: a1={a1_mm:.2f}mm, beam_curve={wire_beam_curve} "
                  f"(벽 안쪽 반지름 {WIRE_WALL_OFFSET_R}mm에 임베드)")
    silicone_vols = [tube_vol]

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
    # 와이어/충전재(K1 구간, s=0~a1)가 있으면 그 재질이 끝나는 지점(s=a1)에 내부 노출면(충전재
    # 끝단 캡, 충전재-와이어 경계 원통면)이 추가로 생기는데, 둘 다 y범위가 전체길이의 0.3배보다
    # 작아서(a1<0.3*L인 경우가 많음) 기존 "y범위 짧으면 TIP캡" 판정에 걸려 진짜 팁(s=L)과
    # 혼동될 수 있음(2026-08-18 와이어 추가하며 발견). 그래서 후보를 모아뒀다가 y중심이 가장
    # 큰(=진짜 팁에 가장 가까운) 것 하나만 TIP으로 인정하고 나머지는 태그하지 않음(자유표면으로
    # 남아도 별도 하중/구속이 없는 내부면이라 무방).
    tube_surfaces = gmsh.model.getBoundary([(3, v) for v in silicone_vols], oriented=False, combined=True)
    base_y = pts[0]["y"]
    path_y_span = max(p["y"] for p in pts) - min(p["y"] for p in pts)
    fixed_faces, cap_candidates, wall_candidates = [], [], []
    for dim, tag in tube_surfaces:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(dim, tag)
        if abs(ymin - base_y) < 1e-6 and abs(ymax - base_y) < 1e-6:
            fixed_faces.append(tag)
        elif (ymax - ymin) < 0.3 * path_y_span:
            cap_candidates.append((tag, (ymin + ymax) / 2))
        else:
            wall_candidates.append((tag, zmax - zmin))

    cap_candidates.sort(key=lambda t: t[1], reverse=True)  # y중심 큰(팁에 가까운) 순
    tip_faces = [cap_candidates[0][0]] if cap_candidates else []
    ignored_caps = [t for t, _ in cap_candidates[1:]]

    wall_candidates.sort(key=lambda t: t[1], reverse=True)  # z범위 큰 순
    outer_faces = [wall_candidates[0][0]] if wall_candidates else []
    if verbose:
        print(f"  면 분류: FIXED={fixed_faces}, TIP={tip_faces}, "
              f"OUTER={outer_faces}(z범위 {wall_candidates[0][1]:.2f}), "
              f"INNER={[t for t,_ in wall_candidates[1:]]}")
        if ignored_caps:
            print(f"  [참고] TIP 후보 중 팁이 아닌 내부 노출면으로 판단해 제외: {ignored_caps}")

    surfaces_ball = gmsh.model.getBoundary([(3, ball_tag)], oriented=False)
    ball_faces = [tag for dim, tag in surfaces_ball]

    gmsh.model.addPhysicalGroup(3, silicone_vols, name="TUBE")
    if wire_beam_curve is not None:
        gmsh.model.addPhysicalGroup(1, [wire_beam_curve], name="WIRE_BEAM")
    gmsh.model.addPhysicalGroup(3, [ball_tag], name="BALL")
    gmsh.model.addPhysicalGroup(2, fixed_faces, name="FIXED")
    gmsh.model.addPhysicalGroup(2, outer_faces, name="TUBE_OUTER")
    gmsh.model.addPhysicalGroup(2, ball_faces, name="BALL_SURF")
    # 팁(자유단) 캡 면 - 접촉으로 인한 팁 변위를 직접 뽑아서(홀센서가 보는 자석 위치 변화와
    # 동급인 값) 빔이론 단순모델이 예측한 것과 실제(FEA) 변위가 얼마나 다른지 비교하기 위함.
    gmsh.model.addPhysicalGroup(2, tip_faces, name="TIP")

    gmsh.model.mesh.setSize(gmsh.model.getBoundary([(3, ball_tag)], recursive=True), mesh_size_ball)
    # 1D 빔(wire_beam_curve)은 3D 솔리드처럼 극단적으로 세밀할 필요 없음(embed 제약이 요소
    # 크기와 무관하게 호스트 요소 안 위치로 절점을 묶어줌) - 전역 Min/Max 범위 내 기본 분할로 충분.
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
    # 와이어 빔 커브(WIRE_BEAM)를 gmsh는 기본으로 T3D3(3절점 트러스, 축력만 저항)로 내보내는데,
    # 굽힘강성 기여가 목적이므로 CalculiX의 B32(3절점 빔, 굽힘 포함)로 재지정 필요. 이 모델에는
    # T3D3를 쓰는 다른 요소가 없으므로(2D는 전부 CPS->S로 이미 처리됨) 전역 치환으로 충분.
    filtered = [line.replace("type=T3D3", "type=B32") for line in filtered]

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
