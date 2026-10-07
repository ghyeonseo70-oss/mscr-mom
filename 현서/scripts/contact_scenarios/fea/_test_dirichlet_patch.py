"""2026-10-06: 사용자 제안 - 접촉(ball-tube contact pair)을 아예 없애고, 접촉이 일어날
자리의 튜브 바깥면 절점들에 "구가 이만큼 눌렀을 때 생기는 변위"를 직접 경계조건(Dirichlet)
으로 강제. 접촉탐색/침투 비선형성을 통째로 제거해서 평범한 경계값 문제로 바꾸는 시도.

패치에 주는 변위는 "평평하게 depth만큼"이 아니라 구면 형상을 따라가게 계산함(인위적인
꺾임/특이점 방지):
  r_perp = 접촉축에서 수직거리, a = 접촉반경 = sqrt(depth*(2*ball_r - depth))
  r_perp <= a 인 절점만: indent(r_perp) = depth - (ball_r - sqrt(ball_r^2 - r_perp^2))
  (r_perp=0에서 indent=depth, r_perp=a에서 indent=0으로 자연스럽게 0 수렴 - 접촉반경
  경계에서 기울기도 연속이라 인위적 각이 생기지 않음)

제일 안 되던 케이스(L_M=12.5,phi=-120,s=80,depth=0.20mm, ball_r=0.4mm - 접촉방식으론
4전 4패)로 테스트.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_bent_contact_scene as scene
import run_contact as rc
import material_convert as mat

L_M, PHI, S, DEPTH = 12.5, -120.0, 80.0, 0.20
BALL_R = 0.4
GAP0 = 0.02  # make_bent_contact_scene.py의 기본 gap0과 반드시 같아야 함
TAG = "DIRICHLET_s80"

centerline_path = os.path.join(HERE, f"matv2_centerline_{TAG}.json")
import subprocess
subprocess.run(
    [sys.executable, os.path.join(HERE, "get_bent_centerline.py"),
     "--L_M", str(L_M), "--phi", str(PHI), "--out", centerline_path],
    check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
)
inp_name = f"matv2_mesh_{TAG}.inp"
sets_name = f"matv2_node_sets_{TAG}.inp"
job_name = f"matv2test_{TAG}"

print(f"=== {TAG}: L_M={L_M}, phi={PHI}, s={S}, depth={DEPTH}mm, ball_r={BALL_R}mm "
      f"(접촉 없이 Dirichlet 패치로 시도) ===")
t0 = time.time()

scene_info = scene.build_mesh(
    contact_s=S, ball_r=BALL_R, verbose=False, centerline_path=centerline_path,
    beta_deg=0.0, inp_name=inp_name, sets_name=sets_name,
    contact_mesh_size=0.08, contact_refine_radius=BALL_R + 0.3,
)
normal = np.array(scene_info["normal"])
contact_point = np.array(scene_info["contact_point"])  # 중심선 위의 점 (표면점 아님)
surface_point = contact_point + normal * (scene.D_OUT / 2.0)

effective_depth = DEPTH - GAP0
a_contact = (max(0.0, effective_depth * (2 * BALL_R - effective_depth))) ** 0.5
print(f"유효 압입깊이(gap 보정)={effective_depth:.4f}mm, 접촉반경 a={a_contact:.4f}mm")

inp_path = os.path.join(HERE, inp_name)
sets_path = os.path.join(HERE, sets_name)
outer_ids = rc._parse_nset(sets_path, "N_TUBE_OUTER")
coords = rc._parse_node_coords(inp_path, outer_ids)

# 2026-10-06 수정: 첫 시도(3방향 전부 고정)는 접선방향까지 "용접"해버려서 실제 접촉(s=30
# 검증, F_mag 32배 과다/팁변위 2.6배 과소/회전 부호반전)과 전혀 안 맞았음 - 마찰 없는 접촉은
# 법선 방향만 구속하고 접선 방향은 자유로워야 함. *TRANSFORM으로 패치 전용 국소좌표계를
# 만들어서(국소 1축=법선) 국소 DOF 1만 구속하고 2,3은 자유롭게 둠.
patch_lines = []
patch_ids = []
for nid, (x, y, z) in coords.items():
    vec = np.array([x, y, z]) - surface_point
    r_along = float(np.dot(vec, normal))
    r_perp_vec = vec - r_along * normal
    r_perp = float(np.linalg.norm(r_perp_vec))
    if r_perp <= a_contact:
        indent = effective_depth - (BALL_R - (BALL_R ** 2 - r_perp ** 2) ** 0.5)
        indent = max(0.0, indent)
        patch_ids.append(nid)
        patch_lines.append(f"{nid}, 1, 1, {-indent:.6E}")  # 국소 1축=normal, 안쪽(-)으로 indent만큼

print(f"패치 절점 수: {len(patch_ids)}개 (N_TUBE_OUTER 전체 {len(outer_ids)}개 중)")
if len(patch_ids) == 0:
    print("패치 절점이 0개 - 메쉬 해상도 대비 접촉반경이 너무 작을 수 있음. 중단.")
    sys.exit(1)

nset_patch_block = "*NSET, NSET=N_PATCH\n" + "".join(
    ",".join(str(i) for i in patch_ids[j:j + 10]) + ",\n" for j in range(0, len(patch_ids), 10)
)

# 국소좌표계: 1축=normal. b점은 normal과 평행하지 않은 아무 벡터(평면만 정해주면 됨 -
# 2,3축 방향은 안 쓰므로 정확할 필요 없음).
b_vec = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
transform_block = (f"*TRANSFORM, NSET=N_PATCH, TYPE=R\n"
                    f"{normal[0]:.6E}, {normal[1]:.6E}, {normal[2]:.6E}, "
                    f"{b_vec[0]:.6E}, {b_vec[1]:.6E}, {b_vec[2]:.6E}\n")

with open(inp_path, encoding="latin-1") as f:
    has_wire = "ELSET=WIRE_BEAM" in f.read().upper()
wire_block = f"""*MATERIAL, NAME=NITINOL
*ELASTIC
{rc.E_NITINOL_MM}, {rc.NU_NITINOL}
**
*BEAM SECTION, ELSET=WIRE_BEAM, MATERIAL=NITINOL, SECTION=CIRC
{rc.WIRE_R_MM}
0.,0.,1.
**
*EMBEDDED ELEMENT, HOST ELSET=TUBE
WIRE_BEAM
**""" if has_wire else "**"

boundary_block = "\n".join(patch_lines)

inp_content = f"""*INCLUDE, INPUT={inp_name}
*INCLUDE, INPUT={sets_name}
**
{nset_patch_block}**
{transform_block}**
*MATERIAL, NAME=SILICONE
*HYPERELASTIC, NEO HOOKE
{mat.C10_MM:.6E}, {mat.D1_MM:.6E}
**
*MATERIAL, NAME=RIGID_MAT
*ELASTIC
{rc.E_RIGID_MM}, 0.3
**
{wire_block}
*SOLID SECTION, ELSET=TUBE, MATERIAL=SILICONE
*SOLID SECTION, ELSET=BALL, MATERIAL=RIGID_MAT
*SHELL SECTION, ELSET=BALL_SURF, MATERIAL=RIGID_MAT
0.01
**
*BOUNDARY
N_FIXED, 1, 3
N_BALL_ALL, 1, 3
**
*STEP, NLGEOM, INC=100
*STATIC
0.01, 1.0, 1e-8, 1.0
*BOUNDARY
{boundary_block}
*NODE PRINT, NSET=N_PATCH
U, RF
*NODE PRINT, NSET=N_TIP
U
*NODE PRINT, NSET=N_MOM
U
*NODE FILE
U
*EL FILE
S, E
*END STEP
"""
job_inp_path = os.path.join(HERE, f"{job_name}.inp")
with open(job_inp_path, "w") as f:
    f.write(inp_content)

print("CalculiX 실행 중...")
result = rc.run_ccx(job_name, timeout=1800, n_threads=4)
dt = time.time() - t0

dat_path = os.path.join(HERE, f"{job_name}.dat")
if not os.path.exists(dat_path):
    print(f"[{TAG}] 실패 - .dat 없음 ({dt:.1f}s). stdout 마지막:")
    print(result.stdout[-1500:])
    sys.exit(1)

converged, last_time = rc.check_converged(job_name)
if not converged:
    print(f"[{TAG}] 실패 - 수렴 안 됨 (TOTAL TIME={last_time:.4f}/1.0, {dt:.1f}s)")
    sys.exit(1)

# N_PATCH 기준으로 반력 합(=접촉력에 해당) 파싱 - parse_results는 N_BALL_ALL 기준이라
# 여기선 직접 N_PATCH로 재구현.
with open(dat_path, encoding="latin-1") as f:
    dat = f.read()


def last_block(marker):
    idx = dat.rfind(marker)
    if idx == -1:
        return None
    chunk = dat[idx + len(marker):]
    for other in ["displacements (vx,vy,vz)", "forces (fx,fy,fz)"]:
        pos = chunk.find(other)
        if pos != -1:
            chunk = chunk[:pos]
    return chunk


def sum_col(chunk, col):
    total = 0.0
    for line in (chunk or "").splitlines():
        parts = line.split()
        if len(parts) == 4:
            try:
                total += float(parts[col])
            except ValueError:
                pass
    return total


force_chunk = last_block("forces (fx,fy,fz) for set N_PATCH and time")
fx = sum_col(force_chunk, 1)
fy = sum_col(force_chunk, 2)
fz = sum_col(force_chunk, 3)
f_mag = (fx ** 2 + fy ** 2 + fz ** 2) ** 0.5

tip = rc.get_rotation_deg(job_name, inp_name, sets_name, nset_name="N_TIP", key_prefix="tip")
tip_chunk = last_block("displacements (vx,vy,vz) for set N_TIP and time")
tux = sum_col(tip_chunk, 1)
tuy = sum_col(tip_chunk, 2)
n_tip = sum(1 for line in (tip_chunk or "").splitlines() if len(line.split()) == 4)
tux_avg = tux / n_tip if n_tip else float("nan")
tuy_avg = tuy / n_tip if n_tip else float("nan")
tip_mag = (tux_avg ** 2 + tuy_avg ** 2) ** 0.5

print(f"\n[{TAG}] 성공! F_mag={f_mag*1000:.4f}mN (Fx={fx*1000:.4f}, Fy={fy*1000:.4f}mN), "
      f"tip_ux={tux_avg:.4f}mm, tip_uy={tuy_avg:.4f}mm, 팁변위={tip_mag:.4f}mm, "
      f"팁회전={tip['tip_theta_deg']:.3f}deg ({dt:.1f}s)")
