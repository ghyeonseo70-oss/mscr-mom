"""2026-10-06: Dirichlet 패치 방식의 등방(원형) 가정이 틀렸다는 게 밝혀짐 - 공 곡률만 쓰고
튜브 자체의 둘레방향 곡률(반지름 1mm, 공 곡률과 비슷한 크기!)을 완전히 빼먹어서 원형 패치를
썼는데, 실제로는 타원형이어야 함:
  - 둘레방향(원주를 감싸는 방향): 공+튜브 둘 다 볼록 -> 곡률이 더해짐(1/ball_r + 1/R_tube)
    -> 접촉면이 더 좁아야 함
  - 길이방향(튜브를 따라가는 방향): 전체 휨 곡률이 아주 완만함(여기선 ~37mm 반경) ->
    거의 평평 -> 접촉면이 더 넓어야 함
포물면 근사(paraboloid approx)로 타원형 패치/프로파일을 계산:
  gap(x_circ, x_axial) = (kappa_circ/2)*x_circ^2 + (kappa_axial/2)*x_axial^2
  indent = max(0, depth - gap)
beta_deg=0 가정 하에이 방향들은 다음처럼 바로 구해짐(_point_and_normal_at_s 공식에서 유도):
  axial_dir = (normal_y, -normal_x, 0)  (법선을 xy평면에서 90도 회전)
  circum_dir = (0, 0, 1)  (항상 전역 z축 - 굽힘이 항상 xy평면 안에서만 일어난다는 가정 때문)
축방향 곡률(kappa_axial = |dtheta/ds|)은 force_model로 직접 계산(해당 L_M,phi 조합마다 다를
수 있으므로 하드코딩 안 함).
"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "force_model"))
import make_bent_contact_scene as scene
import run_contact as rc
import material_convert as mat
import force_model as fm

L_M, PHI, S, DEPTH = 12.5, -120.0, 30.0, 0.80
BALL_R = 0.8  # depth=0.8mm인데 ball_r=0.4면 depth/ball_r=2배라 포물면 근사 자체가 깨짐
# (공 반지름보다 더 깊이 파고드는 꼴) - 접촉 쪽에서 이미 검증된 ball_r=0.8을 그대로 씀
GAP0 = 0.02  # make_bent_contact_scene.py의 기본 gap0과 반드시 같아야 함
TAG = "DIRICHLET_ANISO_DEEP80"
# 접촉 방식 최고 기록(ball_r=0.8~1.0, depth=0.6mm)보다 깊게 - 접촉 수렴 문제 자체가
# 없는 Dirichlet 방식이 얼마나 더 깊이 갈 수 있는지 탐색. 참고: 이 설정(L_M=12.5,phi=-120)은
# 다중해 있는 조합이라 성분 부호가 안 맞을 수 있음 - 일단 "수렴하는지/크기가 말이 되는지"에
# 집중.

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

# beta_deg=0 가정 하의 국소 직교프레임 (make_bent_contact_scene._point_and_normal_at_s 공식 유도)
axial_dir = np.array([normal[1], -normal[0], 0.0])
axial_dir = axial_dir / np.linalg.norm(axial_dir)
circum_dir = np.array([0.0, 0.0, 1.0])

# 축방향 곡률(kappa_axial=|dtheta/ds|, rad/mm) - force_model로 직접 계산(하드코딩 안 함)
_free = fm.solve_shape(L_M=L_M, phi_deg=PHI, loads=[], return_curve=True)
_cs = np.sort(_free["curve_s_mm"])
_cth = np.array(_free["curve_theta_deg"])[np.argsort(_free["curve_s_mm"])]
_s_lo, _s_hi = max(0.5, S - 1.0), min(99.5, S + 1.0)
kappa_axial = abs(np.radians(np.interp(_s_hi, _cs, _cth) - np.interp(_s_lo, _cs, _cth)) / (_s_hi - _s_lo))

R_tube_out = scene.D_OUT / 2.0
kappa_circ = 1.0 / BALL_R + 1.0 / R_tube_out
kappa_axial_combined = 1.0 / BALL_R + kappa_axial

effective_depth = DEPTH - GAP0
a_circ = math.sqrt(2 * effective_depth / kappa_circ) if effective_depth > 0 else 0.0
a_axial = math.sqrt(2 * effective_depth / kappa_axial_combined) if effective_depth > 0 else 0.0
print(f"유효 압입깊이(gap 보정)={effective_depth:.4f}mm")
print(f"kappa_circ(둘레)={kappa_circ:.4f}/mm (a_circ={a_circ:.4f}mm), "
      f"kappa_axial(길이, 공+휨 합성)={kappa_axial_combined:.4f}/mm (a_axial={a_axial:.4f}mm) "
      f"[휨 곡률 자체={kappa_axial:.5f}/mm]")

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
    x_c = float(np.dot(r_perp_vec, circum_dir))
    x_a = float(np.dot(r_perp_vec, axial_dir))
    gap = 0.5 * kappa_circ * x_c ** 2 + 0.5 * kappa_axial_combined * x_a ** 2
    indent = effective_depth - gap
    if indent > 0:
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
        if len(parts) in (4, 5):  # 국소좌표계 노드는 끝에 "L" 표시가 붙어 5개로 나옴
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
