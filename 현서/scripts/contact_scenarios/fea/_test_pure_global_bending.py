"""2026-10-06: Dirichlet 패치 보정을 위해 "순수 전역 휨" 기준값을 구함.
force_model.py(해석모델)는 접촉성 점하중에서 휨을 100배 과소평가해서 기준으로 못 씀 -
대신 공(ball)/접촉(contact) 없이, 실측된 힘(Fx,Fy,Fz)을 접촉점 근처 절점 하나에 직접
*CLOAD로 걸어서 진짜 FEA로 "이 힘만으로 생기는 전역 휨"을 구함. 접촉 비선형이 없어서
거의 바로 수렴할 것으로 예상.

이 결과를 실제 접촉 결과(matv2_EXTRACTPATCH_d40, ①+②)에서 빼면 순수 국소 눌림(①)만 남음.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_contact as rc
import material_convert as mat

inp_name = "matv2_mesh_DEPTH040_r40.inp"
sets_name = "matv2_node_sets_DEPTH040_r40.inp"
job_name = "matv2_PUREGLOBAL_d40"

# 실측 힘 (_test_extract_real_patch.py 결과에서 뽑은 값, N 단위)
FX, FY, FZ = 2.7476969427703686e-05, 1.4047612454650941e-06, -8.979456659532886e-07

outer_ids = rc._parse_nset(sets_name, "N_TUBE_OUTER")
coords = rc._parse_node_coords(inp_name, outer_ids)

normal = np.array([-0.9994336846170133, -0.0336498150494231, 0.0])
D_OUT = 2.0
sys.path.insert(0, os.path.join(HERE, "..", "..", "force_model"))
import force_model as fm
free = fm.solve_shape(L_M=12.5, phi_deg=-120.0, loads=[], return_curve=True)
cs = np.sort(free["curve_s_mm"])
cx = np.array(free["curve_x_mm"])[np.argsort(free["curve_s_mm"])]
cy = np.array(free["curve_y_mm"])[np.argsort(free["curve_s_mm"])]
x0 = np.interp(30.0, cs, cx)
y0 = np.interp(30.0, cs, cy)
bx0, by0 = fm.to_board_frame(x0, y0)
surface_point = np.array([bx0, by0, 3.0]) + normal * (D_OUT / 2.0)

# 접촉점에 제일 가까운 절점 하나 찾기
best_nid, best_d = None, 1e9
for nid, (x, y, z) in coords.items():
    d = np.linalg.norm(np.array([x, y, z]) - surface_point)
    if d < best_d:
        best_d = d
        best_nid = nid
print(f"하중 적용 절점: {best_nid} (접촉점에서 거리 {best_d:.4f}mm)")

with open(inp_name, encoding="latin-1") as f:
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

inp_content = f"""*INCLUDE, INPUT={inp_name}
*INCLUDE, INPUT={sets_name}
**
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
*CLOAD
{best_nid}, 1, {FX:.6E}
{best_nid}, 2, {FY:.6E}
{best_nid}, 3, {FZ:.6E}
*NODE PRINT, NSET=N_TUBE_OUTER
U
*NODE PRINT, NSET=N_TIP
U
*NODE PRINT, NSET=N_MOM
U
*END STEP
"""
job_inp_path = os.path.join(HERE, f"{job_name}.inp")
with open(job_inp_path, "w") as f:
    f.write(inp_content)

print("실행 중 (접촉 없이 순수 점하중만, 빠르게 수렴할 것으로 예상)...")
result = rc.run_ccx(job_name, timeout=1800, n_threads=4)
converged, last_time = rc.check_converged(job_name)
print(f"수렴={converged}, TOTAL TIME={last_time:.4f}")
if not converged:
    print("stdout 마지막:", result.stdout[-1500:])
