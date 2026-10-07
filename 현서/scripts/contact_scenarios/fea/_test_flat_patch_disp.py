"""2026-10-07: 단일 절점 변위 방식은 35배 과다 힘(점 변위 특이점)으로 실패. 실제 접촉 데이터
(_test_extract_real_patch.py)는 접촉점 주변 넓은 영역이 균일하게(~0.364mm) 같이 움직였으므로
(포물면으로 얕아지는 Hertz 모양이 아니라 "바닥이 평평한" 패치), 반경 R 안의 외표면 절점 전부에
균일한 법선방향 변위를 주고(접선방향은 *TRANSFORM으로 자유) 반력을 역산.
s=30, 실제 접촉 depth=0.40mm 케이스(F=0.0275mN, 팁변위 1.3277mm)와 비교.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_contact as rc
import material_convert as mat

sys.path.insert(0, os.path.join(HERE, "..", "..", "force_model"))
import force_model as fm

inp_name = "matv2_mesh_DEPTH040_r40.inp"
sets_name = "matv2_node_sets_DEPTH040_r40.inp"
RADIUS = float(os.environ.get("PATCH_R", "0.5"))     # mm
INDENT = float(os.environ.get("PATCH_INDENT", "0.364"))  # mm, 실제 접촉에서 측정된 중심부 눌림
job_name = f"matv2_FLATPATCH_r{int(RADIUS*100)}"

normal = np.array([-0.9994336846170133, -0.0336498150494231, 0.0])
D_OUT = 2.0
free = fm.solve_shape(L_M=12.5, phi_deg=-120.0, loads=[], return_curve=True)
cs = np.sort(free["curve_s_mm"])
cx = np.array(free["curve_x_mm"])[np.argsort(free["curve_s_mm"])]
cy = np.array(free["curve_y_mm"])[np.argsort(free["curve_s_mm"])]
x0 = np.interp(30.0, cs, cx)
y0 = np.interp(30.0, cs, cy)
bx0, by0 = fm.to_board_frame(x0, y0)
surface_point = np.array([bx0, by0, 3.0]) + normal * (D_OUT / 2.0)  # centerline z=3

outer_ids = rc._parse_nset(sets_name, "N_TUBE_OUTER")
coords = rc._parse_node_coords(inp_name, outer_ids)

patch_ids = []
for nid, (x, y, z) in coords.items():
    vec = np.array([x, y, z]) - surface_point
    r_perp = vec - np.dot(vec, normal) * normal
    if np.linalg.norm(r_perp) <= RADIUS:
        patch_ids.append(nid)
print(f"패치 반경 {RADIUS}mm, 균일 눌림 {INDENT}mm, 절점 {len(patch_ids)}개")

nset_block = "*NSET, NSET=N_PATCH\n" + "".join(
    ",".join(str(i) for i in patch_ids[j:j + 10]) + ",\n" for j in range(0, len(patch_ids), 10))
b_vec = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
transform_block = (f"*TRANSFORM, NSET=N_PATCH, TYPE=R\n"
                   f"{normal[0]:.6E}, {normal[1]:.6E}, {normal[2]:.6E}, "
                   f"{b_vec[0]:.6E}, {b_vec[1]:.6E}, {b_vec[2]:.6E}\n")

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
{nset_block}**
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
0.01, 1.0, 1e-10, 1.0
*BOUNDARY
N_PATCH, 1, 1, {-INDENT:.6E}
*NODE PRINT, NSET=N_PATCH
RF
*NODE PRINT, NSET=N_TIP
U
*END STEP
"""
with open(f"{job_name}.inp", "w") as fo:
    fo.write(inp_content)

print("실행 중...")
result = rc.run_ccx(job_name, timeout=900, n_threads=4)
converged, last_time = rc.check_converged(job_name)
print(f"수렴={converged}, TOTAL TIME={last_time:.4f}")
if not converged:
    print("stdout 마지막:", result.stdout[-1200:])
else:
    with open(f"{job_name}.dat", encoding="latin-1") as fd:
        dat = fd.read()

    def last_block(marker):
        idx = dat.rfind(marker)
        if idx == -1:
            return ""
        chunk = dat[idx + len(marker):]
        for other in ["displacements (vx,vy,vz)", "forces (fx,fy,fz)"]:
            p = chunk.find(other)
            if p != -1:
                chunk = chunk[:p]
        return chunk

    def sum_cols(chunk):
        s = [0.0, 0.0, 0.0]
        n = 0
        for line in chunk.splitlines():
            parts = line.split()
            if len(parts) in (4, 5):
                try:
                    for i in range(3):
                        s[i] += float(parts[i + 1])
                    n += 1
                except ValueError:
                    pass
        return s, n

    f_vec, _ = sum_cols(last_block("forces (fx,fy,fz) for set N_PATCH and time"))
    t_vec, nt = sum_cols(last_block("displacements (vx,vy,vz) for set N_TIP and time"))
    ux, uy = t_vec[0] / nt, t_vec[1] / nt
    print(f"\n[평평한 패치] 반력 F_mag={np.linalg.norm(f_vec)*1000:.4f}mN "
          f"(법선성분={f_vec[0]*1000:.4f}mN)")
    print(f"팁변위: tip_ux={ux:.4f}mm, tip_uy={uy:.4f}mm, 크기={(ux**2+uy**2)**0.5:.4f}mm")
    print("--- 비교(실제 접촉): F=0.0275mN, tip_ux=1.0100, tip_uy=0.8619, 크기=1.3277mm ---")
