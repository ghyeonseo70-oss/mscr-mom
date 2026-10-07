"""2026-10-07: 사용자 제안 - "눌린 점의 변위를 입력으로 주고 힘을 역산"하는 방법 검증.
접촉(contact)도, 여러 점짜리 Dirichlet 패치도 없이, 접촉점 근처 절점 딱 하나에만
변위 경계조건을 주고(그 점이 depth만큼 움직였다고 가정), 거기 버티는 데 필요한 반력(=힘)을
역산. s=30, depth=0.40mm(실제 접촉 실측 F=0.0275mN을 아는 케이스)로 검증.
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
job_name = "matv2_SINGLENODE_d40"

normal = np.array([-0.9994336846170133, -0.0336498150494231, 0.0])
DEPTH = 0.40  # mm, 실제 접촉 케이스와 동일
LOAD_NODE = 18264  # 접촉점에서 제일 가까운 절점(이전 분석에서 확인된 값)

dx, dy, dz = (-normal * DEPTH).tolist()

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
*NSET, NSET=N_ONE
{LOAD_NODE},
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
0.01, 1.0, 1e-10, 1.0
*BOUNDARY
{LOAD_NODE}, 1, 1, {dx:.6E}
{LOAD_NODE}, 2, 2, {dy:.6E}
{LOAD_NODE}, 3, 3, {dz:.6E}
*NODE PRINT, NSET=N_ONE
U, RF
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

    f_vec, nf = sum_cols(last_block("forces (fx,fy,fz) for set N_ONE and time"))
    f_mag = float(np.linalg.norm(f_vec))
    t_vec, nt = sum_cols(last_block("displacements (vx,vy,vz) for set N_TIP and time"))
    ux, uy = t_vec[0] / nt, t_vec[1] / nt
    print(f"\n[단일 절점 변위] 반력 합 F_mag={f_mag*1000:.4f}mN (Fx={f_vec[0]*1000:.4f}, "
          f"Fy={f_vec[1]*1000:.4f}, Fz={f_vec[2]*1000:.4f}mN)")
    print(f"팁변위: tip_ux={ux:.4f}mm, tip_uy={uy:.4f}mm, 크기={(ux**2+uy**2)**0.5:.4f}mm")
    print("--- 비교(실제 접촉, s=30, depth=0.40mm): F=0.0275mN, tip_ux=1.0100, tip_uy=0.8619, 크기=1.3277mm ---")
