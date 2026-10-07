"""2026-10-07: "힘만 걸기" 방식에 상한선이 있는지 탐색. s=80 메쉬(이미 생성됨) 재사용해서
힘 크기를 0.005mN(검증된 기준)부터 점점 키워가며 어디서 깨지는지(수렴 실패 또는
팁변위가 카테터 전체 길이(100mm)에 비해 말이 안 되게 커지는 지점) 확인.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_contact as rc
import material_convert as mat

inp_name = "matv2_mesh_S80_FORCEONLY.inp"
sets_name = "matv2_node_sets_S80_FORCEONLY.inp"

# s=80의 정확한 법선(normal)을 force_model로 재계산(이전 실행값 재사용 대신 정확히 구함)
sys.path.insert(0, os.path.join(HERE, "..", "..", "force_model"))
import force_model as fm
L_M, PHI, S = 12.5, -120.0, 80.0
free = fm.solve_shape(L_M=L_M, phi_deg=PHI, loads=[], return_curve=True)
cs = np.sort(free["curve_s_mm"])
cth = np.array(free["curve_theta_deg"])[np.argsort(free["curve_s_mm"])]
theta = np.radians(np.interp(S, cs, cth))
normal_local = np.array([-np.cos(theta), np.sin(theta), 0.0])  # _point_and_normal_at_s 공식과 동일
normal = normal_local
print(f"s=80 법선(normal)={normal}")

# 이전 실행에서 쓴 force 적용 절점 재사용(로그에서 확인된 값)
LOAD_NODE = 24959

FORCE_LEVELS_MN = [0.005, 0.05, 0.25, 1.0, 5.0, 20.0]  # mN

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

for f_mn in FORCE_LEVELS_MN:
    f_n = f_mn * 1e-3
    Fx, Fy, Fz = (-normal * f_n).tolist()
    job_name = f"matv2_FSCAN_{int(f_mn*1000)}"

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
*STEP, NLGEOM, INC=200
*STATIC
0.01, 1.0, 1e-10, 1.0
*CLOAD
{LOAD_NODE}, 1, {Fx:.6E}
{LOAD_NODE}, 2, {Fy:.6E}
{LOAD_NODE}, 3, {Fz:.6E}
*NODE PRINT, NSET=N_TIP
U
*END STEP
"""
    with open(f"{job_name}.inp", "w") as fo:
        fo.write(inp_content)

    print(f"\n=== F={f_mn}mN ===")
    result = rc.run_ccx(job_name, timeout=900, n_threads=4)
    converged, last_time = rc.check_converged(job_name)
    if not converged:
        print(f"  실패 (TOTAL TIME={last_time:.4f})")
        continue
    with open(f"{job_name}.dat", encoding="latin-1") as fd:
        dat = fd.read()
    marker = "displacements (vx,vy,vz) for set N_TIP and time  0.1000000E+01"
    idx = dat.find(marker)
    chunk = dat[idx+len(marker):]
    p = chunk.find("forces")
    if p != -1:
        chunk = chunk[:p]
    ux=uy=0.0; n=0
    for line in chunk.splitlines():
        parts = line.split()
        if len(parts) in (4,5):
            try:
                ux += float(parts[1]); uy += float(parts[2]); n += 1
            except ValueError:
                pass
    if n:
        tip_mag = ((ux/n)**2+(uy/n)**2)**0.5
        print(f"  성공! tip_ux={ux/n:.4f}mm, tip_uy={uy/n:.4f}mm, 팁변위={tip_mag:.4f}mm "
              f"(카테터 전체길이 100mm의 {tip_mag/100*100:.1f}%)")

print("\n=== 테스트 완료 ===")
