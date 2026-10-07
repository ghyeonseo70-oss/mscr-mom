"""2026-10-07: "힘만 걸기" 검증 범위 확장. s=30에서 실제 접촉으로 얻은 가장 큰 변형 케이스
(ball_r=1.0mm, depth=0.6mm: F_mag=0.0423mN, 팁변위=2.0532mm, 팁회전=-1.679deg,
_test_bigball_scan.py)를 같은 힘을 한 점에 직접 걸어서 재현되는지 확인."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_contact as rc
import material_convert as mat

inp_name = "matv2_mesh_DEPTH040_r40.inp"
sets_name = "matv2_node_sets_DEPTH040_r40.inp"
LOAD_NODE = 18264
normal = np.array([-0.9994336846170133, -0.0336498150494231, 0.0])

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

F_MN = float(os.environ.get("F_MN", "0.0423"))
Fx, Fy, Fz = (-normal * F_MN * 1e-3).tolist()
job_name = f"matv2_S30_BIG_{int(F_MN*10000)}"
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

print(f"=== s=30, F={F_MN}mN (힘만 걸기) ===")
rc.run_ccx(job_name, timeout=900, n_threads=4)
converged, last_time = rc.check_converged(job_name)
print(f"수렴={converged}, TOTAL TIME={last_time:.4f}")
if converged:
    with open(f"{job_name}.dat", encoding="latin-1") as fd:
        dat = fd.read()
    marker = "displacements (vx,vy,vz) for set N_TIP and time  0.1000000E+01"
    chunk = dat[dat.find(marker) + len(marker):]
    p = chunk.find("forces")
    if p != -1:
        chunk = chunk[:p]
    ux = uy = 0.0
    n = 0
    for line in chunk.splitlines():
        parts = line.split()
        if len(parts) in (4, 5):
            try:
                ux += float(parts[1]); uy += float(parts[2]); n += 1
            except ValueError:
                pass
    print(f"tip_ux={ux/n:.4f}mm, tip_uy={uy/n:.4f}mm, 팁변위={((ux/n)**2+(uy/n)**2)**0.5:.4f}mm")
    print("--- 비교(실제 접촉, ball_r=1.0, depth=0.6mm): F=0.0423mN, 팁변위=2.0532mm ---")
