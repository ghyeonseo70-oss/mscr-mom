"""2026-10-07: s=80mm, F=0.3mN(힘만 걸기) - "과도함" 구간이 실제로 얼마나 비현실적인지
그림으로 보여주기 위한 계산. 기존 s=80 메쉬 재사용."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_contact as rc
import material_convert as mat

inp_name = "matv2_mesh_S80_FORCEONLY.inp"
sets_name = "matv2_node_sets_S80_FORCEONLY.inp"
LOAD_NODE = 24959

sys.path.insert(0, os.path.join(HERE, "..", "..", "force_model"))
import force_model as fm
L_M, PHI, S = 12.5, -120.0, 80.0
free = fm.solve_shape(L_M=L_M, phi_deg=PHI, loads=[], return_curve=True)
cs = np.sort(free["curve_s_mm"])
cth = np.array(free["curve_theta_deg"])[np.argsort(free["curve_s_mm"])]
theta = np.radians(np.interp(S, cs, cth))
normal = np.array([-np.cos(theta), np.sin(theta), 0.0])

F_MN = 0.03
f_n = F_MN * 1e-3
Fx, Fy, Fz = (-normal * f_n).tolist()
job_name = "matv2_S80_F003"

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

print(f"=== F={F_MN}mN, s={S} ===")
result = rc.run_ccx(job_name, timeout=900, n_threads=4)
converged, last_time = rc.check_converged(job_name)
print(f"수렴={converged}, TOTAL TIME={last_time:.4f}")
if converged:
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
        print(f"tip_ux={ux/n:.4f}mm, tip_uy={uy/n:.4f}mm, 팁변위={tip_mag:.4f}mm "
              f"(전체길이 100mm의 {tip_mag:.1f}%)")
