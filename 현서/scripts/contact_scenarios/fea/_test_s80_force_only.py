"""2026-10-07: s=80(접촉 5전 5패)에서 "힘만 걸기"(접촉/패치 없이 점하중) 방식이 수렴하는지
확인. 정확한 힘 보정은 나중에 하고, 일단 수렴 자체가 되는지가 핵심.
임의의 작은 힘(s=30,depth~0.1~0.15mm 수준과 비슷한 크기)으로 테스트.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_bent_contact_scene as scene
import run_contact as rc
import material_convert as mat

L_M, PHI, S = 12.5, -120.0, 80.0
BALL_R = 0.4
inp_name = "matv2_mesh_S80_FORCEONLY.inp"
sets_name = "matv2_node_sets_S80_FORCEONLY.inp"
job_name = "matv2_S80_FORCEONLY"

import subprocess
centerline_path = os.path.join(HERE, "matv2_centerline_S80_FORCEONLY.json")
subprocess.run(
    [sys.executable, os.path.join(HERE, "get_bent_centerline.py"),
     "--L_M", str(L_M), "--phi", str(PHI), "--out", centerline_path],
    check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
)
scene_info = scene.build_mesh(
    contact_s=S, ball_r=BALL_R, verbose=False, centerline_path=centerline_path,
    beta_deg=0.0, inp_name=inp_name, sets_name=sets_name,
    contact_mesh_size=0.08, contact_refine_radius=BALL_R + 0.3,
)
normal = np.array(scene_info["normal"])

# 임의의 작은 시험 힘 (방향: -normal, 즉 안쪽으로 미는 방향과 같은 부호 관례를 맞추기 위해
# 실측에서 Fx(법선성분)가 양수로 나왔던 것과 같은 부호로 둠)
F_TEST_N = 0.005e-3  # 5uN = 0.005mN, 임의의 작은 테스트값

outer_ids = rc._parse_nset(sets_name, "N_TUBE_OUTER")
coords = rc._parse_node_coords(inp_name, outer_ids)
D_OUT = 2.0
contact_point = np.array(scene_info["contact_point"])
surface_point = contact_point + normal * (D_OUT / 2.0)
best_nid, best_d = None, 1e9
for nid, (x, y, z) in coords.items():
    d = np.linalg.norm(np.array([x, y, z]) - surface_point)
    if d < best_d:
        best_d = d
        best_nid = nid
print(f"하중 적용 절점: {best_nid} (거리 {best_d:.4f}mm), 시험 힘={F_TEST_N*1000:.4f}mN along -normal")

Fx, Fy, Fz = (-normal * F_TEST_N).tolist()

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
{best_nid}, 1, {Fx:.6E}
{best_nid}, 2, {Fy:.6E}
{best_nid}, 3, {Fz:.6E}
*NODE PRINT, NSET=N_TIP
U
*END STEP
"""
job_inp_path = os.path.join(HERE, f"{job_name}.inp")
with open(job_inp_path, "w") as f:
    f.write(inp_content)

print("실행 중...")
result = rc.run_ccx(job_name, timeout=1800, n_threads=4)
converged, last_time = rc.check_converged(job_name)
print(f"수렴={converged}, TOTAL TIME={last_time:.4f}")
if not converged:
    print("stdout 마지막:", result.stdout[-1000:])
else:
    with open(f"{job_name}.dat", encoding="latin-1") as f:
        dat = f.read()
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
        print(f"tip_ux={ux/n:.4f}mm, tip_uy={uy/n:.4f}mm, 팁변위={(((ux/n)**2+(uy/n)**2)**0.5):.4f}mm")

try:
    os.remove(centerline_path)
except OSError:
    pass
