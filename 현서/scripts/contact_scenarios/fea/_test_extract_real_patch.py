"""2026-10-06: Dirichlet 패치 보정 공식(Hertz 이론, 반무한고체 가정)이 얇은 쉘(벽두께 0.5mm)에
안 맞을 수 있음 - 실제 접촉(contact) 결과에서 "진짜 접촉 패치 크기/모양"을 직접 뽑아서 내
이론값(a_circ, a_axial)과 비교. 기존 메쉬 파일(matv2_mesh_DEPTH040_r40.inp, 이미 생성됨,
gmsh 재실행 불필요)을 재사용해서 真 접촉으로 재실행하되, N_TUBE_OUTER 전체의 U(변위)를
*NODE PRINT로 추가 요청함.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_contact as rc

inp_name = "matv2_mesh_DEPTH040_r40.inp"
sets_name = "matv2_node_sets_DEPTH040_r40.inp"
job_name = "matv2_EXTRACTPATCH_d40"

rc.build_inp(0.40, inp_name=inp_name, sets_name=sets_name, job_name=job_name,
             push_dir=(-0.9994336846170133, -0.0336498150494231, 0.0),  # s=30 케이스 기존 normal 재사용
             print_tip=True, print_mom=True, min_inc=1e-8, stabilize=True)

inp_path = os.path.join(HERE, f"{job_name}.inp")
with open(inp_path, encoding="latin-1") as f:
    content = f.read()
extra = "*NODE PRINT, NSET=N_TUBE_OUTER\nU\n"
content = content.replace("*END STEP", extra + "*END STEP")
with open(inp_path, "w", encoding="latin-1") as f:
    f.write(content)

print("실행 중 (기존 메쉬 재사용, N_TUBE_OUTER 전체 U 추가 출력)...")
result = rc.run_ccx(job_name, timeout=1800, n_threads=4)
converged, last_time = rc.check_converged(job_name)
print(f"수렴={converged}, TOTAL TIME={last_time:.4f}")
if converged:
    res = rc.parse_results(job_name, print_tip=True, print_mom=True)
    print(f"F_mag={res['F_mag_N']*1000:.4f}mN")
