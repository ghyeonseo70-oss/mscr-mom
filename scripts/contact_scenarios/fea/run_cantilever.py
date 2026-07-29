"""
외팔보 검증: 짧은 실리콘관 시험편(30mm) 자유단에 작은 힘을 걸어서
CalculiX(비선형 초탄성 FEA) 결과와 force_model.py가 쓰는 선형 보 이론 예측을 비교한다.
두 값이 비슷하게 나오면(작은 변형 영역이므로) FEA 파이프라인(메쉬/재료/경계조건) 자체가
말이 되게 세팅됐다는 뜻 -> 그 다음에 접촉(비선형 국소변형) 시나리오로 확장해도 신뢰할 수 있다.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import material_convert as mat

L_TEST_M = 0.030  # m, make_tube_mesh.py의 L_TEST(30mm)와 일치해야 함
F_TOTAL = 2.0e-4  # N, 자유단에 거는 총 힘(작게 잡아 선형이론이 잘 맞는 영역으로)
MATERIAL_MODE = "neohookean"  # "linear"(디버그용, *ELASTIC) 또는 "neohookean"(*HYPERELASTIC)

# node_sets.inp에서 N_FREE 절점 개수 세기 (힘을 노드 수만큼 나눠서 걸어야 하므로)
with open(os.path.join(HERE, "node_sets.inp")) as f:
    content = f.read()
free_block = content.split("*NSET, NSET=N_FREE")[1]
free_nodes = [int(x) for x in re.findall(r"\d+", free_block)]
n_free = len(free_nodes)
F_PER_NODE = F_TOTAL / n_free
print(f"N_FREE 절점 수: {n_free}, 절점당 힘: {F_PER_NODE:.4e} N")

if MATERIAL_MODE == "linear":
    E_MM = mat.E * 1e-6  # Pa -> N/mm^2
    material_card = f"*ELASTIC\n{E_MM:.6E}, {mat.NU}"
    step_header = "*STEP\n*STATIC"  # 선형(NLGEOM 없이) - 보 이론과 1:1 비교
else:
    material_card = f"*HYPERELASTIC, NEO HOOKE\n{mat.C10_MM:.6E}, {mat.D1_MM:.6E}"
    step_header = "*STEP, NLGEOM\n*STATIC"

inp_content = f"""*INCLUDE, INPUT=tube_mesh.inp
*INCLUDE, INPUT=node_sets.inp
**
*MATERIAL, NAME=SILICONE
{material_card}
**
*SOLID SECTION, ELSET=TUBE, MATERIAL=SILICONE
**
*BOUNDARY
N_FIXED, 1, 3
**
{step_header}
*CLOAD
N_FREE, 2, {F_PER_NODE:.6E}
*NODE PRINT, NSET=N_FREE
U
*NODE FILE
U
*EL FILE
S, E
*END STEP
"""
inp_path = os.path.join(HERE, "cantilever_test.inp")
with open(inp_path, "w") as f:
    f.write(inp_content)
print(f"작성: {inp_path}")

# ── CalculiX 실행 (conda 환경 활성화 필요 - PowerShell엔 conda hook이 없어서 .bat로 감쌈) ──
# subprocess에 인자 리스트로 넘기면 Windows에서 따옴표가 이중이스케이프되는 문제가 있어서,
# 임시 .bat 파일을 만들어 그걸 실행하는 방식이 훨씬 안전함.
conda_root = os.path.join(os.path.expanduser("~"), "miniconda3")
job_name = "cantilever_test"
bat_path = os.path.join(HERE, "_run_ccx.bat")
with open(bat_path, "w") as f:
    f.write(f'@echo off\r\n')
    f.write(f'call "{conda_root}\\Scripts\\activate.bat" "{conda_root}"\r\n')
    f.write(f'cd /d "{HERE}"\r\n')
    f.write(f'ccx {job_name}\r\n')

print(f"\nCalculiX 실행 중...")
result = subprocess.run([bat_path], capture_output=True, text=True, shell=True)
print(result.stdout[-3000:])
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
os.remove(bat_path)

# ── 결과(.dat) 파싱: N_FREE 절점들의 평균 U2(y방향 변위) ──
dat_path = os.path.join(HERE, f"{job_name}.dat")
if not os.path.exists(dat_path):
    print(f"\n[실패] {dat_path} 가 생성되지 않음 — 위 CalculiX 출력에서 에러 확인 필요")
    sys.exit(1)

with open(dat_path) as f:
    dat = f.read()

uy_values = []
for line in dat.splitlines():
    parts = line.split()
    if len(parts) == 4:
        try:
            node_id = int(parts[0])
            uy = float(parts[2])
            if node_id in free_nodes:
                uy_values.append(uy)
        except ValueError:
            continue

if not uy_values:
    print("\n[실패] .dat에서 U2 값을 못 찾음. 파일 내용을 직접 확인하세요:")
    print(dat[:2000])
    sys.exit(1)

fea_deflection_mm = sum(uy_values) / len(uy_values)  # .dat 값은 이미 mm 단위(메쉬가 mm이므로)
beam_deflection_mm = F_TOTAL * L_TEST_M**3 / (3 * mat.K2) * 1000  # beam 공식은 SI(m)라 mm로 환산

print(f"\n=== 결과 비교 (자유단 y방향 처짐) ===")
print(f"FEA(CalculiX, {MATERIAL_MODE})   : {fea_deflection_mm:.4f} mm  (절점 {len(uy_values)}개 평균)")
print(f"선형 보 이론(force_model K2 기준): {beam_deflection_mm:.4f} mm")
diff_pct = abs(fea_deflection_mm - beam_deflection_mm) / abs(beam_deflection_mm) * 100
print(f"차이: {diff_pct:.2f}%")
