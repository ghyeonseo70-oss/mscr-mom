"""
실리콘관(중공 원통) 3D 메쉬를 Gmsh로 생성해서 CalculiX용 .inp로 내보냄.

첫 검증용이라 짧은 시험편(30mm)으로 만듦 — 나중에 실제 로봇 길이(100mm)나 굽은 형상으로
확장할 때는 이 스크립트의 L_TEST, 그리고 곡선 형상 부분만 바꾸면 됨.

물리적 그룹(면/부피 이름)을 지정해서 나중에 CalculiX 입력파일(.inp)에서
"FIXED"(고정단), "FREE_END"(자유단, 하중 적용면), "TUBE"(재료 적용 대상)로 참조할 수 있게 함.
"""
import os
import gmsh

D_OUT = 2.0   # mm
D_IN = 1.0    # mm
L_TEST = 30.0  # mm, 검증용 시험편 길이 (짧게 잡아 처짐이 선형이론 범위 안에 들게)
MESH_SIZE = 0.3  # mm, 대략적 요소 크기 (튜브 두께 0.5mm 대비 촘촘하게)

gmsh.initialize()
gmsh.model.add("silicone_tube")

# 단면(환형, annulus): 바깥원 - 안쪽원
outer = gmsh.model.occ.addDisk(0, 0, 0, D_OUT / 2, D_OUT / 2)
inner = gmsh.model.occ.addDisk(0, 0, 0, D_IN / 2, D_IN / 2)
annulus = gmsh.model.occ.cut([(2, outer)], [(2, inner)])
gmsh.model.occ.synchronize()

# z방향으로 압출(extrude)해서 3D 튜브 생성
ext = gmsh.model.occ.extrude(annulus[0], 0, 0, L_TEST)
gmsh.model.occ.synchronize()

# ext 리스트에서 부피(dim=3)와 옆면들(dim=2)을 찾음. 맨 앞(annulus 복사본, dim=2)은 free end 반대쪽 면.
vol_tag = [e[1] for e in ext if e[0] == 3][0]

# 시작면(z=0, 고정단)과 끝면(z=L, 자유단) 찾기: 바운딩박스로 판별
surfaces = gmsh.model.getBoundary([(3, vol_tag)], oriented=False)
fixed_faces, free_faces = [], []
for dim, tag in surfaces:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(dim, tag)
    if abs(zmax - zmin) < 1e-6:  # 평평한 원환면(z=const)만 후보
        if abs(zmin) < 1e-3:
            fixed_faces.append(tag)
        elif abs(zmin - L_TEST) < 1e-3:
            free_faces.append(tag)

gmsh.model.addPhysicalGroup(3, [vol_tag], name="TUBE")
gmsh.model.addPhysicalGroup(2, fixed_faces, name="FIXED")
gmsh.model.addPhysicalGroup(2, free_faces, name="FREE_END")

gmsh.option.setNumber("Mesh.MeshSizeMin", MESH_SIZE)
gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE)
gmsh.model.mesh.generate(3)
# 1차 사면체(C3D4)는 굽힘 문제에서 실제보다 뻣뻣하게 나오는 고질적 문제(shear locking)가 있어서
# 2차(중간절점 추가, C3D10)로 올림 - 외팔보 검증에서 이 문제를 실제로 확인함.
gmsh.model.mesh.setOrder(2)

out_dir = os.path.dirname(os.path.abspath(__file__))
inp_path = os.path.join(out_dir, "tube_mesh.inp")
msh_path = os.path.join(out_dir, "tube_mesh.msh")
gmsh.write(inp_path)
gmsh.write(msh_path)  # 시각화 확인용(gmsh GUI로 열어볼 수 있음)

n_nodes = len(gmsh.model.mesh.getNodes()[0])
print(f"메쉬 생성 완료: 절점 {n_nodes}개")

# gmsh가 부피요소(C3D4)뿐 아니라 겉면 삼각형(CPS3, 2D 평면요소)도 같이 .inp에 써버림.
# 이 CPS3 요소들을 CalculiX가 "평면문제(plane stress)"로 오해해서 z=0 평면 체크 에러를
# 내므로, C3D4(3D 부피요소)만 남기고 CPS3 관련 *ELEMENT/*ELSET 블록은 걸러냄.
with open(inp_path, encoding="latin-1") as f:
    lines = f.readlines()

filtered = []
skip = False
for line in lines:
    stripped = line.strip()
    if stripped.startswith("*ELEMENT") and ("CPS3" in stripped or "CPS6" in stripped):
        skip = True
        continue
    if stripped.startswith("*ELSET") and ("Surface1" in stripped or "Surface4" in stripped
                                           or "FIXED" in stripped or "FREE_END" in stripped):
        skip = True
        continue
    if skip and stripped.startswith("*"):
        skip = False
    if skip:
        continue
    filtered.append(line)

with open(inp_path, "w", encoding="latin-1") as f:
    f.writelines(filtered)

print(f"저장(CPS3 필터링됨): {inp_path}")
print(f"저장: {msh_path}")

# CalculiX *BOUNDARY/*CLOAD는 절점집합(NSET)이 필요한데, gmsh가 .inp에 자동으로 써주는 건
# 요소집합(ELSET)뿐이라 별도로 NSET을 뽑아 include 파일로 저장해둠.
def nodes_of(dim, name):
    for d, t in gmsh.model.getPhysicalGroups(dim):
        if gmsh.model.getPhysicalName(d, t) == name:
            return gmsh.model.mesh.getNodesForPhysicalGroup(d, t)[0]
    raise RuntimeError(f"physical group {name} not found")

fixed_nodes = nodes_of(2, "FIXED")
free_nodes = nodes_of(2, "FREE_END")

sets_path = os.path.join(out_dir, "node_sets.inp")
with open(sets_path, "w") as f:
    f.write("*NSET, NSET=N_FIXED\n")
    for i in range(0, len(fixed_nodes), 10):
        f.write(",".join(str(n) for n in fixed_nodes[i:i + 10]) + ",\n")
    f.write("*NSET, NSET=N_FREE\n")
    for i in range(0, len(free_nodes), 10):
        f.write(",".join(str(n) for n in free_nodes[i:i + 10]) + ",\n")
print(f"저장: {sets_path}  (N_FIXED={len(fixed_nodes)}개, N_FREE={len(free_nodes)}개)")

gmsh.finalize()
