"""
force_model.py가 쓰는 영률(E, 교수님 제공 MATLAB 코드(code(2).txt) 값)에서 FEA용
초탄성(Neo-Hookean) 재료물성을 역산.

2026-08-18 수정: 예전엔 force_model.py의 "논문 그림 디지타이징 피팅" K2를 여기서 K2=E*I로
역산해서 E를 구했음(K2=9.8872e-07 고정값). 이제 force_model.py가 반대 방향(E=850kPa를
MATLAB 코드에서 직접 가져와 K2=E*I로 계산)으로 바뀌었으므로, 여기서도 E를 K2에서 역산하지
않고 force_model.py와 동일한 E를 직접 씀 - 두 파일이 서로 다른 K2를 참조해서 FEA 자유단
형상과 접촉힘 재료가 어긋나는 일을 원천적으로 막기 위함(같은 상수 하나를 양쪽이 공유).

선형 보 이론: K = E * I. I(중공원통 단면 2차모멘트) = pi/64 * (D_out^4 - D_in^4).
이렇게 얻은 E는 "선형 보 이론 기준" 등가 영률이라, FEA에서 쓸 초탄성(Neo-Hookean) 모델로
바꿔줘야 함. 실리콘처럼 거의 비압축(포아송비 ~0.49~0.499)인 재료는 소변형 극한에서
전단탄성계수 mu = E / (2*(1+nu)) 이고, Neo-Hookean의 C10 = mu/2 관계를 씀.
(주의: 이건 "소변형 근사" 변환이라 대변형에서는 실제 응력-변형 곡선과 다를 수 있음.
나중에 인장시험 데이터가 생기면 이 파일의 E, NU 값만 실측값으로 바꾸면 됨.)
"""
import numpy as np

# ── force_model.py와 동일한 값 (scripts/force_model/force_model.py의 E_YOUNG 참고) ──
E = 850 * 1000  # Pa, 교수님 제공 MATLAB 코드(code(2).txt) 값

# ── 실리콘 튜브 단면 (논문 Section II-B 기준, force_model.py의 D_OUTER/D_INNER와 동일) ──
D_OUT = 2.0e-3   # m
D_IN = 1.0e-3    # m
I_SILICONE = np.pi / 64 * (D_OUT**4 - D_IN**4)  # m^4

K2 = E * I_SILICONE  # N*m^2, 참고용 - force_model.py의 K2와 반드시 같아야 함
NU = 0.49  # 실리콘 고무의 전형적인 포아송비 (거의 비압축)

MU = E / (2 * (1 + NU))   # 전단탄성계수 (Pa, SI)
C10 = MU / 2               # Neo-Hookean C10 (Pa, SI)
D1 = (1 - 2 * NU) / MU     # 압축성 계수 (1/Pa, SI)

# ── FEA 메쉬는 mm 단위로 만들기 때문에(make_tube_mesh.py), 그 모델에서는
# 힘=N, 길이=mm, 따라서 압력/탄성계수는 N/mm^2(=MPa) 단위로 넣어야 함.
# 1 Pa = 1e-6 N/mm^2 이므로 C10은 1e-6을 곱하고, D1은 (1/압력) 차원이라 반대로 1e6을 곱함.
# 이 변환을 빼먹으면 재료가 100만배 뻣뻣하게 들어가는 단위버그가 생김(실제로 한번 겪음).
C10_MM = C10 * 1e-6   # N/mm^2
D1_MM = D1 * 1e6      # mm^2/N

if __name__ == "__main__":
    print(f"I_silicone = {I_SILICONE:.4e} m^4")
    print(f"등가 영률 E = {E:.4e} Pa = {E/1e6:.3f} MPa")
    print(f"전단탄성계수 mu = {MU:.4e} Pa")
    print(f"Neo-Hookean C10 = {C10:.4e} Pa  (SI)")
    print(f"Neo-Hookean D1  = {D1:.4e} 1/Pa  (SI)")
    print(f"[FEA용, mm-N 단위계] C10 = {C10_MM:.6e} N/mm^2,  D1 = {D1_MM:.6e} mm^2/N")
