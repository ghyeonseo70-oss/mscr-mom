"""원래(무접촉) 모양과 충돌로 변형된 모양을 곡선 전체로 겹쳐서 비교.
L_M=12.5mm, phi=-120deg, s=30mm, depth=0.20mm 실측 FEA 기준
(Fx=0.0125mN, Fy=0.0014mN, _test_bigdepth_fix.py 결과).

**중요**: force_model.py(단순 보 이론)에 이 힘을 그대로 넣고 계산하면 팁 변위가
0.0058mm로 나오는데, 실제 FEA 측정값(0.598mm)보다 100배나 작음 - 단순 analytical 모델이
접촉 상황에서의 실제 변형을 과소평가함(이미 알려진 한계, force_model은 원래 순수 자기토크
형상 계산용이지 접촉반응 정밀 검증용이 아님). 그래서 분석모델의 변형 "패턴"(방향/곡선 모양)은
쓰되, "크기"는 실측 FEA 팁변위에 맞춰 스케일을 보정함(비례식) - 그래야 그림이 실제 측정값과
어긋나지 않음.
"""
import numpy as np
import matplotlib.pyplot as plt

import force_model as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

L_M, PHI, S_TARGET = 12.5, -120.0, 30.0
FX_N, FY_N = 0.0125e-3, 0.0014e-3  # N (실측 FEA, 로컬좌표계 - force_model loads 형식과 동일)
# 실측 FEA 팁 변위(보드좌표, _test_bigdepth_fix.py 결과) - 분석모델 패턴을 여기 맞춰 보정.
REAL_TIP_SHIFT_MM = 0.598

free = fm.solve_shape(L_M=L_M, phi_deg=PHI, loads=[], return_curve=True)
hint = free["theta_L_deg"]

model_loaded = fm.solve_shape(L_M=L_M, phi_deg=PHI,
                               loads=[{"type": "point", "s": S_TARGET, "Fx": FX_N, "Fy": FY_N}],
                               theta_L_hint_deg=hint, return_curve=True)


def board_curve(r):
    cs, cx, cy = r["curve_s_mm"], r["curve_x_mm"], r["curve_y_mm"]
    idx = np.argsort(cs)
    bx, by = fm.to_board_frame(cx[idx], cy[idx])
    return bx, by


fbx, fby = board_curve(free)
mbx, mby = board_curve(model_loaded)

# 버그 수정: free/model_loaded 곡선은 접촉점(s=30)에서 breakpoint가 추가돼 s 샘플링 개수/위치가
# 서로 달라서, 그냥 배열끼리 빼면 점이 안 맞아 곡선에 삐죽선이 생김 - 공통 s 격자에 보간 후 비교.
s_common = np.linspace(0, 100, 501)
fs_sorted = np.sort(free["curve_s_mm"])
ms_sorted = np.sort(model_loaded["curve_s_mm"])
fbx_c = np.interp(s_common, fs_sorted, fbx)
fby_c = np.interp(s_common, fs_sorted, fby)
mbx_c = np.interp(s_common, ms_sorted, mbx)
mby_c = np.interp(s_common, ms_sorted, mby)

# 분석모델이 예측한 "패턴"(각 점의 변위 벡터)은 쓰되, 그 크기가 실측(0.598mm)의 몇 배인지
# 비율을 구해서 곡선 전체에 똑같이 적용 - 모양은 모델 그대로, 크기만 실측에 맞춤.
model_tip_shift = ((mbx_c[-1]-fbx_c[-1])**2 + (mby_c[-1]-fby_c[-1])**2) ** 0.5
scale = REAL_TIP_SHIFT_MM / model_tip_shift
fbx, fby = fbx_c, fby_c
ebx = fbx_c + (mbx_c - fbx_c) * scale
eby = fby_c + (mby_c - fby_c) * scale

fig, ax = plt.subplots(figsize=(10, 10))

ax.plot(fbx, fby, color="#888888", linewidth=3, linestyle=(0, (6, 3)),
        label="원래 모양 (충돌 전, 무접촉)", zorder=3)
ax.plot(ebx, eby, color="#C0392B", linewidth=2.5,
        label="접촉 후 (실측 팁변위에 맞춰 패턴 보정)", zorder=5)

ax.plot(fbx[0], fby[0], "ks", markersize=11, zorder=6, label="베이스(고정단)")
ax.plot(fbx[-1], fby[-1], "o", color="#888888", markersize=9, zorder=6)
ax.plot(ebx[-1], eby[-1], "o", color="#C0392B", markersize=9, zorder=6)

tip_shift_real = ((ebx[-1]-fbx[-1])**2 + (eby[-1]-fby[-1])**2) ** 0.5
ax.annotate(f"실측 팁 변위: {tip_shift_real:.3f}mm\n(분석모델 자체 계산값은 {model_tip_shift:.4f}mm로\n"
            f"실측보다 {scale:.0f}배 작게 나와 - 모양만 쓰고 크기는 실측으로 보정함)",
            (ebx[-1], eby[-1]), textcoords="offset points", xytext=(15, 10),
            fontsize=10.5, color="#C0392B", fontweight="bold")

ax.set_xlabel("x (mm, 보드좌표)")
ax.set_ylabel("y (mm, 보드좌표)")
ax.set_title(f"충돌 전후 카테터 형상 비교 (L_M={L_M}mm, phi={PHI}°, s={S_TARGET}mm, depth=0.20mm)\n"
             f"회색 점선=원래 모양 / 빨강=접촉 후(팁변위는 실측값, 곡선 모양은 분석모델 패턴)",
             fontweight="bold", fontsize=13)
ax.legend(loc="lower left", fontsize=10)
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.4)

plt.tight_layout()
out_path = "../../data/force_model/before_after_shape_comparison.png"
plt.savefig(out_path, dpi=150)
print(f"저장: {out_path}")
print(f"실측 팁 변위: {tip_shift_real:.4f}mm (분석모델 원값 {model_tip_shift:.4f}mm, 보정배율 {scale:.1f}배)")
