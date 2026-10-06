"""원래(무접촉) 모양과 충돌로 변형된 모양을 곡선 전체로 겹쳐서 비교.
L_M=12.5mm, phi=-120deg, s=30mm, depth=0.40mm 실측 FEA 기준
(F_mag=0.0275mN, 팁변위=1.3277mm, 팁회전=-1.086deg, _test_depth040.py 결과, 2026-10-06).

**중요**: force_model.py(단순 보 이론)에 작은 힘을 넣고 계산한 변형 패턴(방향/곡선 모양)만
쓰고, "크기"는 실측 FEA 팁변위에 맞춰 스케일을 보정함(plot_before_after_shape.py와 동일한
방식) - 분석모델 자체의 절대 크기는 접촉 상황에서 신뢰할 수 없지만 패턴(어느 쪽으로 휘는가)은
유용하기 때문.

push_depth는 0.40mm였는데 팁은 1.33mm나 움직임 - 누른 깊이보다 팁 움직임이 3배 이상 큼.
이게 "국소적으로 눌리는 것"보다 "전체가 지렛대처럼 크게 휘는 것"이 압도적으로 더 큰 효과라는
직접적인 증거."""
import numpy as np
import matplotlib.pyplot as plt

import force_model as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

L_M, PHI, S_TARGET, DEPTH = 12.5, -120.0, 30.0, 0.40
FX_N, FY_N = 0.0125e-3, 0.0014e-3  # N (패턴 방향용 - 크기는 아래서 실측으로 재보정되므로 무관)
# 실측 FEA 팁 변위(보드좌표, _test_depth040.py 결과, 2026-10-06) - 분석모델 패턴을 여기 맞춰 보정.
REAL_TIP_SHIFT_MM = 1.3277

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

# plot_before_after_shape.py와 동일한 버그 수정: 공통 s 격자에 보간 후 비교.
s_common = np.linspace(0, 100, 501)
fs_sorted = np.sort(free["curve_s_mm"])
ms_sorted = np.sort(model_loaded["curve_s_mm"])
fbx_c = np.interp(s_common, fs_sorted, fbx)
fby_c = np.interp(s_common, fs_sorted, fby)
mbx_c = np.interp(s_common, ms_sorted, mbx)
mby_c = np.interp(s_common, ms_sorted, mby)

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
ax.annotate(f"실측 팁 변위: {tip_shift_real:.3f}mm (누른 깊이 {DEPTH:.2f}mm의 {tip_shift_real/DEPTH:.1f}배!)\n"
            f"분석모델 자체 계산값은 {model_tip_shift:.4f}mm로 실측보다 {scale:.0f}배 작게 나와\n"
            f"- 모양(패턴)만 쓰고 크기는 실측으로 보정함",
            (ebx[-1], eby[-1]), textcoords="offset points", xytext=(15, 10),
            fontsize=10.5, color="#C0392B", fontweight="bold")

ax.set_xlabel("x (mm, 보드좌표)")
ax.set_ylabel("y (mm, 보드좌표)")
ax.set_title(f"충돌 전후 카테터 형상 비교 (L_M={L_M}mm, phi={PHI}°, s={S_TARGET}mm, depth=0.40mm, 실측)\n"
             f"회색 점선=원래 모양 / 빨강=접촉 후 — 작은 압입이 지렛대처럼 큰 휘어짐으로 증폭됨",
             fontweight="bold", fontsize=13)
ax.legend(loc="lower left", fontsize=10)
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.4)

plt.tight_layout()
out_path = "../../data/force_model/before_after_shape_comparison_depth040.png"
plt.savefig(out_path, dpi=150)
print(f"저장: {out_path}")
print(f"실측 팁 변위: {tip_shift_real:.4f}mm (분석모델 원값 {model_tip_shift:.4f}mm, 보정배율 {scale:.1f}배)")
