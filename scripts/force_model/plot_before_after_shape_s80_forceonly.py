"""원래(무접촉) 모양과 충돌로 변형된 모양을 곡선 전체로 겹쳐서 비교.
L_M=12.5mm, phi=-120deg, s=80mm(팁 근처 - 접촉 방식으로는 5전 5패였던 위치),
"힘만 걸기(force-only)" 방식, F=0.005mN(검증된 안전 범위, 실제 접촉과 99.5% 일치 확인된
구간) 기준 (_test_s80_force_only.py 결과, 2026-10-07: tip_ux=0.5623mm, tip_uy=0.7659mm,
팁변위=0.9502mm).

이 위치(s=80)는 오늘 하루 종일 접촉(contact) 방식으로 단 한 번도 수렴하지 못했던 곳인데,
"힘만 걸기" 방식으로는 바로 풀렸고, 다른 검증 케이스(s=30)에서 실제 접촉과 99.5% 일치가
확인된 안전한 힘 범위를 그대로 적용한 결과임.

force_model.py(단순 보 이론)로 변형 패턴(방향/곡선 모양)만 구하고, 크기는 이 실측(FEA)
팁변위에 맞춰 보정함 - 이전 플롯들과 동일한 방식."""
import numpy as np
import matplotlib.pyplot as plt

import force_model as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

L_M, PHI, S_TARGET = 12.5, -120.0, 80.0
F_APPLIED_MN = 0.005
FX_N, FY_N = 0.0125e-3, 0.0014e-3  # 패턴 방향용 - 크기는 아래서 실측으로 재보정되므로 무관
REAL_TIP_UX, REAL_TIP_UY = 0.5623, 0.7659  # 실측(힘만 걸기) 결과, 보드좌표계
REAL_TIP_SHIFT_MM = (REAL_TIP_UX**2 + REAL_TIP_UY**2) ** 0.5

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

# 접촉점(s=80) 위치도 표시
contact_idx = np.argmin(np.abs(s_common - S_TARGET))

fig, ax = plt.subplots(figsize=(10, 10))

ax.plot(fbx, fby, color="#888888", linewidth=3, linestyle=(0, (6, 3)),
        label="원래 모양 (충돌 전, 무접촉)", zorder=3)
ax.plot(ebx, eby, color="#2451A3", linewidth=2.5,
        label="충돌 후 (힘만 걸기, F=0.005mN)", zorder=5)

ax.plot(fbx[0], fby[0], "ks", markersize=11, zorder=6, label="베이스(고정단)")
ax.plot(fbx[contact_idx], fby[contact_idx], "^", color="#E67E22", markersize=13, zorder=6,
        label="충돌 지점 (s=80mm)")
ax.plot(fbx[-1], fby[-1], "o", color="#888888", markersize=9, zorder=6)
ax.plot(ebx[-1], eby[-1], "o", color="#2451A3", markersize=9, zorder=6)

ax.annotate(f"팁 변위: {REAL_TIP_SHIFT_MM:.3f}mm\n(\"힘만 걸기\" 실측, 접촉 방식으론\n"
            f"이 위치에서 한 번도 안 풀렸음)",
            (ebx[-1], eby[-1]), textcoords="offset points", xytext=(15, 10),
            fontsize=10.5, color="#2451A3", fontweight="bold")

ax.set_xlabel("x (mm, 보드좌표)")
ax.set_ylabel("y (mm, 보드좌표)")
ax.set_title(f"충돌 전후 카테터 형상 (L_M={L_M}mm, phi={PHI}°, s={S_TARGET}mm — 팁 근처)\n"
             f"접촉(contact) 방식은 5전 5패였던 위치 — \"힘만 걸기\"로 대신 계산함",
             fontweight="bold", fontsize=13)
ax.legend(loc="lower left", fontsize=10)
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.4)

plt.tight_layout()
out_path = "../../data/force_model/before_after_shape_s80_forceonly.png"
plt.savefig(out_path, dpi=150)
print(f"저장: {out_path}")
print(f"실측(힘만걸기) 팁 변위: {REAL_TIP_SHIFT_MM:.4f}mm")
