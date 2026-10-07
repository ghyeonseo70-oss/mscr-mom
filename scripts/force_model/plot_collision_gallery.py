"""충돌이 발생했을 때 카테터가 실제로 어떤 모양이 되는지 - 검증된(믿을 수 있는) 결과들만
모아서 한 장에 비교. L_M=12.5mm, phi=-120deg 고정, 베이스 쪽(s=30)과 팁 쪽(s=80) 충돌을
같이 보여줌.

- s=30, depth=0.40mm: 실제 접촉(contact) FEA 실측 (_test_depth040.py, 2026-10-06)
  F_mag=0.0275mN, 팁변위=1.328mm
- s=80, F=0.005mN: "힘만 걸기" 방식, 검증된 안전범위 하한 (_test_s80_force_only.py, 2026-10-07)
  팁변위=0.950mm - 접촉 방식으로는 이 위치에서 5전 5패였음
- s=80, F=0.03mN: "힘만 걸기" 방식, 검증된 안전범위 상한 (_test_s80_f003.py, 2026-10-07)
  팁변위=5.462mm

전부 force_model.py 분석모델의 변형 패턴 위에 실측 팁변위로 크기를 보정한 것 - 패턴은
작은 변형 가정이라 변형이 클수록(특히 s=80 F=0.03mN) 정확도가 떨어질 수 있음(별도 논의됨)."""
import numpy as np
import matplotlib.pyplot as plt

import force_model as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

L_M, PHI = 12.5, -120.0

CASES = [
    dict(s=30.0, label="s=30mm 충돌 (베이스 쪽), depth=0.40mm\n실제 접촉(contact) 실측 — 가장 신뢰도 높음",
         tip_ux=1.0100, tip_uy=0.8619, color="#C0392B", fx=0.0125e-3, fy=0.0014e-3),
    dict(s=80.0, label="s=80mm 충돌 (팁 쪽), F=0.005mN\n\"힘만 걸기\" 검증범위 하한",
         tip_ux=0.5623, tip_uy=0.7659, color="#2451A3", fx=0.0125e-3, fy=0.0014e-3),
    dict(s=80.0, label="s=80mm 충돌 (팁 쪽), F=0.03mN\n\"힘만 걸기\" 검증범위 상한",
         tip_ux=3.3671, tip_uy=4.3003, color="#16A085", fx=0.0125e-3, fy=0.0014e-3),
]

free = fm.solve_shape(L_M=L_M, phi_deg=PHI, loads=[], return_curve=True)
hint = free["theta_L_deg"]


def board_curve(r):
    cs, cx, cy = r["curve_s_mm"], r["curve_x_mm"], r["curve_y_mm"]
    idx = np.argsort(cs)
    bx, by = fm.to_board_frame(cx[idx], cy[idx])
    return bx, by


s_common = np.linspace(0, 100, 501)
fs_sorted = np.sort(free["curve_s_mm"])
fbx, fby = board_curve(free)
fbx_c = np.interp(s_common, fs_sorted, fbx)
fby_c = np.interp(s_common, fs_sorted, fby)

fig, ax = plt.subplots(figsize=(11, 11))

ax.plot(fbx_c, fby_c, color="#888888", linewidth=3.5, linestyle=(0, (6, 3)),
        label="원래 모양 (충돌 전, 무접촉)", zorder=3)
ax.plot(fbx_c[0], fby_c[0], "ks", markersize=13, zorder=7, label="베이스(고정단)")

for case in CASES:
    S_TARGET = case["s"]
    real_tip = (case["tip_ux"]**2 + case["tip_uy"]**2) ** 0.5
    model_loaded = fm.solve_shape(L_M=L_M, phi_deg=PHI,
                                   loads=[{"type": "point", "s": S_TARGET, "Fx": case["fx"], "Fy": case["fy"]}],
                                   theta_L_hint_deg=hint, return_curve=True)
    ms_sorted = np.sort(model_loaded["curve_s_mm"])
    mbx, mby = board_curve(model_loaded)
    mbx_c = np.interp(s_common, ms_sorted, mbx)
    mby_c = np.interp(s_common, ms_sorted, mby)

    model_tip_shift = ((mbx_c[-1]-fbx_c[-1])**2 + (mby_c[-1]-fby_c[-1])**2) ** 0.5
    scale = real_tip / model_tip_shift
    ebx = fbx_c + (mbx_c - fbx_c) * scale
    eby = fby_c + (mby_c - fby_c) * scale

    ax.plot(ebx, eby, color=case["color"], linewidth=2.5, zorder=5, label=case["label"])
    contact_idx = np.argmin(np.abs(s_common - S_TARGET))
    ax.plot(fbx_c[contact_idx], fby_c[contact_idx], "^", color=case["color"], markersize=12, zorder=6)
    ax.plot(ebx[-1], eby[-1], "o", color=case["color"], markersize=9, zorder=6)

ax.set_xlabel("x (mm, 보드좌표)", fontsize=12)
ax.set_ylabel("y (mm, 보드좌표)", fontsize=12)
ax.set_title("충돌 났을 때 카테터 모양 — 검증된 결과 모음\n"
             "(회색 점선=원래 모양, 세모=충돌 지점, 동그라미=변형된 팁 위치)",
             fontweight="bold", fontsize=14)
ax.legend(loc="lower left", fontsize=9.5)
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.4)

plt.tight_layout()
out_path = "../../data/force_model/collision_gallery.png"
plt.savefig(out_path, dpi=150)
print(f"저장: {out_path}")
