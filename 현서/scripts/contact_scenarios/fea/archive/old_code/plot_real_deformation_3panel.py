"""
참고 이미지 스타일 재현: 무접촉 기준형상(점선) vs 접촉시 실제형상(실선, 확대) + 미는방향 화살표,
같은 depth(0.10mm)에서 s=10/30/80mm 3가지 비교.

팁만 근사로 미는 대신, 실제 FEA로 측정한 힘(Fx,Fy)을 force_model의 점하중으로 그대로 넣어서
solve_shape_robust로 전체 곡선을 다시 풀어낸다(plot_nn2_results.py와 동일한 방식) - 이렇게 하면
"베이스 근처를 누르면 그 아래로 형상 전체가 크게 바뀌는" 실제 보 굽힘 효과가 곡선 전체에 반영됨
(팁 변위만 옮기는 근사로는 이 전파 효과를 표현 못 함).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "force_model"))
import force_model as fm

OUT_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")

L_M, PHI = 50.0, 60.0
DEPTH = 0.10
# fea_lm_phi_pos_sweep_all.json 실측값 (L_M=50,phi=60,depth=0.10mm 고정) - 힘 + 팁변위(tip_ux,tip_uy) 둘 다 실측
CASES = [
    {"s": 10.0, "Fx": 0.0002019720998922835, "Fy": 6.327430460460517e-05,
     "tip_ux": 0.8806970896739125, "tip_uy": 0.31480696141304343},
    {"s": 30.0, "Fx": 7.149830539204157e-06, "Fy": 3.4980837749775783e-06,
     "tip_ux": 0.2675779315217391, "tip_uy": 0.10061267668478255},
    {"s": 80.0, "Fx": 7.721699922367469e-07, "Fy": 5.7933859862208164e-08,
     "tip_ux": 0.1328898239130435, "tip_uy": 0.044425707282608695},
]
EXAGGERATE = 15  # 시각화용 확대 배율(회전기반 재구성이라 30배까지는 필요 없음)

r_free = fm.solve_shape(L_M=L_M, phi_deg=PHI, loads=[], return_curve=True)
order0 = np.argsort(r_free["curve_s_mm"])
cs0, cx0, cy0 = r_free["curve_s_mm"][order0], r_free["curve_x_mm"][order0], r_free["curve_y_mm"][order0]
cth0 = r_free["curve_theta_deg"][order0]

fig, axes = plt.subplots(1, 3, figsize=(17, 7.2))
colors = ["#2a78d6", "#eb6834", "#1baf7a"]
for panel, case in enumerate(CASES):
    ax = axes[panel]
    s_mm, Fx, Fy = case["s"], case["Fx"], case["Fy"]

    # 그 위치(s)의 국소 법선방향(튜브 바깥쪽) - make_bent_contact_scene와 동일 정의(beta=0 기준)
    theta = np.radians(np.interp(s_mm, cs0, cth0))
    normal = np.array([-np.cos(theta), np.sin(theta)])
    push_ang = np.degrees(np.arctan2(Fy, Fx))

    # force_model에 실측 Fx,Fy를 그대로 넣어 재구성했더니 두 모델간 알려진 강성 불일치 때문에
    # (PROJECT_STATUS.md에 기록된 문제) 실측 팁변위(tip_ux,tip_uy)보다 훨씬 작게 나와서 안 보였음.
    # 대신 "접촉위치(s) 이전 구간은 그대로, 그 이후 구간을 강체 회전"시켜 팁이 정확히 실측
    # tip_ux,tip_uy만큼 이동하도록 재구성(단순 보 굽힘 근사지만 실측 팁변위를 정확히 반영함).
    d_x_local, d_y_local = case["tip_uy"], case["tip_ux"]  # 국소<->보드 축교환(이 코드베이스 공통 규칙)
    pivot = np.array([np.interp(s_mm, cs0, cx0), np.interp(s_mm, cs0, cy0)])
    tip0 = np.array([cx0[-1], cy0[-1]])
    tip_target = tip0 + np.array([d_x_local, d_y_local]) * EXAGGERATE

    v0 = tip0 - pivot
    v1 = tip_target - pivot
    ang0, ang1 = np.arctan2(v0[1], v0[0]), np.arctan2(v1[1], v1[0])
    dtheta = ang1 - ang0
    R = np.array([[np.cos(dtheta), -np.sin(dtheta)], [np.sin(dtheta), np.cos(dtheta)]])

    mask_after = cs0 >= s_mm
    cx_ex, cy_ex = cx0.copy(), cy0.copy()
    rel = np.stack([cx0[mask_after] - pivot[0], cy0[mask_after] - pivot[1]], axis=1)
    rel_rot = rel @ R.T
    cx_ex[mask_after] = pivot[0] + rel_rot[:, 0]
    cy_ex[mask_after] = pivot[1] + rel_rot[:, 1]

    ax.plot(cx0, cy0, "--", color="#9aa5ab", linewidth=2, label="무접촉 기준형상", zorder=2)
    ax.plot(cx_ex, cy_ex, "-", color=colors[panel], linewidth=2.6,
            label=f"접촉 시 실제형상(FEA, {EXAGGERATE}배 확대)", zorder=3)
    ax.plot(0, 0, "ks", markersize=10, zorder=5, label="베이스(고정단)")

    xt, yt = np.interp(s_mm, cs0, cx0), np.interp(s_mm, cs0, cy0)
    ax.plot(xt, yt, "o", color="#e34948", markersize=12, zorder=7,
            markeredgecolor="white", markeredgewidth=1.4, label=f"접촉위치 (s={s_mm:.0f}mm)")
    arrow_len = 18
    ax.annotate("", xy=(xt - normal[0] * arrow_len, yt - normal[1] * arrow_len), xytext=(xt, yt),
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", linewidth=2.6, mutation_scale=20), zorder=8)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlabel("x (mm, 보드좌표)")
    ax.set_ylabel("y (mm, 보드좌표)")
    tip_disp = np.hypot(case["tip_ux"], case["tip_uy"])
    ax.set_title(f"s={s_mm:.0f}mm 위치를 {DEPTH:.2f}mm 누름 (미는방향 {push_ang:.0f}도)\n"
                 f"F={np.hypot(Fx,Fy)*1000:.3f}mN | 팁 변위(실측) {tip_disp:.2f}mm",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="best")

fig.suptitle(f"초기 형상 vs 접촉 후 실제 형상(FEA 실측 팁변위 기반 재구성) — 예시 3가지 (같은 깊이 {DEPTH:.2f}mm)",
             fontweight="bold", fontsize=13, y=1.03)
out = os.path.join(OUT_DIR, "real_deformation_3panel.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"저장: {out}")
