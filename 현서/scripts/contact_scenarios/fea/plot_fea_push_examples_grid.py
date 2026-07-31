"""fea_push_examples.png을 direction_diversity_cases.png과 같은 스타일(사례별 서브플롯,
두꺼운 회색 튜브 + 점선 기준형상, 힘방향 화살표)로 다시 그린 버전.

plot_direction_diversity.py와 달리 "예측 접촉위치(초록 X)"는 그리지 않음 - 여기서는
예측모델과의 비교가 아니라 실제 FEA 결과 3개(s=10/40/80mm, 같은 깊이 0.10mm)를 있는
그대로 보여주는 것이 목적이라 비교 대상 자체가 없음.
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans CJK KR'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "force_model"))
import force_model as fm

DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
BOARD_BASE_X = 90.0  # get_bent_centerline.py의 보드좌표 변환 기준과 동일

with open(os.path.join(DATA_DIR, "fea_bent_contact_sweep.json")) as f:
    rows = json.load(f)

TARGET_DEPTH = 0.10
EXAMPLE_S = [10.0, 40.0, 80.0]
examples = []
for s in EXAMPLE_S:
    cand = [r for r in rows if r["contact_s_mm"] == s
            and abs(r["push_depth_mm"] - TARGET_DEPTH) < 1e-6]
    assert cand, f"s={s}, depth={TARGET_DEPTH} 케이스를 찾을 수 없음"
    examples.append(cand[0])

L_M, PHI_DEG = 50.0, 60.0  # fea_bent_contact_sweep.json 생성 시 고정한 형상
r_free = fm.solve_shape(L_M=L_M, phi_deg=PHI_DEG, loads=[], return_curve=True)
fcs, fcx, fcy = r_free["curve_s_mm"], r_free["curve_x_mm"], r_free["curve_y_mm"]
forder = np.argsort(fcs)
fcs, fcx, fcy = fcs[forder], fcx[forder], fcy[forder]

TIP_SCALE = 30  # 팁 변위 과장 배율(실제 0.1~0.9mm라 그대로면 안 보임)

fig, axes = plt.subplots(1, 3, figsize=(17, 6.3))
for ax, r in zip(axes, examples):
    s = r["contact_s_mm"]
    bx, by, _ = r["ball_center"]
    # 보드좌표 -> 로컬좌표 (get_bent_centerline.py의 x_b=BASE+y_l, y_b=x_l 을 역으로)
    x_push, y_push = by, bx - BOARD_BASE_X
    Fx_local, Fy_local = r["Fy_total_N"], r["Fx_total_N"]  # 같은 축 교환 규칙 적용
    tip_dx_local = r["tip_uy_avg_mm"] * TIP_SCALE
    tip_dy_local = r["tip_ux_avg_mm"] * TIP_SCALE
    tip_mag = np.hypot(r["tip_ux_avg_mm"], r["tip_uy_avg_mm"])

    # 무접촉 기준형상 (두꺼운 회색 튜브 + 점선)
    ax.plot(fcx, fcy, "-", color="#cfd8e3", linewidth=9, solid_capstyle="round", zorder=2)
    ax.plot(fcx, fcy, "--", color="#8a93a6", linewidth=1.6, zorder=2.5, label="무접촉 기준형상")
    ax.plot(0, 0, "ks", markersize=9, zorder=5, label="베이스(고정단)")

    # 미는 힘 방향 화살표 + 실제 접촉위치
    f_dir = np.array([Fx_local, Fy_local]) / (np.hypot(Fx_local, Fy_local) + 1e-12)
    arrow_len = 16.0
    ax.annotate("", xy=(x_push, y_push),
                xytext=(x_push - f_dir[0] * arrow_len, y_push - f_dir[1] * arrow_len),
                arrowprops=dict(arrowstyle="-|>", color="#B23A32", linewidth=2.6, mutation_scale=20),
                zorder=8)
    ax.plot(x_push, y_push, "o", color="#e34948", markersize=12, zorder=7,
            markeredgecolor="white", markeredgewidth=1.5, label=f"접촉위치 (s={s:.0f}mm)")

    # 팁 변위 (실제 FEA 값, 과장 배율로 확대 표시)
    tip_x0, tip_y0 = r_free["x_L"], r_free["y_L"]
    ax.annotate("", xy=(tip_x0 + tip_dx_local, tip_y0 + tip_dy_local), xytext=(tip_x0, tip_y0),
                arrowprops=dict(arrowstyle="-|>", color="#2a78d6", linewidth=2.6, mutation_scale=20),
                zorder=8)
    ax.plot(tip_x0, tip_y0, "o", color="#0b0b0b", markersize=8, zorder=6, label="팁(원래 위치)")
    ax.plot(tip_x0 + tip_dx_local, tip_y0 + tip_dy_local, "o", color="#2a78d6", markersize=9,
            zorder=8, label=f"팁 변위 {TIP_SCALE}배 확대 (실제 {tip_mag:.2f}mm)")

    # 화살표 끝점까지 autoscale에 포함되도록(안 그러면 화살표가 축 밖으로 잘림) 투명 점 추가
    ax.plot([x_push - f_dir[0] * arrow_len, tip_x0 + tip_dx_local],
            [y_push - f_dir[1] * arrow_len, tip_y0 + tip_dy_local], alpha=0)
    ax.margins(0.12)

    f_ang_local = np.degrees(np.arctan2(Fy_local, Fx_local))

    ax.set_xlabel("x (mm, 로컬)")
    ax.set_ylabel("y (mm, 로컬)")
    ax.set_title(f"s={s:.0f}mm 위치를 {TARGET_DEPTH:.2f}mm 누름 (미는방향 {f_ang_local:.0f}도)\n"
                 f"F={r['F_mag_N']*1000:.3f}mN | 팁변위 {tip_mag:.2f}mm",
                 fontweight="bold", fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=7.5, loc="best")

fig.suptitle("접촉위치에 따른 팁 반응 차이 - FEA 예시 3가지 (같은 깊이 0.10mm)",
             fontweight="bold", fontsize=14.5, y=0.98)
fig.text(0.5, 0.91,
         "미는방향 = 로컬 x축(그림의 x) 기준 힘벡터 각도 / 원주방향 접촉위치(beta)는 3개 모두 0도(굽힘평면 바깥쪽)로 동일, 튜브가 휘어있어 s마다 방향이 달라 보임",
         ha="center", fontsize=9, color="#52514e")
fig.tight_layout(rect=[0, 0, 1, 0.85])

out_path = os.path.join(DATA_DIR, "fea_push_examples_grid.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"저장: {out_path}")
