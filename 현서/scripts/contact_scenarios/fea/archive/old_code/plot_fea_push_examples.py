""""이쪽을 밀면 이렇게 된다"를 보여주는 예시 3개짜리 그림.

같은 push_depth(=0.10mm)로 카테터를 서로 다른 위치(s=10/40/80mm, 베이스에 가까움~팁에
가까움)에서 눌렀을 때 팁(끝단)이 얼마나/어느 방향으로 움직이는지를 실제 FEA 결과값으로
그린다. 팁 변위는 실제로는 0.1~0.9mm 수준(전체 길이 100mm 대비)이라 그대로 그리면 안 보여서
과장 배율(SCALE)로 확대해 표시하고, 그 사실을 그림에 명시한다.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans CJK KR'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]  # 팔레트 슬롯 1,2,3 (고정 순서)

with open(os.path.join(HERE, "bent_centerline.json")) as f:
    cl = json.load(f)
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

SCALE = 80  # 팁 변위 과장 배율 (실제 0.1~0.9mm -> 그림에서 8~70mm 정도로 보이게)

cx = [p["x"] for p in cl["points"]]
cy = [p["y"] for p in cl["points"]]
tip_x, tip_y = cl["x_L"], cl["y_L"]

fig, ax = plt.subplots(figsize=(8, 7.5))
ax.plot(cx, cy, "-", color=INK, linewidth=2.5, alpha=0.75, zorder=3,
        label=f"카테터 중심선 (L_M={cl['L_M']:.0f}mm, phi={cl['phi_deg']:.0f}deg, 무하중)")
ax.plot(cx[0], cy[0], "ks", markersize=11, zorder=6, label="베이스(고정단)")
ax.plot(tip_x, tip_y, "o", color=INK, markersize=9, zorder=6, label="팁(원래 위치)")

for r, color in zip(examples, CAT):
    bx, by = r["ball_center"][0], r["ball_center"][1]
    ax.plot(bx, by, "*", color=color, markersize=20, zorder=7,
            markeredgecolor="white", markeredgewidth=0.8)

    dx, dy = r["tip_ux_avg_mm"] * SCALE, r["tip_uy_avg_mm"] * SCALE
    tip_mag = np.hypot(r["tip_ux_avg_mm"], r["tip_uy_avg_mm"])
    ax.annotate("", xy=(tip_x + dx, tip_y + dy), xytext=(tip_x, tip_y),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=2.5,
                                 mutation_scale=22), zorder=8)
    ax.plot(tip_x + dx, tip_y + dy, "o", color=color, markersize=7, zorder=8)

    label = f"s={r['contact_s_mm']:.0f}mm 위치를 {TARGET_DEPTH:.2f}mm 누르면\n팁이 {tip_mag:.2f}mm 움직임"
    ax.annotate(label, xy=(bx, by), xytext=(12, 10), textcoords="offset points",
                fontsize=9.5, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.9))

ax.set_xlabel("x (mm, 보드좌표)", color=INK)
ax.set_ylabel("y (mm, 보드좌표)", color=INK)
ax.set_aspect("equal")
ax.grid(True, linestyle=":", color=GRID, alpha=0.9)
ax.tick_params(colors=MUTED)
for spine in ax.spines.values():
    spine.set_color(GRID)

ax.set_title(f"같은 깊이(0.10mm)로 눌러도 누르는 위치에 따라 팁 반응이 달라짐 — FEA 예시 3가지",
             fontsize=13, fontweight="bold", color=INK)
fig.text(0.5, 0.015,
         f"★ = 누른 위치(접촉점) / 화살표 = 팁 변위 방향 (실제 크기의 {SCALE}배 확대, 실제는 0.1~0.9mm 수준)",
         ha="center", fontsize=9, color=MUTED)
ax.legend(loc="lower left", fontsize=9, frameon=True)
fig.tight_layout(rect=[0, 0.03, 1, 1])

out_path = os.path.join(DATA_DIR, "fea_push_examples.png")
plt.savefig(out_path, dpi=150, facecolor="#fcfcfb")
print(f"저장: {out_path}")
