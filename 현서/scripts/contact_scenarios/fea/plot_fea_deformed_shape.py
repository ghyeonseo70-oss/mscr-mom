"""초기(무접촉) 카테터 형상과, 실제 FEA로 계산된 접촉 후 형상을 나란히 겹쳐서 보여주는 그림.
extract_fea_deformed_shape.py가 원본 .frd(CalculiX 결과)에서 직접 뽑아낸 "실제 변형된
중심선"을 사용 - 팁 한 점만 화살표로 보여주던 이전 버전과 달리 전체 길이를 따라 어떻게
휘는지 보여줌. 변위가 실제로는 0.1~1mm 수준(길이 100mm 대비)이라 그대로면 안 보여서
과장 배율로 확대하고, 그 사실을 명시함.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans CJK KR'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")

with open(os.path.join(HERE, "bent_centerline.json")) as f:
    cl = json.load(f)
with open(os.path.join(DATA_DIR, "fea_deformed_centerline_examples.json")) as f:
    deformed = json.load(f)
with open(os.path.join(DATA_DIR, "fea_bent_contact_sweep.json")) as f:
    sweep_rows = json.load(f)

TARGET_DEPTH = 0.10
TAGS = ["s10", "s30", "s80"]  # s40/s60은 .frd가 이후 재시도로 덮어써져서 제외(스크립트 주석 참고)
SCALE = 30

CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

fcx = [p["x"] for p in cl["points"]]
fcy = [p["y"] for p in cl["points"]]

fig, axes = plt.subplots(1, 3, figsize=(17, 6.8))
for ax, tag, color in zip(axes, TAGS, CAT):
    s_val = deformed[tag]["contact_s_mm"]
    rec = deformed[tag]
    orig = np.array(rec["orig_xyz"])  # (40, 3) 보드좌표
    defm = np.array(rec["deformed_xyz"])
    delta = defm - orig
    exagg_x = orig[:, 0] + delta[:, 0] * SCALE
    exagg_y = orig[:, 1] + delta[:, 1] * SCALE

    row = [r for r in sweep_rows if r["contact_s_mm"] == s_val
           and abs(r["push_depth_mm"] - TARGET_DEPTH) < 1e-6][0]
    bx, by, _ = row["ball_center"]
    Fx, Fy = row["Fx_total_N"], row["Fy_total_N"]
    f_ang = np.degrees(np.arctan2(Fy, Fx))
    tip_delta_mag = np.linalg.norm(delta[-1, :2])

    ax.plot(fcx, fcy, "-", color="#cfd8e3", linewidth=9, solid_capstyle="round", zorder=2)
    ax.plot(fcx, fcy, "--", color="#8a93a6", linewidth=1.6, zorder=2.5, label="무접촉 기준형상")
    ax.plot(exagg_x, exagg_y, "-", color=color, linewidth=2.2, zorder=4,
            label=f"접촉 시 실제형상 (FEA, {SCALE}배 확대)")
    ax.plot(fcx[0], fcy[0], "ks", markersize=10, zorder=6, label="베이스(고정단)")

    f_dir = np.array([Fx, Fy]) / (np.hypot(Fx, Fy) + 1e-12)
    arrow_len = 16.0
    ax.annotate("", xy=(bx, by), xytext=(bx - f_dir[0] * arrow_len, by - f_dir[1] * arrow_len),
                arrowprops=dict(arrowstyle="-|>", color="#B23A32", linewidth=2.6, mutation_scale=20),
                zorder=8)
    ax.plot(bx, by, "o", color="#e34948", markersize=11, zorder=7,
            markeredgecolor="white", markeredgewidth=1.5, label=f"접촉위치 (s={s_val:.0f}mm)")

    ax.plot([bx - f_dir[0] * arrow_len, exagg_x.min(), exagg_x.max()],
            [by - f_dir[1] * arrow_len, exagg_y.min(), exagg_y.max()], alpha=0)
    ax.margins(0.12)

    ax.set_xlabel("x (mm, 보드좌표)")
    ax.set_ylabel("y (mm, 보드좌표)")
    ax.set_title(f"s={s_val:.0f}mm 위치를 {TARGET_DEPTH:.2f}mm 누름 (미는방향 {f_ang:.0f}도)\n"
                 f"F={row['F_mag_N']*1000:.3f}mN | 팁 변위 {tip_delta_mag:.2f}mm",
                 fontweight="bold", fontsize=11, color=INK)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", color=GRID, alpha=0.6)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(fontsize=7.5, loc="best")

fig.suptitle("초기 형상 vs 접촉 후 실제 형상(FEA) - 예시 3가지 (같은 깊이 0.10mm)",
             fontweight="bold", fontsize=14.5, y=0.985)
fig.text(0.5, 0.915,
         f"굵은 회색+점선 = 무접촉 기준형상 / 색 실선 = 접촉 시 실제 FEA 변형 형상(원본 CalculiX 결과에서 직접 추출, {SCALE}배 확대) / 빨간 화살표 = 미는 힘 방향",
         ha="center", fontsize=9, color=MUTED)
fig.tight_layout(rect=[0, 0, 1, 0.86])

out_path = os.path.join(DATA_DIR, "fea_deformed_shape_examples.png")
fig.savefig(out_path, dpi=150, facecolor="#fcfcfb", bbox_inches="tight")
print(f"저장: {out_path}")
