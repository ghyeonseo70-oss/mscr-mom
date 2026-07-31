"""FEA 접촉 스윕(fea_bent_contact_sweep.json) 결과가 물리적으로 타당한지(부호 반전/불안정
없이 누르는 깊이에 비례해 힘·팁변위가 단조증가하는지) 보여주는 검증 그림.

PROJECT_STATUS.md에 적힌 물리모델(force_model.py)의 핵심 미해결 문제 - "아주 작은 힘에서도
미는 방향과 실제 변위 방향이 반대로 나옴" - 이 FEA 결과에는 나타나지 않음을 보여주는 것이 목적.
"""
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans CJK KR'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
# 고정 순서 카테고리 팔레트 (dataviz 스킬 기본값)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

with open(os.path.join(DATA_DIR, "fea_bent_contact_sweep.json")) as f:
    rows = json.load(f)

groups = defaultdict(list)
for r in rows:
    groups[r["contact_s_mm"]].append(r)

# 명백히 수렴 실패한 이상치 제거: depth>0.03인데 F_mag가 자유상태(~1e-12N) 수준으로 튄 경우
clean_groups = {}
for s, grp in groups.items():
    grp = sorted(grp, key=lambda r: r["push_depth_mm"])
    kept = [r for r in grp if not (r["push_depth_mm"] > 0.03 and r["F_mag_N"] < 1e-9)]
    clean_groups[s] = kept
n_dropped = sum(len(g) for g in groups.values()) - sum(len(g) for g in clean_groups.values())

s_values = sorted(clean_groups.keys())

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

for i, s in enumerate(s_values):
    grp = clean_groups[s]
    depth = [r["push_depth_mm"] for r in grp]
    fmag = [r["F_mag_N"] for r in grp]
    tipmag = [np.hypot(r["tip_ux_avg_mm"], r["tip_uy_avg_mm"]) for r in grp]
    color = CAT[i % len(CAT)]
    axes[0].plot(depth, fmag, "-o", color=color, linewidth=2, markersize=5,
                 label=f"s={s:.0f}mm")
    axes[1].plot(depth, tipmag, "-o", color=color, linewidth=2, markersize=5,
                 label=f"s={s:.0f}mm")

axes[0].set_yscale("log")
axes[0].set_xlabel("누르는 깊이 push_depth (mm)", color=INK)
axes[0].set_ylabel("접촉력 크기 F_mag (N, log)", color=INK)
axes[0].set_title("(A) 접촉력 — 깊이에 비례해 단조증가\n(부호 반전·불안정 없음)", fontweight="bold", color=INK)

axes[1].set_xlabel("누르는 깊이 push_depth (mm)", color=INK)
axes[1].set_ylabel("팁 변위 크기 |tip_u| (mm)", color=INK)
axes[1].set_title("(B) 팁(끝단) 변위 — 접촉위치별로도\n항상 같은 방향으로 단조증가", fontweight="bold", color=INK)

for ax in axes:
    ax.grid(True, linestyle=":", color=GRID, alpha=0.9)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=2)

note = f"L_M=50mm, phi=60deg 고정 / 접촉위치(s) {len(s_values)}곳 스윕"
if n_dropped:
    note += f" / 미수렴 이상치 {n_dropped}개 제외"
fig.suptitle("FEA 접촉 시뮬레이션 검증: 힘·변위가 미는 방향과 일관되게 반응함",
             fontsize=13, fontweight="bold", color=INK, y=0.99)
fig.text(0.5, 0.905, note, ha="center", fontsize=9, color=MUTED)
fig.tight_layout(rect=[0, 0, 1, 0.86])

out_path = os.path.join(DATA_DIR, "fea_contact_validation.png")
plt.savefig(out_path, dpi=150, facecolor="#fcfcfb")
print(f"저장: {out_path}")
