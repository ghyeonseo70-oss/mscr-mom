""""왜 접촉힘 계산에 FEA가 필요한가"를 뒷받침하는 설득용 그림.

해석적 모델(force_model.py, 빔이론+슈팅법)은 find_safe_force_range.py로 실측한 결과
400개 중 85.5%가 첫 테스트 힘(0.5mN)에서 이미 "미는 방향과 반대로" 뒤집히고, 중앙값은
0mN(=항상 즉시 뒤집힘)이다. 반면 FEA(CalculiX, 접촉 비침투 조건을 실제로 푸는 방식)는
지금까지 돌린 접촉 스윕 413개(공통 형상 스윕+geom스윕+각도스윕) 중 단 1건만 예외였고
그 1건도 확인해보면 방향반전이 아니라 CalculiX가 수렴 못 한 케이스(F_mag가 자유상태 수준인
~9e-13N으로 되돌아감, .sta의 TOT TIME<1.0으로 확인됨)라 실질적으로는 0/413.

핵심 메시지: 해석모델은 "점하중을 준 빔 방정식"을 풀 뿐이라 좌굴에 가까운 불안정한 해로도
수렴할 수 있는 반면, FEA는 접촉면이 침투할 수 없다는 물리적 제약을 실제로 풀기 때문에
이런 반전이 원천적으로 생기기 어렵다.
"""
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans CJK KR'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
FORCE_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "force_model")

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
RED = "#e34948"

# ── 해석적 모델: find_safe_force_range.py 결과 ──────────────────────────
thresholds = np.load(os.path.join(FORCE_DATA_DIR, "safe_force_range.npz"))["thresholds"]
n_analytic = len(thresholds)
frac_immediate = (thresholds < 0.5).mean() * 100  # 첫 테스트 힘(0.5mN)에서 이미 반전
median_analytic = np.median(thresholds)

# ── FEA: 접촉 스윕 3개 합쳐서 같은 방식으로 "반전(불안정)" 탐지 ──────────
all_rows = []
for fname in ["fea_bent_contact_sweep.json", "fea_geom_sweep_all.json", "fea_angle_sweep_all.json"]:
    all_rows.extend(json.load(open(os.path.join(FEA_DATA_DIR, fname))))

groups = defaultdict(list)
for r in all_rows:
    key = (round(r.get("L_M_mm", 50.0), 1), round(r.get("phi_deg", 60.0), 1),
           round(r.get("beta_deg", 0.0), 1), round(r["contact_s_mm"], 1))
    groups[key].append(r)

n_points, n_reversal = 0, 0
for grp in groups.values():
    grp = sorted(grp, key=lambda r: r["push_depth_mm"])
    if len(grp) < 2:
        continue
    n_points += len(grp)
    fmags = [r["F_mag_N"] for r in grp]
    for i in range(1, len(grp)):
        if fmags[i] < fmags[i - 1] * 0.5 and fmags[i - 1] > 1e-9:
            n_reversal += 1
n_fea_total = len(all_rows)

# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2), gridspec_kw={"width_ratios": [1.3, 1]})

# (A) 해석모델 히스토그램
ax = axes[0]
bins = [0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 21]
ax.hist(thresholds, bins=bins, color=RED, alpha=0.85, edgecolor="white")
ax.axvline(median_analytic, color=INK, linestyle="--", linewidth=1.5)
ax.text(median_analytic + 0.3, ax.get_ylim()[1] * 0.9, f"중앙값 {median_analytic:.1f}mN",
        fontsize=9, color=INK)
ax.set_xlabel("방향이 안 뒤집히고 버티는 최대 힘 (mN)")
ax.set_ylabel("케이스 수 (전체 400개)")
ax.set_title("(A) 해석적 모델(force_model.py)\n무작위 400개 조합 테스트 결과", fontweight="bold", color=INK)
ax.text(0.97, 0.7, f"{frac_immediate:.1f}%가\n첫 테스트 힘(0.5mN)에서\n이미 반전",
        transform=ax.transAxes, ha="right", fontsize=10.5, color=RED, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=RED))
ax.grid(True, linestyle=":", color=GRID, alpha=0.7, axis="y")
for spine in ax.spines.values():
    spine.set_color(GRID)
ax.tick_params(colors=MUTED)

# (B) 반전 발생 비율 막대비교
ax2 = axes[1]
labels = ["해석적 모델\n(빔이론+슈팅법)", "FEA\n(CalculiX 접촉해석)"]
rates = [frac_immediate, n_reversal / n_points * 100]
colors = [RED, BLUE]
bars = ax2.bar(labels, rates, color=colors, width=0.55, edgecolor="white")
for b, r in zip(bars, rates):
    ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{r:.1f}%",
              ha="center", fontweight="bold", fontsize=12, color=INK)
ax2.set_ylabel("방향 반전(불안정) 발생 비율 (%)")
ax2.set_ylim(0, 100)
ax2.set_title(f"(B) 방향 반전 발생 비율\n해석모델(n={n_analytic}) vs FEA(n={n_points}, 스윕점 기준)",
              fontweight="bold", color=INK)
ax2.grid(True, linestyle=":", color=GRID, alpha=0.7, axis="y")
for spine in ax2.spines.values():
    spine.set_color(GRID)
ax2.tick_params(colors=MUTED)

fig.suptitle("해석적 물리모델은 접촉힘 근처에서 부호가 뒤집히지만, FEA는 그렇지 않다",
             fontweight="bold", fontsize=14.5, y=0.99)
fig.text(0.5, 0.885,
         f"FEA는 형상/위치/각도를 바꿔가며 실행한 접촉 시뮬레이션 {n_fea_total}건(스윕 {len(groups)}그룹) 중 "
         f"단 1건만 예외였고, 그 1건도 방향반전이 아니라 CalculiX 미수렴(.sta 확인)으로 확인됨 → 실질 반전 0건.\n"
         "이유: 해석모델은 빔 방정식에 점하중을 얹어 풀 뿐이라 좌굴에 가까운 불안정 영역에서 잘못된 부호로도 수렴 가능한 반면, "
         "FEA는 접촉면이 서로 뚫고 들어갈 수 없다는 물리적 제약을 직접 풀어서 이런 반전이 원천적으로 나오기 어려움.",
         ha="center", fontsize=9.3, color=MUTED, linespacing=1.6)
fig.tight_layout(rect=[0, 0, 1, 0.78])

out_path = os.path.join(FEA_DATA_DIR, "fea_vs_analytical_reliability.png")
fig.savefig(out_path, dpi=150, facecolor="#fcfcfb", bbox_inches="tight")
print(f"저장: {out_path}")
print(f"해석모델: n={n_analytic}, 중앙값={median_analytic:.2f}mN, 즉시반전비율={frac_immediate:.1f}%")
print(f"FEA: 스윕그룹={len(groups)}, 스윕점={n_points}, 반전={n_reversal} ({n_reversal/n_points*100:.2f}%)")
