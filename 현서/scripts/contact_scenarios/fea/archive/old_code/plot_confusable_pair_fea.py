"""
'헷갈리는 쌍' FEA 검증 결과 시각화 (지금까지 표로만 보고했던 핵심 발견).
단순모델(빔이론)이 "s=10mm를 세게(16mN) vs s=30mm를 약하게(2mN) 누르면 결과가 거의 같다"고
예측한 걸, 실제 연속탄성체 FEA로 각각 눌러봐서 확인 - 크기 비율은 다르지만 방향은 거의 같아서,
단순모델의 예측(진짜 물리적 한계)이 FEA로도 확인됨을 보여주는 핵심 그림.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")

with open(os.path.join(DATA_DIR, "tip_confusable_test.json"), encoding="utf-8") as f:
    rows = json.load(f)
valid = [r for r in rows if r["tip_shift_mag_mm"] > 1e-6]

C_S10, C_S30 = "#2a78d6", "#e34948"
color = {10.0: C_S10, 30.0: C_S30}

fig = plt.figure(figsize=(15, 5.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 0.9], wspace=0.38)

# ── (A) 힘 vs 팁변위 ──────────────────────────────
ax = fig.add_subplot(gs[0])
for s_val in [10.0, 30.0]:
    pts = sorted([r for r in valid if r["s"] == s_val], key=lambda r: r["F_mag_N"])
    F = [r["F_mag_N"] * 1000 for r in pts]
    shift = [r["tip_shift_mag_mm"] for r in pts]
    ax.plot(F, shift, "o-", color=color[s_val], linewidth=2.2, markersize=9,
            label=f"s={s_val:.0f}mm")
    slope = shift[-1] / F[-1] if F[-1] else 0
    ax.annotate(f"{slope:.1f} mm/mN", (F[-1], shift[-1]), textcoords="offset points",
                xytext=(8, -4), fontsize=10, color=color[s_val], fontweight="bold")
ax.set_xlabel("접촉력 크기 |F| (mN)")
ax.set_ylabel("팁 변위 크기 (mm)")
ax.set_title("(A) FEA 실측: 힘 대비 팁변위 - 위치마다 기울기가 8배 차이", fontweight="bold", fontsize=11.5)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(fontsize=10)

# ── (B) 변위 방향(단위벡터) 비교 ──────────────────────────────
ax = fig.add_subplot(gs[1])
labels = ["x성분", "y성분", "z성분"]
x = np.arange(3)
w = 0.32
for i, s_val in enumerate([10.0, 30.0]):
    pts = [r for r in valid if r["s"] == s_val]
    dirs = []
    for r in pts:
        v = np.array([r["tip_ux_mm"], r["tip_uy_mm"], r["tip_uz_mm"]])
        dirs.append(v / np.linalg.norm(v))
    avg_dir = np.mean(dirs, axis=0)
    ax.bar(x + (i - 0.5) * w, avg_dir, w, color=color[s_val], label=f"s={s_val:.0f}mm", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("변위 방향(단위벡터) 성분")
ax.set_title("(B) 변위 '방향'은 두 위치가 거의 동일", fontweight="bold", fontsize=11.5)
ax.grid(True, axis="y", linestyle=":", alpha=0.5)
ax.legend(fontsize=10)
ax.set_ylim(-0.1, 1.05)

# ── (C) 핵심 결론 ──────────────────────────────
ax = fig.add_subplot(gs[2])
ax.axis("off")
ax.set_ylim(0, 1)
lines_top = [
    (0.98, "핵심 결론", 15, True, "#111"),
    (0.86, "단순모델(빔이론) 예측:", 11.5, True, "#444"),
    (0.79, "  s=10mm, F=16mN", 11, False, "#444"),
    (0.73, "  ~= s=30mm, F=2mN", 11, False, "#444"),
    (0.67, "  (힘 비율 8배 -> 결과 동일)", 11, False, "#444"),
    (0.56, "FEA 실측:", 11.5, True, "#444"),
    (0.49, "  s=10mm 민감도 = 4.2 mm/mN", 11, False, "#444"),
    (0.43, "  s=30mm 민감도 = 35.0 mm/mN", 11, False, "#444"),
    (0.37, "  -> 비율 8.3배", 11, False, "#444"),
]
for yy, txt, fs, bold, col in lines_top:
    ax.text(0.0, yy, txt, fontsize=fs, fontweight="bold" if bold else "normal",
            color=col, transform=ax.transAxes, va="top")

ax.add_patch(plt.Rectangle((0.0, 0.02), 1.0, 0.24, transform=ax.transAxes,
                            facecolor="#fbeae8", edgecolor="#B23A32", linewidth=1.3))
ax.text(0.05, 0.22, "8배 ~= 8.3배", fontsize=15, fontweight="bold", color="#B23A32",
        transform=ax.transAxes, va="top")
ax.text(0.05, 0.14, "단순모델 예측이 FEA로도 확인됨\n-> 진짜 물리적 한계",
        fontsize=10.5, color="#B23A32", transform=ax.transAxes, va="top", linespacing=1.5, fontweight="bold")

fig.suptitle("FEA 검증: \"헷갈리는 쌍\"(s=10mm 세게 vs s=30mm 약하게)이 실제로도 구별 안 됨",
             fontweight="bold", fontsize=14.5, y=1.03)

out_path = os.path.join(DATA_DIR, "confusable_pair_fea_verification.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"저장: {out_path}")
