"""2026-10-07: "힘만 걸기(force-only)" 방식의 사용 가능 범위를 보여주는 그림.
s=80mm(접촉으로는 5전 5패였던 곳)에서 힘 크기를 0.005~20mN까지 스캔한 결과
(_test_force_scan.py). 0.005~0.03mN은 실제 접촉과 99.5% 일치가 확인된 "검증된 범위"
(s=30, depth=0.2~0.4mm 기준), 그 이상은 팁변위가 급격히 커져서 못 믿는 영역,
5mN부터는 계산 자체가 수렴하지 않음(1mN은 성공, 5mN은 실패 - 그 사이 어딘가가 한계)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

CATHETER_LEN_MM = 100.0

# 실측 스캔 결과 (_test_force_scan.py, s=80mm)
F_mN = np.array([0.005, 0.05, 0.25, 1.0])
TIP_MM = np.array([0.9502, 8.8004, 29.8109, 51.1174])

# 검증된 범위(s=30, depth=0.2~0.4mm, 실제 접촉과 99.5% 일치 확인됨)
VALIDATED_LO, VALIDATED_HI = 0.005, 0.03

fig, ax = plt.subplots(figsize=(11, 8))

# 배경 영역 색칠
ax.axvspan(0.003, VALIDATED_HI, color="#2ECC71", alpha=0.15, zorder=0)
ax.axvspan(VALIDATED_HI, 0.05, color="#F1C40F", alpha=0.15, zorder=0)
ax.axvspan(0.05, 1.5, color="#E67E22", alpha=0.12, zorder=0)
ax.axvspan(1.5, 30, color="#C0392B", alpha=0.15, zorder=0)

ax.plot(F_mN, TIP_MM, "o-", color="#2451A3", linewidth=2.5, markersize=10, zorder=5,
        label="힘만 걸기(force-only) 결과 (s=80mm)")

# 5mN 실패 지점 표시
ax.plot(5.0, 65, "x", color="#C0392B", markersize=20, markeredgewidth=4, zorder=6)
ax.annotate("5.0mN: 계산 자체가\n수렴 안 됨 (실패)", (5.0, 65), textcoords="offset points",
            xytext=(15, 10), fontsize=12, color="#C0392B", fontweight="bold")

for f, t in zip(F_mN, TIP_MM):
    pct = t / CATHETER_LEN_MM * 100
    ax.annotate(f"{t:.2f}mm\n({pct:.1f}%)", (f, t), textcoords="offset points",
                xytext=(10, -22) if f < 0.1 else (-55, 10), fontsize=11, fontweight="bold",
                color="#1A2238")

ax.axhline(CATHETER_LEN_MM, color="#555", linestyle=":", linewidth=1.5)
ax.annotate("카테터 전체 길이 (100mm)", (0.004, CATHETER_LEN_MM), textcoords="offset points",
            xytext=(0, 6), fontsize=10.5, color="#555")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.003, 30)
ax.set_ylim(0.3, 150)
# 로그축 음수지수 틱이 Malgun Gothic에서 깨져서(유니코드 마이너스) 직접 라벨 지정
xticks = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
ax.set_xticks(xticks)
ax.set_xticklabels([f"{v:g}" for v in xticks])

ax.set_xlabel("적용한 힘 F (mN, 로그축)", fontsize=13)
ax.set_ylabel("팁 변위 (mm, 로그축)", fontsize=13)
ax.set_title("힘만 걸기(force-only) 방식 — 어디까지 믿을 수 있나 (s=80mm 스캔)",
             fontsize=15, fontweight="bold")

# 영역 라벨
label_y = 0.4
ax.text(0.0115, label_y, "검증됨\n(실측 99.5%\n일치 확인)", ha="center", fontsize=10,
        color="#1E8449", fontweight="bold")
ax.text(0.039, label_y, "미검증\n(0.03~0.05mN)", ha="center", fontsize=10,
        color="#9A7D0A", fontweight="bold")
ax.text(0.27, label_y, "과도함\n(팁이 너무 크게 움직임)", ha="center", fontsize=10,
        color="#AF601A", fontweight="bold")
ax.text(5.5, label_y, "붕괴\n(계산 불가 영역)", ha="center", fontsize=10,
        color="#922B21", fontweight="bold")

ax.legend(loc="upper right", fontsize=11)
ax.grid(True, which="both", linestyle=":", alpha=0.3)

plt.tight_layout()
out_path = "../../data/force_model/force_only_limit_scan.png"
plt.savefig(out_path, dpi=150)
print(f"저장: {out_path}")
