"""fea_contact_sweep.json(접촉위치 x 누르는깊이 FEA 결과)을 그래프로 시각화."""
import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea", "fea_contact_sweep.json")
with open(data_path, encoding="utf-8") as f:
    rows = json.load(f)

# z별로 묶기
by_z = {}
for r in rows:
    by_z.setdefault(r["contact_z_mm"], []).append(r)

L_TUBE = max(r["contact_z_mm"] for r in rows) / 0.75 if rows else 100.0  # 대략 추정(25/50/75% 지점 가정)

COLORS = {}
palette = ["#2a78d6", "#e34948", "#2e7d32", "#8e44ad", "#e08e0b"]
z_sorted = sorted(by_z.keys())
for i, z in enumerate(z_sorted):
    COLORS[z] = palette[i % len(palette)]

fig, ax = plt.subplots(figsize=(7.5, 6))
for z in z_sorted:
    group = sorted(by_z[z], key=lambda r: r["push_depth_mm"])
    x = [r["push_depth_mm"] for r in group]
    y = [abs(r["Fx_total_N"]) * 1000 for r in group]  # N -> mN
    pct = z / L_TUBE * 100
    label = f"z={z:.1f}mm (튜브 길이의 {pct:.0f}% 지점)"
    ax.plot(x, y, "o-", color=COLORS[z], linewidth=2.2, markersize=8, label=label)

ax.set_xlabel("누르는 깊이 push depth (mm)")
ax.set_ylabel("총 접촉력 |Fx| (mN)")
ax.set_yscale("log")
# Malgun Gothic 폰트가 지수표기의 유니코드 마이너스(U+2212) 글리프가 없어서 깨지는 문제 회피:
# 지수표기 대신 그냥 소수 표기(0.01, 0.1, 1 ...)로 라벨을 바꿈
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_title(f"CalculiX 접촉해석: 접촉위치 x 깊이별 접촉력\n(튜브 전체길이 {L_TUBE:.0f}mm 기준)",
             fontweight="bold")
ax.grid(True, which="both", linestyle=":", alpha=0.5)
ax.legend(fontsize=9, loc="upper left")

plt.tight_layout()
out_path = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea", "fea_contact_sweep.png")
plt.savefig(out_path, dpi=150)
print(f"저장: {out_path}")
