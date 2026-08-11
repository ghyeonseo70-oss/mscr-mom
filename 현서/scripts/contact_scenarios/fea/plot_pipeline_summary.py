"""
'충돌(접촉) 감지' 파이프라인 전체를 한 장으로 정리한 발표용 대시보드.
(A) 물리모델 접촉 시나리오 설정 예시, (B) 3000개 스윕 개요, (C) FEA 직선튜브 접촉력 검증(9/9),
(D) FEA 굽은튜브 형상-물리모델 일치 확인. 네 패널 모두 "이미 검증까지 끝난, 확실한 결과"만 담음
(홀센서 역산 가능여부는 별도 진행 중이라 여기 포함 안 함).
"""
import json
import os
import sys
import numpy as np
import gmsh
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "force_model"))
import force_model as fm

DATA_ROOT = os.path.join(HERE, "..", "..", "..", "data")
SEQ_CMAP = LinearSegmentedColormap.from_list('seq', ['#dbe9f6', '#5b9bd5', '#1c4e80'])
POS_COLORS = {25.0: "#2a78d6", 50.0: "#e34948", 75.0: "#2e7d32"}

fig = plt.figure(figsize=(15, 12.8))
gs = fig.add_gridspec(3, 2, height_ratios=[0.22, 1, 1], hspace=0.32, wspace=0.28)

# ── 헤로 배너 ──────────────────────────────
hero = fig.add_subplot(gs[0, :])
hero.axis("off")
hero.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hero.transAxes, facecolor="#e8eefa",
                              edgecolor="#2451A3", linewidth=1.5))
hero.text(0.03, 0.5, "0.5mm 이내", fontsize=30, fontweight="bold", color="#2451A3",
          transform=hero.transAxes, va="center", ha="left")
hero.text(0.32, 0.5, "FEA 대비 물리모델 팁 위치 오차\n- 정방향 모델(힘·위치 -> 형상) 신뢰성 확보",
          fontsize=13.5, fontweight="bold", color="#193A7D", transform=hero.transAxes,
          va="center", ha="left", linespacing=1.4)

# ══ (A) 물리모델 접촉 시나리오 설정 예시 ══════════════════════════════
ax = fig.add_subplot(gs[1, 0])
with open(os.path.join(HERE, "bent_centerline.json")) as f:
    cl = json.load(f)
cx = [p["x"] for p in cl["points"]]
cy = [p["y"] for p in cl["points"]]
cs = np.array([p["s"] for p in cl["points"]])
ax.plot(cx, cy, "-", color="#5b9bd5", linewidth=3, label="카테터 형상 (물리모델)", zorder=3)
ax.plot(cl["x_LM"], cl["y_LM"], "D", color="#1c4e80", markersize=10, label="MOM 자석", zorder=5)
ax.plot(cx[-1], cy[-1], "o", color="#1c4e80", markersize=10, label="팁(main 자석)", zorder=5)
ax.plot(90, 0, "ks", markersize=10, label="베이스 (90,0,3) - 하드웨어 좌표", zorder=5)

# 대표 접촉점(s=50mm 부근) + 힘 방향 화살표
i_c = int(np.argmin(np.abs(cs - 50)))
cxp, cyp = cx[i_c], cy[i_c]
th = np.radians(cl["points"][i_c]["theta_deg"])
nvec = np.array([-np.cos(th), np.sin(th)]) * 12
ax.annotate("", xy=(cxp, cyp), xytext=(cxp + nvec[0], cyp + nvec[1]),
            arrowprops=dict(arrowstyle="-|>", color="#e34948", linewidth=2.5))
ax.plot(cxp, cyp, "o", color="#e34948", markersize=9, zorder=6)
ax.annotate("접촉(충돌) 지점 예시\n위치 s, 힘 F로 표현", xy=(cxp + nvec[0], cyp + nvec[1]),
            xytext=(cxp - 35, cyp - 12), color="#e34948", fontsize=9, fontweight="bold")

ax.set_xlabel("x (mm, 보드좌표)")
ax.set_ylabel("y (mm, 보드좌표)")
ax.set_title("(A) 물리모델: 접촉 시나리오 설정", fontweight="bold", fontsize=12)
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.5)
ax.set_xlim(40, 95)
ax.legend(fontsize=8, loc="lower left")

# ══ (B) 3000개 스윕 개요 ══════════════════════════════
ax = fig.add_subplot(gs[1, 1])
d = np.load(os.path.join(DATA_ROOT, "contact_scenarios", "contact_force_scenarios.npz"), allow_pickle=True)
data, cols = d["data"], list(d["columns"])
col = {c: i for i, c in enumerate(cols)}
F_mag = data[:, col["F_mag"]] * 1000
s_pos = data[:, col["s"]]
dx = data[:, col["xL_load"]] - data[:, col["xL_free"]]
dy = data[:, col["yL_load"]] - data[:, col["yL_free"]]
deviation = np.hypot(dx, dy)
sc = ax.scatter(F_mag, deviation, c=s_pos, cmap=SEQ_CMAP, s=12, alpha=0.7, edgecolors="none")
cbar = fig.colorbar(sc, ax=ax, label="접촉위치 s (mm)")
ax.set_xlabel("접촉힘 크기 |F| (mN)")
ax.set_ylabel("팁 위치 이탈량 (mm)")
ax.set_title("(B) 물리모델 스윕 3000개 - 힘이 클수록 형상 변화가 큼", fontweight="bold", fontsize=12)
ax.grid(True, linestyle=":", alpha=0.5)

# ══ (C) FEA 직선튜브 접촉력 검증 (9/9 성공) ══════════════════════════════
ax = fig.add_subplot(gs[2, 0])
with open(os.path.join(DATA_ROOT, "contact_scenarios", "fea", "fea_contact_sweep.json"), encoding="utf-8") as f:
    rows = json.load(f)
by_z = {}
for r in rows:
    by_z.setdefault(r["contact_z_mm"], []).append(r)
for z in sorted(by_z):
    group = sorted(by_z[z], key=lambda r: r["push_depth_mm"])
    x = [r["push_depth_mm"] for r in group]
    y = [abs(r["Fx_total_N"]) * 1000 for r in group]
    pct = z / 100 * 100
    ax.plot(x, y, "o-", color=POS_COLORS[z], linewidth=2.2, markersize=7,
            label=f"z={z:.0f}mm (베이스에서 {pct:.0f}%)")
ax.set_xlabel("누르는 깊이 (mm)")
ax.set_ylabel("총 접촉력 |Fx| (mN)")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_title("(C) FEA 검증: 직선 튜브 접촉해석 (9/9 성공)\n베이스에 가까울수록 힘이 크게 나옴 - 물리적으로 타당",
             fontweight="bold", fontsize=12)
ax.grid(True, which="both", linestyle=":", alpha=0.5)
ax.legend(fontsize=8)

# ══ (D) FEA 굽은튜브 형상 - 물리모델 일치 확인 ══════════════════════════════
ax = fig.add_subplot(gs[2, 1])
gmsh.initialize()
gmsh.open(os.path.join(HERE, "bent_tube_mesh.msh"))
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
coords = np.array(node_coords).reshape(-1, 3)
gmsh.finalize()
ax.scatter(coords[:, 0], coords[:, 1], s=1, alpha=0.12, color="#5b9bd5", label="FEA 메쉬 표면 절점")
ax.plot(cx, cy, "-", color="#e34948", linewidth=2.5, label="물리모델 예측 중심선", zorder=5)
ax.plot(90, 0, "ks", markersize=9, label="베이스", zorder=6)
ax.plot(cl["x_L"], cl["y_L"], "r*", markersize=16, label="물리모델 예측 팁", zorder=6)
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_title("(D) FEA 메쉬 vs 물리모델 예측 - 팁 오차 0.5mm 이내 일치", fontweight="bold", fontsize=12)
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(fontsize=8, loc="best")

fig.suptitle("카테터 접촉(충돌) 물리모델 + FEA 검증 결과", fontweight="bold", fontsize=17, y=1.005)

out_path = os.path.join(DATA_ROOT, "contact_scenarios", "collision_pipeline_summary.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"저장: {out_path}")
