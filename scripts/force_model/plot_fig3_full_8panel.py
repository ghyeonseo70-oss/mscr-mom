"""논문 Fig.3(a)-(h) 전체 재현 (fig_embed_0.jpeg 스타일 그대로):
(a)-(d) L_M/L=0,0.25,0.5,0.75 xy평면 형상, (e)-(h) 같은 조건의 yL-thetaL workspace 그래프
(phi=0,±30,±60,±90,±120,±150 각 점을 별표로, 점선으로 연결 - MOM 있을 때(b~d,f~h)는
phi=0이 다중해라 논문처럼 위/아래 두 갈래로 끊어서 그림)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import force_model as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

LM_RATIOS = [0.0, 0.25, 0.5, 0.75]
PHI_LIST = [0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150]
PHI_COLORS = {0: "#0072BD", 30: "#D95319", 60: "#EDB120", 90: "#7E2F8E", 120: "#77AC30", 150: "#4DBEEE"}


def continuation_sweep(L_M, phi_targets):
    """phi=0에서 시작해 양방향으로 5도씩 연속법 추적."""
    def step(phi, hint):
        try:
            return fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[], theta_L_hint_deg=hint, window_deg=40.0)
        except RuntimeError:
            return fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])

    STEP = 5
    r0 = fm.solve_shape(L_M=L_M, phi_deg=0, loads=[])
    results = {0: r0}
    hint = r0["theta_L_deg"]
    for phi in range(STEP, max(phi_targets) + 1, STEP):
        r = step(phi, hint)
        results[phi] = r
        hint = r["theta_L_deg"]
    hint = r0["theta_L_deg"]
    for phi in range(-STEP, min(phi_targets) - 1, -STEP):
        r = step(phi, hint)
        results[phi] = r
        hint = r["theta_L_deg"]
    return results


fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.2)
axes_top = [fig.add_subplot(gs[0, i]) for i in range(4)]
axes_bot = [fig.add_subplot(gs[1, i]) for i in range(4)]

for col, ratio in enumerate(LM_RATIOS):
    ax_top, ax_bot = axes_top[col], axes_bot[col]
    L_M = max(ratio * fm.L, 0.5)
    trace = continuation_sweep(L_M, PHI_LIST)

    # (a)-(d): xy평면 곡선
    for phi_deg in PHI_LIST:
        if phi_deg not in trace:
            continue
        c = PHI_COLORS[abs(phi_deg)]
        hint = trace[phi_deg]["theta_L_deg"]
        r = fm.solve_shape(L_M=L_M, phi_deg=phi_deg, loads=[], return_curve=True,
                            theta_L_hint_deg=hint, window_deg=40.0)
        ax_top.plot(r["curve_x_mm"] / 10, r["curve_y_mm"] / 10, color=c, linewidth=1.8)
    ax_top.plot(0, 0, "ks", markersize=6)
    title = "$L_M/L = 0$" if ratio == 0 else f"$L_M/L = {ratio}$"
    ax_top.text(0.97, 0.95, title, transform=ax_top.transAxes, ha="right", va="top", fontsize=12)
    ax_top.set_xlabel("$x$(cm)")
    ax_top.set_xlim(-2, 11)
    ax_top.set_ylim(-8, 8)
    ax_top.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax_top.set_title("(" + "abcd"[col] + ")", loc="left", fontweight="bold", fontsize=13)

    # (e)-(h): yL-thetaL workspace, phi별 점 + 점선 연결
    # 0 포함해서 phi 오름차순으로 이었을 때 "위쪽(phi>0)"과 "아래쪽(phi<0)" 두 갈래.
    # CMSCR(ratio=0)은 phi=0이 정상해라 하나로 안 끊고 전부 이어줌.
    pos_phis = sorted([p for p in PHI_LIST if p >= 0])
    neg_phis = sorted([p for p in PHI_LIST if p <= 0], reverse=True)
    for chain in (pos_phis, neg_phis):
        xs = [trace[p]["y_L"] / 10 for p in chain if p in trace]
        ys = [trace[p]["theta_L_deg"] for p in chain if p in trace]
        ax_bot.plot(xs, ys, "k--", linewidth=1, zorder=1)
    for phi_deg in PHI_LIST:
        if phi_deg not in trace:
            continue
        c = PHI_COLORS[abs(phi_deg)]
        r = trace[phi_deg]
        ax_bot.plot(r["y_L"] / 10, r["theta_L_deg"], marker="*", color=c, markersize=14,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=3)
    ax_bot.text(0.97, 0.95, title, transform=ax_bot.transAxes, ha="right", va="top", fontsize=12)
    ax_bot.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_bot.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax_bot.set_xlabel("$y_L$(cm)")
    ax_bot.set_xlim(-10, 10)
    ax_bot.set_ylim(-180, 180)
    ax_bot.set_yticks([-150, -100, -50, 0, 50, 100, 150])
    ax_bot.set_title("(" + "efgh"[col] + ")", loc="left", fontweight="bold", fontsize=13)

axes_top[0].set_ylabel("$y$(cm)")
axes_bot[0].set_ylabel(r"$\theta_L$(°)")

legend_handles = [Line2D([0], [0], color=PHI_COLORS[p], lw=3, label=f"$\\varphi=\\pm{p}$" if p else "$\\varphi=0$")
                  for p in [0, 30, 60, 90, 120, 150]]
axes_top[0].legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.9)

fig.suptitle("논문 Fig.3(a)-(h) 전체 재현 - force_model.py (3구간, 논문 식 그대로)",
             fontweight="bold", fontsize=15, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
out_path = "../../data/force_model/fig3_full_8panel_reproduction.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"저장: {out_path}")
