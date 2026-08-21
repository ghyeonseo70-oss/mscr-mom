"""
논문 Fig.3(a)-(d) 스타일 재현: L_M/L = 0(CMSCR, MOM 없음), 0.25, 0.5, 0.75 별로
여러 외부자기장 각도(phi)를 겹쳐 그려서, MOM 유무/위치에 따른 형상 차이를 직접 비교.
"""
import numpy as np
import matplotlib.pyplot as plt
import force_model as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

LM_RATIOS = [0.0, 0.25, 0.5, 0.75]  # L_M/L
PHI_LIST = [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]


def continuation_sweep(L_M, phi_targets):
    """phi=0에서 시작해 1도씩 연속법(continuation)으로 추적하며 다중해 사이 점프를 방지.
    각 스텝은 직전 theta_L을 hint로 넘겨 같은 물리적 가지를 유지한다.
    (전역탐색은 phi가 30도씩 뛸 때 엉뚱한 해로 튈 수 있음 — 논문 Section IV-B의
    다중해 문제 때문에, 특히 K1/K2가 크거나 L_M/L이 클수록 심해짐)
    """
    def step(phi, hint):
        try:
            return fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[], theta_L_hint_deg=hint)
        except RuntimeError:
            return fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])  # 국소 탐색 실패 시 전역 탐색 폴백

    STEP = 5  # deg. 1도씩 하면 정확하지만 계산량이 너무 커서(계단당 ODE 다회 적분) 5도로 절충
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


fig = plt.figure(figsize=(20, 6.7))
gs = fig.add_gridspec(2, 4, height_ratios=[0.26, 1], hspace=0.25, wspace=0.15)

hero = fig.add_subplot(gs[0, :])
hero.axis("off")
hero.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hero.transAxes, facecolor="#e8eefa",
                              edgecolor="#2451A3", linewidth=1.5))
hero.text(0.02, 0.5, "K1, K2, M1", fontsize=28, fontweight="bold", color="#2451A3",
          transform=hero.transAxes, va="center", ha="left")
hero.text(0.19, 0.5, "논문 Fig.3 디지털화 + 최소자승 피팅으로 확보\n"
                     "-> 굽힘강성·자기모멘트 상수를 실측 없이 논문 데이터로부터 역산",
          fontsize=13, fontweight="bold", color="#193A7D", transform=hero.transAxes,
          va="center", ha="left", linespacing=1.4)

axes = [fig.add_subplot(gs[1, i]) for i in range(4)]
for a in axes[1:]:
    a.sharey(axes[0])
colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(PHI_LIST)))

for ax, ratio in zip(axes, LM_RATIOS):
    L_M = max(ratio * fm.L, 0.5)  # L_M=0은 특이점이라 0.5mm로 근사(사실상 MOM 없음/CMSCR)
    trace = continuation_sweep(L_M, PHI_LIST)
    for phi_deg, c in zip(PHI_LIST, colors):
        if phi_deg not in trace:
            continue
        hint = trace[phi_deg]["theta_L_deg"]
        r = fm.solve_shape(L_M=L_M, phi_deg=phi_deg, loads=[], return_curve=True,
                            theta_L_hint_deg=hint)
        ax.plot(r["curve_x_mm"], r["curve_y_mm"], color=c, linewidth=1.8)
        ax.plot(r["x_L"], r["y_L"], "o", color=c, markersize=4)
    ax.plot(0, 0, "ks", markersize=8)
    title = "CMSCR (MOM 없음, L_M/L=0)" if ratio == 0 else f"L_M/L = {ratio}"
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_xlabel("x (mm, 로컬)")
    ax.set_xlim(-30, 100)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.axhline(0, color="gray", linewidth=0.6)

axes[0].set_ylabel("y (mm, 로컬)")

# 컬러바 대신 범례용 텍스트
sm = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(vmin=min(PHI_LIST), vmax=max(PHI_LIST)))
cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.15, aspect=40)
cbar.set_label("외부자기장 방향 φ (deg)")

fig.suptitle("논문 Fig.3(a)-(d) 스타일 재현 - L_M/L에 따른 형상 비교 (φ = -150°~150° 스윕)",
             fontweight="bold", fontsize=14, y=1.005)

out_path = "../../data/force_model/paper_fig3_style.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"저장: {out_path}")
